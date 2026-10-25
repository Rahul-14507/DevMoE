import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Tuple


class TaskTier(str, Enum):
    EASY = "easy"          # Routine tasks -> Zero-cost local Ollama
    MEDIUM = "medium"      # Standard development -> Curated Free Cloud (Gemma / Flash / Groq)
    COMPLEX = "complex"    # Architectural, security, deep reasoning -> Cloud Frontier / Gemma 27B


@dataclass
class ComplexityEvaluation:
    score: int                           # 0 - 100
    tier: TaskTier
    signals: List[str] = field(default_factory=list)
    estimated_tokens: int = 0
    code_block_count: int = 0
    target_recommendation: str = "local" # "local", "cloud_free", "cloud_frontier"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "tier": self.tier.value,
            "signals": self.signals,
            "estimated_tokens": self.estimated_tokens,
            "code_block_count": self.code_block_count,
            "target_recommendation": self.target_recommendation,
        }


# High complexity indicators (+20 to +30 points each)
COMPLEX_PATTERNS = [
    (r"\b(architect(ure|ural)|system design|microservices?|scalab(le|ility))\b", "System architecture & scalability", 25),
    (r"\b(concurren(cy|t)|mutex|deadlock|race condition|thread[- ]safe|distributed)\b", "Concurrency & distributed systems", 30),
    (r"\b(security audit|vulnerabilit(y|ies)|exploit|sql injection|xss|csrf|buffer overflow)\b", "Security audit & exploit analysis", 30),
    (r"\b(memory leak|heap dump|valgrind|profiling|gc pause|o(om|ut of memory))\b", "Low-level memory & resource debugging", 25),
    (r"\b(compiler|ast|bytecode|parser generator|lexer|formal verification)\b", "Compiler & AST transformation", 25),
    (r"\b(cryptograph(y|ic)|zero knowledge|zk-snark|elliptic curve|aes|rsa)\b", "Advanced cryptography", 25),
    (r"\b(refactor across (multiple|entire) files|multi-file|codebase migration)\b", "Multi-file architectural refactoring", 30),
    (r"\b(complex algorithm|dynamic programming|np-hard|branch and bound)\b", "Deep algorithmic reasoning", 20),
]

# Medium complexity indicators (+10 to +15 points)
MEDIUM_PATTERNS = [
    (r"\b(refactor|optimize|cleanup|dry principle)\b", "Function refactoring & optimization", 12),
    (r"\b(api endpoint|rest api|graphql|fastapi|express|controller)\b", "API & web backend design", 12),
    (r"\b(unit tests?|integration test|pytest|jest|mocking|test suite)\b", "Comprehensive testing & mocks", 12),
    (r"\b(database schema|foreign key|join|indexing|sql query|migration)\b", "Database schema & query design", 14),
    (r"\b(async|await|promise|event loop|coroutine)\b", "Asynchronous logic", 12),
    (r"\b(dockerfile|ci/cd|kubernetes|helm|github actions)\b", "DevOps & deployment setup", 12),
]

# Low complexity indicators (-10 to -20 points)
EASY_PATTERNS = [
    (r"\b(regex|regular expression|regex pattern)\b", "Regex generation/match", -15),
    (r"\b(docstring|type annotations?|type hint|comment this)\b", "Documentation & type annotations", -20),
    (r"\b(syntax error|typo|indentation error|semicolon)\b", "Simple syntax fix", -20),
    (r"\b(boilerplate|getter|setter|dataclass|json to)\b", "Boilerplate & schema transformation", -15),
    (r"\b(convert (this )?(curl|json|yaml|xml) to)\b", "Format conversion", -15),
    (r"\b(hello world|fizzbuzz|simple loop|basic function)\b", "Basic introductory coding", -25),
]


class CodeComplexityClassifier:
    """Intelligent evaluator that scores prompt complexity for software engineering tasks."""

    def __init__(self, easy_threshold: int = 35, medium_threshold: int = 70):
        self.easy_threshold = easy_threshold
        self.medium_threshold = medium_threshold

    def evaluate(self, prompt: str, system_prompt: str = "") -> ComplexityEvaluation:
        full_text = f"{system_prompt}\n{prompt}".strip()
        lower_text = full_text.lower()
        signals: List[str] = []

        # Token length approximation (roughly 4 characters per token in code/text)
        token_count = max(1, len(full_text) // 4)

        # Baseline score depends on length
        if token_count < 120:
            base_score = 15
            signals.append(f"Short prompt (~{token_count} tokens)")
        elif token_count < 400:
            base_score = 30
            signals.append(f"Standard prompt (~{token_count} tokens)")
        elif token_count < 1500:
            base_score = 50
            signals.append(f"Substantial context (~{token_count} tokens)")
        else:
            base_score = 75
            signals.append(f"Large context / codebase context (~{token_count} tokens)")

        # Count code blocks
        code_blocks = re.findall(r"```[\s\S]*?```", full_text)
        code_block_count = len(code_blocks)
        if code_block_count >= 3:
            base_score += 15
            signals.append(f"Multiple code files/blocks provided ({code_block_count} blocks)")
        elif code_block_count >= 1:
            base_score += 5
            signals.append("Embedded code block present")

        # Scan High Complexity Patterns
        for regex, desc, weight in COMPLEX_PATTERNS:
            if re.search(regex, lower_text):
                base_score += weight
                signals.append(f"+{weight} pts: {desc}")

        # Scan Medium Complexity Patterns
        for regex, desc, weight in MEDIUM_PATTERNS:
            if re.search(regex, lower_text):
                base_score += weight
                signals.append(f"+{weight} pts: {desc}")

        # Scan Low Complexity Patterns (Reduces score)
        for regex, desc, weight in EASY_PATTERNS:
            if re.search(regex, lower_text):
                base_score += weight  # weight is negative
                signals.append(f"{weight} pts: {desc}")

        # Clamp between 5 and 100
        final_score = max(5, min(100, base_score))

        # Assign tier
        if final_score <= self.easy_threshold:
            tier = TaskTier.EASY
            target = "local"
        elif final_score <= self.medium_threshold:
            tier = TaskTier.MEDIUM
            target = "cloud_free"
        else:
            tier = TaskTier.COMPLEX
            target = "cloud_frontier"

        return ComplexityEvaluation(
            score=final_score,
            tier=tier,
            signals=signals,
            estimated_tokens=token_count,
            code_block_count=code_block_count,
            target_recommendation=target,
        )
