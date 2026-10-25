from devmoe.router.classifier import CodeComplexityClassifier, TaskTier


def test_classifier_easy():
    classifier = CodeComplexityClassifier(easy_threshold=35, medium_threshold=70)
    eval_res = classifier.evaluate("Write a regex pattern to extract dates and add type annotations")
    assert eval_res.tier == TaskTier.EASY
    assert eval_res.score <= 35
    assert eval_res.target_recommendation == "local"


def test_classifier_medium():
    classifier = CodeComplexityClassifier(easy_threshold=35, medium_threshold=70)
    prompt = """
    Create a FastAPI rest api endpoint that connects to PostgreSQL database schema.
    Write pytest unit tests with mock database sessions.
    ```python
    @app.get("/users")
    def list_users():
        pass
    ```
    """
    eval_res = classifier.evaluate(prompt)
    assert eval_res.tier in (TaskTier.MEDIUM, TaskTier.COMPLEX)
    assert eval_res.score > 35


def test_classifier_complex():
    classifier = CodeComplexityClassifier(easy_threshold=35, medium_threshold=70)
    prompt = """
    Design a distributed system architecture with Raft consensus protocol.
    Handle mutex concurrency, deadlock prevention, Byzantine fault tolerance,
    and memory leak profiling across multiple files in the codebase.
    """
    eval_res = classifier.evaluate(prompt)
    assert eval_res.tier == TaskTier.COMPLEX
    assert eval_res.score >= 70
    assert eval_res.target_recommendation == "cloud_frontier"
