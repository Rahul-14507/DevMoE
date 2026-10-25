document.addEventListener('DOMContentLoaded', () => {
  // Preset prompts
  const PRESETS = {
    easy_regex: "Write a regex pattern in Python to extract all valid email addresses from an unstructured string and provide strict type hints.",
    med_api: "Create a FastAPI endpoint that queries a PostgreSQL database for active users with pagination, includes foreign key joins, and has a pytest mock test suite.",
    complex_concurrency: "Design a distributed consensus mechanism using Raft algorithm in Rust. Handle leader election, log replication, mutex deadlock prevention, and asynchronous network partition recovery across multiple files.",
    easy_syntax: "Fix the syntax error in this Python function:\ndef calculate_total(items)\n    total = 0\n    for item in items\n        total += item['price']\n    return total"
  };

  // State elements
  const ramValue = document.getElementById('ramValue');
  const vramValue = document.getElementById('vramValue');
  const ollamaStatusLabel = document.getElementById('ollamaStatusLabel');
  const ollamaStatusPill = document.getElementById('ollamaStatusPill');

  const promptInput = document.getElementById('promptInput');
  const btnAnalyzeRoute = document.getElementById('btnAnalyzeRoute');
  const btnExecuteCompletion = document.getElementById('btnExecuteCompletion');

  const scoreNum = document.getElementById('scoreNum');
  const scoreTitle = document.getElementById('scoreTitle');
  const scoreSummary = document.getElementById('scoreSummary');
  const scoreProgress = document.getElementById('scoreProgress');
  const tierBadge = document.getElementById('tierBadge');

  const destProvider = document.getElementById('destProvider');
  const destModel = document.getElementById('destModel');
  const decRationale = document.getElementById('decRationale');
  const fallbackRow = document.getElementById('fallbackRow');
  const decFallbackInfo = document.getElementById('decFallbackInfo');
  const signalsList = document.getElementById('signalsList');

  const outputCard = document.getElementById('outputCard');
  const completionContent = document.getElementById('completionContent');
  const outputProviderBadge = document.getElementById('outputProviderBadge');
  const outputLatency = document.getElementById('outputLatency');
  const outputTokens = document.getElementById('outputTokens');

  const recommendationsTableBody = document.getElementById('recommendationsTableBody');
  const fallbackHeroCard = document.getElementById('fallbackHeroCard');
  const fallbackTargetModel = document.getElementById('fallbackTargetModel');
  const fallbackTargetReason = document.getElementById('fallbackTargetReason');

  const providersGrid = document.getElementById('providersGrid');
  const failoverEventsList = document.getElementById('failoverEventsList');

  // Pipeline step nodes
  const stepNodes = [
    document.getElementById('stepPrompt'),
    document.getElementById('stepComplexity'),
    document.getElementById('stepRouting'),
    document.getElementById('stepFailover')
  ];

  // Tab switching
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPane = document.getElementById(`tab-${tab.dataset.tab}`);
      if (targetPane) targetPane.classList.add('active');
    });
  });

  // Preset buttons
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const type = btn.dataset.type;
      if (PRESETS[type]) {
        promptInput.value = PRESETS[type];
        // Trigger quick analysis
        analyzeRouting();
      }
    });
  });

  // Fetch initial system status & recommendations
  async function loadSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) return;
      const data = await res.json();

      // RAM
      const ramTotal = data.hardware.ram_total_gb;
      const ramAvail = data.hardware.ram_available_gb;
      ramValue.textContent = `${ramAvail} GB / ${ramTotal} GB`;

      // VRAM
      if (data.hardware.has_gpu && data.hardware.gpus.length > 0) {
        const gpu = data.hardware.gpus[0];
        vramValue.textContent = `${gpu.vram_free_gb} GB (${gpu.name.slice(0, 16)}...)`;
      } else {
        vramValue.textContent = "CPU-RAM Unified";
      }

      // Ollama status
      if (data.local_engine.is_alive) {
        ollamaStatusLabel.textContent = `Ollama (${data.local_engine.installed_models.length} pulled)`;
        ollamaStatusPill.querySelector('.status-dot').style.backgroundColor = '#10b981';
      } else {
        ollamaStatusLabel.textContent = "Ollama (Offline/Mock)";
        ollamaStatusPill.querySelector('.status-dot').style.backgroundColor = '#f59e0b';
      }

      // Update Fallback Target
      if (fallbackTargetModel) {
        fallbackTargetModel.textContent = data.local_engine.highest_param_name;
        fallbackTargetReason.textContent = `Auto-fallback tag: ${data.local_engine.highest_param_fit} based on ${ramAvail} GB host memory.`;
      }

      // Render Providers Grid
      renderProvidersGrid(data.providers, data.quota_and_circuit_breakers);

      // Render Failover events
      renderFailoverLogs(data.quota_and_circuit_breakers.recent_events);

    } catch (err) {
      console.warn("Failed to load /api/status:", err);
    }
  }

  async function loadRecommendations() {
    try {
      const res = await fetch('/api/recommendations');
      if (!res.ok) return;
      const data = await res.json();

      if (!recommendationsTableBody) return;
      recommendationsTableBody.innerHTML = '';

      data.recommendations.forEach(m => {
        const tr = document.createElement('tr');
        const statusClass = `status-${m.status}`;

        tr.innerHTML = `
          <td><strong>${m.name}</strong></td>
          <td><span class="badge info">${m.tag.split(':')[0]}</span></td>
          <td>${m.param_size_b} B</td>
          <td>${m.required_ram_gb} GB</td>
          <td>${m.context_window_k}k</td>
          <td><span class="badge-status ${statusClass}">${m.status.toUpperCase()}</span></td>
          <td>${m.est_headroom_gb > 0 ? '+' : ''}${m.est_headroom_gb} GB</td>
          <td>
            <button class="btn-copy-tag" data-cmd="${m.pull_command}">Copy Command</button>
          </td>
        `;
        recommendationsTableBody.appendChild(tr);
      });

      // Add copy listeners
      document.querySelectorAll('.btn-copy-tag').forEach(b => {
        b.addEventListener('click', (e) => {
          navigator.clipboard.writeText(b.dataset.cmd);
          const orig = b.textContent;
          b.textContent = 'Copied!';
          setTimeout(() => b.textContent = orig, 1500);
        });
      });

    } catch (err) {
      console.warn("Failed to load /api/recommendations:", err);
    }
  }

  function renderProvidersGrid(providers, quotaData) {
    if (!providersGrid) return;
    providersGrid.innerHTML = '';

    const providerNames = {
      gemini_gemma_cloud: { name: "Google AI Studio", desc: "Gemini 1.5 Pro, Flash & Gemma 2 27B", tier: "Curated Free" },
      ollama_local: { name: "Local Ollama Engine", desc: "Autonomous local MoE on host hardware", tier: "Local MoE" },
      groq_cloud: { name: "Groq Cloud Free Tier", desc: "Qwen 2.5 Coder 32B & Llama 3.3 70B", tier: "Curated Free" },
      cerebras_cloud: { name: "Cerebras Cloud Free Tier", desc: "Llama 3.1 70B Ultra-Fast", tier: "Curated Free" },
      anthropic_claude: { name: "Anthropic Claude", desc: "Claude 3.5 Sonnet Deep Reasoning", tier: "Frontier Cloud" },
      openai_cloud: { name: "OpenAI / GitHub Models", desc: "GPT-4o & GPT-4o-mini", tier: "Frontier Cloud" }
    };

    Object.entries(providers).forEach(([pid, p]) => {
      const meta = providerNames[pid] || { name: pid, desc: "", tier: p.is_free_tier ? "Free Tier" : "Frontier" };
      const q = (quotaData.providers && quotaData.providers[pid]) || {};
      
      const card = document.createElement('div');
      card.className = 'provider-card';

      const isReady = p.configured && p.is_available;
      const statusBadge = isReady 
        ? '<span class="badge tier-easy">Ready</span>' 
        : (p.configured ? '<span class="badge warning">Cooldown</span>' : '<span class="badge">Unconfigured</span>');

      card.innerHTML = `
        <div class="provider-header">
          <div>
            <div class="provider-name">${meta.name}</div>
            <div class="text-muted" style="font-size:0.75rem">${meta.desc}</div>
          </div>
          ${statusBadge}
        </div>
        <div style="font-size:0.74rem; color:#60a5fa; margin-top:4px;">Tier: ${meta.tier}</div>
        <div class="provider-stats">
          <span>Reqs: <b>${q.requests || 0}</b></span>
          <span>Success: <b>${q.success || 0}</b></span>
          <span>Tokens: <b>${(q.prompt_tokens || 0) + (q.completion_tokens || 0)}</b></span>
          <span>Rate Limits: <b>${q.rate_limits || 0}</b></span>
        </div>
      `;
      providersGrid.appendChild(card);
    });
  }

  function renderFailoverLogs(events) {
    if (!failoverEventsList) return;
    if (!events || events.length === 0) {
      failoverEventsList.innerHTML = '<p class="text-muted">No rate limits or failover events recorded. All providers operational.</p>';
      return;
    }

    failoverEventsList.innerHTML = events.map(ev => {
      const timeStr = new Date(ev.timestamp * 1000).toLocaleTimeString();
      return `<div style="padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
        <span style="color:#f59e0b;">[${timeStr}]</span> 
        <strong>HTTP ${ev.status_code}</strong> on <code>${ev.provider_id}</code>. 
        Failover cascade initiated (cooldown ${ev.cooldown_seconds}s).
      </div>`;
    }).join('');
  }

  // Analyze Route
  async function analyzeRouting() {
    const text = promptInput.value.trim();
    if (!text) {
      alert("Please enter a prompt to analyze.");
      return;
    }

    // Animate pipeline
    stepNodes.forEach(node => node.classList.remove('active'));
    stepNodes[0].classList.add('active');
    setTimeout(() => stepNodes[1].classList.add('active'), 150);

    btnAnalyzeRoute.disabled = true;
    btnAnalyzeRoute.textContent = "Analyzing...";

    try {
      const res = await fetch('/api/test-route', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      stepNodes[2].classList.add('active');
      if (data.is_air_gapped) stepNodes[3].classList.add('active');

      // Update score & UI
      scoreNum.textContent = data.score;
      scoreProgress.style.width = `${data.score}%`;

      // Tier styling
      tierBadge.className = `badge tier-${data.tier}`;
      tierBadge.textContent = `${data.tier.toUpperCase()} COMPLEXITY`;

      scoreTitle.textContent = `${data.tier.toUpperCase()} Complexity Task`;
      scoreSummary.textContent = `Estimated tokens: ~${data.estimated_tokens} | Code blocks: ${data.code_block_count}`;

      // Destination
      destProvider.textContent = data.target_provider;
      destModel.textContent = data.target_model;
      decRationale.textContent = data.reason;

      // Signals
      signalsList.innerHTML = '';
      if (data.signals && data.signals.length > 0) {
        data.signals.forEach(sig => {
          const chip = document.createElement('span');
          chip.className = 'signal-tag';
          chip.textContent = sig;
          signalsList.appendChild(chip);
        });
      } else {
        signalsList.innerHTML = '<span class="signal-tag empty">No specific keywords triggered</span>';
      }

    } catch (err) {
      alert(`Routing analysis failed: ${err.message}`);
    } finally {
      btnAnalyzeRoute.disabled = false;
      btnAnalyzeRoute.textContent = "Analyze & Test Route";
    }
  }

  // Execute Completion
  async function executeCompletion() {
    const text = promptInput.value.trim();
    if (!text) {
      alert("Please enter a prompt to execute.");
      return;
    }

    btnExecuteCompletion.disabled = true;
    btnExecuteCompletion.textContent = "Generating...";
    outputCard.style.display = 'block';
    completionContent.textContent = "Streaming/Generating completion from routed model...";

    try {
      const startTime = performance.now();
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: "devmoe-auto",
          messages: [{ role: "user", content: text }]
        })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const elapsed = Math.round(performance.now() - startTime);

      const msg = data.choices[0].message.content;
      completionContent.textContent = msg;

      outputProviderBadge.textContent = `${data.model} (${data._devmoe_meta.executed_provider})`;
      outputLatency.textContent = `Latency: ${data._devmoe_meta.latency_ms || elapsed} ms`;
      outputTokens.textContent = `Tokens: ${data.usage.total_tokens}`;

      if (data._devmoe_meta.was_fallback) {
        fallbackRow.style.display = 'flex';
        decFallbackInfo.textContent = data._devmoe_meta.fallback_reason || "Failover triggered";
      } else {
        fallbackRow.style.display = 'none';
      }

      // Refresh status after execution
      loadSystemStatus();

    } catch (err) {
      completionContent.textContent = `Execution failed: ${err.message}`;
    } finally {
      btnExecuteCompletion.disabled = false;
      btnExecuteCompletion.textContent = "Execute Completion";
    }
  }

  btnAnalyzeRoute.addEventListener('click', analyzeRouting);
  btnExecuteCompletion.addEventListener('click', executeCompletion);

  // Initialize
  loadSystemStatus();
  loadRecommendations();

  // Polling every 10 seconds for real-time memory & provider telemetry
  setInterval(loadSystemStatus, 10000);
});
