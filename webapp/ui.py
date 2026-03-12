"""Self-contained HTML UI for the inference verification API."""


def get_ui_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TEE Inference Verification</title>
<style>
  :root {
    --safe: #22c55e; --suspicious: #f59e0b; --dangerous: #ef4444;
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8;
    --safe-bg: rgba(34,197,94,0.18); --suspicious-bg: rgba(245,158,11,0.18); --dangerous-bg: rgba(239,68,68,0.18);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  .container { max-width: 960px; margin: 0 auto; padding: 2rem 1rem; }
  h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
  .subtitle { color: var(--muted); font-size: 0.875rem; margin-bottom: 1.5rem; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 0.5rem; padding: 1.25rem; margin-bottom: 1rem; }
  .card h2 { font-size: 1rem; margin-bottom: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
  .config-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.5rem; }
  .config-item { font-size: 0.875rem; }
  .config-item .label { color: var(--muted); }
  .config-item .value { font-weight: 600; }
  .form-row { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: end; margin-bottom: 0.75rem; }
  .form-group { display: flex; flex-direction: column; gap: 0.25rem; }
  .form-group label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .form-group input, .form-group textarea { background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 0.5rem 0.75rem; border-radius: 0.375rem; font-size: 0.875rem; }
  .form-group input { width: 120px; }
  .form-group textarea { width: 100%; min-height: 80px; resize: vertical; font-family: inherit; }
  .form-group input:focus, .form-group textarea:focus { outline: none; border-color: #3b82f6; }
  .toggle-btn { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 0.8rem; padding: 0.25rem 0; }
  .toggle-btn:hover { color: var(--text); }
  .advanced { display: none; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border); }
  .advanced.show { display: block; }
  button.primary { background: #3b82f6; color: white; border: none; padding: 0.5rem 1.25rem; border-radius: 0.375rem; cursor: pointer; font-size: 0.875rem; font-weight: 600; }
  button.primary:hover { background: #2563eb; }
  button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
  button.secondary { background: var(--bg); color: var(--muted); border: 1px solid var(--border); padding: 0.4rem 0.75rem; border-radius: 0.375rem; cursor: pointer; font-size: 0.75rem; font-weight: 600; }
  button.secondary:hover { color: var(--text); border-color: var(--muted); }
  .error-banner { background: #7f1d1d; border: 1px solid var(--dangerous); border-radius: 0.375rem; padding: 0.75rem 1rem; margin-bottom: 1rem; display: none; font-size: 0.875rem; }
  .results { display: none; }
  .results.show { display: block; }
  .summary { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .stat { flex: 1; min-width: 120px; text-align: center; padding: 1rem; border-radius: 0.375rem; background: var(--bg); }
  .stat .count { font-size: 1.75rem; font-weight: 700; }
  .stat .pct { font-size: 0.8rem; color: var(--muted); }
  .stat .stat-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; margin-top: 0.25rem; }
  .stat.safe .count { color: var(--safe); }
  .stat.suspicious .count { color: var(--suspicious); }
  .stat.dangerous .count { color: var(--dangerous); }
  .meta { font-size: 0.8rem; color: var(--muted); margin-bottom: 1rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  th { text-align: left; padding: 0.5rem; border-bottom: 2px solid var(--border); color: var(--muted); text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.05em; }
  td { padding: 0.4rem 0.5rem; border-bottom: 1px solid var(--border); }
  .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; }
  .badge.safe { background: #14532d; color: var(--safe); }
  .badge.suspicious { background: #78350f; color: var(--suspicious); }
  .badge.dangerous { background: #7f1d1d; color: var(--dangerous); }
  .table-wrap { max-height: 400px; overflow-y: auto; border-radius: 0.375rem; }

  /* Tabs */
  .tabs { display: flex; gap: 0; margin-bottom: 1rem; }
  .tab-btn { background: var(--bg); border: 1px solid var(--border); color: var(--muted); padding: 0.5rem 1rem; cursor: pointer; font-size: 0.875rem; font-weight: 600; transition: all 0.15s; }
  .tab-btn:first-child { border-radius: 0.375rem 0 0 0.375rem; }
  .tab-btn:last-child { border-radius: 0 0.375rem 0.375rem 0; }
  .tab-btn.active { background: #3b82f6; color: white; border-color: #3b82f6; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* Progress stages */
  .progress { display: none; margin-bottom: 1rem; }
  .progress.show { display: block; }
  .progress-stages { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
  .stage { display: flex; align-items: center; gap: 0.35rem; font-size: 0.85rem; color: var(--muted); }
  .stage.active { color: #3b82f6; font-weight: 600; }
  .stage.done { color: var(--safe); }
  .stage-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--border); flex-shrink: 0; }
  .stage.active .stage-dot { background: #3b82f6; animation: pulse 1s ease-in-out infinite; }
  .stage.done .stage-dot { background: var(--safe); }
  .stage-arrow { color: var(--border); font-size: 0.7rem; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

  /* Token highlights */
  .verified-text { margin-bottom: 1rem; }
  .verified-text h3 { font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
  .token-display { background: var(--bg); border: 1px solid var(--border); border-radius: 0.375rem; padding: 1rem; line-height: 1.8; font-size: 0.9rem; white-space: pre-wrap; word-break: break-word; }
  .token-span { padding: 1px 0; border-radius: 2px; cursor: default; position: relative; }
  .token-span.safe { background: var(--safe-bg); }
  .token-span.suspicious { background: var(--suspicious-bg); }
  .token-span.dangerous { background: var(--dangerous-bg); }
  .token-tooltip { display: none; position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); background: #0f172a; border: 1px solid var(--border); border-radius: 0.25rem; padding: 0.35rem 0.5rem; font-size: 0.7rem; white-space: nowrap; z-index: 10; color: var(--text); pointer-events: none; }
  .token-span:hover .token-tooltip { display: block; }

  /* Full-width elements */
  .prompt-group { width: 100%; }

  /* Response preview */
  .response-preview { background: var(--bg); border: 1px solid var(--border); border-radius: 0.375rem; padding: 1rem; font-size: 0.875rem; line-height: 1.6; white-space: pre-wrap; word-break: break-word; margin-bottom: 0.75rem; max-height: 300px; overflow-y: auto; }

  /* Section header with actions */
  .section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }
  .section-header h2 { margin-bottom: 0; }
</style>
</head>
<body>
<div class="container">
  <h1>TEE Inference Verification</h1>
  <p class="subtitle">Verify LLM outputs for model weight exfiltration detection</p>

  <div class="card" id="config-card">
    <h2>Configuration</h2>
    <div class="config-grid" id="config-grid">
      <div class="config-item"><span class="label">Loading...</span></div>
    </div>
  </div>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('local')">Verify Local (vLLM)</button>
    <button class="tab-btn" onclick="switchTab('openrouter')">Query & Verify (OpenRouter)</button>
  </div>

  <!-- ============ Local vLLM tab ============ -->
  <div class="tab-panel active" id="tab-local">
    <div class="card">
      <h2>Run Verification</h2>
      <div class="form-row">
        <div class="form-group">
          <label>Prompts</label>
          <input type="number" id="n_prompts" value="5" min="1" max="100">
        </div>
        <div class="form-group">
          <label>Max Tokens</label>
          <input type="number" id="max_tokens" value="50" min="1" max="500">
        </div>
        <button class="primary" id="run-btn-local" onclick="runVerifyLocal()">Run Verification</button>
      </div>
      <button class="toggle-btn" onclick="toggleAdvanced('adv-local')">&#9660; Advanced Options</button>
      <div class="advanced" id="adv-local">
        <div class="form-row">
          <div class="form-group"><label>Temperature</label><input type="number" id="temperature" value="1.0" step="0.1" min="0"></div>
          <div class="form-group"><label>Top K</label><input type="number" id="top_k" value="50" min="1"></div>
          <div class="form-group"><label>Top P</label><input type="number" id="top_p" value="0.95" step="0.05" min="0" max="1"></div>
          <div class="form-group"><label>Seed</label><input type="number" id="seed" value="42"></div>
          <div class="form-group"><label>GLS Threshold</label><input type="number" id="gls_threshold" value="-5.0" step="0.5"></div>
          <div class="form-group"><label>Rank Threshold</label><input type="number" id="logit_rank_threshold" value="10" min="1"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ============ OpenRouter tab ============ -->
  <div class="tab-panel" id="tab-openrouter">

    <!-- Step 1: Query -->
    <div class="card">
      <h2>1. Query OpenRouter</h2>
      <div class="form-row">
        <div class="form-group prompt-group">
          <label>Prompt</label>
          <textarea id="or_prompt" placeholder="Enter your prompt for Llama 3.1 8B..."></textarea>
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>Max Tokens</label>
          <input type="number" id="or_max_tokens" value="100" min="1" max="500">
        </div>
        <button class="primary" id="run-btn-query" onclick="runQuery()">Query OpenRouter</button>
      </div>
      <button class="toggle-btn" onclick="toggleAdvanced('adv-query')">&#9660; Query Options</button>
      <div class="advanced" id="adv-query">
        <div class="section-header"><span></span><button class="secondary" onclick="resetQueryDefaults()">Reset to Defaults</button></div>
        <div class="form-row">
          <div class="form-group"><label>Temperature</label><input type="number" id="or_q_temperature" value="1.0" step="0.1" min="0"></div>
          <div class="form-group"><label>Top K</label><input type="number" id="or_q_top_k" value="50" min="1"></div>
          <div class="form-group"><label>Top P</label><input type="number" id="or_q_top_p" value="0.95" step="0.05" min="0" max="1"></div>
          <div class="form-group"><label>Seed</label><input type="number" id="or_q_seed" value="42"></div>
        </div>
      </div>
    </div>

    <!-- Response preview (hidden until query completes) -->
    <div class="card" id="response-card" style="display:none">
      <h2>Response</h2>
      <div class="response-preview" id="response-preview"></div>
    </div>

    <!-- Step 2: Verify (hidden until query completes) -->
    <div class="card" id="verify-card" style="display:none">
      <h2>2. Verify Response</h2>
      <div class="form-row">
        <button class="primary" id="run-btn-verify" onclick="runVerifyText()">Run Verification</button>
      </div>
      <button class="toggle-btn" onclick="toggleAdvanced('adv-verify')">&#9660; Verification Options</button>
      <div class="advanced" id="adv-verify">
        <div class="section-header"><span></span><button class="secondary" onclick="resetVerifyDefaults()">Reset to Defaults</button></div>
        <div class="form-row">
          <div class="form-group"><label>Temperature</label><input type="number" id="or_v_temperature" value="1.0" step="0.1" min="0"></div>
          <div class="form-group"><label>Top K</label><input type="number" id="or_v_top_k" value="50" min="1"></div>
          <div class="form-group"><label>Top P</label><input type="number" id="or_v_top_p" value="0.95" step="0.05" min="0" max="1"></div>
          <div class="form-group"><label>Seed</label><input type="number" id="or_v_seed" value="42"></div>
          <div class="form-group"><label>GLS Threshold</label><input type="number" id="or_v_gls_threshold" value="-5.0" step="0.5"></div>
          <div class="form-group"><label>Rank Threshold</label><input type="number" id="or_v_logit_rank_threshold" value="10" min="1"></div>
        </div>
      </div>
    </div>
  </div>

  <div class="error-banner" id="error"></div>

  <div class="progress" id="progress">
    <div class="card">
      <div class="progress-stages" id="progress-stages"></div>
    </div>
  </div>

  <div class="results" id="results">
    <div class="card">
      <h2>Results</h2>
      <div class="meta" id="results-meta"></div>
      <div class="summary" id="summary"></div>
      <div class="verified-text" id="verified-text-section" style="display:none">
        <h3>Verified Text</h3>
        <div class="token-display" id="token-display"></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>#</th><th>Token</th><th>GLS Score</th><th>Logit Rank</th><th>Classification</th></tr></thead>
          <tbody id="tokens-body"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
let currentTab = 'local';

// State for the two-step OpenRouter flow
let queriedPrompt = '';
let queriedResponseText = '';

const DEFAULTS = {
  temperature: 1.0, top_k: 50, top_p: 0.95, seed: 42,
  gls_threshold: -5.0, logit_rank_threshold: 10,
};

const STAGES_LOCAL = ['generating', 'loading_model', 'verifying', 'done'];
const STAGES_VERIFY = ['loading_model', 'verifying', 'done'];
const STAGE_LABELS = {
  generating: 'Generating',
  loading_model: 'Loading Model',
  verifying: 'Verifying',
  querying_openrouter: 'Querying OpenRouter',
  done: 'Done',
};

async function loadConfig() {
  try {
    const res = await fetch('/config');
    const cfg = await res.json();
    document.getElementById('config-grid').innerHTML = Object.entries(cfg).map(([k, v]) =>
      `<div class="config-item"><span class="label">${k.replace(/_/g, ' ')}:</span> <span class="value">${v}</span></div>`
    ).join('');
    if (cfg.seed !== undefined) DEFAULTS.seed = cfg.seed;
    if (cfg.gls_threshold !== undefined) DEFAULTS.gls_threshold = cfg.gls_threshold;
    if (cfg.logit_rank_threshold !== undefined) DEFAULTS.logit_rank_threshold = cfg.logit_rank_threshold;
    // Apply defaults to all fields
    resetLocalDefaults();
    resetQueryDefaults();
    resetVerifyDefaults();
  } catch (e) {
    document.getElementById('config-grid').innerHTML = '<div class="config-item"><span class="label">Failed to load config</span></div>';
  }
}

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach((btn, i) => {
    btn.classList.toggle('active', (i === 0 && tab === 'local') || (i === 1 && tab === 'openrouter'));
  });
  document.getElementById('tab-local').classList.toggle('active', tab === 'local');
  document.getElementById('tab-openrouter').classList.toggle('active', tab === 'openrouter');
}

function toggleAdvanced(id) {
  document.getElementById(id).classList.toggle('show');
}

function resetLocalDefaults() {
  document.getElementById('temperature').value = DEFAULTS.temperature;
  document.getElementById('top_k').value = DEFAULTS.top_k;
  document.getElementById('top_p').value = DEFAULTS.top_p;
  document.getElementById('seed').value = DEFAULTS.seed;
  document.getElementById('gls_threshold').value = DEFAULTS.gls_threshold;
  document.getElementById('logit_rank_threshold').value = DEFAULTS.logit_rank_threshold;
}

function resetQueryDefaults() {
  document.getElementById('or_q_temperature').value = DEFAULTS.temperature;
  document.getElementById('or_q_top_k').value = DEFAULTS.top_k;
  document.getElementById('or_q_top_p').value = DEFAULTS.top_p;
  document.getElementById('or_q_seed').value = DEFAULTS.seed;
}

function resetVerifyDefaults() {
  document.getElementById('or_v_temperature').value = DEFAULTS.temperature;
  document.getElementById('or_v_top_k').value = DEFAULTS.top_k;
  document.getElementById('or_v_top_p').value = DEFAULTS.top_p;
  document.getElementById('or_v_seed').value = DEFAULTS.seed;
  document.getElementById('or_v_gls_threshold').value = DEFAULTS.gls_threshold;
  document.getElementById('or_v_logit_rank_threshold').value = DEFAULTS.logit_rank_threshold;
}

function showError(msg) {
  const el = document.getElementById('error');
  el.textContent = msg;
  el.style.display = 'block';
}

function hideError() {
  document.getElementById('error').style.display = 'none';
}

function setAllButtons(disabled) {
  for (const id of ['run-btn-local', 'run-btn-query', 'run-btn-verify']) {
    const el = document.getElementById(id);
    if (el) el.disabled = disabled;
  }
}

function showProgress(stages) {
  const el = document.getElementById('progress');
  document.getElementById('progress-stages').innerHTML = stages.map((s, i) => {
    const arrow = i < stages.length - 1 ? '<span class="stage-arrow">&#9654;</span>' : '';
    return `<span class="stage" id="stage-${s}"><span class="stage-dot"></span>${STAGE_LABELS[s]}</span>${arrow}`;
  }).join('');
  el.classList.add('show');
}

function updateStage(stage, stages) {
  document.querySelectorAll('.stage').forEach(el => el.classList.remove('active', 'done'));
  let found = false;
  for (const s of stages) {
    const el = document.getElementById('stage-' + s);
    if (!el) continue;
    if (s === stage) {
      el.classList.add(stage === 'done' ? 'done' : 'active');
      found = true;
    } else if (!found) {
      el.classList.add('done');
    }
  }
}

function hideProgress() {
  document.getElementById('progress').classList.remove('show');
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

async function consumeSSE(url, body, stages) {
  setAllButtons(true);
  hideError();
  document.getElementById('results').classList.remove('show');
  showProgress(stages);

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({detail: res.statusText}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          if (data.stage === 'error') throw new Error(data.detail);
          updateStage(data.stage, stages);
          if (data.stage === 'done' && data.result) renderResults(data.result);
        }
      }
    }
  } catch (e) {
    showError('Failed: ' + e.message);
  } finally {
    setAllButtons(false);
    hideProgress();
  }
}

async function runVerifyLocal() {
  const body = {
    n_prompts: parseInt(document.getElementById('n_prompts').value),
    max_tokens: parseInt(document.getElementById('max_tokens').value),
    config: {
      temperature: parseFloat(document.getElementById('temperature').value),
      top_k: parseInt(document.getElementById('top_k').value),
      top_p: parseFloat(document.getElementById('top_p').value),
      seed: parseInt(document.getElementById('seed').value),
      gls_threshold: parseFloat(document.getElementById('gls_threshold').value),
      logit_rank_threshold: parseInt(document.getElementById('logit_rank_threshold').value),
    },
  };
  await consumeSSE('/verify-stream', body, STAGES_LOCAL);
}

async function runQuery() {
  const prompt = document.getElementById('or_prompt').value.trim();
  if (!prompt) { showError('Please enter a prompt.'); return; }

  setAllButtons(true);
  hideError();
  document.getElementById('results').classList.remove('show');
  document.getElementById('response-card').style.display = 'none';
  document.getElementById('verify-card').style.display = 'none';

  const body = {
    prompt: prompt,
    max_tokens: parseInt(document.getElementById('or_max_tokens').value),
    temperature: parseFloat(document.getElementById('or_q_temperature').value),
    top_k: parseInt(document.getElementById('or_q_top_k').value),
    top_p: parseFloat(document.getElementById('or_q_top_p').value),
    seed: parseInt(document.getElementById('or_q_seed').value),
  };

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({detail: res.statusText}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    queriedPrompt = data.prompt;
    queriedResponseText = data.response_text;

    // Show the response and verify card
    document.getElementById('response-preview').textContent = queriedResponseText;
    document.getElementById('response-card').style.display = 'block';
    document.getElementById('verify-card').style.display = 'block';
  } catch (e) {
    showError('Query failed: ' + e.message);
  } finally {
    setAllButtons(false);
  }
}

async function runVerifyText() {
  if (!queriedResponseText) { showError('No response to verify. Query OpenRouter first.'); return; }

  const body = {
    prompt: queriedPrompt,
    response_text: queriedResponseText,
    temperature: parseFloat(document.getElementById('or_v_temperature').value),
    top_k: parseInt(document.getElementById('or_v_top_k').value),
    top_p: parseFloat(document.getElementById('or_v_top_p').value),
    seed: parseInt(document.getElementById('or_v_seed').value),
    gls_threshold: parseFloat(document.getElementById('or_v_gls_threshold').value),
    logit_rank_threshold: parseInt(document.getElementById('or_v_logit_rank_threshold').value),
  };
  await consumeSSE('/verify-text-stream', body, STAGES_VERIFY);
}

function renderResults(data) {
  document.getElementById('results-meta').textContent =
    `Model: ${data.model_name} | Seed: ${data.seed} | Prompts: ${data.n_prompts} | Total tokens: ${data.total_tokens} | GLS threshold: ${data.gls_threshold} | Rank threshold: ${data.logit_rank_threshold}`;

  document.getElementById('summary').innerHTML = [
    {cls: 'safe', count: data.num_safe, pct: data.safe_pct},
    {cls: 'suspicious', count: data.num_suspicious, pct: data.suspicious_pct},
    {cls: 'dangerous', count: data.num_dangerous, pct: data.dangerous_pct},
  ].map(s => `<div class="stat ${s.cls}"><div class="count">${s.count}</div><div class="pct">${s.pct}%</div><div class="stat-label">${s.cls}</div></div>`).join('');

  const textSection = document.getElementById('verified-text-section');
  const tokenDisplay = document.getElementById('token-display');
  const hasTokenText = data.tokens && data.tokens.some(t => t.token_text);

  if (hasTokenText) {
    tokenDisplay.innerHTML = data.tokens.map(t => {
      const cls = t.classification;
      const text = escapeHtml(t.token_text);
      const tooltip = `GLS: ${t.gls_score.toFixed(4)} | Rank: ${t.logit_rank} | ${cls}`;
      return `<span class="token-span ${cls}"><span class="token-tooltip">${escapeHtml(tooltip)}</span>${text}</span>`;
    }).join('');
    textSection.style.display = 'block';
  } else {
    textSection.style.display = 'none';
  }

  document.getElementById('tokens-body').innerHTML = data.tokens.map((t, i) =>
    `<tr><td>${i + 1}</td><td><code>${escapeHtml(t.token_text || '-')}</code></td><td>${t.gls_score.toFixed(4)}</td><td>${t.logit_rank}</td><td><span class="badge ${t.classification}">${t.classification}</span></td></tr>`
  ).join('');

  document.getElementById('results').classList.add('show');
}

loadConfig();
</script>
</body>
</html>"""
