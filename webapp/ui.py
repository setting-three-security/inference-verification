"""Self-contained HTML UI for the inference verification API (TEE)."""


def get_ui_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Inference Verification in a TEE</title>
<style>
  :root {
    --safe: #22c55e; --suspicious: #f59e0b; --dangerous: #ef4444;
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8;
    --safe-bg: rgba(34,197,94,0.45); --suspicious-bg: rgba(245,158,11,0.4); --dangerous-bg: rgba(239,68,68,0.4);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  .page-layout { display: flex; min-height: 100vh; }
  .container { max-width: 960px; margin: 0 auto; padding: 2rem 1rem; flex: 1; min-width: 0; }

  /* Sidebar */
  .sidebar { width: 420px; flex-shrink: 0; background: var(--surface); border-left: 1px solid var(--border); padding: 1.25rem; overflow-y: auto; transition: width 0.2s, padding 0.2s; }
  .sidebar.collapsed { width: 0; padding: 0; overflow: hidden; }
  .sidebar-toggle { position: fixed; top: 1rem; right: 1rem; z-index: 100; background: var(--surface); border: 1px solid var(--border); color: var(--muted); width: 36px; height: 36px; border-radius: 0.375rem; cursor: pointer; font-size: 1rem; display: flex; align-items: center; justify-content: center; transition: all 0.15s; }
  .sidebar-toggle:hover { color: var(--text); border-color: var(--muted); }
  .sidebar h3 { font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem; }
  .sidebar .verification-iframe { width: 100%; height: 500px; border: 1px solid var(--border); border-radius: 0.375rem; margin-bottom: 1rem; }
  .sidebar .trust-chain { font-size: 0.8rem; line-height: 1.6; color: var(--text); }
  .sidebar .trust-chain h4 { font-size: 0.8rem; color: var(--text); margin-bottom: 0.5rem; font-weight: 600; }
  .sidebar .trust-chain ol { padding-left: 1.25rem; margin-bottom: 0.75rem; }
  .sidebar .trust-chain li { margin-bottom: 0.5rem; }
  .sidebar .trust-chain a { color: #3b82f6; text-decoration: none; }
  .sidebar .trust-chain a:hover { text-decoration: underline; }
  .sidebar .trust-chain .insight { background: var(--bg); border: 1px solid var(--border); border-radius: 0.375rem; padding: 0.75rem; margin-top: 0.75rem; font-size: 0.8rem; line-height: 1.6; }
  @media (max-width: 1200px) {
    .sidebar { position: fixed; top: 0; right: 0; height: 100vh; z-index: 50; box-shadow: -4px 0 20px rgba(0,0,0,0.3); }
    .sidebar.collapsed { box-shadow: none; }
  }
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
<button class="sidebar-toggle" id="sidebar-toggle" onclick="toggleSidebar()" title="Toggle TEE Verification">&#9776;</button>
<div class="page-layout">
<div class="container">
  <h1>Inference Verification in a Trusted Execution Environment (TEE)</h1>
  <p class="subtitle">We use <a href="https://arxiv.org/abs/2511.02620" target="_blank" style="color: #3b82f6;">inference verification</a> to verify that text actually came from a model. This runs on an Nvidia H200 TEE, with the TEE code built and hosted by <a href="https://tinfoil.sh" target="_blank" style="color: #3b82f6;">tinfoil.sh</a>.</p>

  <div class="card">
    <p style="font-size: 0.9rem; line-height: 1.7;">
      Inference verification is the task of verifying that a piece of text was actually generated by a specific model with specific sampling parameters. We use <a href="https://arxiv.org/abs/2511.02620" target="_blank" style="color: #3b82f6;">Token-DiFR</a> to accomplish inference verification.
    </p>
    <details style="margin-top: 0.75rem;">
      <summary style="cursor: pointer; font-size: 0.8rem; color: var(--muted); font-weight: 600;">Why does this matter?</summary>
      <div style="margin-top: 0.5rem; font-size: 0.85rem; line-height: 1.7; color: var(--muted);">
        <p style="margin-bottom: 0.5rem;">Inference verification has two key applications:</p>
        <ul style="padding-left: 1.25rem; margin-bottom: 0.5rem;">
          <li style="margin-bottom: 0.35rem;"><strong>Detecting steganography</strong> &mdash; identifying covert channels hidden in LLM outputs (<a href="https://arxiv.org/abs/2511.02620" target="_blank" style="color: #3b82f6;">arXiv:2511.02620</a>)</li>
          <li style="margin-bottom: 0.35rem;"><strong>Verifying inference providers</strong> &mdash; ensuring a provider actually ran the claimed model and didn't substitute a cheaper one (<a href="https://arxiv.org/abs/2511.20621" target="_blank" style="color: #3b82f6;">arXiv:2511.20621</a>)</li>
        </ul>
        <p>Importantly, for inference verification to be trustworthy, you also need to trust the verifier itself. That's why we run this inside a <strong>Trusted Execution Environment (TEE)</strong> &mdash; the verification code runs in hardware-isolated memory that even the server operator cannot inspect or tamper with.</p>
      </div>
    </details>
  </div>

  <div class="tabs">
    <button class="tab-btn" onclick="switchTab('local')">Verify Local Text</button>
    <button class="tab-btn active" onclick="switchTab('openrouter')">Query & Verify (OpenRouter)</button>
  </div>

  <!-- ============ Verify Local Text tab ============ -->
  <div class="tab-panel" id="tab-local">
    <div class="card">
      <h2>Verify Local Text</h2>
      <p style="font-size: 0.85rem; color: var(--muted); margin-bottom: 0.75rem;">This example text is stored on the device. Click "Run Verification" to verify it was generated by the model.</p>
      <div style="margin-bottom: 0.5rem;">
        <div style="font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem;">Prompt</div>
        <div class="response-preview" id="local-prompt-display" style="max-height: 100px;"></div>
      </div>
      <div style="margin-bottom: 0.75rem;">
        <div style="font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem;">Response Text to Verify</div>
        <div class="response-preview" id="local-response-display"></div>
      </div>
      <div class="form-row">
        <button class="primary" id="run-btn-local" onclick="runVerifyLocal()">Run Verification</button>
      </div>
      <button class="toggle-btn" onclick="toggleAdvanced('adv-local')">&#9660; Advanced Options</button>
      <div class="advanced" id="adv-local">
        <div class="form-row">
          <div class="form-group"><label>Temperature</label><input type="number" id="temperature" value="1.0" step="0.1" min="0"></div>
          <div class="form-group"><label>Top K</label><input type="number" id="top_k" value="50" min="1"></div>
          <div class="form-group"><label>Top P</label><input type="number" id="top_p" value="0.95" step="0.05" min="0" max="1"></div>
          <div class="form-group"><label>Seed</label><input type="number" id="seed" value="42"></div>
          <div class="form-group"><label>FSSL Threshold</label><input type="number" id="gls_threshold" value="-6.0" step="0.5"></div>
          <div class="form-group"><label>Rank Threshold</label><input type="number" id="logit_rank_threshold" value="10" min="1"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ============ OpenRouter tab ============ -->
  <div class="tab-panel active" id="tab-openrouter">

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
        <button class="primary" id="run-btn-query" onclick="runQuery()">Query Llama 3.1 8B</button>
      </div>
      <details style="margin-top: 0.5rem; margin-bottom: 0.5rem;">
        <summary style="cursor: pointer; font-size: 0.8rem; color: var(--suspicious); font-weight: 600;">Simulate a dishonest provider</summary>
        <div style="margin-top: 0.5rem; font-size: 0.85rem; line-height: 1.6; color: var(--muted);">
          <p style="margin-bottom: 0.5rem;">Consider: you requested a response from Llama 3.1 8B, but the provider wants to save money so they serve you Llama 3.2 3B instead. Click below to simulate such a query &mdash; verification should flag the mismatch.</p>
          <button class="secondary" id="run-btn-dishonest" onclick="runQuery('meta-llama/llama-3.2-3b-instruct')" style="color: var(--suspicious); border-color: var(--suspicious);">Query Llama 3.2 3B (dishonest)</button>
        </div>
      </details>
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
          <div class="form-group"><label>FSSL Threshold</label><input type="number" id="or_v_gls_threshold" value="-6.0" step="0.5"></div>
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
    </div>
  </div>
</div>

<!-- ============ TEE Verification Sidebar ============ -->
<aside class="sidebar" id="sidebar">
  <h3>TEE Verification Guarantees</h3>
  <p style="font-size: 0.8rem; color: var(--muted); margin-bottom: 1rem;">Verifying that inference verification is taking place properly.</p>
  <iframe
    class="verification-iframe"
    id="tinfoil-verification"
    src="https://verification-center.tinfoil.sh?darkMode=true&showHeader=true"
    title="TEE Verification Guarantees"
    sandbox="allow-scripts allow-same-origin"
  ></iframe>
  <div class="trust-chain">
    <h4>Chain of Trust</h4>
    <ol>
      <li>The source code lives at <a href="https://github.com/setting-three-security/inference-verification" target="_blank">setting-three-security/inference-verification</a>.</li>
      <li>The <a href="https://github.com/setting-three-security/inference-verification-deployer" target="_blank">deployer repo</a> (public) has <code>tinfoil-config.yml</code> which specifies the exact Docker image + SHA256 digest.</li>
      <li>On tag push, Tinfoil's <code>pri-build-action</code> builds the enclave image and publishes a <strong>Sigstore bundle</strong> &mdash; a signed attestation linking the config to the expected binary measurements.</li>
      <li>At boot, the Nvidia H200's CPU measures the enclave (firmware, kernel, app binary) and produces a <strong>hardware-signed attestation report</strong>.</li>
      <li>On every connection, the Tinfoil SDK compares the Sigstore measurements against the enclave's attestation &mdash; if they don't match, the connection is refused.</li>
    </ol>
    <div class="insight">
      <strong>The deployer repo is the public proof.</strong> Anyone can look at <a href="https://github.com/setting-three-security/inference-verification-deployer/blob/main/tinfoil-config.yml" target="_blank">tinfoil-config.yml</a>, see the exact container image + digest being run, and verify the running enclave matches that config.
    </div>

    <h4 style="margin-top: 1rem;">Verify This TEE</h4>
    <p style="font-size: 0.8rem; color: var(--muted); margin-bottom: 0.5rem;">You can independently verify that this enclave is running the exact code from the public repos using the <a href="https://docs.tinfoil.sh/sdk/cli-sdk" target="_blank" style="color: #3b82f6;">Tinfoil CLI</a>:</p>
    <pre style="background: var(--bg); border: 1px solid var(--border); border-radius: 0.375rem; padding: 0.75rem; font-size: 0.7rem; overflow-x: auto; white-space: pre-wrap; word-break: break-all; color: var(--text); line-height: 1.5;">tinfoil attestation verify \
  -e inference-verification-updated-ui.debug.rinberg-lab.containers.tinfoil.dev \
  -r setting-three-security/inference-verification-deployer</pre>
    <p style="font-size: 0.75rem; color: var(--muted); margin-top: 0.5rem;">This checks that the Sigstore measurements from the public repo match the hardware attestation report from the running enclave.</p>
  </div>
</aside>

</div><!-- /page-layout -->

<script>
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const btn = document.getElementById('sidebar-toggle');
  sidebar.classList.toggle('collapsed');
  btn.innerHTML = sidebar.classList.contains('collapsed') ? '&#9776;' : '&#10005;';
}

let currentTab = 'openrouter';

// State for the two-step OpenRouter flow
let queriedPrompt = '';
let queriedResponseText = '';

const DEFAULTS = {
  temperature: 1.0, top_k: 50, top_p: 0.95, seed: 42,
  gls_threshold: -6.0, logit_rank_threshold: 10,
};

const STAGES_VERIFY = ['loading_model', 'verifying', 'done'];
const STAGE_LABELS = {
  loading_model: 'Loading Model',
  verifying: 'Verifying',
  done: 'Done',
};

async function loadConfig() {
  try {
    const res = await fetch('/config');
    const cfg = await res.json();
    if (cfg.seed !== undefined) DEFAULTS.seed = cfg.seed;
    if (cfg.gls_threshold !== undefined) DEFAULTS.gls_threshold = cfg.gls_threshold;
    if (cfg.logit_rank_threshold !== undefined) DEFAULTS.logit_rank_threshold = cfg.logit_rank_threshold;
    resetLocalDefaults();
    resetQueryDefaults();
    resetVerifyDefaults();
  } catch (e) { /* defaults are fine */ }
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
  for (const id of ['run-btn-local', 'run-btn-query', 'run-btn-verify', 'run-btn-dishonest']) {
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

// Example prompt + response from Llama 3.1 8B via OpenRouter for local verification demo
const LOCAL_EXAMPLE = {
  prompt: "Explain how trusted execution environments work in cloud computing, including the role of hardware attestation and why they matter for running sensitive workloads on shared infrastructure.",
  response: `Trusted Execution Environments (TEEs) are a critical component of cloud computing, providing a secure and isolated execution environment for sensitive workloads. Here's a detailed explanation of how they work, the role of hardware attestation, and why they matter:

**What are Trusted Execution Environments (TEEs)?**

TEEs are hardware-based secure execution environments that provide a trusted, isolated, and secure execution space for applications and sensitive data. They are designed to run in parallel with the main operating system and ensure that the execution of sensitive workloads is isolated from the rest of the system, preventing unauthorized access and tampering.

**How do TEEs work?**

There are two primary types of TEEs:

1. **System-on-Chip (SoC) TEEs**: These are integrated into the System-on-Chip (SoC) of a cloud server, providing a dedicated, secure environment for running workloads.
2. **Co-Processor TEEs**: These are separate processing units, designed to run securely alongside the main CPU, providing a sandboxed environment for workloads.

TEEs typically use a combination of hardware and software mechanisms to ensure isolation and protection:

* **Virtualization**: TEEs use hardware-assisted virtualization to create isolated partitions or domains, which are managed by the TEE's own hypervisor.
* **Memory encryption**: TEEs use cryptographic techniques to encrypt data stored in the TEE's memory, preventing unauthorized access.
* **Intrusion detection and prevention**: TEEs often include intrusion detection and prevention capabilities, monitoring for malicious activity within the isolated environment.`,
};

// Populate the local example displays
document.getElementById('local-prompt-display').textContent = LOCAL_EXAMPLE.prompt;
document.getElementById('local-response-display').textContent = LOCAL_EXAMPLE.response;

async function runVerifyLocal() {
  const body = {
    prompt: LOCAL_EXAMPLE.prompt,
    response_text: LOCAL_EXAMPLE.response,
    temperature: parseFloat(document.getElementById('temperature').value),
    top_k: parseInt(document.getElementById('top_k').value),
    top_p: parseFloat(document.getElementById('top_p').value),
    seed: parseInt(document.getElementById('seed').value),
    gls_threshold: parseFloat(document.getElementById('gls_threshold').value),
    logit_rank_threshold: parseInt(document.getElementById('logit_rank_threshold').value),
  };
  await consumeSSE('/verify-text-stream', body, STAGES_VERIFY);
}

async function runQuery(model) {
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
    model: model || 'meta-llama/llama-3.1-8b-instruct',
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
    const queriedModel = body.model;
    const modelLabel = queriedModel === 'meta-llama/llama-3.1-8b-instruct' ? 'Llama 3.1 8B' : 'Llama 3.2 3B (dishonest)';
    document.getElementById('response-preview').textContent = queriedResponseText;
    document.getElementById('response-card').querySelector('h2').textContent = 'Response from ' + modelLabel;
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
    `Model: ${data.model_name} | Seed: ${data.seed} | Prompts: ${data.n_prompts} | Total tokens: ${data.total_tokens} | FSSL threshold: ${data.gls_threshold} | Rank threshold: ${data.logit_rank_threshold}`;

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
      const glsStr = t.gls_score != null ? t.gls_score.toFixed(4) : 'N/A';
      const tooltip = `FSSL: ${glsStr} | Rank: ${t.logit_rank} | ${cls}`;
      return `<span class="token-span ${cls}"><span class="token-tooltip">${escapeHtml(tooltip)}</span>${text}</span>`;
    }).join('');
    textSection.style.display = 'block';
  } else {
    textSection.style.display = 'none';
  }

  document.getElementById('results').classList.add('show');
}

loadConfig();
</script>
</body>
</html>"""
