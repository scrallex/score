const config = window.__STM_CONFIG__ || {};
const API_BASE = typeof config.apiBase === 'string' ? config.apiBase.replace(/\/$/, '') : '/api';
const DEMO_ENDPOINT = `${API_BASE}/demo`;
const LIVE_ENDPOINT = `${API_BASE}/demo/live`;

const toastEl = document.getElementById('toast');
let liveSocket = null;
let liveEventSource = null;

function showToast(message, kind = 'info') {
  if (!toastEl) return;
  toastEl.textContent = message;
  toastEl.dataset.kind = kind;
  toastEl.hidden = false;
  toastEl.dataset.state = 'visible';
  setTimeout(() => {
    toastEl.dataset.state = 'hidden';
    setTimeout(() => {
      toastEl.hidden = true;
    }, 240);
  }, 2600);
}

function fmtNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—';
  }
  return Number(value).toFixed(digits);
}

function fmtPercent(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—';
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function fmtInteger(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—';
  }
  return Number(value).toLocaleString();
}

async function fetchJson(url) {
  const res = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

function renderPatternSnapshot(pattern) {
  const liftEl = document.getElementById('snapshot-lift');
  const listEl = document.getElementById('snapshot-pattern');
  if (!liftEl || !listEl || !pattern) return;
  const selected = pattern.selected || {};
  const lift = selected.lift ? `${fmtNumber(selected.lift, 2)}×` : '—';
  liftEl.textContent = lift;
  listEl.innerHTML = '';
  const items = [
    { label: 'Signature', value: selected.label || selected.signature || '(n/a)' },
    { label: 'Event Count', value: fmtInteger(selected.event_count) },
    { label: 'History Count', value: fmtInteger(selected.history_count) },
    { label: 'Coverage', value: fmtPercent(pattern.coverage, 1) },
  ];
  items.forEach((item) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${item.label}</span><span class="value">${item.value}</span>`;
    listEl.appendChild(li);
  });
}

function renderTwinSnapshot(twinFinder) {
  const statEl = document.getElementById('snapshot-twins');
  const listEl = document.getElementById('snapshot-twin-list');
  if (!statEl || !listEl || !twinFinder) return;
  const top = (twinFinder.top_matches || [])[0] || {};
  statEl.textContent = fmtInteger(top.twin_windows);
  listEl.innerHTML = '';
  const entries = [
    { label: 'String', value: top.string || '(n/a)' },
    { label: 'ANN distance', value: fmtNumber(top.mean_distance, 6) },
    { label: 'Mean Coherence', value: fmtNumber(top.mean_coherence, 4) },
    { label: 'Mean Stability', value: fmtNumber(top.mean_stability, 4) },
  ];
  entries.forEach((entry) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${entry.label}</span><span class="value">${entry.value}</span>`;
    listEl.appendChild(li);
  });
}

function renderContextSnapshot(context) {
  const shareEl = document.getElementById('snapshot-struct-share');
  const listEl = document.getElementById('snapshot-struct');
  if (!shareEl || !listEl || !context) return;
  shareEl.textContent = fmtPercent(context.structural_share, 2);
  listEl.innerHTML = '';
  (context.top_structural || []).slice(0, 3).forEach((token) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${token.string}</span><span class="value">occ ${fmtInteger(token.occurrences)}</span>`;
    listEl.appendChild(li);
  });
}

function renderRecommendations(context) {
  const table = document.getElementById('recommendations-table');
  if (!table || !context) return;
  table.innerHTML = '';
  (context.top_proposals || []).forEach((item) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${item.string}</td>
      <td>${fmtNumber(item.score, 3)}</td>
      <td>${fmtNumber(item.patternability, 3)}</td>
      <td>${fmtNumber(item.connector, 3)}</td>
      <td>${fmtInteger(item.occurrences)}</td>
    `;
    table.appendChild(tr);
  });
}

function updateGeneratedAt(timestamp) {
  const metaTime = document.getElementById('dashboard-generated');
  if (!metaTime || !timestamp) return;
  const date = new Date(timestamp);
  metaTime.textContent = date.toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'medium',
  });
}

async function loadSnapshot() {
  const payload = await fetchJson(DEMO_ENDPOINT);
  updateGeneratedAt(payload.generated_at);
  const demos = payload.demos || {};
  renderPatternSnapshot(demos.pattern_prophet);
  renderTwinSnapshot(demos.twin_finder);
  renderContextSnapshot(demos.context_refinery);
  renderRecommendations(demos.context_refinery);
  showToast('Dashboard snapshot refreshed');
}

function appendLiveFrame(frame) {
  const list = document.getElementById('dashboard-live-feed');
  if (!list) return;
  const li = document.createElement('li');
  li.innerHTML = `
    <span class="label">${fmtInteger(frame.index)}</span>
    <div>
      <div><strong>${frame.signature || '(none)'}</strong></div>
      <div style="font-size: 0.8rem; opacity: 0.8;">
        coh ${fmtNumber(frame.metrics?.coherence, 4)} ·
        stab ${fmtNumber(frame.metrics?.stability, 4)} ·
        ent ${fmtNumber(frame.metrics?.entropy, 4)} ·
        λ ${fmtNumber(frame.metrics?.lambda_hazard, 4)}
      </div>
    </div>
  `;
  list.prepend(li);
  while (list.childElementCount > 25) {
    list.removeChild(list.lastElementChild);
  }
}

function startLiveStream(button) {
  if (liveSocket || liveEventSource) return;
  const fallback = () => startSseStream(button);
  try {
    liveSocket = new WebSocket(resolveWsEndpoint());
  } catch (error) {
    console.error('Dashboard WebSocket init failed, falling back to SSE', error);
    fallback();
    return;
  }
  liveSocket.onopen = () => {
    if (button) {
      button.dataset.state = 'running';
      button.textContent = 'Stop Stream';
      button.classList.remove('btn--ghost');
      button.classList.add('btn');
    }
    showToast('Live stream started');
  };
  liveSocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      appendLiveFrame(data);
    } catch (error) {
      console.error('Invalid dashboard WebSocket payload', error);
    }
  };
  liveSocket.onerror = (error) => {
    console.error('Dashboard WebSocket error', error);
  };
  liveSocket.onclose = () => {
    const hadSocket = liveSocket !== null;
    liveSocket = null;
    if (hadSocket && !liveEventSource) {
      fallback();
    }
  };
}

function stopLiveStream(button, notify = true) {
  if (liveSocket) {
    try {
      liveSocket.close();
    } catch (error) {
      console.error('Error closing dashboard WebSocket', error);
    }
    liveSocket = null;
  }
  if (liveEventSource) {
    liveEventSource.close();
    liveEventSource = null;
  }
  if (button) {
    button.dataset.state = 'idle';
    button.textContent = 'Start Stream';
    button.classList.add('btn--ghost');
  }
  if (notify) {
    showToast('Live stream stopped');
  }
}

function resolveWsEndpoint() {
  if (typeof window !== 'undefined' && window.location && window.location.host) {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}/ws/stream`;
  }
  return 'ws://127.0.0.1/ws/stream';
}

function startSseStream(button) {
  if (liveEventSource) return;
  liveEventSource = new EventSource(LIVE_ENDPOINT);
  liveEventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      appendLiveFrame(data);
    } catch (error) {
      console.error('Invalid dashboard SSE payload', error);
    }
  };
  liveEventSource.onerror = (error) => {
    console.error('Dashboard SSE stream error', error);
    stopLiveStream(button, false);
    setTimeout(() => startSseStream(button), 1500);
  };
  if (button) {
    button.dataset.state = 'running';
    button.textContent = 'Stop Stream';
    button.classList.remove('btn--ghost');
    button.classList.add('btn');
  }
  showToast('Live stream started (SSE fallback)');
}

function setupControls() {
  const toggle = document.getElementById('dashboard-live-toggle');
  if (!toggle) return;
  toggle.addEventListener('click', () => {
    if (toggle.dataset.state === 'running') {
      stopLiveStream(toggle);
    } else {
      startLiveStream(toggle);
    }
  });
}

window.addEventListener('DOMContentLoaded', () => {
  loadSnapshot().catch((error) => {
    console.error('Failed to load dashboard snapshot', error);
    showToast('Failed to load dashboard snapshot', 'error');
  });
  setupControls();
});
