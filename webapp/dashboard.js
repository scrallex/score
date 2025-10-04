const config = window.__STM_CONFIG__ || {};
const API_BASE = typeof config.apiBase === 'string' ? config.apiBase.replace(/\/$/, '') : '/api';
const DEMO_ENDPOINT = `${API_BASE}/demo`;
const LIVE_ENDPOINT = `${API_BASE}/demo/live`;
const BACKTEST_STATUS_ENDPOINT = `${API_BASE}/backtests/status`;
const BACKTEST_LATEST_ENDPOINT = `${API_BASE}/backtests/latest`;
const BACKTEST_STATIC_LATEST = typeof config.backtestsLatest === 'string'
  ? config.backtestsLatest
  : '/output/backtests/latest.json';
const BACKTEST_STATIC_PARTIAL = typeof config.backtestsPartial === 'string'
  ? config.backtestsPartial
  : BACKTEST_STATIC_LATEST.replace(/latest\.json$/, 'latest.partial.json');

const toastEl = document.getElementById('toast');
let liveSocket = null;
let liveEventSource = null;
let backtestPollHandle = null;

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

function fmtBps(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—';
  }
  return Number(value).toFixed(digits);
}

function fmtPercentPoint(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—';
  }
  return `${Number(value).toFixed(digits)}%`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    switch (char) {
      case '&':
        return '&amp;';
      case '<':
        return '&lt;';
      case '>':
        return '&gt;';
      case '"':
        return '&quot;';
      case "'":
        return '&#39;';
      default:
        return char;
    }
  });
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

function renderBacktest(payload, status) {
  const statusEl = document.getElementById('backtest-status-text');
  const updatedEl = document.getElementById('backtest-updated');
  const runEl = document.getElementById('backtest-run-name');
  const errorEl = document.getElementById('backtest-error');
  if (!statusEl || !updatedEl || !runEl || !errorEl) {
    return;
  }

  errorEl.hidden = true;
  errorEl.textContent = '';

  const state = (status && typeof status.state === 'string' ? status.state : payload?.status) || 'unknown';
  let label = state.charAt(0).toUpperCase() + state.slice(1);
  if (status && typeof status.progress === 'number') {
    const pct = Math.max(0, Math.min(1, Number(status.progress)));
    label += ` · ${(pct * 100).toFixed(0)}%`;
  }
  statusEl.textContent = label;

  const timestamp = payload?.generated_at;
  if (timestamp) {
    const date = new Date(timestamp);
    updatedEl.textContent = `Updated ${date.toLocaleString(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
    })}`;
  } else {
    updatedEl.textContent = '—';
  }

  const runs = Array.isArray(payload?.runs) ? payload.runs : [];
  const activeRun = runs.length ? runs[runs.length - 1] : null;
  if (activeRun) {
    const instruments = Array.isArray(activeRun.instruments) ? activeRun.instruments : [];
    const qualified = instruments.filter((entry) => entry?.metrics?.qualified).length;
    const summary = [`Run: ${activeRun.name || 'Unnamed'}`, `${qualified}/${instruments.length} qualified`];
    runEl.innerHTML = `<strong>${escapeHtml(summary[0])}</strong><span>${escapeHtml(summary[1])}</span>`;
  } else {
    runEl.textContent = 'No backtest runs available.';
  }

  renderBacktestTable(activeRun);
}

function renderBacktestTable(run) {
  const tbody = document.getElementById('backtest-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  const instruments = run?.instruments;
  if (!Array.isArray(instruments) || instruments.length === 0) {
    const row = document.createElement('tr');
    row.innerHTML = '<td colspan="9">No instrument results yet.</td>';
    tbody.appendChild(row);
    return;
  }

  instruments.forEach((entry) => {
    if (entry && typeof entry.error === 'string' && entry.error.length) {
      const row = document.createElement('tr');
      row.classList.add('backtest-row-error');
      row.innerHTML = `
        <td>${escapeHtml(entry.instrument || '—')}</td>
        <td colspan="8" class="backtest-error-cell">${escapeHtml(entry.error)}</td>
      `;
      tbody.appendChild(row);
      return;
    }

    const metrics = entry?.metrics || {};
    const row = document.createElement('tr');
    if (metrics.qualified) {
      row.classList.add('backtest-row-qualified');
    }
    const winRate = fmtPercent(metrics.win_rate, 1);
    const coverage = fmtPercent(entry?.gate_coverage, 1);
    row.innerHTML = `
      <td>${escapeHtml(entry?.instrument || '—')}</td>
      <td>${fmtInteger(metrics.trade_count)}</td>
      <td>${fmtBps(metrics.avg_return_bps, 1)}</td>
      <td>${winRate}</td>
      <td>${fmtNumber(metrics.profit_factor, 2)}</td>
      <td>${fmtNumber(metrics.sharpe, 2)}</td>
      <td>${fmtPercentPoint(metrics.max_drawdown_pct, 1)}</td>
      <td>${coverage}</td>
      <td>${escapeHtml((entry?.source || '').toUpperCase() || '—')}</td>
    `;
    tbody.appendChild(row);
  });
}

function renderBacktestError(message) {
  const statusEl = document.getElementById('backtest-status-text');
  const updatedEl = document.getElementById('backtest-updated');
  const runEl = document.getElementById('backtest-run-name');
  const errorEl = document.getElementById('backtest-error');
  const tbody = document.getElementById('backtest-table-body');
  if (!statusEl || !updatedEl || !runEl || !errorEl || !tbody) {
    return;
  }
  statusEl.textContent = 'Error';
  updatedEl.textContent = '—';
  runEl.textContent = 'Backtest results unavailable.';
  tbody.innerHTML = '<tr><td colspan="9">No data.</td></tr>';
  errorEl.hidden = false;
  errorEl.textContent = message || 'Failed to load backtest results.';
}

async function fetchBacktestLatestPayload() {
  try {
    return await fetchJson('/backtests/latest');
  } catch (error) {
    try {
      const latestRes = await fetch(BACKTEST_STATIC_LATEST, {
        headers: { Accept: 'application/json' },
      });
      if (latestRes.ok) {
        return latestRes.json();
      }
      const partialRes = await fetch(BACKTEST_STATIC_PARTIAL, {
        headers: { Accept: 'application/json' },
      });
      if (partialRes.ok) {
        return partialRes.json();
      }
    } catch (staticError) {
      console.debug('Static backtest payload unavailable', staticError);
    }
    throw error;
  }
}

function scheduleBacktestPolling() {
  if (backtestPollHandle) return;
  backtestPollHandle = setInterval(() => {
    refreshBacktests(false).catch(() => {});
  }, 5000);
}

function stopBacktestPolling() {
  if (!backtestPollHandle) return;
  clearInterval(backtestPollHandle);
  backtestPollHandle = null;
}

async function refreshBacktests(showToastMessage = true) {
  try {
    const [statusResult, latestResult] = await Promise.allSettled([
      fetchJson('/backtests/status'),
      fetchBacktestLatestPayload(),
    ]);

    const status = statusResult.status === 'fulfilled' ? statusResult.value : null;
    if (statusResult.status !== 'fulfilled' && statusResult.reason) {
      console.debug('Backtest status unavailable', statusResult.reason);
    }

    if (latestResult.status !== 'fulfilled') {
      throw latestResult.reason || new Error('Failed to load latest backtest');
    }

    const latest = latestResult.value;
    renderBacktest(latest, status);

    if (status && status.state === 'running') {
      scheduleBacktestPolling();
    } else {
      stopBacktestPolling();
    }

    if (showToastMessage) {
      showToast('Backtests refreshed');
    }
  } catch (error) {
    console.error('Failed to refresh backtests', error);
    stopBacktestPolling();
    renderBacktestError(error?.message || 'Unable to load backtest data');
    if (showToastMessage) {
      showToast('Failed to refresh backtests', 'error');
    }
  }
}

function setupBacktestControls() {
  const refreshBtn = document.getElementById('backtest-refresh');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      refreshBacktests(true);
    });
  }
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
  setupBacktestControls();
  refreshBacktests(false).catch((error) => {
    console.error('Failed to load initial backtests', error);
  });
});
