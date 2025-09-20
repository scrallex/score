const config = window.__STM_CONFIG__ || {};
const API_BASE = typeof config.apiBase === 'string' ? config.apiBase.replace(/\/$/, '') : '/api';

const toastEl = document.getElementById('toast');

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
    }, 220);
  }, 2600);
}

async function fetchJson(path) {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, {
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`Request failed (${response.status})`);
  }
  return response.json();
}

function fmtPercent(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—';
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function fmtNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—';
  }
  return Number(value).toFixed(digits);
}

function fmtInteger(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—';
  }
  return Number(value).toLocaleString();
}

function setMeta(meta) {
  const timeEl = document.getElementById('meta-generated');
  const sourcesEl = document.getElementById('meta-sources');
  if (timeEl && typeof meta?.generated_at === 'string') {
    const date = new Date(meta.generated_at);
    timeEl.textContent = date.toLocaleString(undefined, {
      dateStyle: 'medium',
      timeStyle: 'medium',
    });
  }
  if (sourcesEl && meta?.sources) {
    const entries = Object.entries(meta.sources)
      .map(([key, path]) => `${key}: ${path}`)
      .join(' · ');
    sourcesEl.textContent = entries ? `Sources: ${entries}` : 'Sources: —';
  }
}

function renderPatternProphet(data) {
  const section = document.querySelector('[data-demo="pattern_prophet"]');
  if (!section || !data) return;
  const selected = data.selected || {};
  section.querySelector('[data-field="label"]').textContent = selected.label || selected.signature || '—';
  section.querySelector('[data-field="event_count"]').textContent = fmtInteger(selected.event_count);
  section.querySelector('[data-field="history_count"]').textContent = fmtInteger(selected.history_count);
  section.querySelector('[data-field="lift"]').textContent = selected.lift ? selected.lift.toFixed(2) : '—';

  const metricsBody = section.querySelector('[data-field="metrics"]');
  metricsBody.innerHTML = '';
  const eventMetrics = selected.event_metrics || {};
  const historyMetrics = selected.history_metrics || {};
  const metricKeys = ['coherence', 'stability', 'entropy', 'rupture', 'lambda_hazard'];
  metricKeys.forEach((key) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${key.replace('_', ' ')}</td>
      <td>${fmtNumber(eventMetrics[key], 4)}</td>
      <td>${fmtNumber(historyMetrics[key], 4)}</td>
    `;
    metricsBody.appendChild(row);
  });

  const samplesBody = section.querySelector('[data-field="samples"]');
  samplesBody.innerHTML = '';
  (selected.samples || []).forEach((sample) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${fmtInteger(sample.index)}</td>
      <td>${fmtNumber(sample.coherence, 4)}</td>
      <td>${fmtNumber(sample.stability, 4)}</td>
      <td>${fmtNumber(sample.entropy, 4)}</td>
      <td>${fmtNumber(sample.lambda_hazard, 4)}</td>
    `;
    samplesBody.appendChild(row);
  });

  const footer = section.querySelector('[data-field="chart"]');
  if (footer) {
    const coverage = typeof data.coverage === 'number' ? fmtPercent(data.coverage, 1) : '—';
    footer.textContent = `Event window coverage: ${coverage} · Total windows: ${fmtInteger(data.total_windows)}`;
  }
}

function renderTwinFinder(data, assets) {
  const section = document.querySelector('[data-demo="twin_finder"]');
  if (!section || !data) return;
  const tbody = section.querySelector('[data-field="rows"]');
  tbody.innerHTML = '';
  (data.top_matches || []).forEach((match) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${match.string}</td>
      <td>${fmtInteger(match.twin_windows)}</td>
      <td>${fmtNumber(match.mean_distance, 6)}</td>
      <td>${fmtNumber(match.mean_coherence, 4)}</td>
      <td>${fmtNumber(match.mean_stability, 4)}</td>
    `;
    tbody.appendChild(row);
  });

  const footer = section.querySelector('[data-field="chart"]');
  if (footer) {
    const assetPath = assets?.twin_finder;
    if (assetPath) {
      footer.innerHTML = `<a href="/${assetPath}" target="_blank" rel="noreferrer">Open structural overlay →</a>`;
    } else {
      footer.textContent = 'Visual overlay available in docs.';
    }
  }
}

function renderContextRefinery(data, assets) {
  const section = document.querySelector('[data-demo="context_refinery"]');
  if (!section || !data) return;
  const shareEl = section.querySelector('[data-field="structural_share"]');
  if (shareEl) {
    shareEl.textContent = fmtPercent(data.structural_share || 0, 2);
  }

  const renderList = (selector, items, formatter) => {
    const el = section.querySelector(selector);
    if (!el) return;
    el.innerHTML = '';
    (items || []).forEach((item) => {
      const li = document.createElement('li');
      li.innerHTML = formatter(item);
      el.appendChild(li);
    });
  };

  renderList('[data-field="top_structural"]', data.top_structural, (item) => `
    <span>${item.string}</span>
    <span class="value">occ ${fmtInteger(item.occurrences)}</span>
  `);

  renderList('[data-field="top_proposals"]', data.top_proposals, (item) => `
    <span>${item.string}</span>
    <span class="value">score ${fmtNumber(item.score, 3)}</span>
  `);

  const footer = section.querySelector('[data-field="chart"]');
  if (footer) {
    const assetPath = assets?.context_refinery;
    if (assetPath) {
      footer.innerHTML = `<a href="/${assetPath}" target="_blank" rel="noreferrer">Open foreground heat strip →</a>`;
    } else {
      footer.textContent = '';
    }
  }
}

async function loadDemo() {
  try {
    const payload = await fetchJson('/demo');
    setMeta({
      generated_at: payload.generated_at,
      sources: payload.sources,
    });
    const demos = payload.demos || {};
    renderPatternProphet(demos.pattern_prophet);
    renderTwinFinder(demos.twin_finder, payload.assets);
    renderContextRefinery(demos.context_refinery, payload.assets);
    showToast('Demo payload loaded');
  } catch (error) {
    console.error('Failed to load demo payload', error);
    showToast('Failed to load demo payload. Check API connectivity.', 'error');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  loadDemo();
  const metaEl = document.getElementById('meta');
  if (metaEl) {
    metaEl.addEventListener('click', () => {
      loadDemo();
      showToast('Refreshing demo payload…');
    });
  }
});
