const config = window.__STM_CONFIG__ || {};
const API_BASE = typeof config.apiBase === 'string' ? config.apiBase.replace(/\/$/, '') : '/api';
const LIVE_ENDPOINT = `${API_BASE}/demo/live`;
const QUICK_ANALYZE_ENDPOINT = `${API_BASE}/analyze/quick`;

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

function appendLiveFrame(frame) {
  const list = document.getElementById('live-feed-items');
  if (!list) return;
  const item = document.createElement('li');
  const signature = frame.signature || '(none)';
  item.innerHTML = `
    <span class="label">${fmtInteger(frame.index)}</span>
    <div>
      <div><strong>${signature}</strong></div>
      <div style="font-size: 0.8rem; opacity: 0.8;">
        coh ${fmtNumber(frame.metrics?.coherence, 4)} ·
        stab ${fmtNumber(frame.metrics?.stability, 4)} ·
        ent ${fmtNumber(frame.metrics?.entropy, 4)} ·
        λ ${fmtNumber(frame.metrics?.lambda_hazard, 4)}
      </div>
    </div>
  `;
  list.prepend(item);
  while (list.childElementCount > 20) {
    list.removeChild(list.lastElementChild);
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
      console.error('Invalid SSE payload', error);
    }
  };
  liveEventSource.onerror = (event) => {
    console.error('SSE stream error', event);
    stopLiveStream(button, false);
    setTimeout(() => startSseStream(button), 1500);
  };
  if (button) {
    button.dataset.state = 'running';
    button.textContent = 'Stop Live Stream';
    button.classList.remove('btn--ghost');
    button.classList.add('btn');
  }
  showToast('Live stream started (SSE fallback)');
}

function startLiveStream(button) {
  if (liveSocket || liveEventSource) return;
  try {
    liveSocket = new WebSocket(resolveWsEndpoint());
  } catch (error) {
    console.error('WebSocket init failed, falling back to SSE', error);
    startSseStream(button);
    return;
  }
  liveSocket.onopen = () => {
    if (button) {
      button.dataset.state = 'running';
      button.textContent = 'Stop Live Stream';
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
      console.error('Invalid WebSocket payload', error);
    }
  };
  liveSocket.onerror = (event) => {
    console.error('WebSocket stream error', event);
  };
  liveSocket.onclose = () => {
    const wasActive = liveSocket !== null;
    liveSocket = null;
    if (wasActive) {
      // Attempt fallback to SSE when WS drops unexpectedly
      if (!liveEventSource) {
        startSseStream(button);
      }
    }
  };
}

function stopLiveStream(button, notify = true) {
  if (liveSocket) {
    try {
      liveSocket.close();
    } catch (error) {
      console.error('Error closing WebSocket', error);
    }
    liveSocket = null;
  }
  if (liveEventSource) {
    liveEventSource.close();
    liveEventSource = null;
  }
  if (button) {
    button.dataset.state = 'idle';
    button.textContent = 'Start Live Stream';
    button.classList.add('btn--ghost');
  }
  if (notify) {
    showToast('Live stream stopped');
  }
}

async function handleQuickAnalyze(file) {
  const summaryCard = document.getElementById('quick-summary');
  const summaryEl = document.getElementById('quick-summary-text');
  const metricsCard = document.getElementById('quick-metrics');
  const numericEl = document.getElementById('quick-numeric');
  const recommendCard = document.getElementById('quick-recommend');
  const recommendationsEl = document.getElementById('quick-recommendations');
  const signatureCard = document.getElementById('quick-signature-card');
  const signaturesEl = document.getElementById('quick-signatures');
  const manifoldMetaEl = document.getElementById('quick-manifold-meta');
  if (!file || !summaryCard || !summaryEl || !metricsCard || !numericEl || !recommendCard || !recommendationsEl || !signatureCard || !signaturesEl || !manifoldMetaEl) {
    return;
  }
  summaryCard.hidden = false;
  summaryEl.textContent = 'Analyzing…';
  metricsCard.hidden = true;
  numericEl.innerHTML = '';
  recommendCard.hidden = true;
  recommendationsEl.innerHTML = '';
  signatureCard.hidden = true;
  signaturesEl.innerHTML = '';
  manifoldMetaEl.textContent = '';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(QUICK_ANALYZE_ENDPOINT, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      if (response.status === 413) {
        throw new Error('Uploaded file exceeds server size limits');
      }
      const text = await response.text();
      throw new Error(text || response.statusText);
    }
    const result = await response.json();
    summaryEl.textContent = `${result.rows ?? 0} rows · ${result.columns?.length ?? 0} columns`;
    metricsCard.hidden = false;

    (result.numeric_columns || []).forEach((name) => {
      const li = document.createElement('li');
      li.innerHTML = `<span>${name}</span>`;
      numericEl.appendChild(li);
    });
    if (numericEl.childElementCount === 0) {
      numericEl.innerHTML = '<li><span>No numeric columns detected</span></li>';
    }

    (result.metrics || []).forEach((metric) => {
      const li = document.createElement('li');
      li.innerHTML = `
        <span>${metric.column}</span>
        <span class="value">μ ${fmtNumber(metric.mean, 4)} · σ ${fmtNumber(metric.std_dev, 4)}</span>
      `;
      numericEl.appendChild(li);
    });

    Object.entries(result.recommendations || {}).forEach(([key, value]) => {
      const li = document.createElement('li');
      li.innerHTML = `<span>${key}</span><span class="value">${value}</span>`;
      recommendationsEl.appendChild(li);
    });
    recommendCard.hidden = recommendationsEl.childElementCount === 0;

    const manifold = result.manifold || {};
    (manifold.top_signatures || []).forEach((entry) => {
      const li = document.createElement('li');
      li.innerHTML = `
        <span>${entry.signature}</span>
        <span class="value">${fmtInteger(entry.count)} hits · coh ${fmtNumber(entry.mean_coherence, 4)}</span>
      `;
      signaturesEl.appendChild(li);
    });
    manifoldMetaEl.textContent = `Windows: ${fmtInteger(manifold.total_windows || 0)} · window=${manifold.window_bytes} stride=${manifold.stride}`;
    signatureCard.hidden = signaturesEl.childElementCount === 0;

    showToast('Quick analysis complete');
  } catch (error) {
    console.error('Quick analyze failed', error);
    summaryEl.textContent = 'Analysis failed';
    metricsCard.hidden = true;
    recommendCard.hidden = true;
    signatureCard.hidden = true;
    showToast(error.message || 'Failed to analyze file', 'error');
  }
}

async function analyzeText() {
  const textInput = document.getElementById('text-input');
  const summaryCard = document.getElementById('text-summary');
  const metricsEl = document.getElementById('text-metrics');
  const patternsCard = document.getElementById('text-patterns');
  const patternList = document.getElementById('text-pattern-list');
  const repeatingCard = document.getElementById('text-repeating');
  const repeatingList = document.getElementById('text-sequence-list');
  const interpretationCard = document.getElementById('text-interpretation');
  const interpretationList = document.getElementById('text-interpretation-list');
  if (!textInput || !summaryCard || !metricsEl || !patternsCard || !patternList || !repeatingCard || !repeatingList || !interpretationCard || !interpretationList) {
    return;
  }

  const text = textInput.value.trim();
  if (!text) {
    showToast('Please paste some text to analyze', 'error');
    return;
  }

  summaryCard.style.display = 'block';
  metricsEl.innerHTML = '<div><dt>Status</dt><dd>Analyzing…</dd></div>';
  patternsCard.style.display = 'none';
  patternList.innerHTML = '';
  repeatingCard.style.display = 'none';
  repeatingList.innerHTML = '';
  interpretationCard.style.display = 'none';
  interpretationList.innerHTML = '';

  try {
    const response = await fetch(`${API_BASE}/analyze/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || response.statusText);
    }
    const result = await response.json();

    const metrics = result.metrics || {};
    metricsEl.innerHTML = `
      <div><dt>Total Tokens</dt><dd>${fmtInteger(metrics.total_tokens ?? 0)}</dd></div>
      <div><dt>Unique Tokens</dt><dd>${fmtInteger(metrics.unique_tokens ?? 0)}</dd></div>
      <div><dt>Structural Coverage</dt><dd>${fmtPercent(metrics.structural_coverage ?? 0, 1)}</dd></div>
      <div><dt>Repetition Ratio</dt><dd>${fmtPercent(metrics.repetition_ratio ?? 0, 1)}</dd></div>
    `;

    const patterns = result.structural_patterns || [];
    if (patterns.length) {
      patternsCard.style.display = 'block';
      patternList.innerHTML = patterns
        .map(
          (pattern) => `
            <li>
              <span class="pattern-sig">${pattern.signature}</span>
              <span class="pattern-stats">${fmtInteger(pattern.count)} hits · coh ${fmtNumber(pattern.avg_coherence, 4)} · stab ${fmtNumber(pattern.avg_stability, 4)}</span>
            </li>
          `,
        )
        .join('');
    }

    const sequences = result.repeating_sequences || [];
    if (sequences.length) {
      repeatingCard.style.display = 'block';
      repeatingList.innerHTML = sequences
        .map(
          (sequence) => `
            <li>
              <span class="seq-token">"${sequence.token}"</span>
              <span class="seq-stats">${fmtInteger(sequence.frequency)} hits${
                sequence.periodicity ? ` · period ${fmtInteger(sequence.periodicity)}` : ''
              }</span>
            </li>
          `,
        )
        .join('');
    }

    const interpretation = result.interpretation || [];
    if (interpretation.length) {
      interpretationCard.style.display = 'block';
      interpretationList.innerHTML = interpretation.map((entry) => `<li>${entry}</li>`).join('');
    }

    showToast('Text analysis complete!');
  } catch (error) {
    console.error('Text analysis failed', error);
    metricsEl.innerHTML = '<div><dt>Status</dt><dd>Analysis failed</dd></div>';
    patternsCard.style.display = 'none';
    repeatingCard.style.display = 'none';
    interpretationCard.style.display = 'none';
    showToast(error.message || 'Text analysis failed', 'error');
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

  const liveToggle = document.getElementById('live-toggle');
  if (liveToggle) {
    liveToggle.addEventListener('click', () => {
      if (liveToggle.dataset.state === 'running') {
        stopLiveStream(liveToggle);
      } else {
        startLiveStream(liveToggle);
      }
    });
  }

  const quickBtn = document.getElementById('quick-upload-btn');
  const quickInput = document.getElementById('quick-upload');
  if (quickBtn && quickInput) {
    quickBtn.addEventListener('click', () => {
      const file = quickInput.files?.[0];
      if (!file) {
        showToast('Select a CSV file first', 'error');
        return;
      }
      void handleQuickAnalyze(file);
    });
  }

  const textInput = document.getElementById('text-input');
  const charCount = document.getElementById('text-char-count');
  if (textInput && charCount) {
    const updateCount = () => {
      charCount.textContent = `${textInput.value.length} characters`;
    };
    textInput.addEventListener('input', updateCount);
    updateCount();
  }

  const analyzeTextBtn = document.getElementById('text-analyze-btn');
  if (analyzeTextBtn) {
    analyzeTextBtn.addEventListener('click', () => {
      void analyzeText();
    });
  }

  const sampleBtn = document.getElementById('text-sample-btn');
  if (sampleBtn && textInput) {
    sampleBtn.addEventListener('click', () => {
      const sample = `Dear customer, thank you for your order.\nYour order number is 12345.\nDear customer, your shipment is ready.\nYour tracking number is 67890.`;
      textInput.value = sample;
      const event = new Event('input');
      textInput.dispatchEvent(event);
    });
  }
});
