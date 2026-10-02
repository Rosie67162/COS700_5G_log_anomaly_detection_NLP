/* Keep navigation and fetched research data in one local state. */
'use strict';
const state = {
  dataset: null, runs: [], runCache: new Map(), saved: null,
  page: 'logs', logPage: 1, logPages: 1, logRequest: 0, inspected: null,
  evalRun: '', evalModel: '', evalPage: 1, evalPages: 1, evalRequest: 0,
  compareGroup: 'logs', compareRun: '', comparisonRows: [],
  predictRun: '', predictRecord: null, jobId: null, jobTimer: null, jobKind: 'training',
  analysisRun: '', latestAnalysis: null, analysisPage: 1, analysisPages: 1,
};
const $ = id => document.getElementById(id);
const metrics = [
  ['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'],
  ['f1', 'F1'], ['fpr', 'False positive rate'],
];
const colours = ['var(--cyan)', 'var(--purple)', 'var(--pink)', 'var(--lime)', 'var(--yellow)'];

/* Escape dataset content before placing it in an HTML template. */
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[char]));
}
function number(value) { return value == null ? '—' : Number(value).toLocaleString(); }
function percent(value, places = 2) {
  return value == null || !Number.isFinite(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(places)}%`;
}
function pretty(value) { return String(value ?? '').replaceAll('_', ' '); }
function clip(value, max = 200) { const text = String(value ?? ''); return text.length > max ? `${text.slice(0, max)}…` : text; }
function query(params) {
  return new URLSearchParams(Object.entries(params).filter(([, value]) => value !== null && value !== undefined && value !== '')).toString();
}

/* Standardise labels supplied by the data and model endpoints. */
function groupName(group) { return group === 'sessions' ? 'TF-IDF / LogBERT-style' : 'TF-IDF / Word2Vec'; }
function classifierKey(value) {
  const key = String(value ?? '').toLowerCase().replaceAll('_', ' ');
  return key === 'rf' || key.includes('forest') ? 'rf' : key === 'lr' || key.includes('logistic') ? 'lr' : key;
}
function classifierName(value) {
  const key = classifierKey(value);
  return key === 'lr' ? 'Logistic Regression' : key === 'rf' ? 'Random Forest' : pretty(value);
}
function representationName(value, group = 'logs') {
  const embed = group === 'sessions' ? 'LogBERT-style' : 'Word2Vec';
  return ({ tfidf: 'TF-IDF', embedding: embed, concat: `TF-IDF + ${embed} concatenation`,
    weighted: `TF-IDF-weighted ${embed}`, fusion: `TF-IDF + ${embed} decision fusion`,
    word2vec: 'Word2Vec', logbert: 'LogBERT-style' })[value] || String(value ?? 'Unknown representation');
}
function shortRepresentation(value, group) {
  const name = representationName(value, group);
  if (/concat/i.test(name)) return 'Concatenation';
  if (/weighted/i.test(name)) return 'TF-IDF-weighted';
  if (/fusion/i.test(name)) return 'Decision fusion';
  return name;
}
function binaryLabel(value) {
  const text = String(value ?? '').toLowerCase();
  if (['1', 'abnormal', 'anomaly', 'anomalous', 'true'].includes(text)) return 'Anomaly';
  if (['0', 'normal', 'false'].includes(text)) return 'Normal';
  return 'Unlabelled';
}
function labelTag(value) {
  const label = binaryLabel(value);
  return `<span class="tag ${label === 'Anomaly' ? 'tag-anomaly' : label === 'Unlabelled' ? 'tag-unknown' : ''}">${label}</span>`;
}
function runGroup(run) { return run?.config?.group || run?.group || 'logs'; }
function completedRuns(group) {
  return state.runs.filter(run => (!run.status || run.status === 'completed') &&
    (!run.dataset_id || run.dataset_id === state.dataset?.id) && (!group || runGroup(run) === group));
}
function allCompletedRuns() { return state.runs.filter(run => !run.status || run.status === 'completed'); }
function runLabel(run) {
  const date = run.created_at ? new Date(run.created_at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : run.id;
  return `${groupName(runGroup(run))} · ${run.config?.profile || 'research'} · ${date}`;
}
function rowMessage(row) { return row.log_text ?? row.message ?? row.clean_text ?? ''; }

/* Surface backend failures without inventing fallback results. */
function alertUser(message, success = false) {
  $('alert-message').textContent = message;
  $('app-alert').classList.toggle('success', success);
  $('app-alert').hidden = false;
  $('app-alert').scrollIntoView({ block: 'nearest' });
}
function guarded(action) {
  return async event => {
    try { await action(event); }
    catch (error) { alertUser(error.message || 'The request could not be completed.'); }
  };
}
async function api(path, options = {}) {
  const response = await fetch(path, { credentials: 'same-origin', ...options });
  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json') ? await response.json() : null;
  if (!response.ok) throw new Error(data?.error || data?.message || `Request failed (${response.status}). Please check that the Flask server is running.`);
  if (data == null) throw new Error('The server returned an unexpected response. Please refresh the page.');
  return data;
}
function post(path, data) {
  return api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
}
async function getRun(id) {
  if (!id) return null;
  if (!state.runCache.has(id)) state.runCache.set(id, await api(`/api/runs/${encodeURIComponent(id)}`));
  return state.runCache.get(id);
}

/* Switch pages while preserving experiment selections and jobs. */
async function showPage(page, focus = false) {
  const allowed = ['logs', 'context', 'train', 'evaluate', 'compare', 'manual', 'dashboard'];
  state.page = allowed.includes(page) ? page : 'logs';
  document.querySelectorAll('.page').forEach(section => { section.hidden = section.id !== `page-${state.page}`; });
  document.querySelectorAll('[data-page]').forEach(link => {
    if (link.dataset.page === state.page) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  const title = $(`page-${state.page}`).querySelector('h1').textContent;
  document.title = `${title} · 5G Research Lab`;
  if (focus) $('main-content').focus({ preventScroll: true });
  if (!state.dataset) return;
  if (state.page === 'logs') await loadLogs();
  if (state.page === 'context') await loadContextTimeline();
  if (state.page === 'evaluate') await loadEvaluation();
  if (state.page === 'compare') await loadComparison();
  if (state.page === 'context') await loadContextTimeline();
  if (state.page === 'manual') await renderManualPage();
  if (state.page === 'dashboard') await loadDashboard();
}

/* Load dataset context and retain compatible saved-run choices. */
async function refreshState() {
  const previousDataset = state.dataset?.id;
  const data = await api('/api/state');
  state.dataset = data.dataset || null;
  state.latestAnalysis = data.latest_analysis && (!data.latest_analysis.dataset_id || data.latest_analysis.dataset_id === data.dataset?.id) ? data.latest_analysis : null;
  state.runs = (data.runs || []).slice().sort((a, b) => String(b.created_at || b.id).localeCompare(String(a.created_at || a.id)));
  if (previousDataset && previousDataset !== state.dataset?.id) {
    state.logPage = 1; state.inspected = null; state.predictRecord = null; state.evalRun = ''; state.evalModel = ''; state.compareRun = ''; state.predictRun = '';
    $('raw-log').textContent = 'Select a record from this dataset.'; $('clean-log').textContent = '—'; $('log-tokens').innerHTML = '';
    $('inspector-record').textContent = 'Select a row above to inspect it.';
  }
  if (!state.dataset) {
    $('connection-status').textContent = 'No dataset uploaded · waiting for CSV';
    $('dashboard-empty').hidden = false; $('dashboard-content').hidden = true;
    $('context-empty').hidden = false; $('context-content').hidden = true;
    $('manual-empty').hidden = false; $('manual-content').hidden = true;
    $('file-state').textContent = 'No dataset loaded. Upload a CSV file to begin.';
    renderRunSelectors(); renderTrainingSetup(); renderRunList();
    if (data.active_job && ['queued', 'running'].includes(data.active_job.status)) {
      state.jobKind = data.active_job.kind || 'training'; renderJob(data.active_job); if (state.jobId !== data.active_job.id) startPolling(data.active_job.id);
    }
    return data;
  }
  const hasModel = completedRuns().length > 0;
  $('dashboard-empty').hidden = hasModel;
  $('dashboard-content').hidden = !hasModel;
  if (!hasModel) {
    $('dashboard-empty').querySelector('h2').textContent = 'Train a model before viewing dashboard results';
    $('dashboard-empty').querySelector('p').textContent = 'Upload complete. Train at least one model in Train models before dashboard results are shown.';
    $('dashboard-empty').querySelector('a').textContent = 'Train a model ↗';
    $('dashboard-empty').querySelector('a').href = '#train';
  }
  $('context-empty').hidden = true; $('context-content').hidden = false;
  $('manual-empty').hidden = !hasModel; $('manual-content').hidden = hasModel ? false : true;
  $('connection-status').textContent = `${state.dataset.name} · ${number(state.dataset.row_count)} logs`;
  renderDataset(); renderRunSelectors(); renderTrainingSetup(); renderRunList();
  if (data.active_job && ['queued', 'running'].includes(data.active_job.status)) {
    state.jobKind = data.active_job.kind || 'training'; renderJob(data.active_job); if (state.jobId !== data.active_job.id) startPolling(data.active_job.id);
  }
  return data;
}

/* Derive class counts from the current dataset's actual labels. */
function datasetLabels() {
  const counts = state.dataset?.label_counts || {};
  let normal = 0; let anomaly = 0; let unlabelled = 0;
  Object.entries(counts).forEach(([key, value]) => {
    const label = binaryLabel(key);
    if (label === 'Normal') normal += Number(value) || 0;
    else if (label === 'Anomaly') anomaly += Number(value) || 0;
    else unlabelled += Number(value) || 0;
  });
  if (!Object.keys(counts).length) unlabelled = Number(state.dataset?.row_count || 0);
  return { normal, anomaly, unlabelled };
}
function pairsHtml(entries) {
  return entries.map(([label, value]) => `<div class="pair"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join('');
}
function distributionHtml(values, colour = 'var(--cyan)', omitNone = false) {
  const entries = Object.entries(values || {}).filter(([key]) => !omitNone || !['none', 'normal', ''].includes(key.toLowerCase())).sort((a, b) => Number(b[1]) - Number(a[1]));
  if (!entries.length) return '<p class="note">No category information is available in this dataset.</p>';
  const maximum = Math.max(...entries.map(([, value]) => Number(value)), 1);
  return entries.map(([label, value]) => `<div class="distribution-row"><div class="distribution-label"><span>${escapeHtml(pretty(label))}</span><strong>${number(value)}</strong></div><div class="distribution-track"><span style="width:${Math.max(0, Number(value) / maximum * 100)}%;background:${colour}"></span></div></div>`).join('');
}
function formatDetail(value) {
  if (value == null) return 'Not available';
  if (typeof value === 'number') return number(value);
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return value.map(item => typeof item === 'object' ? JSON.stringify(item) : String(item)).join(', ') || 'None';
  return Object.entries(value).map(([key, item]) => `${pretty(key)}: ${formatDetail(item)}`).join(' · ');
}

/* Render dashboard totals and the Data Context page from the API. */
function renderDataset() {
  const dataset = state.dataset;
  if (!dataset) return;
  const total = Number(dataset.row_count || 0);
  const labels = datasetLabels();
  const normalPct = total ? labels.normal / total : 0;
  const anomalyPct = total ? labels.anomaly / total : 0;
  $('context-total').textContent = number(total);
  $('context-normal').textContent = dataset.has_labels === false ? '—' : number(labels.normal);
  $('context-anomaly').textContent = dataset.has_labels === false ? '—' : number(labels.anomaly);
  $('context-normal-percent').textContent = dataset.has_labels === false ? 'Labels not supplied' : `${percent(normalPct, 1)} of records`;
  $('context-anomaly-percent').textContent = dataset.has_labels === false ? 'Labels not supplied' : `${percent(anomalyPct, 1)} of records`;
  $('label-balance').innerHTML = dataset.has_labels === false ? '<div class="empty-state">This dataset is unlabelled. Add known labels to train and evaluate models.</div>' :
    `<div class="balance-track" role="img" aria-label="${percent(normalPct, 1)} normal, ${percent(anomalyPct, 1)} anomalous"><span style="width:${normalPct * 100}%;background:var(--lime)"></span><span style="width:${anomalyPct * 100}%;background:var(--pink)"></span></div><div class="legend"><span><i class="swatch" style="background:var(--lime)"></i>Normal · ${(normalPct * 100).toFixed(1)}%</span><span><i class="swatch" style="background:var(--pink)"></i>Anomaly · ${(anomalyPct * 100).toFixed(1)}%</span>${labels.unlabelled ? `<span>Unlabelled · ${number(labels.unlabelled)}</span>` : ''}</div>`;
  $('context-unlabelled').textContent = number(labels.unlabelled);
  $('context-sessions').textContent = number(dataset.session_count);
  $('context-templates').textContent = number(dataset.template_count);
  $('context-unique').textContent = number(dataset.unique_clean_texts);
  $('data-source-description').textContent = typeof dataset.source === 'string' ? dataset.source : formatDetail(dataset.source || dataset.name);
  const partitions = Object.entries(dataset.split_counts || {});
  $('partition-list').innerHTML = partitions.length ? pairsHtml(partitions.map(([name, count]) => [`${pretty(name)} · ${percent(Number(count) / (total || 1), 1)}`, `${number(count)} logs`])) : '<p class="note">No pre-assigned partitions. Choose a custom grouped split when training.</p>';
  $('column-list').innerHTML = (dataset.columns || []).map(column => `<span class="chip">${escapeHtml(column)}</span>`).join('');
  $('function-bars').innerHTML = distributionHtml(dataset.network_function_counts);
  $('category-list').innerHTML = distributionHtml(dataset.anomaly_type_counts, 'var(--pink)', true);
  const audit = dataset.audit || {};
  $('audit-list').innerHTML = Object.keys(audit).length ? pairsHtml(Object.entries(audit).map(([key, value]) => [pretty(key), formatDetail(value)])) : '<p class="note">Dataset audit details are not available.</p>';
  $('training-dataset').textContent = `${number(total)} logs · full dataset`;
  $('file-state').textContent = `Active dataset: ${dataset.name} · ${number(total)} records · CSV upload`;
}

/* Build selectors from completed runs belonging to this dataset. */
function selectOptions(select, options, chosen, placeholder = 'Train a model first') {
  select.innerHTML = options.length ? options.map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join('') : `<option value="">${escapeHtml(placeholder)}</option>`;
  if (options.some(([value]) => value === chosen)) select.value = chosen;
  select.disabled = !options.length;
  return select.value;
}
function renderRunSelectors() {
  const runs = completedRuns();
  const options = runs.map(run => [run.id, runLabel(run)]);
  state.evalRun = selectOptions($('eval-run'), options, state.evalRun);
  const comparisonOptions = completedRuns(state.compareGroup).map(run => [run.id, runLabel(run)]);
  if (state.saved?.groups?.[state.compareGroup]?.length && state.dataset) comparisonOptions.push(['notebook', 'Original notebook saved results · reference']);
  state.compareRun = selectOptions($('comparison-run'), comparisonOptions, state.compareRun, 'No results for this experiment group');
}

/* Retrieve a page of logs from the selected display range. */
async function loadLogs() {
  const request = ++state.logRequest;
  const data = await api(`/api/logs?${query({ dataset_id: state.dataset?.id, limit: $('log-limit').value,
    page: state.logPage, page_size: 20, split: $('log-split').value, search: $('log-search').value.trim() })}`);
  if (request !== state.logRequest) return;
  const rows = data.rows || [];
  state.logPage = Number(data.page || 1); state.logPages = Number(data.pages || 1);
  $('log-rows').innerHTML = rows.length ? rows.map(row => `<tr class="log-row ${String(state.inspected?.record?.row_id) === String(row.row_id) ? 'is-selected' : ''}" data-record="${escapeHtml(row.row_id)}"><td>${escapeHtml(row.row_id)}</td><td>${escapeHtml(row.network_function || '—')}</td><td class="log-message">${escapeHtml(rowMessage(row))}${row.timestamp ? `<span class="mini">${escapeHtml(row.timestamp)}</span>` : ''}</td><td>${labelTag(row.target ?? row.label)}</td><td>${escapeHtml(pretty(row.split || '—'))}</td><td><button class="table-action" data-inspect="${escapeHtml(row.row_id)}" aria-label="Inspect record ${escapeHtml(row.row_id)}">Inspect ↗</button></td></tr>`).join('') : '<tr><td colspan="6" class="empty-cell">No logs match this search and partition.</td></tr>';
  const start = rows.length ? (state.logPage - 1) * 20 + 1 : 0;
  const last = rows.length ? start + rows.length - 1 : 0;
  $('log-range').textContent = `Showing ${number(start)}–${number(last)} of ${number(data.available ?? data.total)} displayed logs · page ${state.logPage} of ${state.logPages} · ${number(data.total)} matching records`;
  $('log-prev').disabled = state.logPage <= 1;
  $('log-next').disabled = state.logPage >= state.logPages || !rows.length;
}

/* Inspect the exact stored and recomputed text for one record. */
async function inspectRecord(id, scroll = true) {
  const data = await api(`/api/record/${encodeURIComponent(id)}?${query({ dataset_id: state.dataset.id })}`);
  state.inspected = data;
  const record = data.record || {};
  $('inspector-record').textContent = `Record ${record.row_id ?? id} · ${record.network_function || 'network function unavailable'}${record.session_id ? ` · ${record.session_id} · ${number(data.session_count)} logs in session` : ''}`;
  $('raw-log').textContent = rowMessage(record);
  $('clean-log').textContent = data.clean_text ?? record.clean_text ?? '';
  const tokens = Array.isArray(data.tokens) ? data.tokens : String(data.clean_text || '').split(/\s+/).filter(Boolean);
  $('log-tokens').innerHTML = tokens.map(token => `<span class="chip">${escapeHtml(token)}</span>`).join('');
  $('use-record').disabled = false;
  document.querySelectorAll('#log-rows tr[data-record]').forEach(row => row.classList.toggle('is-selected', row.dataset.record === String(id)));
  if (scroll) $('preprocessing-inspector').scrollIntoView({ block: 'start', behavior: 'smooth' });
}

/* Upload data without treating uploaded content as instructions. */
async function uploadDataset(event) {
  event.preventDefault();
  const file = $('dataset-file').files[0];
  if (!file) return;
  const form = new FormData(); form.append('file', file);
  $('upload-button').disabled = true; $('upload-button').textContent = 'Loading…';
  try {
    await api('/api/upload', { method: 'POST', body: form });
    state.logPage = 1; $('log-search').value = ''; $('log-split').value = '';
    await refreshState(); await loadLogs();
    alertUser(`Loaded ${state.dataset.name} with ${number(state.dataset.row_count)} records.`, true);
  } finally { $('upload-button').disabled = false; $('upload-button').textContent = 'Load file'; }
}

/* Apply a retained model to the complete active dataset. */
/* Display batch predictions without treating them as test-set metrics. */
/* Explain feature settings before fitting and adapt group controls. */
function renderTrainingSetup() {
  const group = $('train-group').value;
  const representation = $('train-representation').value || 'tfidf';
  const options = [['tfidf', 'TF-IDF'], ['embedding', representationName('embedding', group)],
    ['concat', representationName('concat', group)], ['weighted', representationName('weighted', group)],
    ['fusion', representationName('fusion', group)], ['all', 'All five representations']];
  selectOptions($('train-representation'), options, representation);
  $('custom-split').hidden = $('train-split').value !== 'custom';
  const test = Number($('test-percent').value || 15); const validation = Number($('validation-percent').value || 15);
  const unit = group === 'sessions' ? 'Unit: complete ordered session' : 'Unit: individual log entry';
  const split = $('train-split').value === 'custom' ? `Train ${100 - test - validation}% / validation ${validation}% / test ${test}%` : 'Supplied train / validation / test partitions';
  $('training-unit').textContent = `${unit} · ${split} · related sessions remain together`;
  const selected = $('train-representation').value;
  $('train-vocabulary').textContent = group === 'sessions' ? 'TF-IDF cap 25,000 · training templates' : 'TF-IDF cap 20,000 · training words';
  $('train-dimension').textContent = selected === 'tfidf' ? 'Not used for TF-IDF' : group === 'sessions' ? '64 features · LogBERT-style' : '100 features · Word2Vec';
  $('train-vector').textContent = selected === 'fusion' ? 'Two model scores fused' : selected === 'concat' ? 'TF-IDF features + embedding' : 'Measured after fitting';
  $('profile-note').textContent = $('train-profile').value === 'research' ? 'Full dataset · 300 forest trees · 10 Word2Vec epochs or 12 LogBERT-style epochs. Keep the profile constant for controlled comparisons.' : 'Full dataset · 120 forest trees · 5 Word2Vec or LogBERT-style epochs. Quick-profile results are separate from research-profile runs.';
  const needsLabels = state.dataset?.has_labels === false;
  const needsSessions = group === 'sessions' && state.dataset?.can_sessions === false;
  const running = Boolean(state.jobId);
  $('train-selected').disabled = running || needsLabels || needsSessions;
  $('train-all').disabled = running || needsLabels || needsSessions;
  $('train-availability').textContent = needsLabels ? 'This dataset needs known normal/anomaly labels before training.' : needsSessions ? 'Session experiments need session_id and template_id columns. Choose Category 1 or load a session-ready dataset.' : running ? 'A training job is already running. You can continue exploring the dataset.' : 'Training runs in the background. You can explore other pages while it completes.';
}

/* Start actual training with shared partitions and validation. */
async function beginTraining(all = false) {
  const test = Number($('test-percent').value); const validation = Number($('validation-percent').value);
  if ($('train-split').value === 'custom' && (!Number.isFinite(test) || !Number.isFinite(validation) || test < 5 || test > 40 || validation < 5 || validation > 30 || test + validation >= 70)) {
    throw new Error('Choose 5–40% test data and 5–30% validation data, leaving more than 30% for training.');
  }
  $('train-selected').disabled = true; $('train-all').disabled = true;
  try {
    const result = await post('/api/train', {
      group: $('train-group').value, representation: all ? 'all' : $('train-representation').value,
      classifier: all ? 'all' : $('train-classifier').value, split_mode: $('train-split').value,
      test_size: test / 100, validation_size: validation / 100, profile: $('train-profile').value,
    });
    if (!result.job_id) throw new Error('The server did not provide a training job ID.');
    state.jobKind = 'training';
    renderJob({ id: result.job_id, status: 'queued', progress: 0, message: 'Preparing the selected dataset and shared partitions…' });
    startPolling(result.job_id);
    $('training-progress').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  } catch (error) { renderTrainingSetup(); throw error; }
}

/* Keep real job progress visible, including failure details. */
function renderJob(job) {
  const analysis = job.kind === 'analysis' || state.jobKind === 'analysis';
  const running = ['queued', 'running'].includes(job.status);
  const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
  $('training-progress').hidden = false;
  $('job-title').textContent = job.status === 'completed' ? (analysis ? 'Dataset analysis complete' : 'Experiment complete') : job.status === 'failed' ? 'Job could not finish' : analysis ? 'Dataset analysis progress' : 'Training progress';
  $('job-percent').textContent = `${Math.round(progress)}%`;
  $('job-progress').value = progress;
  $('job-progress').textContent = `${Math.round(progress)}%`;
  $('job-message').textContent = job.error || job.message || 'Preparing models…';
  $('job-actions').hidden = job.status !== 'completed' || analysis;
  $('global-job').hidden = !running;
  $('global-job-title').textContent = `${job.status === 'queued' ? 'Job queued' : analysis ? 'Analysing dataset' : 'Training models'} · ${Math.round(progress)}%`;
  $('global-job-message').textContent = job.message || 'Running the selected experiment';
}
function startPolling(id) {
  clearTimeout(state.jobTimer); state.jobId = id; renderTrainingSetup();
  state.jobTimer = setTimeout(pollJob, 700);
}
async function pollJob() {
  const id = state.jobId;
  if (!id) return;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
    renderJob(job);
    if (job.status === 'completed' || job.status === 'failed') {
      state.jobId = null;
      if (job.status === 'failed') alertUser(job.error || job.message || 'Training failed. Review the dataset and settings, then try again.');
      if (job.run_id && job.status === 'completed' && state.jobKind !== 'analysis') {
        state.runCache.delete(job.run_id);
        const run = await getRun(job.run_id);
        state.evalRun = job.run_id; state.predictRun = job.run_id;
        state.compareGroup = runGroup(run); state.compareRun = job.run_id; state.evalModel = ''; state.evalPage = 1;
        renderFeatureDetails(run);
      }
      await refreshState();
      if (job.analysis_id && job.status === 'completed') {
        state.latestAnalysis = { id: job.analysis_id, dataset_id: state.dataset?.id, run_id: job.run_id };
        state.analysisPage = 1;
      }
      if (job.status === 'completed') await showPage(state.page);
      renderTrainingSetup();
      return;
    }
    state.jobTimer = setTimeout(pollJob, 1500);
  } catch (error) {
    if (!state.jobId) { alertUser(error.message || 'The job finished, but its results could not be loaded. Refresh Saved experiments.'); renderTrainingSetup(); return; }
    $('job-message').textContent = `${error.message} Reconnecting to the training job…`;
    state.jobTimer = setTimeout(pollJob, 5000);
  }
}

/* Show persisted experiments and their measured feature dimensions. */
function renderRunList() {
  const runs = completedRuns();
  $('training-runs').classList.toggle('empty-state', !runs.length);
  $('training-runs').innerHTML = runs.length ? runs.map(run => `<div class="run-card"><div><strong>${escapeHtml(groupName(runGroup(run)))}</strong><small>${escapeHtml(runLabel(run))}</small><small>${escapeHtml(run.id)}${run.results ? ` · ${run.results.length} configurations` : ''}</small></div><div class="row"><button class="button small" data-open-run="${escapeHtml(run.id)}">View results ↗</button><button class="button small" data-features-run="${escapeHtml(run.id)}">Feature settings</button></div></div>`).join('') : 'No experiments trained yet. Your completed runs will be saved here.';
}
function renderFeatureDetails(run) {
  $('fitted-features').hidden = false;
  $('feature-run-label').textContent = runLabel(run);
  const info = run.feature_info || {};
  const entries = Object.entries(info);
  $('feature-details').innerHTML = entries.length ? pairsHtml(entries.map(([key, value]) => [pretty(key), formatDetail(value)])) : '<p class="note">Feature settings were not recorded for this run.</p>';
  if (run.training_history?.length) {
    $('feature-details').innerHTML += `<details class="feature-details"><summary>Training history</summary><pre class="code">${escapeHtml(JSON.stringify(run.training_history, null, 2))}</pre></details>`;
  }
}

/* Load one saved run and reconcile its selected model's test metrics. */
async function loadEvaluation() {
  const request = ++state.evalRequest;
  if (!state.evalRun) {
    $('evaluation-empty').hidden = false; $('evaluation-content').hidden = true;
    $('export-predictions').hidden = true; selectOptions($('eval-model'), [], ''); return;
  }
  const run = await getRun(state.evalRun);
  if (request !== state.evalRequest) return;
  const results = run.results || [];
  state.evalModel = selectOptions($('eval-model'), results.map(result => [result.model_id, `${representationName(result.representation, runGroup(run))} · ${classifierName(result.classifier)}`]), state.evalModel, 'No trained configurations');
  const result = results.find(item => item.model_id === state.evalModel);
  $('evaluation-empty').hidden = Boolean(result); $('evaluation-content').hidden = !result;
  $('export-predictions').hidden = !result;
  if (!result) return;
  const unit = result.unit || (runGroup(run) === 'sessions' ? 'sessions' : 'log entries');
  $('evaluation-description').textContent = `${number(result.test_count)} held-out ${unit} · ${run.config?.profile || 'research'} profile · ${run.id} · threshold selected using validation data`;
  $('evaluation-metrics').innerHTML = [['precision', 'Precision'], ['recall', 'Recall'], ['f1', 'F1'], ['accuracy', 'Accuracy'], ['fpr', 'False positive rate']].map(([key, label]) => `<article class="metric"><span>${label}</span><strong>${percent(result[key])}</strong></article>`).join('');
  const matrix = result.confusion_matrix || [[result.tn, result.fp], [result.fn, result.tp]];
  const [[tn, fp], [fn, tp]] = matrix;
  $('confusion-matrix').innerHTML = `<div></div><div class="matrix-axis">Predicted<br>normal</div><div class="matrix-axis">Predicted<br>anomaly</div><div class="matrix-axis">Actual<br>normal</div><div class="matrix-cell"><strong>${number(tn)}</strong>True negatives</div><div class="matrix-cell error"><strong>${number(fp)}</strong>False positives</div><div class="matrix-axis">Actual<br>anomaly</div><div class="matrix-cell missed"><strong>${number(fn)}</strong>False negatives</div><div class="matrix-cell"><strong>${number(tp)}</strong>True positives</div>`;
  $('confusion-matrix').setAttribute('aria-label', `Actual normal: ${tn} predicted normal, ${fp} predicted anomaly. Actual anomaly: ${fn} predicted normal, ${tp} predicted anomaly.`);
  $('eval-threshold').textContent = result.threshold == null ? '—' : Number(result.threshold).toFixed(4);
  $('export-predictions').href = `/api/export/${encodeURIComponent(state.evalRun)}?${query({ kind: 'predictions', model_id: state.evalModel })}`;
  await loadEvidence(request);
}

/* Fetch documented TP, FP, FN and TN records with pagination. */
async function loadEvidence(request = state.evalRequest) {
  if (!state.evalRun || !state.evalModel) return;
  const data = await api(`/api/runs/${encodeURIComponent(state.evalRun)}/predictions?${query({ model_id: state.evalModel,
    outcome: $('outcome-filter').value, page: state.evalPage, page_size: 20 })}`);
  if (request !== state.evalRequest) return;
  state.evalPage = Number(data.page || 1); state.evalPages = Number(data.pages || 1);
  const rows = data.rows || [];
  $('evidence-rows').innerHTML = rows.length ? rows.map(row => `<tr><td class="log-message"><strong>${escapeHtml(row.unit_id ?? row.row_id ?? row.session_id)}</strong><span class="mini">${escapeHtml(clip(row.log_text, 260))}</span>${row.known_anomaly_type && row.known_anomaly_type !== 'none' ? `<span class="mini">Known category: ${escapeHtml(pretty(row.known_anomaly_type))}</span>` : ''}</td><td>${labelTag(row.actual)}</td><td>${labelTag(row.predicted)}</td><td class="numeric">${percent(row.score)}</td><td><span class="tag ${row.outcome === 'FP' ? 'tag-anomaly' : row.outcome === 'FN' ? 'tag-fn' : ''}">${escapeHtml(row.outcome)}</span></td></tr>`).join('') : '<tr><td colspan="5" class="empty-cell">No test records have this outcome for the selected configuration.</td></tr>';
  const start = rows.length ? (state.evalPage - 1) * 20 + 1 : 0;
  $('evidence-range').textContent = `${number(start)}–${number(rows.length ? start + rows.length - 1 : 0)} of ${number(data.total)} test records · page ${state.evalPage} of ${state.evalPages}`;
  $('evidence-prev').disabled = state.evalPage <= 1;
  $('evidence-next').disabled = state.evalPage >= state.evalPages || !rows.length;
}

/* Compare configurations from one run or clearly labelled notebook results. */
async function loadComparison() {
  document.querySelectorAll('[data-compare-group]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.compareGroup === state.compareGroup)));
  renderRunSelectors();
  let rows = []; let source = ''; let run = null;
  if (state.compareRun === 'notebook') {
    rows = state.saved?.groups?.[state.compareGroup] || [];
    source = `${state.saved?.source || 'Original notebook saved results'} · ${groupName(state.compareGroup)} · reference results from the supplied notebook; these were not retrained in this app`;
  } else if (state.compareRun) {
    run = await getRun(state.compareRun); rows = run.results || [];
    source = `${groupName(runGroup(run))} · ${run.config?.profile || 'research'} profile · ${run.id} · all configurations share the same training, validation and test partitions`;
  } else source = 'No measured results are available for this group. Train configurations to compare them.';
  state.comparisonRows = rows;
  $('comparison-source').textContent = source;
  $('export-metrics').hidden = !run;
  if (run) $('export-metrics').href = `/api/export/${encodeURIComponent(run.id)}?kind=metrics`;
  const best = Object.fromEntries(metrics.map(([key]) => {
    const values = rows.map(row => row[key]).filter(value => value !== null && value !== undefined && Number.isFinite(Number(value))).map(Number);
    return [key, values.length ? (key === 'fpr' ? Math.min(...values) : Math.max(...values)) : null];
  }));
  $('comparison-rows').innerHTML = rows.length ? rows.map(row => `<tr><td>${escapeHtml(representationName(row.representation, state.compareGroup))}<span class="mini">${escapeHtml(classifierName(row.classifier))}</span></td>${metrics.map(([key]) => {
    const winner = row[key] != null && best[key] != null && Math.abs(Number(row[key]) - best[key]) < 1e-12;
    return `<td class="numeric ${winner ? 'best' : ''}">${percent(row[key])}${winner ? '<span class="best-label">BEST</span>' : ''}</td>`;
  }).join('')}</tr>`).join('') : '<tr><td colspan="6" class="empty-cell">No results yet. Train all 10 configurations in this group for a complete comparison.</td></tr>';
  renderCharts();
}

/* Use CSS bars to chart every requested metric without remote libraries. */
function renderCharts() {
  const selected = $('chart-classifier').value;
  const rows = state.comparisonRows.filter(row => selected === 'all' || classifierKey(row.classifier) === selected);
  $('comparison-charts').innerHTML = rows.length ? metrics.map(([key, label]) => `<article class="chart"><h2>${label} ${key === 'fpr' ? '↓' : '↑'}</h2>${rows.map((row, index) => {
    const value = Number(row[key]); const valid = row[key] != null && Number.isFinite(value);
    const width = valid ? Math.max(0, Math.min(100, value * 100)) : 0;
    return `<div class="bar-row"><span class="bar-label">${escapeHtml(shortRepresentation(row.representation, state.compareGroup))}${selected === 'all' ? `<span class="mini">${classifierKey(row.classifier) === 'lr' ? 'Logistic regression' : 'Random forest'}</span>` : ''}</span><div class="bar-track" role="img" aria-label="${escapeHtml(representationName(row.representation, state.compareGroup))} ${escapeHtml(classifierName(row.classifier))}: ${percent(row[key])}"><div class="bar" style="width:${width}%;background:${key === 'fpr' ? 'var(--pink)' : colours[index % colours.length]}"></div></div><span class="bar-value">${valid ? (value * 100).toFixed(2) : '—'}</span></div>`;
  }).join('')}<div class="chart-axis"><span>0%</span><span>100%</span></div></article>`).join('') : '';
}

async function loadContextTimeline() {
  const latest = completedRuns()[0];
  if (!latest) {
    $('context-timeline').className = 'empty-state'; $('context-timeline').textContent = 'Train a model to view predicted anomaly counts over time.';
    $('context-timeline-source').textContent = 'Awaiting a trained model'; $('context-prediction-summary').textContent = ''; return;
  }
  const run = await getRun(latest.id);
  const result = (run.results || []).find(item => item.model_id === run.recommended_model_id) || run.results?.[0];
  if (!result) return;
  const data = await api(`/api/runs/${encodeURIComponent(run.id)}/predictions?${query({ model_id: result.model_id, page: 1, page_size: 1 })}`);
  $('context-timeline-note').textContent = 'Predicted anomalies in the held-out test set for the best model from the latest completed run.';
  $('context-timeline-source').textContent = `${shortRepresentation(result.representation, runGroup(run))} · ${classifierKey(result.classifier).toUpperCase()}`;
  $('context-prediction-summary').textContent = `Predicted normal ${number(data.summary?.normal)} · predicted anomalous ${number(data.summary?.anomaly)} · ${number(result.test_count)} held-out ${result.unit || 'records'}`;
  renderTimelineInto('context-timeline', Array.isArray(data.timeline) ? data.timeline : []);
}

function renderTimelineInto(id, timeline) {
  const target = $(id);
  if (!timeline.length) { target.className = 'empty-state'; target.textContent = 'No timestamped anomaly predictions are available for this configuration.'; return; }
  const max = Math.max(...timeline.map(point => Number(point.count) || 0), 1);
  const chartWidth = 900; const chartHeight = 160; const left = 35; const bottom = 145;
  const span = (chartWidth - left - 12) / timeline.length;
  target.className = '';
  target.innerHTML = `<svg class="timeline-svg" viewBox="0 0 ${chartWidth} ${chartHeight}" role="img" aria-label="Predicted anomaly counts over time. Peak count ${max}."><line x1="${left}" y1="${bottom}" x2="${chartWidth - 8}" y2="${bottom}" stroke="#34425d"/><line x1="${left}" y1="20" x2="${chartWidth - 8}" y2="20" stroke="#26334c" stroke-dasharray="4 5"/><text x="0" y="25">${max}</text><text x="15" y="148">0</text>${timeline.map((point, index) => { const height = (Number(point.count) || 0) / max * 118; return `<rect x="${left + index * span + 2}" y="${bottom - height}" width="${Math.max(.5, span - 4)}" height="${height}" rx="2" fill="var(--pink)"><title>${escapeHtml(point.time)}: ${Number(point.count) || 0} anomalies</title></rect>`; }).join('')}</svg><div class="timeline-labels"><span>${escapeHtml(timeline[0].time)}</span><span>${escapeHtml(timeline[timeline.length - 1].time)}</span></div>`;
}

async function beginDashboardAnalysis(run) {
  if (!state.dataset || !run) return;
  if (state.jobId) return;
  const existing = state.latestAnalysis;
  if (existing && existing.dataset_id === state.dataset.id && existing.run_id === run.id) return;
  $('dashboard-analysis-status').textContent = 'Analysing the uploaded dataset with the best model…';
  $('dashboard-analysis-summary').textContent = 'Please wait while predictions are generated for the complete dataset.';
  state.jobKind = 'analysis';
  try {
    const result = await post('/api/analyze', { run_id: run.id, dataset_id: state.dataset.id });
    renderJob({ id: result.job_id, kind: 'analysis', status: 'queued', progress: 0, message: 'Preparing the full dataset for prediction…' });
    startPolling(result.job_id);
  } catch (error) {
    $('dashboard-analysis-status').textContent = error.message || 'Dataset analysis could not be started.';
    throw error;
  }
}

async function loadDashboard() {
  const run = completedRuns()[0];
  const hasModel = Boolean(state.dataset && run);
  if (!state.dataset || !hasModel) {
    $('dashboard-empty').hidden = false;
    $('dashboard-content').hidden = true;
    $('dashboard-empty').querySelector('h2').textContent = state.dataset ? 'Train a model before viewing dashboard results' : 'Dashboard waiting for your experiment';
    $('dashboard-empty').querySelector('p').textContent = state.dataset ? 'Upload complete. Train at least one model in Train models before dashboard results are shown.' : 'Upload a CSV dataset in Logs & preprocessing and train at least one model before dashboard results are shown.';
    $('dashboard-empty').querySelector('a').textContent = state.dataset ? 'Train a model ↗' : 'Upload CSV ↗';
    $('dashboard-empty').querySelector('a').href = state.dataset ? '#train' : '#logs';
    return;
  }
  $('dashboard-empty').hidden = true; $('dashboard-content').hidden = false;
  const meta = await getRun(run.id);
  const result = (meta.results || []).find(item => item.model_id === meta.recommended_model_id) || meta.results?.[0];
  if (!result) return;
  $('dashboard-total').textContent = number(state.dataset.row_count);
  $('dashboard-dataset-name').textContent = state.dataset.name;
  $('dashboard-model-name').textContent = `${shortRepresentation(result.representation, runGroup(run))} · ${classifierName(result.classifier)}`;
  $('dashboard-model-detail').textContent = `${representationName(result.representation, runGroup(run))} · ${classifierName(result.classifier)} · ${run.id}`;
  $('dashboard-metrics').innerHTML = [['accuracy','Accuracy'],['precision','Precision'],['recall','Recall'],['f1','F1'],['fpr','False positive rate']].map(([key,label]) => `<article class="metric"><span>${label}</span><strong>${percent(result[key])}</strong></article>`).join('');
  const matrix = result.confusion_matrix || [[result.tn, result.fp],[result.fn,result.tp]];
  const [[tn,fp],[fn,tp]] = matrix;
  $('dashboard-matrix').innerHTML = `<div></div><div class="matrix-axis">Predicted<br>normal</div><div class="matrix-axis">Predicted<br>anomaly</div><div class="matrix-axis">Actual<br>normal</div><div class="matrix-cell"><strong>${number(tn)}</strong>True negatives</div><div class="matrix-cell error"><strong>${number(fp)}</strong>False positives</div><div class="matrix-axis">Actual<br>anomaly</div><div class="matrix-cell missed"><strong>${number(fn)}</strong>False negatives</div><div class="matrix-cell"><strong>${number(tp)}</strong>True positives</div>`;
  $('dashboard-counts').textContent = `Held-out test counts · TP ${number(tp)} · FP ${number(fp)} · FN ${number(fn)} · TN ${number(tn)}.`;
  const analysis = state.latestAnalysis && state.latestAnalysis.dataset_id === state.dataset.id && state.latestAnalysis.run_id === run.id
    ? await api(`/api/analyses/${encodeURIComponent(state.latestAnalysis.id)}?page=${state.analysisPage}&page_size=20`)
    : null;
  if (!analysis) {
    $('dashboard-predicted').textContent = '…';
    $('dashboard-normal').textContent = '…';
    $('dashboard-anomaly').textContent = '…';
    $('dashboard-analysis-model').textContent = `${shortRepresentation(result.representation, runGroup(run))} · ${classifierName(result.classifier)}`;
    $('dashboard-analysis-status').textContent = state.jobId && state.jobKind === 'analysis' ? 'Analysing the complete uploaded dataset with the best model…' : 'Preparing automatic dataset analysis with the best model…';
    $('dashboard-analysis-summary').textContent = 'The table will populate automatically when analysis completes.';
    $('dashboard-analysis-rows').innerHTML = '<tr><td colspan="5" class="empty-cell">Analysing the uploaded dataset…</td></tr>';
    $('dashboard-analysis-range').textContent = '—';
    $('dashboard-analysis-prev').disabled = true; $('dashboard-analysis-next').disabled = true;
    if (!state.jobId) await beginDashboardAnalysis(run);
    return;
  }
  $('dashboard-predicted').textContent = number(analysis.total);
  $('dashboard-normal').textContent = number(analysis.summary?.normal);
  $('dashboard-anomaly').textContent = number(analysis.summary?.anomaly);
  $('dashboard-analysis-model').textContent = `${shortRepresentation((analysis.model||{}).representation, runGroup(run))} · ${classifierName((analysis.model||{}).classifier)}`;
  $('dashboard-analysis-status').textContent = `Full dataset analysis · ${number(analysis.total)} ${analysis.unit === 'session' ? 'sessions' : 'logs'} analysed.`;
  $('dashboard-analysis-summary').textContent = `${number(analysis.summary?.normal)} predicted normal · ${number(analysis.summary?.anomaly)} predicted anomalous`;
  const rows = analysis?.rows || [];
  $('dashboard-analysis-rows').innerHTML = rows.length ? rows.map((row,index) => `<tr class="expandable-row" data-dashboard-detail="${index}"><td>${escapeHtml(row.unit_id ?? row.row_id ?? row.session_id)}</td><td class="log-message">${escapeHtml(clip(row.cleaned_message ?? row.log_text, 300))}</td><td>${labelTag(row.predicted)}</td><td class="numeric">${percent(row.score)}</td><td>${labelTag(row.known_label ?? row.actual)}</td></tr><tr class="dashboard-detail" id="dashboard-detail-${index}" hidden><td colspan="5"><strong>English explainability</strong><p class="note">${escapeHtml(row.explainability || 'No explanation was recorded.')}</p></td></tr>`).join('') : '<tr><td colspan="5" class="empty-cell">No dataset analysis results were returned.</td></tr>';
  state.analysisPage = Number(analysis.page || 1); state.analysisPages = Number(analysis.pages || 1);
  $('dashboard-analysis-range').textContent = `${number((state.analysisPage-1)*20 + (rows.length ? 1 : 0))}–${number((state.analysisPage-1)*20 + rows.length)} of ${number(analysis.total)} analysed units · page ${state.analysisPage} of ${state.analysisPages}`;
  $('dashboard-analysis-prev').disabled = state.analysisPage <= 1; $('dashboard-analysis-next').disabled = state.analysisPage >= state.analysisPages;
}

async function renderManualPage() {
  const hasModel = Boolean(state.dataset && completedRuns().length);
  $('manual-empty').hidden = hasModel; $('manual-content').hidden = !hasModel;
  if (!hasModel) return;
  const run = completedRuns()[0];
  const meta = await getRun(run.id);
  const result = (meta.results || []).find(item => item.model_id === meta.recommended_model_id) || meta.results?.[0];
  $('manual-model-note').textContent = result ? `Best model: ${shortRepresentation(result.representation, runGroup(meta))} · ${classifierName(result.classifier)}` : 'No recommended model is available.';
  $('manual-analyse').disabled = !result;
}

function createManualInputs() {
  const count = Math.max(1, Math.min(100, Number($('manual-count').value) || 1));
  $('manual-count').value = count;
  $('manual-input-count').textContent = `${count} log${count === 1 ? '' : 's'}`;
  $('manual-inputs').innerHTML = Array.from({length: count}, (_, index) => `<label class="field"><span>Log ${index + 1}</span><textarea data-manual-log rows="4" placeholder="Enter sample 5G log ${index + 1}…"></textarea></label>`).join('');
  $('manual-status').textContent = `Enter ${count} sample log${count === 1 ? '' : 's'}, then press Analyse logs.`;
  $('manual-results').hidden = true;
}

async function analyseManualLogs() {
  const inputs = [...document.querySelectorAll('[data-manual-log]')];
  if (!inputs.length) throw new Error('Create the requested number of log inputs first.');
  const logs = inputs.map(input => input.value.trim());
  if (logs.some(value => !value)) throw new Error('Please enter text in every sample log field.');
  $('manual-analyse').disabled = true; $('manual-analyse').textContent = 'Analysing…'; $('manual-status').textContent = 'Applying the best trained model to the sample logs…';
  try {
    const data = await post('/api/manual-analyze', { dataset_id: state.dataset.id, logs });
    const run = await getRun(data.run_id);
    const model = data.model || {};
    $('manual-results-model').textContent = `${shortRepresentation(model.representation, runGroup(run))} · ${classifierName(model.classifier)}`;
    $('manual-results-summary').textContent = `${number(data.total)} logs analysed · ${number(data.summary?.normal)} predicted normal · ${number(data.summary?.anomaly)} predicted anomalous`;
    $('manual-result-rows').innerHTML = (data.rows || []).map((row,index) => `<tr class="expandable-row" data-manual-detail="${index}"><td>${escapeHtml(row.record)}</td><td class="log-message">${escapeHtml(row.cleaned_message)}</td><td>${labelTag(row.predicted)}</td><td class="numeric">${percent(row.score)}</td><td>${labelTag(row.known_label)}</td></tr><tr class="manual-detail" id="manual-detail-${index}" hidden><td colspan="5"><strong>English explainability</strong><p class="note">${escapeHtml(row.explainability)}</p></td></tr>`).join('');
    $('manual-results').hidden = false; $('manual-status').textContent = 'Analysis complete. Click a log row to expand its English explainability.';
    $('manual-results').scrollIntoView({block:'nearest', behavior:'smooth'});
  } finally { $('manual-analyse').disabled = false; $('manual-analyse').textContent = 'Analyse logs'; }
}

/* Draw a real held-out anomaly timeline for the latest trained model. */
async function loadDashboardPredictions() { return loadDashboard(); }


/* Attach navigation, data loading and experiment controls. */
window.addEventListener('hashchange', guarded(() => showPage(location.hash.slice(1), true)));
$('dismiss-alert').addEventListener('click', () => { $('app-alert').hidden = true; });
$('upload-form').addEventListener('submit', guarded(uploadDataset));
for (const id of ['log-limit', 'log-split']) $(id).addEventListener('change', guarded(async () => { state.logPage = 1; await loadLogs(); }));
$('log-search-form').addEventListener('submit', guarded(async event => { event.preventDefault(); state.logPage = 1; await loadLogs(); }));
$('log-prev').addEventListener('click', guarded(async () => { state.logPage = Math.max(1, state.logPage - 1); await loadLogs(); }));
$('log-next').addEventListener('click', guarded(async () => { state.logPage = Math.min(state.logPages, state.logPage + 1); await loadLogs(); }));
$('log-rows').addEventListener('click', guarded(async event => { const row = event.target.closest('tr[data-record]'); if (row) await inspectRecord(row.dataset.record); }));
$('use-record').addEventListener('click', guarded(() => { if (state.inspected?.record) location.hash = 'manual'; }));
for (const id of ['train-group', 'train-representation', 'train-classifier', 'train-split', 'train-profile', 'test-percent', 'validation-percent']) $(id).addEventListener('change', renderTrainingSetup);
$('training-form').addEventListener('submit', guarded(async event => { event.preventDefault(); await beginTraining(); }));
$('train-all').addEventListener('click', guarded(() => beginTraining(true)));
$('refresh-runs').addEventListener('click', guarded(async () => { await refreshState(); }));
$('training-runs').addEventListener('click', guarded(async event => {
  const results = event.target.closest('[data-open-run]'); const features = event.target.closest('[data-features-run]');
  if (results) { state.evalRun = results.dataset.openRun; state.evalModel = ''; state.evalPage = 1; renderRunSelectors(); location.hash = 'evaluate'; }
  if (features) { renderFeatureDetails(await getRun(features.dataset.featuresRun)); $('fitted-features').scrollIntoView({ block: 'start', behavior: 'smooth' }); }
}));
$('eval-run').addEventListener('change', guarded(async () => { state.evalRun = $('eval-run').value; state.evalModel = ''; state.evalPage = 1; await loadEvaluation(); }));
$('eval-model').addEventListener('change', guarded(async () => { state.evalModel = $('eval-model').value; state.evalPage = 1; await loadEvaluation(); }));
$('outcome-filter').addEventListener('change', guarded(async () => { state.evalPage = 1; await loadEvidence(++state.evalRequest); }));
$('evidence-prev').addEventListener('click', guarded(async () => { state.evalPage = Math.max(1, state.evalPage - 1); await loadEvidence(++state.evalRequest); }));
$('evidence-next').addEventListener('click', guarded(async () => { state.evalPage = Math.min(state.evalPages, state.evalPage + 1); await loadEvidence(++state.evalRequest); }));
document.querySelectorAll('[data-compare-group]').forEach(button => button.addEventListener('click', guarded(async () => { state.compareGroup = button.dataset.compareGroup; state.compareRun = ''; await loadComparison(); })));
$('comparison-run').addEventListener('change', guarded(async () => { state.compareRun = $('comparison-run').value; await loadComparison(); }));
$('chart-classifier').addEventListener('change', renderCharts);
$('manual-create').addEventListener('click', guarded(createManualInputs));
$('manual-clear').addEventListener('click', () => { $('manual-inputs').innerHTML = ''; $('manual-input-count').textContent = '0 logs'; $('manual-results').hidden = true; $('manual-status').textContent = 'Create the requested number of log inputs first.'; });
$('manual-analyse').addEventListener('click', guarded(analyseManualLogs));
$('manual-inputs').addEventListener('input', () => { $('manual-results').hidden = true; });
$('dashboard-analysis-prev').addEventListener('click', guarded(async () => { state.analysisPage = Math.max(1, state.analysisPage - 1); await loadDashboard(); }));
$('dashboard-analysis-next').addEventListener('click', guarded(async () => { state.analysisPage = Math.min(state.analysisPages, state.analysisPage + 1); await loadDashboard(); }));
$('dashboard-analysis-rows').addEventListener('click', event => { const row = event.target.closest('[data-dashboard-detail]'); if (!row) return; const detail = $(`dashboard-detail-${row.dataset.dashboardDetail}`); if (detail) detail.hidden = !detail.hidden; });
$('manual-result-rows').addEventListener('click', event => { const row = event.target.closest('[data-manual-detail]'); if (!row) return; const detail = $(`manual-detail-${row.dataset.manualDetail}`); if (detail) detail.hidden = !detail.hidden; });

/* Initialise real data and clearly labelled notebook reference results. */
(async function initialise() {
  try {
    const results = await Promise.allSettled([refreshState(), api('/api/saved-results')]);
    if (results[1].status === 'fulfilled') state.saved = results[1].value;
    if (results[0].status === 'rejected') throw results[0].reason;
    renderRunSelectors();
    await showPage(location.hash.slice(1) || 'logs');
  } catch (error) {
    $('connection-status').textContent = 'Workspace connection needs attention';
    alertUser(error.message || 'Could not load the research workspace.');
  }
})();
