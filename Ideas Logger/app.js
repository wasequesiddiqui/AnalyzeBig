// ============================================================
// Ideas Logger — Google Sheets Backed Application
// Stores data in Google Sheets via Apps Script Web App.
// localStorage is used as an optimistic cache.
// ============================================================

// ---- Google Sheets URLs ----
// Published CSV (READ-ONLY — fast, no auth):
const GS_CSV_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQG7rGLJGtxk8lrogGl2Bu-4dcDxujYgFM55YlWkORT7BTcv59GtsEFGVd5ObkUIlg1sNBHX80e0w8r/pub?output=csv';
// Apps Script Web App (WRITE — needs doGet/doPost deployed):
const GS_API_URL = 'https://script.google.com/macros/s/AKfycbyDIg-I8PT4s100mVPGvdU6nX3suGJwgmV09o_ECZFQ5M9IdsVLy9_T9BOKpA8iEnwL/exec';

// ---- localStorage keys ----
const STORAGE_KEY = 'ideas_logger_data';

let allIdeas = [];
let charts   = {};

// ============================================================
// INITIALIZATION
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  // 1) Load localStorage FIRST (sync – app always has data)
  loadFromLocalStorage();

  // 2) Render UI immediately
  updateThemeFilter();
  renderTable();
  updateCount();

  // 3) Set up interactivity
  initTabs();
  initForm();
  initImportButton();

  // 4) Background tasks
  detectIP();
  syncFromGoogleSheets();   // silently pull fresher data from Sheets
});

// ============================================================
// DATA: Google Sheets API
// ============================================================

/** Fetch ideas from the published CSV (read-only, fast, no auth). */
async function fetchFromSheets() {
  const resp = await fetch(GS_CSV_URL, { cache: 'no-store' });
  if (!resp.ok) throw new Error('Sheets CSV fetch failed: ' + resp.status);
  const csvText = await resp.text();
  // Parse the CSV into an array of objects (handles the sheet's column headers)
  return parseCSVText(csvText);
}

/** POST data to Google Sheets via Apps Script (write operations). */
async function postToSheets(params) {
  const formBody = new URLSearchParams();
  for (const [key, val] of Object.entries(params)) {
    formBody.append(key, val);
  }

  const resp = await fetch(GS_API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: formBody.toString()
  });

  if (!resp.ok) throw new Error('Sheets POST failed: ' + resp.status);
  return await resp.json();
}

// ============================================================
// DATA: localStorage (cache)
// ============================================================
function loadFromLocalStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    allIdeas = raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error('localStorage read failed:', e);
    allIdeas = [];
  }
}

function saveIdeasLocal() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(allIdeas));
}

/** Silently sync from Google Sheets in background. */
async function syncFromGoogleSheets() {
  try {
    const sheetData = await fetchFromSheets();
    if (Array.isArray(sheetData) && sheetData.length > 0) {
      allIdeas = sheetData;
      saveIdeasLocal();
      updateThemeFilter();
      renderTable();
      updateCount();
    }
  } catch (e) {
    console.warn('Google Sheets sync failed (using localStorage):', e.message);
  }
}

/** User-initiated refresh: pull from Sheets and show result. */
async function refreshIdeas() {
  try {
    const sheetData = await fetchFromSheets();
    if (Array.isArray(sheetData) && sheetData.length > 0) {
      allIdeas = sheetData;
      saveIdeasLocal();
    }
  } catch (e) {
    console.warn('Refresh failed:', e.message);
  }
  updateThemeFilter();
  renderTable();
  updateCount();
}

function getNextId() {
  if (allIdeas.length === 0) return '1';
  const maxId = Math.max(...allIdeas.map(i => parseInt(i.id) || 0));
  return String(maxId + 1);
}

// ============================================================
// TAB SWITCHING
// ============================================================
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
      if (btn.dataset.tab === 'analytics-tab') loadAnalytics();
      if (btn.dataset.tab === 'list-tab') renderTable();
    });
  });
}

// ============================================================
// IP DETECTION
// ============================================================
async function detectIP() {
  const ipField = document.getElementById('ip');
  try {
    const resp = await fetch('https://api.ipify.org?format=json');
    const data = await resp.json();
    ipField.value = data.ip;
  } catch {
    ipField.value = 'Unable to detect';
  }
}

// ============================================================
// FORM HANDLING
// ============================================================
function initForm() {
  const form = document.getElementById('idea-form');
  if (!form) {
    console.error('FATAL: #idea-form not found in DOM');
    return;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submit-btn');
    const feedback  = document.getElementById('form-feedback');

    try {
      submitBtn.disabled = true;
      submitBtn.textContent = '⏳ Saving...';
      feedback.className = 'feedback';
      feedback.textContent = '';

      const themeEl  = document.getElementById('theme');
      const detailEl = document.getElementById('detail');
      const ipEl     = document.getElementById('ip');
      const userEl   = document.getElementById('username');

      if (!themeEl || !detailEl) {
        throw new Error('Form fields not found');
      }

      const theme    = themeEl.value.trim();
      const detail   = detailEl.value.trim();
      const ip       = ipEl ? ipEl.value : 'unknown';
      const username = userEl ? userEl.value.trim() : 'anonymous';

      if (!theme || !detail) {
        feedback.className = 'feedback error';
        feedback.textContent = '❌ Theme and detail are required.';
        submitBtn.disabled = false;
        submitBtn.textContent = '🚀 Log Idea';
        return;
      }

      // 1) Save to Google Sheets
      let sheetSynced = false;
      try {
        const result = await postToSheets({
          action: 'create',
          theme: theme,
          detail: detail,
          ip: ip,
          username: username
        });
        // User's script returns {"result":"error",...} or {"result":"success",...}
        // Our script returns {"id":"1","Theme":"...",...} on success or {"error":"msg"} on failure
        if (result.result === 'error' || result.error) {
          console.warn('Sheets create warning:', JSON.stringify(result));
        } else {
          sheetSynced = true;
          // Re-sync from Sheets to get the server-assigned ID
          await syncFromGoogleSheets();
          feedback.className = 'feedback success';
          feedback.textContent = '✅ Idea logged & synced to Google Sheets!';
          form.reset();
          detectIP();
          submitBtn.disabled = false;
          submitBtn.textContent = '🚀 Log Idea';
          setTimeout(() => { feedback.className = 'feedback'; }, 5000);
          return;
        }
      } catch (err) {
        console.warn('Sheets create failed, saving locally:', err.message);
      }

      // 2) Always save locally
      const newIdea = {
        id: getNextId(),
        theme: theme,
        detail: detail,
        ip: ip || 'unknown',
        username: username || 'anonymous',
        status: 'open',
        createdAt: new Date().toISOString(),
        ModifiedAt: new Date().toISOString(),
        ImplementedAt: ''
      };

      allIdeas.push(newIdea);
      saveIdeasLocal();
      updateThemeFilter();
      renderTable();
      updateCount();

      feedback.className = 'feedback success';
      feedback.textContent = sheetSynced
        ? '✅ Idea logged & synced to Google Sheets!'
        : '✅ Idea saved locally (Sheets sync pending).';

      form.reset();
      detectIP();

    } catch (err) {
      console.error('Form submit error:', err);
      feedback.className = 'feedback error';
      feedback.textContent = '❌ Error: ' + err.message;
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = '🚀 Log Idea';
      setTimeout(() => { feedback.className = 'feedback'; }, 5000);
    }
  });
}

// ============================================================
// THEME FILTER
// ============================================================
function updateThemeFilter() {
  const themes = [...new Set(allIdeas.map(i => i.theme || i.Theme || '').filter(Boolean))].sort();
  const select = document.getElementById('theme-filter');
  const currentVal = select.value;
  select.innerHTML = '<option value="all">All</option>';
  themes.forEach(t => {
    select.innerHTML += `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`;
  });
  if ([...select.options].some(o => o.value === currentVal)) {
    select.value = currentVal;
  }
}

// ============================================================
// TABLE RENDERING
// ============================================================
function renderTable() {
  const statusFilter = document.getElementById('status-filter').value;
  const themeFilter  = document.getElementById('theme-filter').value;
  const tbody = document.getElementById('table-body');

  let filtered = [...allIdeas];
  if (statusFilter !== 'all') filtered = filtered.filter(i => getField(i, 'Status') === statusFilter);
  if (themeFilter !== 'all')  filtered = filtered.filter(i => getField(i, 'Theme') === themeFilter);
  filtered.sort((a, b) => parseInt(getField(b, 'ID')) - parseInt(getField(a, 'ID')));

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" class="empty-state">No ideas found. ${allIdeas.length === 0 ? 'Go to "Log Idea" tab to add one!' : 'Try adjusting filters.'}</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(idea => {
    const status   = getField(idea, 'Status');
    const id       = getField(idea, 'ID');
    const theme    = getField(idea, 'Theme');
    const detail   = getField(idea, 'Detail');
    const username = getField(idea, 'Username');
    const ip       = getField(idea, 'IP');
    const createdAt = getField(idea, 'CreatedAt');
    const implAt   = getField(idea, 'ImplementedAt');

    const timeHtml = status === 'open'
      ? getTimeElapsedHtml(createdAt)
      : `<span class="status-badge status-implemented">✅ Implemented</span>`;

    const createdDate = createdAt
      ? new Date(createdAt).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
      : '—';
    const implDate = implAt
      ? new Date(implAt).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
      : '—';

    const sc = status === 'open' ? 'status-open' : 'status-implemented';
    const sl = status === 'open' ? 'Open' : 'Implemented';

    return `
      <tr>
        <td><strong>#${id}</strong></td>
        <td><strong>${escapeHtml(theme)}</strong></td>
        <td class="detail-cell" title="${escapeHtml(detail)}">${escapeHtml(truncate(detail, 80))}</td>
        <td>${escapeHtml(username)}</td>
        <td><code>${escapeHtml(ip)}</code></td>
        <td><span class="status-badge ${sc}">${sl}</span></td>
        <td>${timeHtml}</td>
        <td title="Implemented: ${implDate}">${createdDate}</td>
        <td class="action-cell">
          ${status === 'open' ? `
            <input type="checkbox" class="implement-checkbox" data-id="${id}" title="Mark as implemented">
            <button class="btn btn-sm btn-success mark-impl-btn" data-id="${id}" style="display:none;">✅ Submit</button>
          ` : `
            <button class="btn btn-sm btn-warning reopen-btn" data-id="${id}">↩ Reopen</button>
          `}
        </td>
      </tr>`;
  }).join('');

  tbody.querySelectorAll('.implement-checkbox').forEach(cb => {
    cb.addEventListener('change', function () {
      this.closest('tr').querySelector('.mark-impl-btn').style.display = this.checked ? 'inline-flex' : 'none';
    });
  });
  tbody.querySelectorAll('.mark-impl-btn').forEach(btn => {
    btn.addEventListener('click', () => markImplemented(btn.dataset.id));
  });
  tbody.querySelectorAll('.reopen-btn').forEach(btn => {
    btn.addEventListener('click', () => reopenIdea(btn.dataset.id));
  });
}

// ============================================================
// MARK IMPLEMENTED / REOPEN
// ============================================================
async function markImplemented(id) {
  const idea = allIdeas.find(i => getField(i, 'ID') === id);
  if (!idea) return;

  // Try Sheets first
  try {
    await postToSheets({ action: 'implement', id: id });
  } catch (e) {
    console.warn('Sheets implement failed:', e.message);
  }

  // Update local
  setField(idea, 'Status', 'implemented');
  setField(idea, 'ImplementedAt', new Date().toISOString());
  setField(idea, 'ModifiedAt', new Date().toISOString());
  saveIdeasLocal();
  renderTable();
  updateCount();
}

async function reopenIdea(id) {
  const idea = allIdeas.find(i => getField(i, 'ID') === id);
  if (!idea) return;

  // Try Sheets first
  try {
    await postToSheets({ action: 'reopen', id: id });
  } catch (e) {
    console.warn('Sheets reopen failed:', e.message);
  }

  // Update local
  setField(idea, 'Status', 'open');
  setField(idea, 'ImplementedAt', '');
  setField(idea, 'ModifiedAt', new Date().toISOString());
  saveIdeasLocal();
  renderTable();
  updateCount();
}

// ============================================================
// CSV IMPORT
// ============================================================
function initImportButton() {
  const importInput = document.getElementById('csv-import-input');
  if (!importInput) return;
  importInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    try {
      // Try Sheets import first
      let sheetResult = null;
      try {
        sheetResult = await postToSheets({ action: 'import', csvdata: text });
      } catch (err) {
        console.warn('Sheets import failed:', err.message);
      }

      if (sheetResult && sheetResult.success) {
        await syncFromGoogleSheets();
        alert(`✅ Imported ${sheetResult.added} idea(s) to Google Sheets. ${sheetResult.skipped} skipped.`);
      } else {
        // Fallback: parse locally
        const imported = parseCSVText(text);
        if (imported.length === 0) { alert('No valid ideas found.'); return; }
        const existingIds = new Set(allIdeas.map(i => getField(i, 'ID')));
        let added = 0;
        imported.forEach(idea => {
          const iid = getField(idea, 'ID');
          if (!existingIds.has(iid)) {
            allIdeas.push(idea);
            existingIds.add(iid);
            added++;
          }
        });
        allIdeas.sort((a, b) => parseInt(getField(a, 'ID')) - parseInt(getField(b, 'ID')));
        saveIdeasLocal();
        updateThemeFilter();
        renderTable();
        updateCount();
        alert(`✅ Imported ${added} idea(s) locally. ${imported.length - added} skipped.`);
      }
    } catch (err) {
      alert('❌ Failed: ' + err.message);
    }
    importInput.value = '';
  });
}

// ============================================================
// CSV EXPORT
// ============================================================
function exportCSV() {
  if (allIdeas.length === 0) { alert('No ideas to export.'); return; }
  const headers = ['ID', 'Theme', 'Detail', 'IP', 'Username', 'Status', 'CreatedAt', 'ModifiedAt', 'ImplementedAt'];
  const rows = allIdeas.map(i => [
    getField(i, 'ID'),
    csvEscape(getField(i, 'Theme')),
    csvEscape(getField(i, 'Detail')),
    getField(i, 'IP'),
    getField(i, 'Username'),
    getField(i, 'Status'),
    getField(i, 'CreatedAt'),
    getField(i, 'ModifiedAt'),
    getField(i, 'ImplementedAt')
  ].join(','));
  const csvText = [headers.join(','), ...rows].join('\n') + '\n';
  const blob = new Blob(['\uFEFF' + csvText], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ideas_export_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ============================================================
// CSV PARSING (for import fallback)
// ============================================================
function parseCSVText(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
  const ideas = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = parseCSVLine(lines[i]);
    if (cols.length === 0) continue;
    const obj = {};
    headers.forEach((h, idx) => { obj[h] = (cols[idx] || '').trim(); });
    ideas.push(obj);
  }
  return ideas;
}
function parseCSVLine(line) {
  const result = []; let current = '', inQ = false;
  for (const ch of line) {
    if (inQ) { if (ch === '"') { if (result.length < line.length && line[result.length+1] === '"') { current += '"'; } else inQ = false; } else current += ch; }
    else { if (ch === '"') inQ = true; else if (ch === ',') { result.push(current); current = ''; } else current += ch; }
  }
  result.push(current);
  return result;
}
function csvEscape(str) {
  if (!str) return '';
  const s = String(str);
  return (s.includes(',') || s.includes('"') || s.includes('\n')) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

// ============================================================
// FIELD HELPERS (handles both casing: "Theme" vs "theme")
// ============================================================
function getField(obj, name) {
  if (obj[name] !== undefined) return obj[name];
  const lower = name.toLowerCase();
  for (const key of Object.keys(obj)) {
    if (key.toLowerCase() === lower) return obj[key];
  }
  return '';
}
function setField(obj, name, value) {
  if (obj[name] !== undefined) { obj[name] = value; return; }
  const lower = name.toLowerCase();
  for (const key of Object.keys(obj)) {
    if (key.toLowerCase() === lower) { obj[key] = value; return; }
  }
  obj[name] = value;
}

// ============================================================
// TIME ELAPSED
// ============================================================
function getTimeElapsedHtml(createdAt) {
  if (!createdAt) return '<span>—</span>';
  const created = new Date(createdAt);
  const now = new Date();
  const diffMs = now - created;
  if (diffMs < 0) return '<span>Just now</span>';
  const s = Math.floor(diffMs / 1000), m = Math.floor(s / 60), h = Math.floor(m / 60);
  const d = Math.floor(h / 24), w = Math.floor(d / 7), mo = Math.floor(d / 30);
  let text;
  if (s < 60) text = 'Just now';
  else if (m < 60) text = `${m}m ago`;
  else if (h < 24) text = `${h}h ${m % 60}m ago`;
  else if (d < 7) text = `${d}d ${h % 24}h ago`;
  else if (w < 4) text = `${w}w ${d % 7}d ago`;
  else text = `${mo}mo ${d % 30}d ago`;
  let cls = d > 30 ? 'very-old' : (d > 7 ? 'old' : '');
  return `<span class="time-elapsed ${cls}">${text}</span>`;
}

// ============================================================
// ANALYTICS
// ============================================================
function loadAnalytics() {
  const ideas = allIdeas;
  const total = ideas.length;
  const implemented = ideas.filter(i => getField(i, 'Status') === 'implemented').length;
  const openCount = total - implemented;

  const themeCounts = {};
  ideas.forEach(i => { const t = getField(i, 'Theme'); themeCounts[t] = (themeCounts[t] || 0) + 1; });

  const monthlyData = {};
  ideas.forEach(i => {
    const ca = getField(i, 'CreatedAt');
    const month = ca ? ca.substring(0, 7) : 'unknown';
    const st = getField(i, 'Status');
    if (!monthlyData[month]) monthlyData[month] = { open: 0, implemented: 0 };
    monthlyData[month][st]++;
  });
  const sortedMonths = Object.keys(monthlyData).sort();
  const monthlyLabels = sortedMonths;
  const monthlyOpen = sortedMonths.map(m => monthlyData[m].open || 0);
  const monthlyImplemented = sortedMonths.map(m => monthlyData[m].implemented || 0);

  let cumOpen = 0, cumImpl = 0;
  const cumData = sortedMonths.map(m => {
    cumOpen += (monthlyData[m].open || 0);
    cumImpl += (monthlyData[m].implemented || 0);
    return { month: m, open: cumOpen, implemented: cumImpl };
  });

  const implIdeas = ideas.filter(i => getField(i, 'Status') === 'implemented' && getField(i, 'ImplementedAt'));
  let avgImplDays = 0;
  if (implIdeas.length > 0) {
    const totalDays = implIdeas.reduce((sum, i) =>
      sum + (new Date(getField(i, 'ImplementedAt')) - new Date(getField(i, 'CreatedAt'))) / 86400000, 0);
    avgImplDays = Math.round(totalDays / implIdeas.length * 10) / 10;
  }

  const userCounts = {};
  ideas.forEach(i => { const u = getField(i, 'Username'); userCounts[u] = (userCounts[u] || 0) + 1; });
  const implRate = total > 0 ? Math.round((implemented / total) * 100) : 0;

  document.getElementById('stat-total').textContent      = total;
  document.getElementById('stat-open').textContent       = openCount;
  document.getElementById('stat-implemented').textContent = implemented;
  document.getElementById('stat-rate').textContent       = implRate + '%';
  document.getElementById('stat-avg-days').textContent   = avgImplDays + ' days';

  const colors = { success: '#10b981', warning: '#f59e0b' };
  const chartColors = ['#6366f1','#10b981','#f59e0b','#8b5cf6','#ec4899','#3b82f6','#14b8a6','#f97316','#ef4444','#84cc16','#06b6d4','#a855f7','#e11d48','#22c55e','#64748b'];

  renderOrUpdateChart('chart-monthly', {
    type: 'bar', data: { labels: monthlyLabels, datasets: [
      { label: 'Open', data: monthlyOpen, backgroundColor: colors.warning + 'BB', borderRadius: 4 },
      { label: 'Implemented', data: monthlyImplemented, backgroundColor: colors.success + 'BB', borderRadius: 4 }
    ]}, options: barOptions()
  });
  renderOrUpdateChart('chart-cumulative', {
    type: 'line', data: { labels: cumData.map(d => d.month), datasets: [
      { label: 'Total Open', data: cumData.map(d => d.open), borderColor: colors.warning, backgroundColor: colors.warning + '20', fill: true, tension: 0.3, pointRadius: 4 },
      { label: 'Total Implemented', data: cumData.map(d => d.implemented), borderColor: colors.success, backgroundColor: colors.success + '20', fill: true, tension: 0.3, pointRadius: 4 }
    ]}, options: lineOptions()
  });
  const themeEntries = Object.entries(themeCounts).sort((a, b) => b[1] - a[1]);
  renderOrUpdateChart('chart-themes', {
    type: 'bar', data: { labels: themeEntries.map(e => e[0]), datasets: [
      { label: 'Ideas', data: themeEntries.map(e => e[1]), backgroundColor: themeEntries.map((_, i) => chartColors[i % chartColors.length]), borderRadius: 4 }
    ]}, options: { ...barOptions(), indexAxis: 'y' }
  });
  renderOrUpdateChart('chart-status', {
    type: 'doughnut', data: { labels: ['Open', 'Implemented'], datasets: [
      { data: [openCount, implemented], backgroundColor: [colors.warning, colors.success], borderWidth: 2, borderColor: '#fff' }
    ]}, options: doughnutOptions()
  });
  const userEntries = Object.entries(userCounts).sort((a, b) => b[1] - a[1]);
  renderOrUpdateChart('chart-users', {
    type: 'bar', data: { labels: userEntries.map(e => e[0]), datasets: [
      { label: 'Ideas', data: userEntries.map(e => e[1]), backgroundColor: chartColors.slice(0, userEntries.length), borderRadius: 4 }
    ]}, options: barOptions()
  });
  const timelineIdeas = ideas.filter(i => getField(i, 'Status') === 'implemented' && getField(i, 'CreatedAt'))
    .sort((a, b) => new Date(getField(a, 'CreatedAt')) - new Date(getField(b, 'CreatedAt')));
  renderOrUpdateChart('chart-timeline', {
    type: 'bar', data: { labels: timelineIdeas.map(i => `#${getField(i, 'ID')}: ${getField(i, 'Theme')}`), datasets: [{
      label: 'Days to Implement',
      data: timelineIdeas.map(i => Math.round((new Date(getField(i, 'ImplementedAt')) - new Date(getField(i, 'CreatedAt'))) / 86400000 * 10) / 10),
      backgroundColor: timelineIdeas.map((_, idx) => chartColors[idx % chartColors.length]), borderRadius: 4
    }]}, options: { ...barOptions(),
      plugins: { ...barOptions().plugins, title: { display: true, text: 'Days from creation to implementation per idea', font: { size: 12 }, color: '#64748b' } }
    }
  });
}

// ============================================================
// CHART HELPERS
// ============================================================
function renderOrUpdateChart(canvasId, config) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  if (charts[canvasId]) charts[canvasId].destroy();
  charts[canvasId] = new Chart(canvas.getContext('2d'), config);
}
function barOptions() {
  return { responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: true, position: 'top', labels: { usePointStyle: true, padding: 20 } } },
    scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } };
}
function lineOptions() {
  return { responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: true, position: 'top', labels: { usePointStyle: true, padding: 20 } } },
    scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } };
}
function doughnutOptions() {
  return { responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, padding: 20 } } } };
}

// ============================================================
// UTILITY
// ============================================================
function updateCount() {
  document.getElementById('idea-count').textContent = `${allIdeas.length} idea${allIdeas.length !== 1 ? 's' : ''}`;
}
function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div'); div.textContent = str; return div.innerHTML;
}
function truncate(str, maxLen) {
  if (!str) return '';
  return str.length > maxLen ? str.substring(0, maxLen) + '...' : str;
}
