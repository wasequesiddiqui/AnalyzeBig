// ============================================================
// Google Apps Script — Ideas Logger Backend
// Deploy as Web App (Execute as: "Me", Who has access: "Anyone")
// Copy this entire file into your Apps Script project.
// ============================================================

var SHEET_NAME = 'Sheet1';  // change if your sheet tab has a different name

// ---- Helper: get or create sheet ----
function getSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName(SHEET_NAME);
  if (!sheet) {
    sheet = ss.insertSheet(SHEET_NAME);
    sheet.appendRow(['ID', 'Theme', 'Detail', 'IP', 'Username', 'Status', 'CreatedAt', 'ModifiedAt', 'ImplementedAt']);
  }
  return sheet;
}

// ---- GET: return all ideas as JSON ----
function doGet(e) {
  var sheet = getSheet();
  var data = sheet.getDataRange().getValues();
  if (data.length < 2) {
    return ContentService.createTextOutput(JSON.stringify([]))
      .setMimeType(ContentService.MimeType.JSON);
  }

  var headers = data[0];
  var rows = [];
  for (var i = 1; i < data.length; i++) {
    var row = {};
    for (var j = 0; j < headers.length; j++) {
      row[headers[j]] = data[i][j] || '';
    }
    rows.push(row);
  }
  return ContentService.createTextOutput(JSON.stringify(rows))
    .setMimeType(ContentService.MimeType.JSON);
}

// ---- POST: create / update / implement / reopen / import ----
function doPost(e) {
  try {
    var params = e.parameter;
    var action = params.action || 'create';
    var sheet = getSheet();
    var headers = sheet.getDataRange().getValues()[0];

    if (action === 'create') {
      return createIdea(sheet, params, headers);
    } else if (action === 'implement') {
      return updateStatus(sheet, params.id, 'implemented');
    } else if (action === 'reopen') {
      return updateStatus(sheet, params.id, 'open');
    } else if (action === 'import') {
      return importCSV(sheet, params.csvdata);
    } else {
      return jsonError('Unknown action: ' + action);
    }
  } catch (err) {
    return jsonError(err.toString());
  }
}

// ---- Create a new idea ----
function createIdea(sheet, params, headers) {
  var theme    = params.theme || '';
  var detail   = params.detail || '';
  var ip       = params.ip || 'unknown';
  var username = params.username || 'anonymous';

  if (!theme || !detail) {
    return jsonError('Theme and Detail are required');
  }

  var data = sheet.getDataRange().getValues();
  var nextId = 1;
  if (data.length > 1) {
    for (var i = 1; i < data.length; i++) {
      var existingId = parseInt(data[i][0]) || 0;
      if (existingId >= nextId) nextId = existingId + 1;
    }
  }

  var now = new Date().toISOString();
  var newRow = [String(nextId), theme, detail, ip, username, 'open', now, now, ''];

  sheet.appendRow(newRow);

  var idea = rowToObject(newRow, headers);
  return ContentService.createTextOutput(JSON.stringify(idea))
    .setMimeType(ContentService.MimeType.JSON);
}

// ---- Update idea status (implement / reopen) ----
function updateStatus(sheet, ideaId, newStatus) {
  if (!ideaId) return jsonError('Missing id parameter');

  var data = sheet.getDataRange().getValues();
  var headers = data[0];
  var now = new Date().toISOString();

  for (var i = 1; i < data.length; i++) {
    if (String(data[i][0]) === String(ideaId)) {
      var statusCol = headers.indexOf('Status');
      var implCol   = headers.indexOf('ImplementedAt');
      var modCol    = headers.indexOf('ModifiedAt');

      if (statusCol >= 0) sheet.getRange(i + 1, statusCol + 1).setValue(newStatus);
      if (implCol >= 0) sheet.getRange(i + 1, implCol + 1).setValue(newStatus === 'implemented' ? now : '');
      if (modCol >= 0) sheet.getRange(i + 1, modCol + 1).setValue(now);

      // Return updated row
      var updatedRow = sheet.getRange(i + 1, 1, 1, headers.length).getValues()[0];
      return ContentService.createTextOutput(JSON.stringify(rowToObject(updatedRow, headers)))
        .setMimeType(ContentService.MimeType.JSON);
    }
  }
  return jsonError('Idea not found: ' + ideaId);
}

// ---- Import ideas from CSV text ----
function importCSV(sheet, csvData) {
  if (!csvData) return jsonError('Missing csvdata parameter');

  var lines = csvData.split('\n');
  if (lines.length < 2) return jsonError('CSV must have header + at least one row');

  var csvHeaders = lines[0].split(',');
  var existingData = sheet.getDataRange().getValues();
  var existingIds = {};
  for (var i = 1; i < existingData.length; i++) {
    existingIds[String(existingData[i][0])] = true;
  }

  var addedCount = 0;
  var skippedCount = 0;

  for (var i = 1; i < lines.length; i++) {
    var cols = lines[i].split(',');
    if (cols.length < 2) continue;

    var id = (csvHeaders.indexOf('ID') >= 0) ? cols[csvHeaders.indexOf('ID')] : '';
    if (existingIds[String(id)]) { skippedCount++; continue; }

    var theme        = csvHeaders.indexOf('Theme') >= 0        ? cols[csvHeaders.indexOf('Theme')] : '';
    var detail       = csvHeaders.indexOf('Detail') >= 0       ? cols[csvHeaders.indexOf('Detail')] : '';
    var ip           = csvHeaders.indexOf('IP') >= 0           ? cols[csvHeaders.indexOf('IP')] : 'unknown';
    var username     = csvHeaders.indexOf('Username') >= 0     ? cols[csvHeaders.indexOf('Username')] : 'anonymous';
    var status       = csvHeaders.indexOf('Status') >= 0       ? cols[csvHeaders.indexOf('Status')] : 'open';
    var createdAt    = csvHeaders.indexOf('CreatedAt') >= 0    ? cols[csvHeaders.indexOf('CreatedAt')] : new Date().toISOString();
    var modifiedAt   = csvHeaders.indexOf('ModifiedAt') >= 0   ? cols[csvHeaders.indexOf('ModifiedAt')] : '';
    var implementedAt = csvHeaders.indexOf('ImplementedAt') >= 0 ? cols[csvHeaders.indexOf('ImplementedAt')] : '';

    var now = new Date().toISOString();
    sheet.appendRow([id, theme, detail, ip, username, status, createdAt, now, implementedAt]);
    existingIds[String(id)] = true;
    addedCount++;
  }

  return ContentService.createTextOutput(JSON.stringify({
    success: true,
    added: addedCount,
    skipped: skippedCount
  })).setMimeType(ContentService.MimeType.JSON);
}

// ---- Helper: convert row array to object ----
function rowToObject(row, headers) {
  var obj = {};
  for (var k = 0; k < headers.length; k++) {
    obj[headers[k]] = row[k] || '';
  }
  return obj;
}

// ---- Helper: return JSON error ----
function jsonError(msg) {
  return ContentService.createTextOutput(JSON.stringify({ error: msg }))
    .setMimeType(ContentService.MimeType.JSON);
}
