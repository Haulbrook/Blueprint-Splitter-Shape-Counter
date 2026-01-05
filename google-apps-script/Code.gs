/**
 * Blueprint Shape Counter - Main Server Code
 *
 * A Google Apps Script web app for counting symbols in landscape blueprints
 * using Claude AI vision capabilities.
 */

/**
 * Serves the main HTML interface
 */
function doGet() {
  return HtmlService.createTemplateFromFile('Index')
    .evaluate()
    .setTitle('Blueprint Shape Counter')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL)
    .addMetaTag('viewport', 'width=device-width, initial-scale=1');
}

/**
 * Include HTML files (for modular HTML structure)
 */
function include(filename) {
  return HtmlService.createHtmlOutputFromFile(filename).getContent();
}

/**
 * Process a single section image with Claude API
 * @param {string} imageBase64 - Base64 encoded image data
 * @param {number} sectionNumber - Section number (1-6)
 * @param {string} sheetName - Name of the sheet being processed
 * @returns {Object} Analysis results from Claude
 */
function processSection(imageBase64, sectionNumber, sheetName) {
  // Validate API key
  if (!CLAUDE_API_KEY || CLAUDE_API_KEY === 'YOUR_CLAUDE_API_KEY_HERE') {
    throw new Error('Please configure your Claude API key in Config.gs');
  }

  // Remove data URL prefix if present
  const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, '');

  // Build the request payload
  const payload = {
    model: CLAUDE_MODEL,
    max_tokens: MAX_TOKENS,
    system: SYSTEM_PROMPT,
    messages: [{
      role: 'user',
      content: [
        {
          type: 'image',
          source: {
            type: 'base64',
            media_type: 'image/png',
            data: base64Data
          }
        },
        {
          type: 'text',
          text: `Analyze Section ${sectionNumber} of 6 from sheet "${sheetName}".

Count all symbols following the protocol. Remember:
- This is Section ${sectionNumber} in a 2x3 grid layout
- Apply seam rules for edge symbols
- Return EXACT counts or CANNOT COUNT

Begin your systematic analysis.`
        }
      ]
    }]
  };

  // Make API request
  const options = {
    method: 'post',
    contentType: 'application/json',
    headers: {
      'x-api-key': CLAUDE_API_KEY,
      'anthropic-version': '2023-06-01'
    },
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  };

  try {
    const response = UrlFetchApp.fetch(CLAUDE_API_ENDPOINT, options);
    const responseCode = response.getResponseCode();
    const responseText = response.getContentText();

    if (responseCode !== 200) {
      const errorData = JSON.parse(responseText);
      throw new Error(`API Error (${responseCode}): ${errorData.error?.message || responseText}`);
    }

    const data = JSON.parse(responseText);
    const analysisText = data.content[0].text;

    // Parse the response to extract counts
    const parsedResults = parseClaudeResponse(analysisText, sectionNumber);

    return {
      success: true,
      sectionNumber: sectionNumber,
      rawResponse: analysisText,
      parsed: parsedResults
    };

  } catch (error) {
    return {
      success: false,
      sectionNumber: sectionNumber,
      error: error.message
    };
  }
}

/**
 * Parse Claude's response to extract symbol counts
 * @param {string} responseText - Raw response from Claude
 * @param {number} sectionNumber - Section number
 * @returns {Object} Parsed counts and flags
 */
function parseClaudeResponse(responseText, sectionNumber) {
  const result = {
    symbolCounts: {},
    flaggedAreas: [],
    status: 'COMPLETE'
  };

  // Extract symbol types and counts using regex patterns
  const symbolPattern = /SYMBOL TYPE \d+:\s*([^\n]+)\s*\n\s*Count:\s*(\d+|CANNOT COUNT[^\n]*)/gi;
  let match;

  while ((match = symbolPattern.exec(responseText)) !== null) {
    const symbolType = match[1].trim();
    const countValue = match[2].trim();

    if (countValue.startsWith('CANNOT COUNT')) {
      result.flaggedAreas.push({
        symbolType: symbolType,
        reason: countValue,
        section: sectionNumber
      });
      result.status = 'INCOMPLETE';
    } else {
      const count = parseInt(countValue, 10);
      if (!isNaN(count)) {
        result.symbolCounts[symbolType] = count;
      }
    }
  }

  // Check for section status
  if (responseText.includes('INCOMPLETE') || responseText.includes('CANNOT COUNT')) {
    result.status = 'INCOMPLETE';
  }

  // Extract any flagged areas mentioned
  const flagPattern = /CANNOT COUNT\s*-\s*([^\n]+)/gi;
  while ((match = flagPattern.exec(responseText)) !== null) {
    const existingFlag = result.flaggedAreas.find(f => f.reason.includes(match[1]));
    if (!existingFlag) {
      result.flaggedAreas.push({
        reason: match[1].trim(),
        section: sectionNumber
      });
    }
  }

  return result;
}

/**
 * Aggregate results from all 6 sections into sheet totals
 * @param {Array} sectionResults - Array of results from all sections
 * @param {string} sheetName - Name of the sheet
 * @returns {Object} Aggregated totals
 */
function aggregateResults(sectionResults, sheetName) {
  const totals = {
    sheetName: sheetName,
    symbolTotals: {},
    allFlaggedAreas: [],
    sectionDetails: [],
    overallStatus: 'COMPLETE'
  };

  sectionResults.forEach((result, index) => {
    // Store section details
    totals.sectionDetails.push({
      section: index + 1,
      success: result.success,
      status: result.parsed?.status || 'ERROR',
      counts: result.parsed?.symbolCounts || {},
      flags: result.parsed?.flaggedAreas || [],
      error: result.error || null
    });

    if (!result.success) {
      totals.overallStatus = 'ERROR';
      return;
    }

    // Sum up symbol counts
    const counts = result.parsed?.symbolCounts || {};
    for (const [symbolType, count] of Object.entries(counts)) {
      totals.symbolTotals[symbolType] = (totals.symbolTotals[symbolType] || 0) + count;
    }

    // Collect flagged areas
    const flags = result.parsed?.flaggedAreas || [];
    totals.allFlaggedAreas.push(...flags);

    if (result.parsed?.status === 'INCOMPLETE') {
      totals.overallStatus = 'INCOMPLETE';
    }
  });

  return totals;
}

/**
 * Generate CSV content from results
 * @param {Array} allSheetResults - Array of aggregated results for all sheets
 * @returns {string} CSV content
 */
function generateCSV(allSheetResults) {
  let csv = 'Sheet,Symbol Type,Count,Status\n';

  allSheetResults.forEach(sheet => {
    const sheetName = sheet.sheetName.replace(/,/g, ';');

    // Add symbol totals
    for (const [symbolType, count] of Object.entries(sheet.symbolTotals)) {
      const cleanType = symbolType.replace(/,/g, ';');
      csv += `"${sheetName}","${cleanType}",${count},Counted\n`;
    }

    // Add flagged areas
    sheet.allFlaggedAreas.forEach(flag => {
      const cleanType = (flag.symbolType || 'Unknown').replace(/,/g, ';');
      const cleanReason = (flag.reason || '').replace(/,/g, ';').replace(/"/g, "'");
      csv += `"${sheetName}","${cleanType}",FLAGGED,"${cleanReason}"\n`;
    });
  });

  // Add summary section
  csv += '\n\nSUMMARY\n';
  csv += 'Symbol Type,Total Across All Sheets\n';

  const grandTotals = {};
  allSheetResults.forEach(sheet => {
    for (const [symbolType, count] of Object.entries(sheet.symbolTotals)) {
      grandTotals[symbolType] = (grandTotals[symbolType] || 0) + count;
    }
  });

  for (const [symbolType, total] of Object.entries(grandTotals)) {
    csv += `"${symbolType}",${total}\n`;
  }

  return csv;
}

/**
 * Generate PDF content (returns HTML that can be converted to PDF)
 * @param {Array} allSheetResults - Array of aggregated results for all sheets
 * @returns {string} HTML content for PDF generation
 */
function generatePDFContent(allSheetResults) {
  let html = `
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: Arial, sans-serif; padding: 20px; }
    h1 { color: #333; border-bottom: 2px solid #4285f4; padding-bottom: 10px; }
    h2 { color: #4285f4; margin-top: 30px; }
    table { border-collapse: collapse; width: 100%; margin: 15px 0; }
    th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
    th { background: #4285f4; color: white; }
    tr:nth-child(even) { background: #f9f9f9; }
    .flagged { background: #fff3cd; color: #856404; }
    .summary { background: #d4edda; }
    .total-row { font-weight: bold; background: #e8f5e9; }
    .timestamp { color: #666; font-size: 12px; margin-top: 20px; }
  </style>
</head>
<body>
  <h1>Blueprint Symbol Count Report</h1>
  <p class="timestamp">Generated: ${new Date().toLocaleString()}</p>
`;

  // Add each sheet's results
  allSheetResults.forEach(sheet => {
    html += `<h2>${escapeHtml(sheet.sheetName)}</h2>`;
    html += '<table>';
    html += '<tr><th>Symbol Type</th><th>Count</th><th>Status</th></tr>';

    for (const [symbolType, count] of Object.entries(sheet.symbolTotals)) {
      html += `<tr><td>${escapeHtml(symbolType)}</td><td>${count}</td><td>Counted</td></tr>`;
    }

    sheet.allFlaggedAreas.forEach(flag => {
      html += `<tr class="flagged"><td>${escapeHtml(flag.symbolType || 'Area')}</td><td>FLAGGED</td><td>${escapeHtml(flag.reason)}</td></tr>`;
    });

    html += '</table>';
  });

  // Grand totals
  html += '<h2>Grand Totals (All Sheets)</h2>';
  html += '<table>';
  html += '<tr><th>Symbol Type</th><th>Total Count</th></tr>';

  const grandTotals = {};
  allSheetResults.forEach(sheet => {
    for (const [symbolType, count] of Object.entries(sheet.symbolTotals)) {
      grandTotals[symbolType] = (grandTotals[symbolType] || 0) + count;
    }
  });

  for (const [symbolType, total] of Object.entries(grandTotals)) {
    html += `<tr class="total-row"><td>${escapeHtml(symbolType)}</td><td>${total}</td></tr>`;
  }

  html += '</table>';
  html += '</body></html>';

  return html;
}

/**
 * Helper function to escape HTML
 */
function escapeHtml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/**
 * Test function to verify API connectivity
 */
function testAPIConnection() {
  if (!CLAUDE_API_KEY || CLAUDE_API_KEY === 'YOUR_CLAUDE_API_KEY_HERE') {
    return { success: false, message: 'API key not configured' };
  }

  const payload = {
    model: CLAUDE_MODEL,
    max_tokens: 100,
    messages: [{
      role: 'user',
      content: 'Say "Connection successful" and nothing else.'
    }]
  };

  const options = {
    method: 'post',
    contentType: 'application/json',
    headers: {
      'x-api-key': CLAUDE_API_KEY,
      'anthropic-version': '2023-06-01'
    },
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  };

  try {
    const response = UrlFetchApp.fetch(CLAUDE_API_ENDPOINT, options);
    const responseCode = response.getResponseCode();

    if (responseCode === 200) {
      return { success: true, message: 'API connection successful!' };
    } else {
      const errorData = JSON.parse(response.getContentText());
      return { success: false, message: `API Error: ${errorData.error?.message || 'Unknown error'}` };
    }
  } catch (error) {
    return { success: false, message: `Connection failed: ${error.message}` };
  }
}
