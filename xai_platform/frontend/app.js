/**
 * app.js - XAI Model Explanation Platform
 * Handles all frontend logic: file uploads, API calls, chart rendering, what-if simulation.
 */

const API_BASE = '';  // Same origin

// ==================== State ====================
let appState = {
    modelFile: null,
    datasetFile: null,
    columns: [],
    featureImportance: {},
    featureNames: [],
    numSamples: 0
};

// ==================== DOM Elements ====================
const modelDropZone = document.getElementById('modelDropZone');
const datasetDropZone = document.getElementById('datasetDropZone');
const modelFileInput = document.getElementById('modelFile');
const datasetFileInput = document.getElementById('datasetFile');
const modelFileName = document.getElementById('modelFileName');
const datasetFileName = document.getElementById('datasetFileName');
const settingsSection = document.getElementById('settingsSection');
const targetSelect = document.getElementById('targetSelect');
const analyzeBtn = document.getElementById('analyzeBtn');
const heroSection = document.getElementById('heroSection');
const dashboard = document.getElementById('dashboard');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');

// ==================== File Upload ====================
function setupUploadZone(zone, input, nameEl, type) {
    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = '#3b82f6'; });
    zone.addEventListener('dragleave', () => { zone.style.borderColor = ''; });
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.style.borderColor = '';
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            handleFileSelect(input, nameEl, zone, type);
        }
    });
    input.addEventListener('change', () => handleFileSelect(input, nameEl, zone, type));
}

function handleFileSelect(input, nameEl, zone, type) {
    const file = input.files[0];
    if (!file) return;
    nameEl.textContent = file.name;
    zone.classList.add('has-file');
    if (type === 'model') appState.modelFile = file;
    else appState.datasetFile = file;
    checkReadyToUpload();
}

async function checkReadyToUpload() {
    if (appState.modelFile && appState.datasetFile) {
        showLoading('Uploading files...');
        try {
            const formData = new FormData();
            formData.append('model', appState.modelFile);
            formData.append('dataset', appState.datasetFile);

            const resp = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: formData });
            const data = await resp.json();

            if (data.error) throw new Error(data.error);

            // Populate target column selector
            targetSelect.innerHTML = '';
            data.columns.forEach(col => {
                const opt = document.createElement('option');
                opt.value = col;
                opt.textContent = col;
                targetSelect.appendChild(opt);
            });

            settingsSection.style.display = 'block';
            analyzeBtn.disabled = false;
            document.getElementById('rowIndex').max = data.rows - 1;
            appState.columns = data.columns;

            hideLoading();
        } catch (err) {
            hideLoading();
            alert('Upload Error: ' + err.message);
        }
    }
}

setupUploadZone(modelDropZone, modelFileInput, modelFileName, 'model');
setupUploadZone(datasetDropZone, datasetFileInput, datasetFileName, 'dataset');

// ==================== Analyze ====================
analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
    const targetCol = targetSelect.value;
    if (!targetCol) return alert('Please select a target column.');

    showLoading('Computing model metrics...');
    try {
        const resp = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_column: targetCol })
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        appState.featureImportance = data.feature_importance;
        appState.featureNames = data.feature_names;
        appState.numSamples = data.num_samples;

        // Hide hero, show dashboard
        heroSection.style.display = 'none';
        dashboard.style.display = 'block';

        // Render everything
        renderSummary(data);
        renderMetrics(data.metrics);
        renderFeatureImportance(data.feature_importance);
        renderShapSummary(data.shap_summary, data.feature_names);
        buildWhatIfInputs(data.feature_names);

        // Auto-load local explanation for row 0
        const rowIdx = parseInt(document.getElementById('rowIndex').value) || 0;
        document.getElementById('localRowInput').value = rowIdx;
        await loadLocalExplanation(rowIdx);

        hideLoading();
    } catch (err) {
        hideLoading();
        alert('Analysis Error: ' + err.message);
    }
}

// ==================== Rendering Functions ====================

function renderSummary(data) {
    const container = document.getElementById('summaryCards');
    const cards = [
        { label: 'Model Type', value: data.model_type, color: 'blue' },
        { label: 'Dataset Rows', value: data.num_samples.toLocaleString(), color: 'green' },
        { label: 'Features', value: data.num_features, color: 'purple' },
        { label: 'Target', value: document.getElementById('targetSelect').value, color: 'orange' }
    ];
    container.innerHTML = cards.map(c => `
        <div class="metric-card">
            <div class="metric-label">${c.label}</div>
            <div class="metric-value ${c.color}">${c.value}</div>
        </div>
    `).join('');
}

function renderMetrics(metrics) {
    const container = document.getElementById('metricsCards');
    const colors = ['blue', 'green', 'purple', 'orange'];
    let html = '';
    let idx = 0;

    for (const [key, value] of Object.entries(metrics)) {
        if (key === 'confusion_matrix') continue;
        const display = typeof value === 'number' ? value.toFixed(4) : value;
        html += `
            <div class="metric-card">
                <div class="metric-label">${key.replace(/_/g, ' ')}</div>
                <div class="metric-value ${colors[idx % colors.length]}">${display}</div>
            </div>`;
        idx++;
    }
    container.innerHTML = html;

    // Confusion matrix
    if (metrics.confusion_matrix) {
        const card = document.getElementById('confusionCard');
        card.style.display = 'block';
        const cm = metrics.confusion_matrix;
        const container = document.getElementById('confusionMatrix');

        // Use Plotly heatmap
        const trace = {
            z: cm,
            type: 'heatmap',
            colorscale: [[0, '#111827'], [1, '#3b82f6']],
            showscale: true,
            hovertemplate: 'Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
        };
        const layout = {
            ...darkLayout(),
            xaxis: { title: 'Predicted', color: '#9ca3af' },
            yaxis: { title: 'Actual', color: '#9ca3af', autorange: 'reversed' },
            height: 350,
            margin: { t: 20, b: 60, l: 60, r: 20 }
        };
        Plotly.newPlot('confusionMatrix', [trace], layout, { responsive: true });
    }
}

function renderFeatureImportance(importance) {
    const entries = Object.entries(importance).slice(0, 20);
    const features = entries.map(e => e[0]).reverse();
    const values = entries.map(e => e[1]).reverse();

    const trace = {
        type: 'bar',
        x: values,
        y: features,
        orientation: 'h',
        marker: {
            color: values.map(v => {
                const ratio = v / Math.max(...values);
                return `rgba(59, 130, 246, ${0.3 + ratio * 0.7})`;
            }),
            line: { color: 'rgba(59, 130, 246, 0.8)', width: 1 }
        },
        hovertemplate: '%{y}: %{x:.4f}<extra></extra>'
    };

    const layout = {
        ...darkLayout(),
        xaxis: { title: 'Mean |SHAP Value|', color: '#9ca3af', gridcolor: 'rgba(75,85,99,0.3)' },
        yaxis: { color: '#9ca3af', tickfont: { size: 11 } },
        height: Math.max(400, entries.length * 25),
        margin: { t: 10, b: 50, l: 150, r: 20 }
    };

    Plotly.newPlot('featureImportanceChart', [trace], layout, { responsive: true });
}

function renderShapSummary(shapData, featureNames) {
    // Group data by feature
    const featureMap = {};
    shapData.forEach(d => {
        if (!featureMap[d.feature]) featureMap[d.feature] = { shap: [], fval: [] };
        featureMap[d.feature].shap.push(d.shap_value);
        featureMap[d.feature].fval.push(d.feature_value);
    });

    // Take top 15 features by mean absolute SHAP
    const ranked = Object.entries(featureMap)
        .map(([name, data]) => ({
            name,
            meanAbs: data.shap.reduce((s, v) => s + Math.abs(v), 0) / data.shap.length,
            shap: data.shap,
            fval: data.fval
        }))
        .sort((a, b) => b.meanAbs - a.meanAbs)
        .slice(0, 15);

    const traces = ranked.reverse().map((feat, idx) => {
        // Normalize feature values for color
        const minV = Math.min(...feat.fval);
        const maxV = Math.max(...feat.fval);
        const range = maxV - minV || 1;
        const colors = feat.fval.map(v => {
            const ratio = (v - minV) / range;
            const r = Math.round(59 + ratio * (239 - 59));
            const g = Math.round(130 - ratio * 62);
            const b = Math.round(246 - ratio * 178);
            return `rgb(${r},${g},${b})`;
        });

        return {
            type: 'scatter',
            mode: 'markers',
            x: feat.shap,
            y: feat.shap.map(() => feat.name),
            marker: {
                color: colors,
                size: 5,
                opacity: 0.7
            },
            name: feat.name,
            showlegend: false,
            hovertemplate: `${feat.name}<br>SHAP: %{x:.4f}<extra></extra>`
        };
    });

    const layout = {
        ...darkLayout(),
        xaxis: {
            title: 'SHAP Value (impact on prediction)',
            color: '#9ca3af',
            gridcolor: 'rgba(75,85,99,0.3)',
            zeroline: true,
            zerolinecolor: 'rgba(255,255,255,0.2)'
        },
        yaxis: { color: '#9ca3af', tickfont: { size: 11 } },
        height: Math.max(400, ranked.length * 30),
        margin: { t: 10, b: 50, l: 150, r: 20 }
    };

    Plotly.newPlot('shapSummaryChart', traces, layout, { responsive: true });
}

// ==================== Local Explanation ====================
document.getElementById('explainRowBtn').addEventListener('click', () => {
    const idx = parseInt(document.getElementById('localRowInput').value);
    loadLocalExplanation(idx);
});

async function loadLocalExplanation(index) {
    try {
        const resp = await fetch(`${API_BASE}/api/explain/${index}`);
        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        // Prediction info
        const infoEl = document.getElementById('predictionInfo');
        let infoHtml = `<div class="info-chip">Prediction: ${data.prediction}</div>`;
        if (data.probabilities) {
            for (const [cls, prob] of Object.entries(data.probabilities)) {
                infoHtml += `<div class="info-chip">${cls}: ${(prob * 100).toFixed(1)}%</div>`;
            }
        }
        infoEl.innerHTML = infoHtml;

        // Chart
        const entries = Object.entries(data.contributions).slice(0, 15);
        const features = entries.map(e => e[0]).reverse();
        const values = entries.map(e => e[1]).reverse();

        const trace = {
            type: 'bar',
            x: values,
            y: features,
            orientation: 'h',
            marker: {
                color: values.map(v => v > 0 ? 'rgba(239, 68, 68, 0.8)' : 'rgba(59, 130, 246, 0.8)'),
                line: { width: 0 }
            },
            hovertemplate: '%{y}: %{x:.4f}<extra></extra>'
        };

        const layout = {
            ...darkLayout(),
            xaxis: { title: 'SHAP Contribution', color: '#9ca3af', gridcolor: 'rgba(75,85,99,0.3)', zeroline: true, zerolinecolor: 'rgba(255,255,255,0.2)' },
            yaxis: { color: '#9ca3af', tickfont: { size: 11 } },
            height: Math.max(350, entries.length * 25),
            margin: { t: 10, b: 50, l: 150, r: 20 }
        };

        Plotly.newPlot('localExplanationChart', [trace], layout, { responsive: true });

        // Table
        const tableEl = document.getElementById('localExplanationTable');
        let tableHtml = `<table><thead><tr><th>Feature</th><th>Value</th><th>SHAP Contribution</th></tr></thead><tbody>`;
        for (const [feat, contrib] of Object.entries(data.contributions)) {
            const val = data.feature_values[feat];
            const color = contrib > 0 ? '#ef4444' : '#3b82f6';
            tableHtml += `<tr><td>${feat}</td><td>${typeof val === 'number' ? val.toFixed(4) : val}</td><td style="color:${color};font-weight:600;">${contrib.toFixed(4)}</td></tr>`;
        }
        tableHtml += '</tbody></table>';
        tableEl.innerHTML = tableHtml;

        // Update what-if inputs with the values from this row
        for (const [feat, val] of Object.entries(data.feature_values)) {
            const inp = document.getElementById(`whatif-${feat}`);
            if (inp) inp.value = typeof val === 'number' ? val.toFixed(4) : val;
        }

    } catch (err) {
        alert('Local explanation error: ' + err.message);
    }
}

// ==================== What-If Simulator ====================
function buildWhatIfInputs(featureNames) {
    const container = document.getElementById('whatifInputs');
    container.innerHTML = featureNames.map(name => `
        <div class="whatif-input-group">
            <label>${name}</label>
            <input type="number" step="any" id="whatif-${name}" value="0">
        </div>
    `).join('');
}

document.getElementById('whatifBtn').addEventListener('click', async () => {
    const features = {};
    appState.featureNames.forEach(name => {
        const input = document.getElementById(`whatif-${name}`);
        features[name] = parseFloat(input.value) || 0;
    });

    try {
        const resp = await fetch(`${API_BASE}/api/whatif`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features })
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        const resultEl = document.getElementById('whatifResult');
        resultEl.style.display = 'block';

        let html = `<h4>Simulation Result</h4><div class="prediction-value">${data.prediction}</div>`;
        if (data.probabilities) {
            html += '<div class="prob-bar">';
            for (const [cls, prob] of Object.entries(data.probabilities)) {
                html += `<span class="prob-item">${cls}: ${(prob * 100).toFixed(1)}%</span>`;
            }
            html += '</div>';
        }
        resultEl.innerHTML = html;
    } catch (err) {
        alert('What-if error: ' + err.message);
    }
});

// ==================== AI Insights ====================
document.getElementById('aiInsightBtn').addEventListener('click', async () => {
    const apiKey = document.getElementById('geminiKey').value;
    const resultEl = document.getElementById('aiInsightResult');
    resultEl.textContent = 'Generating insights...';

    try {
        const resp = await fetch(`${API_BASE}/api/ai-insights`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                feature_importance: appState.featureImportance,
                api_key: apiKey
            })
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        // Simple markdown rendering
        resultEl.innerHTML = renderMarkdown(data.explanation);
    } catch (err) {
        resultEl.textContent = 'Error: ' + err.message;
    }
});

// ==================== Utilities ====================
function showLoading(text) {
    loadingText.textContent = text;
    loadingOverlay.style.display = 'flex';
}
function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function darkLayout() {
    return {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#9ca3af', family: 'Inter' }
    };
}

function renderMarkdown(text) {
    return text
        .replace(/### (.*)/g, '<h3>$1</h3>')
        .replace(/## (.*)/g, '<h2>$1</h2>')
        .replace(/# (.*)/g, '<h1>$1</h1>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/^- (.*)/gm, '<li>$1</li>')
        .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
        .replace(/\n/g, '<br>');
}

// ==================== Loading Animation ====================
function showLoading(msg) {
    loadingText.textContent = msg;
    loadingOverlay.style.display = 'flex';
}
