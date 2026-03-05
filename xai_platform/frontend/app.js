/**
 * app.js - XplainAI Studio
 * Handles frontend logic, tab switching, and Plotly charting.
 */

const API_BASE = '';

let appState = {
    modelFile: null,
    datasetFile: null,
    columns: [],
    featureImportance: {},
    featureNames: [],
    numSamples: 0,
    metrics: null,
    previewData: []
};

// ================== DOM Elements ==================
const D = {
    modelZone: document.getElementById('modelDropZone'),
    dataZone: document.getElementById('datasetDropZone'),
    modelInput: document.getElementById('modelFile'),
    dataInput: document.getElementById('datasetFile'),
    modelName: document.getElementById('modelFileName'),
    dataName: document.getElementById('datasetFileName'),
    
    settings: document.getElementById('settingsSection'),
    targetSel: document.getElementById('targetSelect'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    
    hero: document.getElementById('heroSection'),
    dash: document.getElementById('dashboard'),
    topNav: document.getElementById('topNav'),
    navLinks: document.getElementById('navLinks'),
    navBadges: document.getElementById('navBadges'),
    footer: document.getElementById('footer'),
    loading: document.getElementById('loadingOverlay'),
    
    navA: document.querySelectorAll('.nav-links a'),
    views: document.querySelectorAll('.dash-view')
};

// ================== Tab Switching ==================
D.navA.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        D.navA.forEach(a => a.classList.remove('active'));
        D.views.forEach(v => v.classList.remove('active'));
        
        link.classList.add('active');
        const targetId = link.getAttribute('href').substring(1);
        document.getElementById(targetId).classList.add('active');
    });
});

// ================== File Upload ==================
function setupDropZone(zone, input, nameEl, type) {
    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = '#34d399'; });
    zone.addEventListener('dragleave', () => { zone.style.borderColor = ''; });
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.style.borderColor = '';
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            handleFile(input, nameEl, zone, type);
        }
    });
    input.addEventListener('change', () => handleFile(input, nameEl, zone, type));
}

function handleFile(input, nameEl, zone, type) {
    const file = input.files[0];
    if (!file) return;
    nameEl.textContent = file.name;
    zone.classList.add('has-file');
    if (type === 'model') appState.modelFile = file;
    else appState.datasetFile = file;
    checkUploads();
}

async function checkUploads() {
    if (appState.modelFile && appState.datasetFile) {
        showLoading('Uploading and parsing files...');
        try {
            const formData = new FormData();
            formData.append('model', appState.modelFile);
            formData.append('dataset', appState.datasetFile);

            const resp = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: formData });
            const data = await resp.json();
            if (data.error) throw new Error(data.error);

            D.targetSel.innerHTML = '';
            data.columns.forEach(col => {
                const opt = document.createElement('option');
                opt.value = col;
                opt.textContent = col;
                D.targetSel.appendChild(opt);
            });

            D.settings.style.display = 'block';
            D.analyzeBtn.disabled = false;
            appState.previewData = data.preview;
            document.getElementById('localRowInput').max = data.rows - 1;

            hideLoading();
        } catch (err) {
            hideLoading();
            alert('Upload Error: ' + err.message);
        }
    }
}

setupDropZone(D.modelZone, D.modelInput, D.modelName, 'model');
setupDropZone(D.dataZone, D.dataInput, D.dataName, 'dataset');

// ================== Analysis ==================
D.analyzeBtn.addEventListener('click', async () => {
    const targetCol = D.targetSel.value;
    if (!targetCol) return;

    showLoading('Running full XAI analysis...');
    try {
        const resp = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_column: targetCol })
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        appState = { ...appState, ...data };
        appState.targetCol = targetCol;
        appState.data_stats = data.data_stats;
        appState.class_balance = data.class_balance;
        appState.feature_distributions = data.feature_distributions;
        
        // Transition UI
        D.hero.style.display = 'none';
        D.dash.style.display = 'block';
        D.footer.style.display = 'flex';
        D.navLinks.style.display = 'flex';
        D.navBadges.style.display = 'flex';
        
        // Populate
        populateNavBadges();
        populateOverview();
        renderRadarChart();
        renderPieChart();
        renderFeatureImportance();
        populatePerformance();
        
        if (typeof buildWhatIfInputs === 'function') buildWhatIfInputs();
        buildDataInsights();

        // Initial local SHAP
        document.getElementById('localRowInput').value = 0;
        await explainLocalRow(0);

        hideLoading();
    } catch (err) {
        hideLoading();
        alert('Analysis Error: ' + err.message);
    }
});

// ================== Renderers ==================
function populateNavBadges() {
    const isReg = !appState.metrics.accuracy;
    document.getElementById('badgeModel').textContent = appState.model_type;
    document.getElementById('badgeSamples').textContent = appState.num_samples;
    
    if (!isReg) {
        const acc = (appState.metrics.accuracy * 100).toFixed(1);
        document.getElementById('badgeAcc').textContent = acc + '%';
    } else {
        const r2 = appState.metrics.r2_score.toFixed(3);
        document.getElementById('badgeAcc').textContent = r2 + ' (R²)';
    }
}

function fill(id, val) { const el = document.getElementById(id); if(el) el.textContent = val; }

function populateOverview() {
    fill('insightModelType', appState.model_type);
    fill('insightSamples', appState.num_samples);
    fill('insightFeatures', appState.num_features);
    fill('insightTarget', appState.targetCol);

    fill('valModelAcronym', appState.model_type.substring(0,3).toUpperCase());
    fill('valModelFull', appState.model_type);
    fill('valFeatureCount', appState.num_features);
    fill('valTotalSamples', appState.num_samples);

    const m = appState.metrics;
    if (m.accuracy) {
        fill('valAccuracy', (m.accuracy * 100).toFixed(1) + '%');
        fill('valAucRoc', m.auc_roc ? m.auc_roc.toFixed(3) : 'N/A');
    } else {
        fill('valAccuracy', m.r2_score.toFixed(3));
        document.querySelector('#valAccuracy + .m-sub').textContent = 'R2 Score';
        fill('valAucRoc', m.rmse.toFixed(3));
        document.querySelector('#valAucRoc + .m-sub').textContent = 'RMSE';
        document.querySelector('#valAucRoc').parentElement.querySelector('.m-label').textContent = 'RMSE';
    }
}

function renderRadarChart() {
    const m = appState.metrics;
    if(!m.accuracy) {
        document.getElementById('chartRadar').innerHTML = '<p class="text-muted mt-4">Radar chart is only available for classification models.</p>';
        return;
    }
    
    const trace = {
        type: 'scatterpolar',
        r: [m.accuracy, m.precision, m.recall, m.f1_score, m.auc_roc || 0, m.accuracy], 
        theta: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Accuracy'],
        fill: 'toself',
        fillcolor: 'rgba(52, 211, 153, 0.2)',
        line: { color: '#34d399' }
    };
    const layout = {
        ...getPlotLayout(),
        polar: {
            radialaxis: { visible: true, range: [0, 1], color: '#64748b', gridcolor: 'rgba(255,255,255,0.1)' },
            angularaxis: { color: '#94a3b8' },
            bgcolor: 'rgba(0,0,0,0)'
        },
        margin: { t: 20, b: 20, l: 40, r: 40 }
    };
    Plotly.newPlot('chartRadar', [trace], layout, {responsive: true});
}

function renderPieChart() {
    const m = appState.metrics;
    if(!m.confusion_matrix) return;
    
    const cm = m.confusion_matrix;
    let vals, labels, colors;
    
    if (cm.length === 2) {
        vals = [cm[1][1], cm[0][0], cm[0][1], cm[1][0]]; // TP, TN, FP, FN
        labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative'];
        colors = ['#34d399', '#60a5fa', '#f87171', '#fbbf24'];
    } else {
        // Multi-class sum diagonal (correct) vs rest (incorrect)
        let correct = 0, total = 0;
        for(let i=0; i<cm.length; i++) {
            for(let j=0; j<cm.length; j++) {
                total += cm[i][j];
                if(i===j) correct += cm[i][j];
            }
        }
        vals = [correct, total - correct];
        labels = ['Correct', 'Incorrect'];
        colors = ['#34d399', '#f87171'];
    }

    const trace = {
        type: 'pie',
        values: vals,
        labels: labels,
        marker: { colors: colors, line: { color: '#151b2b', width: 2 } },
        textinfo: 'percent',
        hoverinfo: 'label+value',
        hole: 0.4
    };
    const layout = {
        ...getPlotLayout(),
        showlegend: true,
        legend: { orientation: 'h', y: -0.1 },
        margin: { t: 10, b: 10, l: 10, r: 10 }
    };
    Plotly.newPlot('chartPie', [trace], layout, {responsive: true});
}

function renderFeatureImportance() {
    const entries = Object.entries(appState.feature_importance).slice(0, 15);
    const features = entries.map(e => e[0]).reverse();
    const values = entries.map(e => e[1]).reverse();

    const trace = {
        type: 'bar',
        x: values,
        y: features,
        orientation: 'h',
        marker: {
            color: values.map((v, i) => i%2===0 ? '#34d399' : '#f87171') // Alternate style for visual diversity matching mockup
        }
    };
    const layout = { ...getPlotLayout(), margin: {l: 120, r: 20, t: 10, b: 40} };
    Plotly.newPlot('chartFeatureImportance', [trace], layout, {responsive: true});
}

function populatePerformance() {
    const m = appState.metrics;
    if(m.accuracy) {
        fill('perfAccuracy', (m.accuracy * 100).toFixed(1) + '%');
        fill('perfPrecision', (m.precision * 100).toFixed(1) + '%');
        fill('perfRecall', (m.recall * 100).toFixed(1) + '%');
        fill('perfF1', (m.f1_score * 100).toFixed(1) + '%');
        
        // Confusion matrix cards
        const grid = document.getElementById('cmGridCards');
        if(m.confusion_matrix.length === 2) {
            const [[TN, FP], [FN, TP]] = m.confusion_matrix;
            grid.innerHTML = `
                <div class="cm-box cm-tp"><div class="cm-val">${TP}</div><div class="cm-label">True Positive</div></div>
                <div class="cm-box cm-tn"><div class="cm-val">${TN}</div><div class="cm-label">True Negative</div></div>
                <div class="cm-box cm-fp"><div class="cm-val">${FP}</div><div class="cm-label">False Positive</div></div>
                <div class="cm-box cm-fn"><div class="cm-val">${FN}</div><div class="cm-label">False Negative</div></div>
            `;
        } else {
            grid.innerHTML = '<p class="text-muted text-center py-4">Detailed breakdown currently only supports binary classification.</p>';
        }

        // Class report
        const tb = document.querySelector('#classReportTable tbody');
        tb.innerHTML = '';
        if(m.classification_report) {
            for(const [cls, stats] of Object.entries(m.classification_report)) {
                if(cls === 'accuracy' || cls === 'macro avg' || cls === 'weighted avg') continue;
                tb.innerHTML += `
                    <tr>
                        <td class="class-name">${cls}</td>
                        <td>${stats.precision.toFixed(2)}</td>
                        <td>${stats.recall.toFixed(2)}</td>
                        <td>${stats['f1-score'] ? stats['f1-score'].toFixed(2) : stats.f1_score ? stats.f1_score.toFixed(2) : '0.00'}</td>
                        <td>${stats.support}</td>
                    </tr>
                `;
            }
        }
    }
}

// ================== Local Explain (Predict & Explain) ==================
document.getElementById('explainRowBtn').addEventListener('click', () => {
    const idx = parseInt(document.getElementById('localRowInput').value) || 0;
    explainLocalRow(idx);
});

async function explainLocalRow(idx) {
    try {
        fill('dispSampleIdx', idx);
        const resp = await fetch(`${API_BASE}/api/explain/${idx}`);
        const data = await resp.json();
        if(data.error) return;

        fill('dispPredOutcome', data.prediction);
        if(data.probabilities) {
            let pStr = '';
            for(const [k,v] of Object.entries(data.probabilities)) {
                if(v > 0.5) pStr += `(Confidence: ${(v*100).toFixed(1)}%)`;
            }
            fill('dispPredOutcome', `${data.prediction} ${pStr}`);
        }

        const entries = Object.entries(data.contributions).sort((a,b)=> Math.abs(b[1]) - Math.abs(a[1])).slice(0, 10).reverse();
        const base = 0.5; // Dummy base if unavailable
        let diff = entries.reduce((s, e)=>s+e[1], 0);
        
        fill('dispBaseVal', '0.50'); // Default assuming scaled, would need actual base from SHAP expected_value
        fill('dispFinalVal', (0.50 + diff).toFixed(2));

        const trace = {
            type: 'bar', orientation: 'h',
            x: entries.map(e=>e[1]),
            y: entries.map(e=>e[0]),
            marker: { color: entries.map(e=>e[1] > 0 ? '#34d399' : '#f87171') }
        };
        Plotly.newPlot('chartLocalShap', [trace], { ...getPlotLayout(), margin: {l: 120, r:20, t:10, b:30} }, {responsive:true});
        
    } catch(e) {}
}

// ================== What If (Split View) ==================

let whatIfTimeout = null;

document.getElementById('loadWhatIfBtn').addEventListener('click', async () => {
    const idx = parseInt(document.getElementById('whatifRowInput').value) || 0;
    try {
        const resp = await fetch(`${API_BASE}/api/explain/${idx}`);
        const data = await resp.json();
        if(data.error) return alert(data.error);

        // Store original state
        appState.whatIfOriginal = {
            idx: idx,
            features: data.feature_values,
            prediction: data.prediction,
            probabilities: data.probabilities,
            contributions: data.contributions
        };

        // Populate Left Panel (Inputs)
        buildWhatIfSplitInputs();
        
        // Populate Right Panel (Original)
        updatePredBox('whatifOriginalPred', 'whatifOriginalProb', data.prediction, data.probabilities);
        updatePredBox('whatifModifiedPred', 'whatifModifiedProb', data.prediction, data.probabilities);
        
        // Clear impact chart and explanation
        document.getElementById('whatifExplanationText').innerHTML = 'Showing baseline instance. Change a feature value to see its impact.';
        Plotly.purge('chartWhatIfImpact');
        
    } catch(e) {
        alert("Failed to load instance.");
    }
});

document.getElementById('resetWhatIfBtn').addEventListener('click', () => {
    if(!appState.whatIfOriginal) return;
    buildWhatIfSplitInputs(); // Rebuild from original
    // Reset Right Panel
    updatePredBox('whatifModifiedPred', 'whatifModifiedProb', appState.whatIfOriginal.prediction, appState.whatIfOriginal.probabilities);
    document.getElementById('whatifExplanationText').innerHTML = 'Values reset to original baseline.';
    Plotly.purge('chartWhatIfImpact');
});

function buildWhatIfSplitInputs() {
    if(!appState.whatIfOriginal) return;
    const wrap = document.getElementById('whatifInputs');
    wrap.innerHTML = Object.entries(appState.whatIfOriginal.features).map(([f, val]) => `
        <div class="whatif-input-wrapper">
            <label>${f}</label>
            <input type="number" step="any" id="wi-${f}" class="num-input whatif-trigger" value="${val}">
        </div>
    `).join('');

    // Highlight changed inputs for UX
    document.querySelectorAll('.num-input').forEach(input => {
        input.addEventListener('input', () => {
             input.style.borderColor = 'var(--accent-green)';
        });
    });
}

document.getElementById('execWhatIfBtn').addEventListener('click', () => {
    // Reset visual hints
    document.querySelectorAll('.num-input').forEach(input => input.style.borderColor = '');
    runWhatIfSimulation();
});

async function runWhatIfSimulation() {
    if(!appState.whatIfOriginal) return;
    
    // Gather modified features
    const features = {};
    Object.keys(appState.whatIfOriginal.features).forEach(f => {
        features[f] = parseFloat(document.getElementById(`wi-${f}`).value) || 0;
    });

    try {
        const resp = await fetch(`${API_BASE}/api/whatif`, {
            method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({features})
        });
        const data = await resp.json();
        if(data.error) return;

        // Update Right Panel (Modified)
        updatePredBox('whatifModifiedPred', 'whatifModifiedProb', data.prediction, data.probabilities);

        // Compute SHAP Differences
        const origContrib = appState.whatIfOriginal.contributions;
        const modContrib = data.contributions;
        const diffs = [];
        let maxDiffFeat = null;
        let maxAbsDiff = 0;

        Object.keys(origContrib).forEach(f => {
            const d = (modContrib[f] || 0) - origContrib[f];
            if (Math.abs(d) > 0.0001) {
                diffs.push({ feature: f, diff: d });
            }
            if (Math.abs(d) > maxAbsDiff) {
                maxAbsDiff = Math.abs(d);
                maxDiffFeat = { feature: f, diff: d, oldVal: appState.whatIfOriginal.features[f], newVal: features[f] };
            }
        });

        diffs.sort((a,b) => Math.abs(a.diff) - Math.abs(b.diff)); // ascending abs for horizontal bar chart
        
        // Plot Impact
        if(diffs.length > 0) {
            const trace = {
                type: 'bar', orientation: 'h',
                x: diffs.map(d => d.diff),
                y: diffs.map(d => d.feature),
                marker: { color: diffs.map(d => d.diff > 0 ? '#34d399' : '#f87171') } // green positive, red negative
            };
            Plotly.newPlot('chartWhatIfImpact', [trace], { ...getPlotLayout(), margin: {l: 120, r:20, t:10, b:30} }, {responsive:true});
            
            // Build Explanation
            if(maxDiffFeat) {
                const dir = maxDiffFeat.diff > 0 ? '<span class="highlight-dir highlight-up">increased</span>' : '<span class="highlight-dir highlight-down">decreased</span>';
                const origValTxt = typeof maxDiffFeat.oldVal === 'number' ? maxDiffFeat.oldVal.toFixed(3) : maxDiffFeat.oldVal;
                const newValTxt = typeof maxDiffFeat.newVal === 'number' ? maxDiffFeat.newVal.toFixed(3) : maxDiffFeat.newVal;

                document.getElementById('whatifExplanationText').innerHTML = 
                    `The prediction ${dir} primarily because <span class="highlight-feat">${maxDiffFeat.feature}</span> was changed from <strong>${origValTxt}</strong> to <strong>${newValTxt}</strong>.`;
            }

        } else {
            Plotly.purge('chartWhatIfImpact');
            document.getElementById('whatifExplanationText').innerHTML = 'Modifications did not significantly impact the prediction.';
        }

    } catch(e) {}
}

function updatePredBox(valId, probId, prediction, probabilities) {
    document.getElementById(valId).textContent = prediction;
    const pEl = document.getElementById(probId);
    if(probabilities) {
        const maxProb = Math.max(...Object.values(probabilities));
        const conf = (maxProb * 100).toFixed(1);
        pEl.textContent = `Confidence: ${conf}%`;
    } else {
        pEl.textContent = '';
    }
}


// ================== Data Insights ==================
function buildDataPreview() {
    if (!appState.preview || !appState.preview.length) return;
    const cols = appState.columns || Object.keys(appState.preview[0]);
    let h = '<table class="report-table"><thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
    appState.preview.forEach(row => {
        h += '<tr>' + cols.map(c => `<td>${typeof row[c] === 'number' ? row[c].toFixed(3) : row[c]}</td>`).join('') + '</tr>';
    });
    h += '</tbody></table>';
    const el = document.getElementById('dataPreviewTable');
    if (el) el.innerHTML = h;
}

function buildDataInsights() {
    buildDataPreview();

    // 1. Class Balance Pie Chart
    if (appState.class_balance) {
        const labels = Object.keys(appState.class_balance);
        const values = Object.values(appState.class_balance);
        const traces = [{
            type: 'pie',
            labels: labels,
            values: values,
            hole: 0.5,
            marker: { colors: ['#34d399', '#f87171', '#60a5fa', '#fbbf24'] },
            textinfo: 'label+percent',
            hoverinfo: 'label+value'
        }];
        const layout = { ...getPlotLayout(), showlegend: true, legend: {orientation: 'h', y: -0.2}, margin: {t:20, b:40, l:20, r:20} };
        Plotly.newPlot('chartClassBalance', traces, layout, {responsive: true});
    }

    // 2. Dataset Stats Panel
    if (appState.dataset_stats_panel) {
        const p = appState.dataset_stats_panel;
        const rows = [
            ['Total Rows', p.total_rows],
            ['Feature Columns', p.feature_columns],
            ['Missing Values', `${p.missing_values} (${p.missing_pct}%)`],
            ['Duplicate Rows', `${p.duplicate_rows} (${p.duplicate_pct}%)`],
            ['Numeric Features', p.numeric_features],
            ['Categorical Features', p.categorical_features]
        ];
        const panelHTML = rows.map(([label, val]) => `
            <div class="stat-row">
                <div class="stat-label">${label}</div>
                <div class="stat-dots"></div>
                <div class="stat-val">${val}</div>
            </div>`).join('');
        const container = document.getElementById('datasetStatsPanel');
        if (container) container.innerHTML = panelHTML;
    }

    // 3. Feature Distribution Bar Chart
    if (appState.numeric_feature_dist && appState.numeric_feature_dist.length) {
        const features = appState.numeric_feature_dist.map(f => f.feature);
        const means = appState.numeric_feature_dist.map(f => f.mean);
        const hovers = appState.numeric_feature_dist.map(f =>
            `Feature: ${f.feature}<br>Average Value: ${f.mean.toFixed(2)}<br>Count: ${f.count}`);
        const trace = {
            x: features,
            y: means,
            type: 'bar',
            text: hovers,
            hoverinfo: 'text',
            marker: { color: '#60a5fa' }
        };
        const layout = {
            ...getPlotLayout(),
            xaxis: { title: 'Features', tickangle: -40 },
            yaxis: { title: 'Average Value' },
            margin: { t: 20, b: 80, l: 50, r: 20 }
        };
        Plotly.newPlot('chartFeatureDistMod', [trace], layout, {responsive: true});
    }
}

// ================== Helpers ==================
function getPlotLayout() {
    return {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8', family: 'Inter' },
        xaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { gridcolor: 'rgba(255,255,255,0.05)' }
    };
}
function showLoading(t) { document.getElementById('loadingText').textContent = t; D.loading.style.display = 'flex'; }
function hideLoading() { D.loading.style.display = 'none'; }
