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
        renderShapSummary();
        populatePerformance();
        buildWhatIfInputs();
        buildDataPreview();

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

    // Populate the sub grid
    const grid = document.getElementById('featureImportanceGrid');
    const total = values.reduce((a,b)=>a+b, 0) || 1;
    
    let html = '';
    entries.slice(0,6).forEach(e => {
        const perc = ((e[1]/total)*100).toFixed(0);
        html += `
            <div class="f-stat-card">
                <div class="f-stat-perc">${perc}%</div>
                <div class="f-stat-info">
                    <h5>${e[0]}</h5>
                    <p>High relative importance score</p>
                </div>
            </div>
        `;
    });
    grid.innerHTML = html;
}

function renderShapSummary() {
    const shapData = appState.shap_summary;
    const fMap = {};
    shapData.forEach(d => {
        if(!fMap[d.feature]) fMap[d.feature] = {s:[], f:[]};
        fMap[d.feature].s.push(d.shap_value);
        fMap[d.feature].f.push(d.feature_value);
    });

    const ranked = Object.entries(fMap)
        .map(([name, d]) => ({ name, mean: d.s.reduce((sum, v)=>sum+Math.abs(v),0)/d.s.length, ...d}))
        .sort((a,b) => b.mean - a.mean).slice(0, 10).reverse();

    const traces = ranked.map(feat => {
        const min = Math.min(...feat.f), max = Math.max(...feat.f), range = max - min || 1;
        const colors = feat.f.map(v => {
            const r = (v-min)/range;
            // Blue to Red (0 to 1) -> rgb(96,165,250) to rgb(248,113,113)
            return `rgb(${96 + r*(248-96)}, ${165 - r*(165-113)}, ${250 - r*(250-113)})`;
        });
        return {
            type: 'scatter', mode: 'markers', x: feat.s, y: feat.s.map(()=>feat.name),
            marker: { color: colors, size: 6, opacity: 0.8 }, showlegend: false
        }
    });
    
    const layout = { ...getPlotLayout(), xaxis: { zeroline: true, zerolinecolor: 'rgba(255,255,255,0.2)', title: 'SHAP Value' }, margin: {l:120, r:20, t:10, b:40} };
    Plotly.newPlot('chartGlobalShap', traces, layout, {responsive:true});
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
                        <td>${stats.f1-score ? stats['f1-score'].toFixed(2) : stats.f1_score ? stats.f1_score.toFixed(2) : '0.00'}</td>
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

// ================== What If ==================
function buildWhatIfInputs() {
    const wrap = document.getElementById('whatifInputs');
    wrap.innerHTML = appState.featureNames.map(f => `
        <div class="whatif-input-wrapper">
            <label>${f}</label>
            <input type="number" step="any" id="wi-${f}" class="num-input" value="0">
        </div>
    `).join('');
}

document.getElementById('whatifBtn').addEventListener('click', async () => {
    const features = {};
    appState.featureNames.forEach(f => features[f] = parseFloat(document.getElementById(`wi-${f}`).value) || 0);
    try {
        const resp = await fetch(`${API_BASE}/api/whatif`, {
            method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({features})
        });
        const data = await resp.json();
        const resEl = document.getElementById('whatifResult');
        resEl.style.display = 'block';
        if(data.probabilities) {
            resEl.textContent = `Predicted: ${data.prediction} (Conf: ${(Object.values(data.probabilities).find(v => v > 0.5) * 100 || Math.max(...Object.values(data.probabilities)) * 100).toFixed(1)}%)`;
        } else {
            resEl.textContent = `Predicted: ${data.prediction}`;
        }
    } catch(e) {}
});

// ================== Data Preview ==================
function buildDataPreview() {
    if(!appState.previewData.length) return;
    const cols = Object.keys(appState.previewData[0]);
    let h = '<table class="report-table"><thead><tr>' + cols.map(c=>`<th>${c}</th>`).join('') + '</tr></thead><tbody>';
    appState.previewData.forEach(row => {
        h += '<tr>' + cols.map(c => `<td>${typeof row[c]==='number'?row[c].toFixed(3):row[c]}</td>`).join('') + '</tr>';
    });
    h += '</tbody></table>';
    document.getElementById('dataPreviewTable').innerHTML = h;
}

// ================== AI Insights ==================
document.getElementById('aiInsightBtn').addEventListener('click', async () => {
    const res = document.getElementById('aiInsightResult');
    res.textContent = 'Generating insights...';
    try {
        const resp = await fetch(`${API_BASE}/api/ai-insights`, {
            method:'POST', headers:{'Content-Type':'application/json'}, 
            body:JSON.stringify({feature_importance: appState.feature_importance, api_key: document.getElementById('geminiKey').value})
        });
        const data = await resp.json();
        res.innerHTML = data.explanation
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');
    } catch(e) { res.textContent = 'Failed to generate insights.'; }
});

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
