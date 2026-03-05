import React, { useState } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import './index.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [modelFile, setModelFile] = useState(null);
  const [dataFile, setDataFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [targetCol, setTargetCol] = useState('');
  const [localIndex, setLocalIndex] = useState(0);

  const [isLoading, setIsLoading] = useState(false);
  const [globalData, setGlobalData] = useState(null);
  const [localData, setLocalData] = useState(null);
  const [error, setError] = useState('');

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!modelFile || !dataFile || !targetCol) {
      setError('Please provide Model, Dataset, and Target Column.');
      return;
    }

    try {
      setIsLoading(true);
      setError('');
      
      const formData = new FormData();
      formData.append('model_file', modelFile);
      formData.append('data_file', dataFile);
      formData.append('target_col', targetCol);

      // 1. Upload files
      const uploadRes = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      console.log('Upload response:', uploadRes.data);

      // 2. Fetch global explanation
      const globalRes = await axios.get(`${API_BASE}/explain/global`);
      setGlobalData(globalRes.data);

      // 3. Fetch local explanation
      const localRes = await axios.get(`${API_BASE}/explain/local/${localIndex}`);
      setLocalData(localRes.data);

    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFetchLocal = async () => {
    try {
      setIsLoading(true);
      const localRes = await axios.get(`${API_BASE}/explain/local/${localIndex}`);
      setLocalData(localRes.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Prepare chart data for feature importance
  const getGlobalChartData = () => {
    if (!globalData?.feature_importance) return null;
    
    // Sort by value descending
    const sorted = Object.entries(globalData.feature_importance)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15); // Top 15

    return {
      labels: sorted.map(i => i[0]),
      datasets: [
        {
          label: 'Mean Absolute SHAP Value',
          data: sorted.map(i => i[1]),
          backgroundColor: 'rgba(59, 130, 246, 0.8)',
          borderColor: 'rgba(59, 130, 246, 1)',
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    };
  };

  const globalChartData = getGlobalChartData();
  const globalChartOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: true, text: 'Global Feature Importance (SHAP)', color: '#fff' }
    },
    scales: {
      y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
      x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <h2>XAI Platform</h2>
        <p>Interactive Explainer Dashboard</p>

        <form onSubmit={handleUpload}>
          <div className="form-group">
            <label>1. Upload Model (.joblib/.pkl)</label>
            <input type="file" accept=".pkl,.joblib" onChange={e => setModelFile(e.target.files[0])} required />
          </div>

          <div className="form-group">
            <label>2. Upload Dataset (.csv)</label>
            <input type="file" accept=".csv" onChange={e => setDataFile(e.target.files[0])} required />
          </div>

          <div className="form-group">
            <label>3. Target Column Name</label>
            <input 
              type="text" 
              placeholder="e.g. target, label, Y" 
              value={targetCol} 
              onChange={e => setTargetCol(e.target.value)} 
              required 
            />
          </div>

          <div className="form-group">
            <label>Local Explanation Row Index</label>
            <input 
              type="number" 
              min="0" 
              value={localIndex} 
              onChange={e => setLocalIndex(parseInt(e.target.value) || 0)} 
            />
          </div>

          <button type="submit" className="btn btn-primary" disabled={isLoading} style={{width: '100%', marginTop: '1rem'}}>
            {isLoading ? <div className="spinner" /> : 'Run Analysis'}
          </button>
        </form>

        {globalData && (
          <button onClick={handleFetchLocal} className="btn" style={{background: 'rgba(255,255,255,0.1)', color: 'white', marginTop: '1rem'}}>
            Reload Local Row Only
          </button>
        )}

      </aside>

      {/* Main Content */}
      <main className="main-content">
        {error && (
          <div className="alert error">
            <svg style={{width: 20, height: 20}} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            {error}
          </div>
        )}

        {!globalData && !isLoading && !error && (
          <div className="glass-panel" style={{textAlign: 'center', padding: '4rem 2rem'}}>
            <svg style={{width: 64, height: 64, margin: '0 auto 1rem', color: '#3b82f6'}} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path></svg>
            <h2>Welcome to Explainer Platform</h2>
            <p>Upload your trained ML model and dataset in the sidebar to generate beautiful, interactive SHAP explanations modularly extracted from explainerdashboard.</p>
          </div>
        )}

        {globalData && (
          <>
            <div className="dashboard-grid">
              <div className="glass-panel metric-card">
                <div className="metric-label">Model Type</div>
                <h3 style={{color: '#3b82f6', marginTop: '0.5rem'}}>sklearn / pipeline</h3>
              </div>
              <div className="glass-panel metric-card">
                <div className="metric-label">Target Column</div>
                <div className="metric-value" style={{fontSize: '1.5rem', color: '#10b981'}}>{targetCol}</div>
              </div>
            </div>

            <div className="glass-panel">
              <div className="chart-container">
                {globalChartData && <Bar data={globalChartData} options={globalChartOpts} />}
              </div>
            </div>
          </>
        )}

        {localData && (
          <div className="dashboard-grid" style={{gridTemplateColumns: '1fr 2fr'}}>
            <div className="glass-panel">
              <h3>Local Row {localIndex} Prediction</h3>
              <p style={{marginBottom: '2rem'}}>What-if simulation output and expected base margin.</p>
              
              <div style={{marginBottom: '1.5rem'}}>
                <div className="metric-label">Base Value (Expected)</div>
                <div className="metric-value">{localData.base_value?.toFixed(4) || 0}</div>
              </div>

              <div>
                <div className="metric-label">Model Prediction</div>
                <div className="metric-value" style={{color: '#10b981'}}>{localData.prediction?.toFixed(4) || 0}</div>
              </div>
            </div>

            <div className="glass-panel">
              <h3>Feature Contributions</h3>
              <div style={{overflowX: 'auto', marginTop: '1rem'}}>
                <table>
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Value</th>
                      <th>SHAP Contribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(localData.feature_values).map((feat, i) => {
                      const shapVal = localData.shap_values[i];
                      const isPositive = shapVal > 0;
                      return (
                        <tr key={feat}>
                          <td>{feat}</td>
                          <td style={{color: '#94a3b8'}}>{localData.feature_values[feat]}</td>
                          <td style={{color: isPositive ? 'var(--danger)' : 'var(--accent-color)', fontWeight: 500}}>
                            {isPositive ? '+' : ''}{shapVal?.toFixed(4) || 0}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
