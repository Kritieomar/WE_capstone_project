# XplainAI Studio — Explainable AI Platform

> **Understand your model, not just its output.**

XplainAI Studio is a full-stack web application that takes any trained scikit-learn model and a dataset and provides deep, interactive explanations of how and why the model makes its predictions using SHAP (SHapley Additive exPlanations).

---

## 📸 Features

- **Model Overview** — Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix, Radar Chart
- **Feature Importance** — SHAP-based global feature importance bar chart
- **SHAP Analysis** — Local per-sample waterfall chart showing feature contributions
- **What-If Simulator** — Modify any feature value and instantly see how the prediction changes
- **Data Insights** — Feature distribution, class balance, and dataset statistics

---

## 🛠️ Tech Stack

### Backend
| Library | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Flask | REST API Server |
| Flask-CORS | Cross-origin request handling |
| scikit-learn | ML metrics, model & pipeline support |
| SHAP | Model-agnostic feature attribution |
| pandas | Data loading and processing |
| NumPy | Numerical computations |
| joblib | Model serialization |

### Frontend
| Technology | Purpose |
|---|---|
| HTML5 / CSS3 | Structure and dark-themed UI |
| Vanilla JavaScript (ES6+) | DOM manipulation and API calls |
| Plotly.js | Interactive charts |
| Google Fonts | Inter, Syncopate, Space Mono typography |

---

## 📁 Project Structure

```
xai_platform/
├── backend/
│   ├── api_server.py          # Flask routes & API endpoints
│   ├── model_loader.py        # Load .pkl / .joblib model files
│   ├── data_handler.py        # CSV loading, feature prep, pipeline extraction
│   ├── explanation_engine.py  # SHAP computation (KernelExplainer)
│   ├── metrics_engine.py      # sklearn classification metrics
│   └── ai_insight_engine.py   # Optional Gemini AI integration
├── frontend/
│   ├── index.html             # Single-page dashboard
│   ├── styles.css             # Full custom dark UI
│   └── app.js                 # All frontend logic & chart rendering
├── test_gen.py                # Script to generate a sample loan model & dataset
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Kritieomar/WE_capstone_project.git
cd WE_capstone_project/xai_platform
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Sample Model & Dataset (Optional)
If you don't have a model ready, generate one using the loan dataset:
```bash
python test_gen.py
```
This creates `loan_model.joblib` and uses `loan_preprocessed_dataset.csv`.

### 5. Start the Server
```bash
python backend/api_server.py
```

### 6. Open the Dashboard
Navigate to **http://localhost:5000** in your browser.

---

## 🧪 How to Use

1. **Upload** your trained `.pkl` or `.joblib` model file
2. **Upload** your dataset as a `.csv` file
3. **Select** the target column (what the model predicts)
4. Click **"Run Full XAI Analysis"**
5. Explore the dashboard tabs:
   - **Overview** → Model performance summary
   - **Feature Importance** → Which features matter most
   - **SHAP Analysis** → Why a specific prediction was made
   - **What-If Simulator** → What would change the prediction
   - **Data Insights** → Distribution and class balance

---


## ✅ Supported Models

Any scikit-learn compatible model:
- `LogisticRegression`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `SVC`
- `XGBClassifier` (via sklearn wrapper)
- `Pipeline` (with preprocessing + classifier)


---



## 👩‍💻 Authors

Built as part of the **WE Capstone Project**.

