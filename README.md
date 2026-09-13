
# 🌾 KrishiSahayak — Agricultural Risk Prediction System
**Crop Distress Early-Warning System for Maharashtra Districts**

Built for farmers and agricultural officers who need to know — which district, which crop, which year is heading toward distress — before the season ends badly.

---

## Problem Statement
Maharashtra farmers face crop failures every year but get warnings too late.
KrishiSahayak predicts it early —

**Rainfall + Temperature + Yield History → Distress Score → Risk Level → Action**

---

## Modules

| Module | Description |
|---|---|
| Risk Label Engine | Builds Low/Medium/High label from real yield shortfall + rainfall deviation |
| ML Predictor | XGBoost classifier — predicts crop distress per district per year |
| Baseline Comparison | Logistic Regression vs XGBoost — honest accuracy reporting |
| Bilingual Dashboard | English/Marathi Streamlit UI with district map |

---

## Model Results

| Model | Accuracy | F1 (Macro) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 57.5% | 0.570 | 0.761 |
| XGBoost | 63.9% | 0.638 | 0.816 |

---

## Dataset
- Real Maharashtra government agricultural data
- 25 districts · 14 crops · 1999–2017
- 4,684 model-ready rows
- Source: Maharashtra district-wise crop + rainfall records

---

## Tech Stack
- **Language:** Python 3.12
- **Dashboard:** Streamlit (English + Marathi)
- **ML:** XGBoost · Scikit-learn
- **Charts:** Plotly
- **Data:** Pandas + CSV

---

## Setup
```bash
pip install streamlit xgboost scikit-learn pandas numpy plotly joblib
python 01_generate_data.py
python 02_train_model.py
streamlit run app.py
```

---

## Folder Structure
```
## 📁 Repository Guide

Files are grouped here by role — everything still lives in the repo root,
this is just a map of what's what.

**📂 Data**
- `raw_final_crop.csv` — original raw government dataset
- `krishisahayak_dataset.csv` — cleaned, feature-engineered, model-ready data

**🧠 Pipeline Scripts** (run in this order)
- `01_generate_data.py` — cleaning + feature engineering
- `01b_eda.py` — exploratory data analysis, summary stats, outlier detection
- `02_train_model.py` — model training + evaluation
- `reference_data.py` — district coordinates used by the map
- `app.py` — Streamlit dashboard

**🤖 Trained Model Artifacts** (produced by `02_train_model.py`)
- `xgb_model.pkl` — saved XGBoost classifier
- `label_encoder.pkl` — encodes Low/Medium/High ↔ 0/1/2
- `feature_list.pkl` — exact feature order the model expects

**📊 EDA Outputs** (produced by `01b_eda.py`)
- `01_missing_value_map.png`, `02_outlier_boxplots.png`,
  `03_univariate_distributions.png`, `04_year_trends.png`,
  `05_yield_by_crop.png`, `06_target_distribution.png`,
  `07_correlation_heatmap.png`, `08_top_risk_districts.png`
- `missing_value_report.csv`, `outlier_report.csv`,
  `summary_statistics.csv`, `variable_description.csv`

**📈 Model Evaluation Outputs** (produced by `02_train_model.py`)
- `confusion_matrix.png`, `feature_importance.png`

**⚙️ Config**
- `requirements.txt` — dependencies
- `README.md` — this file

---

**Developed by Durva Sagar Ambre · TY Project · 2026**

---
