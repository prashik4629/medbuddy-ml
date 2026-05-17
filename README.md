# 🫀 MedBuddy.ML — Heart Disease Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![Render](https://img.shields.io/badge/Backend-Render-purple?style=flat-square)
![Streamlit Cloud](https://img.shields.io/badge/Frontend-Streamlit%20Cloud-red?style=flat-square)

A professional, production-inspired AI healthcare application that predicts the risk of heart disease using machine learning. Built with a clean architecture separating the ML pipeline, REST API backend, and interactive frontend.

---

## 🌐 Live Demo

| Component | Link |
|---|---|
| 🖥️ Frontend App | [medbuddy-ml-a.streamlit.app](https://medbuddy-ml-a.streamlit.app) |
| ⚡ Backend API | [medbuddy-ml-api.onrender.com](https://medbuddy-ml-api.onrender.com) |
| 📖 API Docs | [medbuddy-ml-api.onrender.com/docs](https://medbuddy-ml-api.onrender.com/docs) |

> ⚠️ The backend is hosted on Render's free tier. First request may take 30–50 seconds to wake up.

---

## 🧠 Project Overview

MedBuddy.ML takes 13 clinical features as input and predicts whether a patient is at risk of heart disease, along with a confidence probability and risk level classification.

This project demonstrates production-level ML engineering practices including model comparison, cross-validation, explainability, API design, authentication, and cloud deployment.

---

## ✨ Features

### Machine Learning
- Compared 4 models: Random Forest, XGBoost, Logistic Regression, Voting Ensemble
- 5-fold Stratified Cross Validation for reliable evaluation
- Automatic best model selection based on ROC-AUC score
- SHAP explainability to understand feature importance
- Group-based train/test split to prevent data leakage from duplicate rows

### Backend
- FastAPI with versioned endpoints (`/api/v1/predict`)
- API key authentication on all prediction endpoints
- Pydantic input validation with clinical range checks
- Structured logging to file and console
- Global exception handling
- CORS middleware for frontend communication
- `/health` endpoint for uptime monitoring

### Frontend
- Dark medical-themed UI built with Streamlit + Plotly
- Interactive risk gauge chart
- 3-column patient input form with medical context
- Session-based prediction history with donut chart
- Risk level badges (Low / Moderate / High)
- Responsive layout with collapsible sidebar

---

## 🏗️ Architecture

```
Patient Input (Streamlit Frontend)
          ↓
    HTTP POST Request
    X-API-Key Header
          ↓
  FastAPI Backend (Render)
    Input Validation
    API Key Auth
          ↓
  XGBoost ML Model
  (joblib pipeline)
          ↓
  Prediction + Risk Level
          ↓
  JSON Response
          ↓
  Gauge Chart + Result Cards
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly |
| Backend | FastAPI, Uvicorn |
| ML Models | XGBoost, Scikit-learn, Voting Ensemble |
| Explainability | SHAP |
| Config | Pydantic Settings, python-dotenv |
| Serialization | Joblib |
| Backend Hosting | Render |
| Frontend Hosting | Streamlit Community Cloud |
| Version Control | GitHub |

---

## 📁 Project Structure

```
medbuddy-ml/
│
├── backend/
│   ├── core/
│   │   ├── config.py          # Centralized settings via pydantic-settings
│   │   └── security.py        # API key authentication
│   ├── routers/
│   │   └── prediction.py      # /api/v1/predict endpoint
│   ├── services/
│   │   └── predictor.py       # Model loading and inference
│   ├── main.py                # FastAPI app, CORS, exception handlers
│   └── training.py            # Full ML training pipeline
│
├── frontend/
│   └── app.py                 # Streamlit UI
│
├── dataset/
│   └── heart.csv              # Cleveland Heart Disease dataset
│
├── model_dir/
│   ├── heart_disease_prediction_model.joblib
│   └── shap_summary.png
│
├── .streamlit/
│   └── secrets.toml           # Streamlit Cloud secrets (not committed)
│
├── .env.example               # Environment variable template
├── .gitignore
├── Procfile                   # Render process file
├── render.yaml                # Render deployment config
├── requirements.txt           # Full requirements
├── requirements-backend.txt   # Backend-only requirements for Render
├── runtime.txt                # Python version
└── README.md
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11+
- Git

### Installation

```bash
# Clone the repo
git clone https://github.com/prashik4629/medbuddy-ml.git
cd medbuddy-ml

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env and set your PROJECT_ROOT path
```

### Train the model

```bash
python -m backend.training
```

### Run the backend

```bash
uvicorn backend.main:app --reload
```

### Run the frontend

```bash
streamlit run frontend/app.py
```

---

## 📊 Model Performance

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.8735 | 0.9158 | 0.8795 | 0.9555 |
| XGBoost | **0.9332** | **0.9966** | **0.9365** | **0.9977** |
| Logistic Regression | 0.8332 | 0.8838 | 0.8426 | 0.9198 |
| Voting Ensemble | 0.9414 | 0.9409 | 0.9418 | 0.9898 |

**Winner: XGBoost** with ROC-AUC of 0.9977 on cross-validation.

Test set ROC-AUC: **0.8577**

---

## 🔒 API Authentication

All prediction requests require an API key in the header:

```bash
curl -X POST https://medbuddy-ml-api.onrender.com/api/v1/predict \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52, "sex": 1, "cp": 0,
    "trestbps": 125, "chol": 212, "fbs": 0,
    "restecg": 1, "thalach": 168, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
  }'
```

---

## 📋 API Response

```json
{
  "prediction": 1,
  "probability": 0.9220,
  "diagnosis": "Heart Disease Detected",
  "risk_level": "High Risk"
}
```

---

## 🔮 Future Improvements

- [ ] Add patient report PDF export
- [ ] Add database to store prediction history
- [ ] Add more datasets for improved generalization
- [ ] Add LIME explainability alongside SHAP
- [ ] Add JWT authentication for multi-user support
- [ ] Add model retraining pipeline
- [ ] Add fairness/bias analysis across demographics
- [ ] Add unit and integration tests

---

## ⚠️ Disclaimer

This application is built for **educational and demonstration purposes only**. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 👤 Author

**Prashik Meshram**
- GitHub: [@prashik4629](https://github.com/prashik4629)

---

## 📄 Dataset

Cleveland Heart Disease Dataset from the UCI Machine Learning Repository.
- 1,025 patient records
- 13 clinical features
- Binary classification target (0 = No Disease, 1 = Disease)
