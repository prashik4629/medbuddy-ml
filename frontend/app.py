import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/predict")
API_KEY = os.getenv("API_KEY", "medbuddy-dev-key-change-in-production")

st.set_page_config(
    page_title="MedBuddy.ML",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0a0f1e; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #1a1f35 100%);
        border-right: 1px solid rgba(99, 179, 237, 0.15);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    .main .block-container { background: #0a0f1e; padding: 2rem 3rem; }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(99, 179, 237, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #63b3ed 0%, #76e4f7 50%, #b794f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }

    .hero-sub {
        font-size: 1.1rem;
        color: #718096;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        color: #63b3ed;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .risk-high {
        background: rgba(245, 101, 101, 0.15);
        border: 1px solid rgba(245, 101, 101, 0.4);
        color: #fc8181;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    .risk-moderate {
        background: rgba(237, 137, 54, 0.15);
        border: 1px solid rgba(237, 137, 54, 0.4);
        color: #f6ad55;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    .risk-low {
        background: rgba(72, 187, 120, 0.15);
        border: 1px solid rgba(72, 187, 120, 0.4);
        color: #68d391;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3182ce, #553c9a);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        letter-spacing: 0.02em;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(49, 130, 206, 0.4);
    }

    hr { border-color: rgba(99, 179, 237, 0.1); margin: 1.5rem 0; }

    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0a0f1e; }
    ::-webkit-scrollbar-thumb { background: #3182ce; border-radius: 4px; }

    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


if "history" not in st.session_state:
    st.session_state.history = []


# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 2rem;'>
        <div style='font-size:3rem;'>❤️</div>
        <div style='font-size:1.3rem; font-weight:700; color:#63b3ed;'>MedBuddy.ML</div>
        <div style='font-size:0.75rem; color:#718096; margin-top:0.25rem;'>Heart Disease Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["Prediction", "History", "About"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.8rem; color:#718096; line-height:1.8;'>
        Model: XGBoost<br>
        Accuracy: 77.6%<br>
        ROC-AUC: 0.858<br>
        Dataset: Cleveland<br>
        Features: 13
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem; color:#4a5568; text-align:center; padding-top:1rem;'>
        For educational purposes only.<br>
        Not a substitute for medical advice.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# PAGE: PREDICTION
# ══════════════════════════════════════════
if page == "Prediction":

    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <p class='hero-title'>Heart Risk Analysis</p>
        <p class='hero-sub'>Enter patient vitals to receive an AI-powered cardiovascular risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Patient Information</div>',
                unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            age = st.number_input("Age", min_value=1, max_value=120, value=52)
            sex = st.selectbox("Sex", options=[0, 1],
                               format_func=lambda x: "Male" if x == 1 else "Female")
            cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                              format_func=lambda x: {
                                  0: "0 — Typical Angina",
                                  1: "1 — Atypical Angina",
                                  2: "2 — Non-Anginal Pain",
                                  3: "3 — Asymptomatic"}[x])
            fbs = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1],
                               format_func=lambda x: "Yes" if x == 1 else "No")

        with col2:
            st.markdown("**Vitals**")
            trestbps = st.number_input("Resting Blood Pressure (mmHg)",
                                       min_value=80, max_value=250, value=125)
            chol = st.number_input("Cholesterol (mg/dl)",
                                   min_value=100, max_value=600, value=212)
            thalach = st.number_input("Max Heart Rate",
                                      min_value=60, max_value=250, value=168)
            oldpeak = st.number_input("ST Depression (Oldpeak)",
                                      min_value=0.0, max_value=10.0,
                                      value=1.0, step=0.1)

        with col3:
            st.markdown("**ECG & Tests**")
            restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                                   format_func=lambda x: {
                                       0: "0 — Normal",
                                       1: "1 — ST-T Abnormality",
                                       2: "2 — LV Hypertrophy"}[x])
            exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                                 format_func=lambda x: "Yes" if x == 1 else "No")
            slope = st.selectbox("ST Slope", options=[0, 1, 2],
                                 format_func=lambda x: {
                                     0: "0 — Upsloping",
                                     1: "1 — Flat",
                                     2: "2 — Downsloping"}[x])
            ca = st.number_input("Major Vessels (0-4)",
                                 min_value=0, max_value=4, value=0)
            thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                                format_func=lambda x: {
                                    0: "0 — Normal",
                                    1: "1 — Fixed Defect",
                                    2: "2 — Reversible Defect",
                                    3: "3 — Unknown"}[x])

        submitted = st.form_submit_button("Run Risk Analysis")

    if submitted:
        payload = {
            "age": age, "sex": sex, "cp": cp,
            "trestbps": trestbps, "chol": chol, "fbs": fbs,
            "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }

        with st.spinner("Analyzing patient data..."):
            try:
                response = requests.post(
                    API_URL,
                    json=payload,
                    headers={"X-API-Key": API_KEY},
                    timeout=10,
                )
                result = response.json()

                if response.status_code != 200:
                    st.error(f"API Error: {result.get('detail', 'Unknown error')}")
                    st.stop()

                prediction  = result["prediction"]
                probability = result["probability"]
                diagnosis   = result["diagnosis"]
                risk_level  = result["risk_level"]

                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "age": age,
                    "sex": "Male" if sex == 1 else "Female",
                    "probability": probability,
                    "risk_level": risk_level,
                    "diagnosis": diagnosis,
                })

                st.markdown("---")
                st.markdown('<div class="section-label">Analysis Results</div>',
                            unsafe_allow_html=True)

                # ── 3 metric cards ──
                r1, r2, r3 = st.columns([1, 1, 1])

                risk_class = {
                    "High Risk": "risk-high",
                    "Moderate Risk": "risk-moderate",
                    "Low Risk": "risk-low",
                }[risk_level]

                card_style = """
                    background:rgba(255,255,255,0.04);
                    border:1px solid rgba(99,179,237,0.2);
                    border-radius:12px;
                    padding:0;
                    text-align:center;
                    height:110px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    align-items:center;
                """

                with r1:
                    st.markdown(f"""
                    <div style='{card_style}'>
                        <div style='font-size:0.7rem;color:#718096;
                            letter-spacing:0.1em;text-transform:uppercase;
                            margin-bottom:0.75rem;'>Diagnosis</div>
                        <div style='font-size:1rem;font-weight:600;
                            color:#e2e8f0;'>{diagnosis}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with r2:
                    st.markdown(f"""
                    <div style='{card_style}'>
                        <div style='font-size:0.7rem;color:#718096;
                            letter-spacing:0.1em;text-transform:uppercase;
                            margin-bottom:0.75rem;'>Risk Level</div>
                        <span class='{risk_class}'>{risk_level}</span>
                    </div>
                    """, unsafe_allow_html=True)

                with r3:
                    st.markdown(f"""
                    <div style='{card_style}'>
                        <div style='font-size:0.7rem;color:#718096;
                            letter-spacing:0.1em;text-transform:uppercase;
                            margin-bottom:0.75rem;'>Probability</div>
                        <div style='font-size:1.8rem;font-weight:700;
                            color:#63b3ed;'>{probability:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Gauge centered ──
                _, g_col, _ = st.columns([1, 2, 1])
                with g_col:
                    gauge_color = (
                        "#fc8181" if probability >= 0.75 else
                        "#f6ad55" if probability >= 0.45 else
                        "#68d391"
                    )
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(probability * 100, 1),
                        number={"suffix": "%",
                                "font": {"size": 48, "color": "#e2e8f0"}},
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickcolor": "#4a5568",
                                "tickfont": {"color": "#718096", "size": 12},
                            },
                            "bar": {"color": gauge_color, "thickness": 0.3},
                            "bgcolor": "rgba(0,0,0,0)",
                            "bordercolor": "rgba(0,0,0,0)",
                            "steps": [
                                {"range": [0, 45],
                                 "color": "rgba(72,187,120,0.1)"},
                                {"range": [45, 75],
                                 "color": "rgba(237,137,54,0.1)"},
                                {"range": [75, 100],
                                 "color": "rgba(245,101,101,0.1)"},
                            ],
                            "threshold": {
                                "line": {"color": gauge_color, "width": 4},
                                "thickness": 0.85,
                                "value": round(probability * 100, 1),
                            },
                        },
                        title={"text": "Risk Score",
                               "font": {"color": "#718096", "size": 14}},
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"family": "Inter"},
                        height=320,
                        margin=dict(t=40, b=0, l=30, r=30),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the backend is running.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")


# ══════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════
elif page == "History":
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <p class='hero-title'>Prediction History</p>
        <p class='hero-sub'>All predictions made in this session</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class='glass-card' style='text-align:center; padding:3rem;'>
            <div style='font-size:2rem;'>📋</div>
            <div style='color:#718096; margin-top:0.5rem;'>
                No predictions yet. Run an analysis first.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_history = pd.DataFrame(st.session_state.history)
        st.markdown(f"""
        <div style='margin-bottom:1rem;'>
            <span style='color:#63b3ed; font-weight:600;'>{len(df_history)}</span>
            <span style='color:#718096;'> predictions this session</span>
        </div>
        """, unsafe_allow_html=True)

        for i, row in df_history.iterrows():
            risk_class = {
                "High Risk": "risk-high",
                "Moderate Risk": "risk-moderate",
                "Low Risk": "risk-low",
            }[row["risk_level"]]

            st.markdown(f"""
            <div class='glass-card' style='display:flex;
                justify-content:space-between; align-items:center;'>
                <div style='display:flex; gap:2rem; align-items:center;'>
                    <div style='color:#4a5568; font-size:0.8rem;'>{row["time"]}</div>
                    <div style='color:#e2e8f0;'>{row["age"]} yrs · {row["sex"]}</div>
                    <div style='color:#63b3ed;
                        font-weight:600;'>{row["probability"]:.1%}</div>
                </div>
                <span class='{risk_class}'>{row["risk_level"]}</span>
            </div>
            """, unsafe_allow_html=True)

        if len(df_history) > 1:
            st.markdown("---")
            st.markdown('<div class="section-label">Risk Distribution</div>',
                        unsafe_allow_html=True)
            risk_counts = df_history["risk_level"].value_counts()
            color_map = {
                "High Risk": "#fc8181",
                "Moderate Risk": "#f6ad55",
                "Low Risk": "#68d391",
            }
            fig_pie = go.Figure(go.Pie(
                labels=risk_counts.index.tolist(),
                values=risk_counts.values.tolist(),
                hole=0.6,
                marker_colors=[color_map.get(l, "#718096")
                               for l in risk_counts.index],
                textfont={"color": "#e2e8f0"},
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#a0aec0"},
                height=300,
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(font={"color": "#a0aec0"}),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()


# ══════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════
elif page == "About":
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <p class='hero-title'>About MedBuddy.ML</p>
        <p class='hero-sub'>A professional heart disease prediction system</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-label'>How it works</div>
            <div style='color:#a0aec0; font-size:0.9rem; line-height:1.9;'>
                1. Patient vitals are entered into the form<br>
                2. Data is sent to the FastAPI backend<br>
                3. XGBoost model runs inference<br>
                4. Risk level and probability are returned<br>
                5. Results are visualized with charts
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-label'>Tech Stack</div>
            <div style='color:#a0aec0; font-size:0.9rem; line-height:1.9;'>
                Frontend: Streamlit + Plotly<br>
                Backend: FastAPI + Uvicorn<br>
                ML: XGBoost + Scikit-learn<br>
                Explainability: SHAP<br>
                Deployment: Render + Streamlit Cloud
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card' style='margin-top:1rem;'>
        <div class='section-label'>Disclaimer</div>
        <div style='color:#718096; font-size:0.85rem; line-height:1.8;'>
            This application is built for educational and demonstration
            purposes only. It is not intended to replace professional medical
            advice, diagnosis, or treatment. Always consult a qualified
            healthcare provider for medical decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)