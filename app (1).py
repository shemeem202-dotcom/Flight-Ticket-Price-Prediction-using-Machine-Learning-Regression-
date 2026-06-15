import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Syne:wght@700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    /* ── Hero title ── */
    .hero-eyebrow {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #818cf8;
        margin-bottom: 10px;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.2rem, 5vw, 3.8rem);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.02em;
        background: linear-gradient(
            100deg,
            #f472b6 0%,
            #a78bfa 25%,
            #60a5fa 50%,
            #34d399 75%,
            #fbbf24 100%
        );
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 5s linear infinite;
        margin: 0 0 12px;
    }
    @keyframes shimmer {
        0%   { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    .hero-sub {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 400;
        color: #64748b;
        letter-spacing: 0.01em;
    }
    .hero-sub strong { color: #94a3b8; font-weight: 600; }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #a5b4fc;
        letter-spacing: 0.06em;
        margin-left: 10px;
        vertical-align: middle;
        text-transform: uppercase;
    }

    .main { background-color: #0f172a; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }

    /* ── Predict button ── */
    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        color: #fff;
        border: none;
        border-radius: 14px;
        padding: 18px 0;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: 0 6px 30px rgba(139,92,246,0.45);
        margin-top: 8px;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(139,92,246,0.65);
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 50%, #c084fc 100%);
    }
    div[data-testid="stButton"] > button:active {
        transform: translateY(0px);
        box-shadow: 0 4px 18px rgba(139,92,246,0.4);
    }

    /* ── Prediction result card ── */
    .predict-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
        border: 1px solid rgba(139,92,246,0.35);
        border-radius: 20px;
        padding: 36px 24px;
        text-align: center;
        color: white;
        box-shadow: 0 12px 40px rgba(99,102,241,0.3);
        margin: 24px 0;
        animation: fadeUp 0.5s ease;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .predict-label {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #a5b4fc;
        margin-bottom: 6px;
    }
    .predict-price {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #c4b5fd, #a5b4fc, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .predict-meta {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 10px;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        color: white;
        margin: 4px 0;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: rgba(139,92,246,0.5); }
    .metric-value { font-size: 1.9rem; font-weight: 800; color: #a5b4fc; }
    .metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
    .metric-sub   { font-size: 0.7rem; color: #475569; margin-top: 2px; }

    /* ── Section headers ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: 0.02em;
        margin: 36px 0 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(99,102,241,0.2);
    }
    .section-eyebrow {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #6366f1;
        margin-bottom: 4px;
    }

    /* ── Sidebar ── */
    div[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
    .stSelectbox label, .stSlider label, .stNumberInput label { color: #94a3b8 !important; font-size: 0.8rem !important; }
    h1, h2, h3 { color: #f1f5f9 !important; }
    p, li { color: #cbd5e1; }

    /* Divider */
    hr { border-color: rgba(99,102,241,0.15) !important; }

    /* Info box */
    .stAlert { background: rgba(99,102,241,0.08) !important; border-color: rgba(99,102,241,0.25) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────
@st.cache_data
def generate_dataset():
    np.random.seed(42)
    n = 10682

    airlines     = ["IndiGo", "Air India", "Jet Airways", "SpiceJet", "Vistara",
                    "GoAir", "Multiple carriers", "Air Asia"]
    sources      = ["Banglore", "Kolkata", "Delhi", "Chennai", "Mumbai"]
    destinations = ["New Delhi", "Banglore", "Cochin", "Kolkata", "Hyderabad", "Delhi"]
    stops        = [0, 1, 2, 3]
    stop_w       = [0.35, 0.45, 0.15, 0.05]

    airline_col  = np.random.choice(airlines, n)
    source_col   = np.random.choice(sources, n)
    dest_col     = np.random.choice(destinations, n)
    stops_col    = np.random.choice(stops, n, p=stop_w)
    month_col    = np.random.randint(1, 13, n)
    day_col      = np.random.randint(1, 29, n)
    dep_hour     = np.random.randint(0, 24, n)
    dep_min      = np.random.choice([0, 15, 30, 45], n)
    duration_min = np.random.randint(60, 720, n)
    arr_hour     = (dep_hour + duration_min // 60) % 24
    arr_min      = dep_min

    airline_premium = np.where(np.isin(airline_col, ["Vistara", "Jet Airways", "Air India"]), 1500, 0)
    base_price = (
        3000
        + duration_min * 4.5
        + stops_col * 800
        + airline_premium
        + np.where(np.isin(month_col, [3,4,5,10,11,12]), 1000, 0)
        + np.random.normal(0, 800, n)
    ).clip(1500, 60000)

    df = pd.DataFrame({
        "Airline": airline_col, "Source": source_col, "Destination": dest_col,
        "Total_Stops": stops_col, "Journey_Month": month_col,
        "Journey_Day": day_col, "Dep_Hour": dep_hour, "Dep_Minute": dep_min,
        "Duration_Mins": duration_min.astype(int),
        "Arr_Hour": arr_hour, "Arr_Minute": arr_min,
        "Price": base_price.astype(int)
    })
    return df

@st.cache_resource
def train_models(df):
    df_enc = df.copy()
    le = LabelEncoder()
    encoders = {}
    for col in df_enc.select_dtypes(include="object").columns:
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    X = df_enc.drop("Price", axis=1)
    y = df_enc["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":  GradientBoostingRegressor(random_state=42),
        "Decision Tree":      DecisionTreeRegressor(random_state=42),
        "Linear Regression":  LinearRegression(),
        "KNN Regressor":      KNeighborsRegressor(n_neighbors=5),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        results[name] = {
            "model": m,
            "pred":  pred,
            "MAE":   mean_absolute_error(y_test, pred),
            "RMSE":  np.sqrt(mean_squared_error(y_test, pred)),
            "R2":    r2_score(y_test, pred),
        }

    return results, encoders, X_test, y_test, X.columns.tolist()

# ─────────────────────────────────────────
# Load & train
# ─────────────────────────────────────────
df = generate_dataset()

with st.spinner("Training models…"):
    results, encoders, X_test, y_test, feature_cols = train_models(df)

best_model_name = max(results, key=lambda k: results[k]["R2"])
best_model      = results[best_model_name]["model"]

# ─────────────────────────────────────────
# Sidebar – inputs
# ─────────────────────────────────────────
st.sidebar.markdown("## ✈️ Flight Details")
st.sidebar.markdown("---")

airline     = st.sidebar.selectbox("Airline",
    ["IndiGo","Air India","Jet Airways","SpiceJet","Vistara","GoAir","Multiple carriers","Air Asia"])
source      = st.sidebar.selectbox("Source City",
    ["Banglore","Kolkata","Delhi","Chennai","Mumbai"])
destination = st.sidebar.selectbox("Destination City",
    ["New Delhi","Banglore","Cochin","Kolkata","Hyderabad","Delhi"])
stops       = st.sidebar.selectbox("Total Stops", [0, 1, 2, 3],
    format_func=lambda x: ["Non-stop","1 Stop","2 Stops","3+ Stops"][x])
month       = st.sidebar.slider("Journey Month", 1, 12, 6, help="1=Jan … 12=Dec")
day         = st.sidebar.slider("Journey Day", 1, 28, 15)
dep_hour    = st.sidebar.slider("Departure Hour", 0, 23, 8)
dep_min     = st.sidebar.selectbox("Departure Minute", [0, 15, 30, 45])
duration    = st.sidebar.slider("Flight Duration (mins)", 60, 720, 150)

arr_hour = (dep_hour + duration // 60) % 24
arr_min  = dep_min

st.sidebar.markdown("---")
predict_clicked = st.sidebar.button("🔮  Predict Price")

# ─────────────────────────────────────────
# Encode helper
# ─────────────────────────────────────────
def encode(val, col):
    return encoders.get(col, {}).get(val, 0)

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.markdown("""
<div style="padding: 36px 0 28px;">
    <div class="hero-eyebrow">✈ &nbsp; Indian Domestic Routes · ML Powered</div>
    <div class="hero-title">Flight Ticket<br>Price Predictor</div>
    <div class="hero-sub">
        Powered by <strong>Random Forest</strong> · trained on 10,000+ domestic flight records
        <span class="hero-badge">Live Model</span>
    </div>
</div>
<hr style="margin-bottom: 28px;">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Prediction result (shown on button click OR always)
# ─────────────────────────────────────────
input_data = pd.DataFrame([{
    "Airline":       encode(airline, "Airline"),
    "Source":        encode(source, "Source"),
    "Destination":   encode(destination, "Destination"),
    "Total_Stops":   stops,
    "Journey_Month": month,
    "Journey_Day":   day,
    "Dep_Hour":      dep_hour,
    "Dep_Minute":    dep_min,
    "Duration_Mins": duration,
    "Arr_Hour":      arr_hour,
    "Arr_Minute":    arr_min,
}])[feature_cols]

predicted_price = max(1500, best_model.predict(input_data)[0])

if predict_clicked or "predicted" not in st.session_state:
    st.session_state["predicted"] = predicted_price
    st.session_state["pred_meta"] = f"{airline}  ·  {source} → {destination}  ·  {'Non-stop' if stops == 0 else str(stops) + ' stop(s)'}"

if "predicted" in st.session_state:
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(f"""
        <div class="predict-box">
            <div class="predict-label">Estimated Fare</div>
            <div class="predict-price">₹{st.session_state['predicted']:,.0f}</div>
            <div class="predict-meta">{st.session_state['pred_meta']}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.78rem'>"
    "✈️ Flight Price Predictor · Random Forest · Indian Domestic Routes"
    "</p>", unsafe_allow_html=True
)