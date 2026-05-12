import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.graph_objects as go
from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import plotly.express as px
import base64

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# ======================================================
# PAGE CONFIG (MUST BE FIRST)
# ======================================================

st.set_page_config(
    page_title="Malaria Risk Forecasting | CSIR",
    layout="wide"
)

MODEL_PATH = None
HORIZON = None

# ======================================================
# NEW: STATE → DATA PATH MAP
# ======================================================

STATE_DATA = {
    "Assam": "data/assam.csv",
    "Tripura": "data/tripura.csv",
    "Meghalaya": "data/meghalaya.csv",
    "Arunachal Pradesh": "data/arunachalpradesh.csv",
    "Mizoram": "data/mizoram.csv",
    "Nagaland": "data/nagaland.csv",
    "Sikkim": "data/sikkim.csv",
}

# ======================================================
# SESSION STATE
# ======================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model(model_path):
    model = NBeats.load_from_checkpoint(model_path)
    model.eval()
    return model

st.markdown("""
<style>

/* ================= GLOBAL RESET ================= */

header {visibility: hidden;}

html, body, [class*="css"] {
    margin: 0 !important;
    padding: 0 !important;
    font-family: "Segoe UI", "Noto Sans", "Arial", sans-serif;
    overflow-x: hidden !important;   /*  prevent horizontal scroll */
}

/* Remove Streamlit page padding completely */
.block-container {
    padding: 0rem 1rem 0rem 1rem !important;
    margin: 0 !important;
    max-width: 100% !important;
}

/* Remove default vertical spacing */
div[data-testid="stVerticalBlock"] {
    gap: 0rem !important;
}

/* ================= SELECTBOX LABEL FIX ================= */

div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stSelectbox"]) {
    background: linear-gradient(90deg, #008080, #20b2aa);
    padding: 20px 25px;
    border-radius: 8px;
    margin-bottom: 15px;
}

/* Make labels white */
label[data-testid="stWidgetLabel"] {
    color: white !important;
    font-weight: 600 !important;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #006666, #008080) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    transition: 0.2s ease-in-out;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}

/* ================= NAVBAR BASE ROW ================= */

/* keep base clean (do NOT color all rows) */
div[data-testid="stHorizontalBlock"] {
    gap: 0rem !important;
    align-items: center !important;
    flex-wrap: wrap !important;   /*  responsive wrap */
}

/* Remove extra top space Streamlit injects */
section.main > div:first-child {
    padding-top: 0rem !important;
}

/* ================= HEADER ================= */

.header {
    width: 100%;
    background-color: white;
    padding: 6px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.header img {
    height: 60px;
}

.header-center {
    flex: 1;
    text-align: center;
}

.header-center h1 {
    margin: 0;
    font-size: 30px;
    font-weight: 700;
    color: #000000;
    word-wrap: break-word;   /*  prevent overflow */
}

.dashboard-card {
    background: white;
    padding: 25px;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-top: 20px;
}

h2 {
    font-weight: 700 !important;
    letter-spacing: 0.5px;
}

h3 {
    font-weight: 600 !important;
    color: #004d4d;
}
/* ================= PREMIUM STICKY NAVBAR ================= */

.nav-marker + div[data-testid="stHorizontalBlock"] {
    position: sticky;
    top: 0;
    z-index: 999;
    background: linear-gradient(90deg, #008080, #20b2aa);
    padding: 14px 40px;
    color: white !important;
    font-size: 12px !important;
    align-items: center !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.18);
}

/* Center columns */
.nav-marker + div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
    display: flex !important;
    justify-content: center !important;
    color: white !important;
    font-size: 12px !important;
    align-items: center !important;
    min-width: 0 !important;
}

/* Navbar buttons */
.nav-marker + div[data-testid="stHorizontalBlock"] button {
    background: transparent !important;
    border: none !important;
    color: white !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.8px !important;
    padding: 8px 16px !important;
    margin: 0 !important;
    line-height: 1 !important;
    border-radius: 6px;
    transition: all 0.25s ease;
    white-space: nowrap;   /*  prevent ugly wrapping */
}

/* Hover */
.nav-marker + div[data-testid="stHorizontalBlock"] button:hover {
    background: rgba(255,255,255,0.15) !important;
}

/*  ACTIVE TAB HIGHLIGHT */
.nav-marker + div[data-testid="stHorizontalBlock"] button[kind="primary"] {
    background: white !important;
    color: #008080 !important;
    font-weight: 800 !important;
}

/* ================= IMAGE SECTION ================= */

.image-section div[data-testid="column"] {
    display: flex;
    align-items: flex-start;
}

/* Remove white block padding */
div[data-testid="stImage"] {
    padding: 0 !important;
}

/* prevent cropping */
.image-section img {
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}

/* ================= GLOBAL RESPONSIVE ================= */

/*  TABLET */
@media (max-width: 1100px) {

    .nav-marker + div[data-testid="stHorizontalBlock"] {
        padding: 12px 20px;
    }

    .nav-marker + div[data-testid="stHorizontalBlock"] button {
        font-size: 13px !important;
        padding: 6px 10px !important;
    }

    .header-center h1 {
        font-size: 24px;
    }
}

body {
    background-color: #f7f9fb;
}

/* ================= FOOTER ================= */

.footer {
    margin-top: 20px;
    padding: 15px;
    background: #f2f2f2;
    font-size: 13px;
    text-align: center;
    border-top: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
# ======================================================
# HEADER WITH LOGOS
# ======================================================
st.markdown("""
<div class="header">
    <img src="data:image/png;base64,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">
    <div class="header-center">
        <h1>Malaria Risk Forecasting and Early Warning System for North-East states of India</h1>
    </div>
    <img
src="data:image/png;base64,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">
    <img src="data:image/png;base64,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">
</div>
""", unsafe_allow_html=True) 

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
# ======================================================
# NAV BAR (ORANGE)
# ======================================================
st.markdown("""
<style>

/* Make navbar background orange */
div[data-testid="stHorizontalBlock"] {
    background-color: #008080;
    padding: 10px 40px;
}

/* Style navbar buttons */
div[data-testid="stHorizontalBlock"] button {
    background-color: transparent !important;
    border: none !important;
    color: white !important;
    font-size: 12px !important;
    font-weight: 500 !important;   /* EXTRA BOLD */
    text-transform: uppercase !important;  /* ALL CAPS */
    letter-spacing: 1px !important;
}

/* Hover effect */
div[data-testid="stHorizontalBlock"] button:hover {
    background-color: rgba(255,255,255,0.2) !important;
    border-radius: 2px;
}

</style>
""", unsafe_allow_html=True)


nav_container = st.container()

with nav_container:
    st.markdown('<div class="nav-marker">', unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        if st.button("Home", type="primary" if st.session_state.page=="home" else "secondary"):
            st.session_state.page = "home"
            st.rerun()

    with col2:
        if st.button("About Us", type="primary" if st.session_state.page=="about" else "secondary"):
            st.session_state.page = "about"
            st.rerun()
        
    with col3:
        if st.button("About the Data", type="primary" if st.session_state.page=="data" else "secondary"):
            st.session_state.page = "data"
            st.rerun()

    with col4:
        if st.button("Methods", type="primary" if st.session_state.page=="methods" else "secondary"):
            st.session_state.page = "methods"
            st.rerun()

    with col5:
        if st.button("Forecasting"):
            st.session_state.page = "forecast"
            st.rerun()

    with col6:
        if st.button("Contact Us"):
            st.session_state.page = "contact"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# MALARIA HERO IMAGE (REQUESTED)
# ======================================================

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

if st.session_state.page == "home":

    left_img = img_to_base64("images/image 10.png")
    right_img = img_to_base64("images/image 4.png")

    st.markdown(f"""
    <style>

    .hero-wrapper {{
        display: flex;
        width: 100%;
        gap: 14px;
        margin-bottom: 28px;
    }}

    .hero-left {{
        flex: 4;        /* 70% */
        height: 650px;
        overflow: hidden;
    }}

    .hero-right {{
        flex: 3;        /* 30% */
        height: 650px;
        overflow: hidden;
    }}

    .hero-left img,
    .hero-right img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center;
        display: block;
        border-radius: 8px;
    }}

    </style>

    <div class="hero-wrapper">
        <div class="hero-left">
            <img src="data:image/png;base64,{left_img}">
        </div>
        <div class="hero-right">
            <img src="data:image/png;base64,{right_img}">
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
# ======================================================
# PAGE CONTENT
# ======================================================
if st.session_state.page == "home":
    st.subheader("Overview")
    st.markdown("""
    Malaria remains a major public health challenge in the North-Eastern states of India.
    This system provides early warning through deep learning–based forecasting to support
    proactive disease control and elimination strategies.
    """)

elif st.session_state.page == "about":
    st.subheader("About Us")
    st.write(
        "Indian Institute of Chemical Technology (IICT), Hyderabad, established in 1944, is a constituent laboratory of the Council of Scientific and Industrial Research (CSIR), New Delhi. With its expertise in chemistry and chemical technology, it provides solutions to challenges faced by Industry, Government Departments and Entrepreneurs through basic and applied research, and process development. The institute is internationally recognized for its contributions to chemistry research and is an ideal place for taking ideas to commercialization through state-of-the-art research and development. CSIR-IICT during its seventy-year journey has made its mark as a dynamic, innovative and result-oriented R&D organization. The clientele spans all comers of the globe. In India, it is CSIR-Indian Institute of Chemical Technology (CSIR-IICT) is one of the oldest National Laboratories the reliable destination of chemical and biotech industries. The reputation that CSIR-IICT could establish amongst the industrial clients as a reliable R&D partner, can be largely attributed to its rich pool of scientists with expertise in broad-ranging research areas and simple and effective business development strategies."
    )

elif st.session_state.page == "data":
    st.subheader("About the Data")
    st.write(
        "Malaria remains a major public health challenge in the North-Eastern states of India, which contribute a disproportionately high share of the country’s Plasmodium falciparum burden and represent some of the most persistent transmission zones nationally. The region accounts for about 15% of India’s malaria cases and roughly 12% of the national P. falciparum cases, the most severe form of malaria infection.Transmission in this region is sustained by efficient vector species such as Anopheles minimus and Anopheles baimaii, along with favorable ecological conditions including forested terrain, high humidity, and perennial hill streams that support continuous mosquito breeding. In addition, the North-East serves as a critical epidemiological corridor linking India with Southeast Asia, facilitating the historical introduction of antimalarial drug-resistant parasite strains into the country. Many malaria-affected areas in the region are located in tribal and difficult-to-access locations, which further complicates surveillance and control efforts.In this context, early warning systems based on deep learning–driven forecasting can play an important role by identifying temporal patterns in malaria transmission and supporting evidence-based decision-making for targeted intervention planning. Such predictive approaches align with India’s national malaria elimination strategy (2016–2030) by strengthening preparedness and improving resource prioritization in high-risk transmission settings."
        
    )

elif st.session_state.page == "methods":
    st.subheader("Methods")
    st.write(
        "Forecasting is performed using the N-BEATS deep learning architecture, "
        
    )

elif st.session_state.page == "contact":
    st.subheader("Contact Us")
    st.write("""
    **CSIR – Indian Institute of Chemical Technology (IICT)**  
    **Academy of Scientific and Innovative Research (AcSIR)**  

     jshraddha888@gmail.com
    """)

    st.markdown(
        "<hr style='border:1px solid #e0e0e0; margin-top:30px; margin-bottom:20px;'>",
        unsafe_allow_html=True
    )
# ======================================================
# FORECASTING PAGE (UPDATED — IMPORTANT)
# ======================================================
elif st.session_state.page == "forecast":

    st.subheader("Malaria Forecasting Dashboard")
    st.markdown("---")

    # ======================================================
    # STEP 1 — STATE SELECTION
    # ======================================================
    selected_state = st.selectbox(
        "Select North-East State",
        list(STATE_DATA.keys()),
        key="state_select"
    )

    # Small gap
    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)

    # ======================================================
    # STEP 2 — LOAD DATA
    # ======================================================
    data_path = STATE_DATA[selected_state]

    try:
        data = pd.read_csv(data_path)
    except Exception:
        st.error(f"Could not load data for {selected_state}")
        st.stop()

    # ======================================================
    # STEP 3 — DATA PREPARATION
    # ======================================================
    data["Month"] = pd.to_datetime(
        data["Month"], format="%d-%m-%Y", errors="coerce"
    )

    data = data.dropna(subset=["Month", "LogCases"])
    data = data.sort_values("Month").reset_index(drop=True)

    data["time_idx"] = np.arange(len(data))
    data["series"] = 0

    # ======================================================
    # STEP 4 — DATA PREVIEW
    # ======================================================
    st.subheader(f"Data Preview — {selected_state}")
    st.dataframe(data.head(), use_container_width=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ======================================================
    # STEP 5 — HORIZON SELECTION
    # ======================================================
    horizon_choice = st.selectbox(
        "Select Forecast Horizon",
        ["3 Months", "6 Months"],
        key="horizon_select"
    )

    if horizon_choice == "3 Months":
        HORIZON = 3
        MODEL_PATH = "models/nbeats_3month.ckpt"
    else:
        HORIZON = 6
        MODEL_PATH = "models/nbeats_6month.ckpt"

    st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)

    # ======================================================
    # STEP 6 — GENERATE FORECAST BUTTON
    # ======================================================
    if st.button("Generate Forecast", key="forecast_button"):

        with st.spinner("Generating forecast... Please wait"):

            model = load_model(MODEL_PATH)

            dataset = TimeSeriesDataSet(
                data,
                time_idx="time_idx",
                target="LogCases",
                group_ids=["series"],
                max_encoder_length=12,
                max_prediction_length=HORIZON,
                time_varying_unknown_reals=["LogCases"],
                target_normalizer=GroupNormalizer(groups=["series"]),
            )

            dl = dataset.to_dataloader(train=False, batch_size=1)
            preds = model.predict(dl).detach().cpu().numpy().flatten()

            # ---------------- Forecast Data ----------------
            last_month = data["Month"].iloc[-1]
            forecast_months = pd.date_range(
                start=last_month + pd.DateOffset(months=1),
                periods=HORIZON,
                freq="MS"
            )

            forecast_df = pd.DataFrame({
                "Month": forecast_months,
                "Forecasted LogCases": preds[-HORIZON:]
            })

            # ---------------- Confidence Intervals (LOG SCALE) ----------------
            hist_std = data["LogCases"].std()

            forecast_df["Lower CI"] = (
                forecast_df["Forecasted LogCases"] - 1.96 * hist_std
            )

            forecast_df["Upper CI"] = (
                forecast_df["Forecasted LogCases"] + 1.96 * hist_std
            )

            # ================= CONVERT BACK TO ORIGINAL SCALE =================

            data["Cases"] = np.exp(data["LogCases"])

            forecast_df["Forecasted Cases"] = np.exp(
                forecast_df["Forecasted LogCases"]
            )

            forecast_df["Lower CI Cases"] = np.exp(
                forecast_df["Lower CI"]
            )

            forecast_df["Upper CI Cases"] = np.exp(
                forecast_df["Upper CI"]
            )

            # ---------------- Metrics ----------------
            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            col1.metric("Forecast Horizon", f"{HORIZON} Months")
            col2.metric("Latest Cases", f"{round(data['Cases'].iloc[-1], 0):,}")
            col3.metric("Average Forecasted Cases", f"{round(forecast_df['Forecasted Cases'].mean(), 0):,}")

            # ---------------- Risk Classification (ORIGINAL SCALE) ----------------
            low_thr = np.percentile(data["Cases"], 33)
            high_thr = np.percentile(data["Cases"], 66)

            def classify(val):
                if val <= low_thr:
                    return "Low Risk"
                elif val <= high_thr:
                    return "Medium Risk"
                else:
                    return "High Risk"

            forecast_df["Risk Level"] = forecast_df[
                "Forecasted Cases"
            ].apply(classify)
            

            # ---------------- Plot ----------------

            plt.style.use("seaborn-v0_8-whitegrid")

            forecast_x = pd.concat([
                pd.Series([data["Month"].iloc[-1]]),
                forecast_df["Month"]
            ])

            forecast_y = np.concatenate([
                [data["Cases"].iloc[-1]],
                forecast_df["Forecasted Cases"]
            ])

            # ================= CREATE FIGURE =================

            fig = go.Figure()

            # ---------------- Historical Line ----------------
            fig.add_trace(
                go.Scatter(
                    x=data["Month"],
                    y=data["Cases"],
                    mode="lines+markers",
                    name="Historical Cases",
                    line=dict(color="#006666", width=3),
                    marker=dict(size=6),
                    hovertemplate="<b>Month:</b> %{x}<br><b>Cases:</b> %{y:.2f}<extra></extra>"
                )
            )

            # ---------------- Empty Forecast Trace (For Animation) ----------------
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#cc3300", width=3, dash="dash"),
                    marker=dict(size=6),
                    hovertemplate="<b>Month:</b> %{x}<br><b>Forecast:</b> %{y:.2f}<extra></extra>"
                )
            )

            # ================= CREATE ANIMATION FRAMES =================

            frames = []

            for i in range(1, len(forecast_x) + 1):
                frames.append(
                    go.Frame(
                        data=[
                            go.Scatter(
                                x=forecast_x[:i],
                                y=forecast_y[:i]
                            )
                        ],
                        traces=[1] 
                    )
                )

            fig.frames = frames

            # ================= CONFIDENCE INTERVAL =================

            fig.add_trace(
                go.Scatter(
                    x=forecast_df["Month"],
                    y=forecast_df["Upper CI"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_df["Month"],
                    y=forecast_df["Lower CI"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(255, 204, 153, 0.4)",
                    line=dict(width=0),
                    name="95% Confidence Interval",
                    hoverinfo="skip"
                )
            )

            # ================= ANIMATION BUTTON WITH EASING =================

            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=True,
                        bgcolor="#006666",          # highlighted background
                        bordercolor="#004c4c",
                        borderwidth=2,
                        font=dict(color="black", size=13),

                        buttons=[
                            dict(
                                label="▶ Show Forecast",
                                method="animate",
                                args=[
                                    None,
                                    dict(
                                        frame=dict(duration=700, redraw=True),
                                        transition=dict(
                                            duration=600,
                                            easing="cubic-in-out"
                                        ),
                                        fromcurrent=True
                                    )
                                ]
                            )
                        ],

                        x=1,
                        y=1.18,
                        xanchor="right",
                        yanchor="top",
                        pad=dict(t=8, r=10)
                    )
                ]
            )
            # ================= LAYOUT STYLING =================

            fig.update_layout(
                title=dict(
                    text=f"{selected_state} Malaria Forecast ({HORIZON} Months)",
                    font=dict(size=20),
                    x=0.02
                ),
                xaxis_title="Month",
                yaxis_title="Cases",
                template="plotly_white",
                hovermode="x unified",
                paper_bgcolor="#f4f7f9",
                plot_bgcolor="#ffffff",
                margin=dict(l=40, r=40, t=80, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0
                )
            )

            fig.update_traces(connectgaps=True)

            # ================= RENDER =================

            st.plotly_chart(fig, use_container_width=True)

            # ---------------- Forecast Table ----------------
            st.subheader(f"{HORIZON}-Month Forecast with Confidence Intervals & Risk")

            def highlight(val):
                if val == "High Risk":
                    return "background-color: #f8d7da"
                elif val == "Medium Risk":
                    return "background-color: #fff3cd"
                else:
                    return "background-color: #d4edda"

            st.dataframe(
                forecast_df.style.applymap(highlight, subset=["Risk Level"]),
                use_container_width=True
            )

            # ================= OVERALL RISK =================
            overall_risk = forecast_df["Risk Level"].value_counts().idxmax()

            st.markdown("---")
            st.subheader("Overall Risk Assessment")

            if overall_risk == "Low Risk":
                risk_color = "#2ECC71"
            elif overall_risk == "Medium Risk":
                risk_color = "#F39C12"
            else:
                risk_color = "#E74C3C"
            st.markdown(f"""
            <div style="
                background: {risk_color};
                padding: 18px;
                border-radius: 12px;
                text-align: center;
                font-size: 20px;
                font-weight: 600;
                color: white;
                box-shadow: 0 6px 18px rgba(0,0,0,0.15);
                margin-bottom: 25px;
            ">
            Forecasted Malaria Risk in {selected_state}: {overall_risk}
            </div>
            """, unsafe_allow_html=True)
            # ================= NORTH EAST SPATIAL MAP =================
            st.markdown("---")

            # ================= FORECAST VALUE =================
            cases_value = round(forecast_df["Forecasted Cases"].mean(), 0)

            # ================= RISK LEVEL =================
            low_thr = np.percentile(data["Cases"], 33)
            high_thr = np.percentile(data["Cases"], 66)

            if cases_value <= low_thr:
                risk_level = "Low"
                risk_color = "#2ECC71"
            elif cases_value <= high_thr:
                risk_level = "Moderate"
                risk_color = "#F39C12"
            else:
                risk_level = "High"
                risk_color = "#E74C3C"

            st.markdown("### GeoAI Spatial Risk Panel")

            # ================= LOAD GEOJSON =================
            with open("india_states.geojson", "r", encoding="utf-8") as f:
                india_geojson = json.load(f)

            # ================= DATA =================
            map_df = pd.DataFrame({
                "State": [selected_state],
                "Average Forecasted Cases": [cases_value]
            })

            # ================= CHOROPLETH MAP =================
            fig_map = px.choropleth_mapbox(
                map_df,
                geojson=india_geojson,
                locations="State",
                featureidkey="properties.NAME_1",
                color="Average Forecasted Cases",

                color_continuous_scale=[
                    [0, "#2ECC71"],     # Low Risk
                    [0.5, "#F39C12"],   # Medium Risk
                    [1, "#E74C3C"]      # High Risk
                ],

                mapbox_style="carto-positron",  # Clean light theme
                zoom=5,
                center={"lat": 22.5, "lon": 80},
                opacity=0.85
            )

            # ================= OPTIONAL HIGH RISK MARKER =================
            if risk_level == "High":
                fig_map.add_trace(
                    go.Scattermapbox(
                        lat=[26.5],
                        lon=[92.5],
                        model="markers",
                        marker=dict(
                            size=30,
                            color="rgba(231,76,60,0.35)"
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

            # ================= LAYOUT CLEANUP =================
            fig_map.update_layout(

                margin=dict(l=0, r=0, t=40, b=0),
                height=600,

                coloraxis_colorbar=dict(
                    title="Average Forecasted Cases",
                    thickness=18
                )
            )
            # ================= ADD INDIA + STATE BORDER OUTLINES =================
            fig_map.update_layout(
                mapbox={
                    "layers": [
                        {
                            "source": india_geojson,
                            "sourcetype": "geojson",
                            "type": "line",
                            "color": "black",
                            "line": {
                                "width": 1.2
                            }
                        }
                    ]
                }
            )
            # ================= DISPLAY MAP =================
            st.plotly_chart(fig_map, use_container_width=True)
# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<div class="footer">
<b>CSIR – Indian Institute of Chemical Technology (IICT)</b><br>
Academy of Scientific and Innovative Research (AcSIR)<br>
</div>
""", unsafe_allow_html=True) 
