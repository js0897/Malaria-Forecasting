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
import sqlite3
from io import BytesIO
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet


def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
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

DB_PATH = r"database/malaria.db"

# ======================================================
# MODEL PATHS
# ======================================================

MODEL_PATHS = {
    "3 Months": "models/nbeats_3month.ckpt",
    "6 Months": "models/nbeats_6month.ckpt"
}

# ======================================================
# LOAD DATA FROM DATABASE
# ======================================================

def load_state_data(state_name):

    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            month AS Month,
            cases AS Cases,
            log_cases AS LogCases
        FROM malaria_cases
        WHERE state = ?
        ORDER BY month
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(state_name,)
    )

    conn.close()
    # Convert Month to datetime
    df["Month"] = pd.to_datetime(df["Month"])
    return df
# ======================================================
# SESSION STATE
# ======================================================

if "page" not in st.session_state:
    st.session_state.page = "home"
# ======================================================
# ADMIN LOGIN SESSION
# ======================================================

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

# NEW
if "admin_page" not in st.session_state:
    st.session_state.admin_page = "dashboard"
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
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* ================= GLOBAL RESET ================= */

*{
    margin:0;
    padding:0;
    box-sizing:border-box;
}

html{
    scroll-behavior:smooth;
}

body{
    font-family:"Poppins","Segoe UI",sans-serif;
    background:#F8FAFC;
    color:#1E293B;
    overflow-x:hidden;
}

/* Hide Streamlit UI */
header{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

#MainMenu{
    visibility:hidden;
}

/* Main container */

.block-container{

    max-width:100% !important;
    padding-top:1rem !important;
    padding-left:2rem !important;
    padding-right:2rem !important;
    padding-bottom:2rem !important;

}

/* Better spacing */
div[data-testid="stVerticalBlock"]{
    gap:1rem;
}

div[data-testid="stHorizontalBlock"]{
    gap:1rem;

}
/*====================================================
            SELECTBOX & BUTTON STYLING
====================================================*/

/* Selectbox Container */
div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stSelectbox"]){

    background:linear-gradient(
        135deg,
        #1E3A8A,
        #2563EB
    );

    padding:22px 24px;
    border-radius:16px;
    margin-bottom:20px;
    box-shadow:
        0 10px 25px rgba(37,99,235,.18);
    transition:all .3s ease;

}

/* Hover Effect */
div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stSelectbox"]):hover{

    transform:translateY(-2px);
    box-shadow:
        0 15px 35px rgba(37,99,235,.28);

}
/* Label */
label[data-testid="stWidgetLabel"]{
    color:white !important;
    font-size:15px !important;
    font-weight:600 !important;
    letter-spacing:.3px;
    margin-bottom:8px !important;
}

/* Button */

div[data-testid="stButton"] > button{
    background:linear-gradient(
        135deg,
        #2563EB,
        #1D4ED8
    ) !important;
    color:white !important;
    font-size:16px !important;
    font-weight:600 !important;
    border:none !important;
    border-radius:14px !important;
    padding:12px 24px !important;
    transition:all .30s ease;
    box-shadow:
        0 8px 20px rgba(37,99,235,.22);
}

/* Button Hover */
div[data-testid="stButton"] > button:hover{
    transform:translateY(-3px);
    background:linear-gradient(
        135deg,
        #1D4ED8,
        #1E40AF
    ) !important;
    box-shadow:
        0 12px 30px rgba(37,99,235,.32);
}

/* Button Click */
div[data-testid="stButton"] > button:active{
    transform:scale(.98);
}

/*====================================================
                NAVBAR BASE LAYOUT
====================================================*/

/* Main horizontal layout used for navbar */

div[data-testid="stHorizontalBlock"]{
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    gap:0.75rem !important;
    flex-wrap:wrap !important;
    width:100%;
    margin:0 auto;
    transition:all .3s ease;

}

/* Navbar columns */

div[data-testid="stHorizontalBlock"] > div[data-testid="column"]{

    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    min-width:0;
    padding:0;
}

/* Remove Streamlit's default top spacing */

section.main > div:first-child{
    padding-top:0 !important;
    margin-top:0 !important;

}

/* Reduce unnecessary spacing between navbar and page */

div[data-testid="stVerticalBlock"] > div:first-child{
    margin-top:0 !important;
}

/*====================================================
                    PREMIUM HEADER
====================================================*/

.header{
    width:100%;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:14px 32px;
    margin-bottom:25px;
    background:rgba(255,255,255,0.92);
    backdrop-filter:blur(16px);
    -webkit-backdrop-filter:blur(16px);
    border:1px solid rgba(226,232,240,0.9);
    border-radius:22px;
    box-shadow:
        0 10px 35px rgba(15,23,42,0.08);
    transition:all .35s ease;
}

/* Hover Effect */
.header:hover{
    transform:translateY(-2px);
    box-shadow:
        0 18px 45px rgba(15,23,42,0.12);
}

/*====================================================
                        LOGO
====================================================*/
.header img{
    height:72px;
    width:auto;
    object-fit:contain;
    transition:.3s ease;
}
.header img:hover{
    transform:scale(1.05);
}

/*====================================================
                    HEADER TITLE
====================================================*/
.header-center{
    flex:1;
    text-align:center;
    padding:0 20px;
}

.header-center h1{
    margin:0;
    font-size:34px;
    font-weight:700;
    letter-spacing:.4px;
    color:#0F172A;
    line-height:1.2;
}

/*====================================================
            HEADER LEFT & RIGHT
====================================================*/
.header-left,
.header-right{
    display:flex;
    align-items:center;
    gap:15px;
}
.header-left img{
    height:80px;
}
.header-right img{
    height:72px;
    transition:.3s;
}
.header-right img:hover{
    transform:scale(1.08);
}
            
/*====================================================
                AI BADGE
====================================================*/
.badge{
    display:inline-block;
    background:linear-gradient(
        135deg,
        #2563EB,
        #38BDF8
    );
    color:white;
    padding:8px 18px;
    border-radius:50px;
    font-size:13px;
    font-weight:600;
    letter-spacing:1px;
    margin-bottom:18px;
    box-shadow:
        0 8px 20px rgba(37,99,235,.25);
}

/*====================================================
            HEADER DESCRIPTION
====================================================*/
.header-center p{
    margin-top:18px;
    color:#64748B;
    font-size:18px;
    line-height:1.7;
    max-width:750px;
    margin-left:auto;
    margin-right:auto;
} 
                       
/*====================================================
                DASHBOARD CARD
====================================================*/

.dashboard-card{
    background:#FFFFFF;
    padding:30px;
    margin-top:24px;
    border-radius:22px;
    border:1px solid #E2E8F0;
    box-shadow:
        0 10px 30px rgba(15,23,42,.08);
    transition:all .35s ease;
}

.dashboard-card:hover{
    transform:translateY(-4px);
    box-shadow:
        0 20px 45px rgba(15,23,42,.14);
}

/*====================================================
                    HEADINGS
====================================================*/

h2{
    color:#0F172A;
    font-size:32px;
    font-weight:700 !important;
    letter-spacing:.4px;
    margin-bottom:12px;
}
h3{
    color:#475569;
    font-size:24px;
    font-weight:600 !important;
    letter-spacing:.3px;
    margin-bottom:10px;
}
            
/*====================================================
            PREMIUM GLASS STICKY NAVBAR
====================================================*/
.nav-marker + div[data-testid="stHorizontalBlock"]{
    position:sticky;
    top:18px;
    z-index:9999;
    width:92%;
    margin:0 auto 25px auto;
    padding:14px 18px;
    display:flex !important;
    justify-content:center;
    align-items:center;
    flex-wrap:nowrap;
    gap:12px;
    background:rgba(15,23,42,0.78);
    backdrop-filter:blur(18px);
    -webkit-backdrop-filter:blur(18px);
    border:1px solid rgba(255,255,255,.08);
    border-radius:22px;
    box-shadow:
        0 12px 35px rgba(15,23,42,.28);
    transition:all .35s ease;
}

/* Slight floating effect */
.nav-marker + div[data-testid="stHorizontalBlock"]:hover{
    box-shadow:
        0 18px 45px rgba(15,23,42,.35);
}

/*====================================================
                NAVBAR COLUMNS
====================================================*/

.nav-marker + div[data-testid="stHorizontalBlock"]
div[data-testid="column"]{
    display:flex !important;
    justify-content:center;
    align-items:center;
    padding:0;
    min-width:0;
}

/*====================================================
                NAVIGATION BUTTONS
====================================================*/
.nav-marker + div[data-testid="stHorizontalBlock"] button{
    background:transparent !important;
    border:none !important;
    color:#E2E8F0 !important;
    font-size:15px !important;
    font-weight:600 !important;
    letter-spacing:.3px;
    padding:10px 20px !important;
    border-radius:14px;
    transition:all .30s ease;
    white-space:nowrap;
}

/*====================================================
                    HOVER
====================================================*/

.nav-marker + div[data-testid="stHorizontalBlock"] button:hover{
    background:rgba(255,255,255,.08) !important;
    color:#FFFFFF !important;
    transform:translateY(-2px);
}

/*====================================================
                ACTIVE PAGE
====================================================*/

.nav-marker + div[data-testid="stHorizontalBlock"] button[kind="primary"]{

    background:linear-gradient(
        135deg,
        #2563EB,
        #1D4ED8
    ) !important;
    color:#FFFFFF !important;
    font-weight:700 !important;
    border:none !important;
    box-shadow:
        0 8px 18px rgba(37,99,235,.35);
}

/*====================================================
            ACTIVE HOVER
====================================================*/

.nav-marker + div[data-testid="stHorizontalBlock"] button[kind="primary"]:hover{

    transform:translateY(-2px);
    box-shadow:
        0 12px 28px rgba(37,99,235,.45);
}
            
/*====================================================
            PREMIUM IMAGE CAROUSEL SECTION
====================================================*/
/* Main Gallery */
.image-section{
    position:relative;
    width:100%;
    margin:35px auto;
}

/* Horizontal Scroll Container */
.image-carousel{
    display:flex;
    gap:25px;
    overflow-x:auto;
    scroll-behavior:smooth;
    scrollbar-width:none;
    padding:15px 8px;
}
.image-carousel::-webkit-scrollbar{
    display:none;
}

/* Individual Image Card */
.image-card{
    flex:0 0 32%;
    min-width:320px;
    background:rgba(255,255,255,.75);
    backdrop-filter:blur(14px);
    -webkit-backdrop-filter:blur(14px);
    border:1px solid rgba(255,255,255,.18);
    border-radius:24px;
    overflow:hidden;
    box-shadow:
        0 12px 35px rgba(15,23,42,.10);
    transition:all .35s ease;
}
.image-card:hover{
    transform:translateY(-6px);
    box-shadow:
        0 20px 45px rgba(15,23,42,.18);
}
/* Streamlit Image */
div[data-testid="stImage"]{
    padding:0 !important;
    background:transparent !important;
    box-shadow:none !important;
}
/* Image */
.image-card img{
    width:100%;
    height:280px;
    object-fit:cover;
    transition:transform .5s ease;
}
.image-card:hover img{
    transform:scale(1.05);
}
/* Caption */
.image-caption{
    padding:20px;
    text-align:center;
}
.image-caption h3{
    color:#0F172A;
    margin-bottom:8px;
    font-size:22px;
    font-weight:700;
}
.image-caption p{
    color:#64748B;
    font-size:15px;
}
/* Navigation Arrows */
.carousel-arrow{
    position:absolute;
    top:45%;
    transform:translateY(-50%);
    width:52px;
    height:52px;
    border-radius:50%;
    border:none;
    background:rgba(15,23,42,.80);
    color:white;
    font-size:22px;
    cursor:pointer;
    display:flex;
    align-items:center;
    justify-content:center;
    backdrop-filter:blur(12px);
    transition:.3s;
    z-index:100;
}
.carousel-arrow:hover{
    background:#2563EB;
    transform:translateY(-50%) scale(1.08);
}
/* Left Arrow */
.carousel-left{
    left:-18px;
}
/* Right Arrow */
.carousel-right{
    right:-18px;
}
/* Mobile */
@media(max-width:991px){
.image-card{
    flex:0 0 85%;
}
.carousel-arrow{
    width:44px;
    height:44px;
}
}

/*====================================================
            GLOBAL BACKGROUND & RESPONSIVE
====================================================*/

/* Premium AI Dashboard Background */
body{
    background:
        linear-gradient(
            180deg,
            #F8FAFC 0%,
            #F1F5F9 40%,
            #EEF4FF 100%
        );
    background-attachment:fixed;
}

/*====================================================
                LARGE TABLETS
====================================================*/

@media screen and (max-width:1200px){
    .main .block-container{
       padding-left:2rem !important;
       padding-right:2rem !important;
    }
}
            
/*====================================================
                    TABLETS
====================================================*/
@media screen and (max-width:992px){
    /* Floating Navbar */
    .nav-marker + div[data-testid="stHorizontalBlock"]{
        width:96%;
        padding:12px 18px;
        border-radius:18px;
    }
    .nav-marker + div[data-testid="stHorizontalBlock"] button{
        font-size:14px !important;
        padding:8px 14px !important;
    }
    /* Header */
    .header{
        padding:16px 20px;
        border-radius:18px;
    }
    .header img{
        height:58px;
    }
    .header-center h1{
        font-size:28px;
    }
}

/*====================================================
                    MOBILE
====================================================*/
@media screen and (max-width:768px){
    .main .block-container{
        padding-left:1rem !important;
        padding-right:1rem !important;
    }
    .nav-marker + div[data-testid="stHorizontalBlock"]{
        width:98%;
        padding:10px;
        border-radius:16px;
        flex-wrap:wrap !important;
        gap:6px;
    }
    .nav-marker + div[data-testid="stHorizontalBlock"] button{
        width:100%;
        font-size:14px !important;
        padding:10px 12px !important;
    }
    .header{
        flex-direction:column;
        text-align:center;
        gap:15px;
    }
    .header img{
        height:50px;
    }
    .header-center{
        padding:0;
    }
    .header-center h1{
        font-size:22px;
        line-height:1.3;
    }
}

/*====================================================
                SMALL MOBILE
====================================================*/

@media screen and (max-width:480px){
    .header-center h1{
        font-size:18px;
    }
    .nav-marker + div[data-testid="stHorizontalBlock"] button{
        font-size:13px !important;
    }
}

/*====================================================
                    PREMIUM FOOTER
====================================================*/
.footer{
    width:100%;
    margin-top:50px;
    padding:24px 30px;
    text-align:center;
    background:rgba(255,255,255,0.80);
    backdrop-filter:blur(14px);
    -webkit-backdrop-filter:blur(14px);
    border:1px solid rgba(226,232,240,0.8);
    border-radius:20px;
    box-shadow:
        0 8px 30px rgba(15,23,42,0.08);
    transition:all .35s ease;
}

/* Hover Effect */
.footer:hover{
    transform:translateY(-2px);
    box-shadow:
        0 15px 40px rgba(15,23,42,0.12);
}

/* Footer Heading */
.footer h4{
    color:#0F172A;
    font-size:18px;
    font-weight:700;
    margin-bottom:10px;
}

/* Footer Text */
.footer p{
    margin:6px 0;
    color:#64748B;
    font-size:14px;
    line-height:1.8;
}

/* Footer Links */
.footer a{
    color:#2563EB;
    text-decoration:none;
    font-weight:600;
    transition:.3s;
}
.footer a:hover{
    color:#1D4ED8;
    text-decoration:underline;
}

/* Divider */
.footer hr{
    border:none;
    height:1px;
    margin:18px 0;
    background:linear-gradient(
        to right,
        transparent,
        #CBD5E1,
        transparent
    );
}

/* Copyright */
.footer .copyright{
    font-size:13px;
    color:#94A3B8;
    margin-top:12px;
}
/*====================================================
        PUSH CONTENT BELOW FIXED NAVBAR
====================================================*/
.main .block-container{
    padding-top:120px !important;
}
.page-header{
    margin:25px 0 35px 0;
    padding:35px;
    background:rgba(255,255,255,.82);
    backdrop-filter:blur(15px);
    -webkit-backdrop-filter:blur(15px);
    border-radius:22px;
    border:1px solid rgba(226,232,240,.80);
    box-shadow:
        0 12px 35px rgba(15,23,42,.08);

}
.page-header h1{
    margin:0;
    color:#0F172A;
    font-size:42px;
    font-weight:700;
}
.page-header p{
    margin-top:12px;
    color:#64748B;
    font-size:18px;
    line-height:1.8;

}
/*====================================================
                SECTION TITLE
====================================================*/

.section-title{
    font-size:28px;
    font-weight:700;
    color:#0F172A;
    margin-top:45px;
    margin-bottom:18px;
    padding-bottom:10px;
    border-bottom:2px solid rgba(37,99,235,.15);
}
            
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD HEADER LOGOS
# ======================================================

def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    

if st.session_state.page == "home":

    logo_csir = img_to_base64(r"assets/CSIR-Logo (1).jpg")
    logo_iict = img_to_base64(r"assets/iict logo.jpg")
    logo_eiacp = img_to_base64(r"assets/eiacp logo.png")
    logo_moefcc = img_to_base64(r"assets/moefcc-logo.png")

    st.markdown(
    f"""
    <div class="header">

    <div class="header-left">
    <img src="data:image/jpeg;base64,{logo_csir}">
    <img src="data:image/jpeg;base64,{logo_iict}">
    </div>

    <div class="header-center">

    <div class="badge">
    AI • EARLY WARNING SYSTEM
    </div>

    <h1>
    Malaria Risk Forecasting and Early Warning System
    for North-East States of India
    </h1>

    </div>

    <div class="header-right">
    <img src="data:image/png;base64,{logo_eiacp}">
    <img src="data:image/png;base64,{logo_moefcc}">
    </div>

    </div>
    """,
    unsafe_allow_html=True
    )
# ======================================================
# NAV BAR (ORANGE)
# ======================================================
st.markdown("""
<style>

/*====================================================
            PREMIUM GLASS AI NAVBAR
====================================================*/

.nav-marker + div[data-testid="stHorizontalBlock"]{
    position:fixed;
    top:15px;
    left:50%;
    transform:translateX(-50%);
    z-index:9999;
    width:94%;
    margin:auto;
    padding:14px 18px;
    display:flex;
    justify-content:center;
    align-items:center;
    background:rgba(15,23,42,.82);
    backdrop-filter:blur(18px);
    -webkit-backdrop-filter:blur(18px);
    border:1px solid rgba(255,255,255,.08);
    border-radius:20px;
    box-shadow:
        0 12px 35px rgba(15,23,42,.20);
}

/* Navbar Buttons */
.nav-marker + div[data-testid="stHorizontalBlock"] button{
    background:transparent !important;
    border:none !important;
    color:#E2E8F0 !important;
    font-size:15px !important;
    font-weight:600 !important;
    letter-spacing:.3px;
    padding:10px 18px !important;
    border-radius:12px;
    transition:all .30s ease;
}

/* Hover */
.nav-marker + div[data-testid="stHorizontalBlock"] button:hover{
    background:rgba(255,255,255,.08) !important;
    color:white !important;
    transform:translateY(-2px);
}

/* Active Page */
.nav-marker + div[data-testid="stHorizontalBlock"] button[kind="primary"]{
    background:linear-gradient(
        135deg,
        #2563EB,
        #1D4ED8
    ) !important;
    color:white !important;
    font-weight:700 !important;
    box-shadow:
        0 8px 20px rgba(37,99,235,.35);
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# PREMIUM NAVIGATION BAR
# ======================================================

nav_container = st.container()

with nav_container:

    st.markdown('<div class="nav-marker">', unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6, col7 = st.columns(
        [1.0, 1.2, 1.9, 1.5, 1.0, 1.2, 1.0]
    )

    # ================= HOME ================= #

    with col1:

        if st.button(
            "Home",
            type="primary" if st.session_state.page == "home" else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = "home"
            st.rerun()

    # ================= ABOUT ================= #

    with col2:

        if st.button(
            " About Us",
            type="primary" if st.session_state.page == "about" else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = "about"
            st.rerun()

    # ================= MALARIA ================= #

    with col3:

        if st.button(
            " Malaria in North-East India",
            type="primary" if st.session_state.page == "data" else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = "data"
            st.rerun()

    # ================= FORECASTING ================= #

    with col4:

        forecast = st.button(
            "Forecasting ▼",
            type="primary"
            if st.session_state.page in ["forecasting", "methods", "forecast_analysis"]
            else "secondary",
            use_container_width=True,
        )

        if forecast:
            st.session_state.show_forecast_menu = not st.session_state.get(
                "show_forecast_menu", False
            )
            # Switch to Forecasting page
            st.session_state.page = "forecasting"
            st.rerun()

    # ================= TEAM ================= #

    with col5:

        if st.button(
            " Team",
            type="primary" if st.session_state.page == "team" else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = "team"
            st.rerun()

    # ================= CONTACT ================= #

    with col6:

        if st.button(
            " Contact Us",
            type="primary" if st.session_state.page == "contact" else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = "contact"
            st.rerun()

    # ================== ADMIN ================== #

    with col7:

        if st.button(
            " Admin",
            type="primary" if st.session_state.page == "admin" else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = "admin"
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# FORECASTING SUB MENU
# ======================================================

    if st.session_state.get("show_forecast_menu", False):

        st.markdown(
            """
            <div style="
                background:rgba(255,255,255,.75);
                backdrop-filter:blur(12px);
                border-radius:16px;
                padding:12px;
                margin-top:10px;
                margin-bottom:20px;
                box-shadow:0 10px 30px rgba(0,0,0,.08);
            ">
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                " Methods",
                key="methods_menu",
                use_container_width=True,
            ):
                st.session_state.page = "methods"
                st.session_state.show_forecast_menu = False
                st.rerun()

        with col2:

            if st.button(
                " Forecast Analysis",
                key="analysis_menu",
                use_container_width=True,
            ):
                st.session_state.page = "forecast"
                st.session_state.show_forecast_menu = False
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
# ======================================================
# MALARIA HERO IMAGE (REQUESTED)
# ======================================================

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ======================================================
# PREMIUM HERO CAROUSEL
# ======================================================

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

if st.session_state.page == "home":

    hero1 = img_to_base64(r"images/image 4.png")
    hero2 = img_to_base64(r"images/image 22.jpg")
    hero3 = img_to_base64(r"images/image 25.jpg")

    st.markdown(f"""
<style>

/*====================================================
                HERO CAROUSEL
====================================================*/

.hero-wrapper{{
    width:100%;
    margin:25px auto 40px auto;
}}

.hero-slider{{
    position:relative;
    width:100%;
    overflow:hidden;
    border-radius:28px;
    box-shadow:
        0 18px 45px rgba(15,23,42,.12);
}}

.hero-slides{{
    display:flex;
    width:300%;
    animation:slide 18s infinite;
}}

.hero-slide{{
    width:100%;
    flex-shrink:0;
    position:relative;
}}

.hero-slide img{{
    width:100%;
    height:520px;
    object-fit:cover;
    display:block;
}}

.hero-overlay{{
    position:absolute;
    inset:0;
    background:
        linear-gradient(
            rgba(15,23,42,.20),
            rgba(15,23,42,.55)
        );
}}

.hero-text{{
    position:absolute;
    left:60px;
    bottom:60px;
    color:white;
    z-index:5;
    max-width:650px;
}}

.hero-text h2{{
    font-size:46px;
    margin:0;
    font-weight:700;
}}

.hero-text p{{
    margin-top:15px;
    font-size:18px;
    line-height:1.7;
}}

.arrow{{
    position:absolute;
    top:50%;
    transform:translateY(-50%);
    width:52px;
    height:52px;
    border-radius:50%;
    background:rgba(255,255,255,.18);
    backdrop-filter:blur(12px);
    display:flex;
    justify-content:center;
    align-items:center;
    font-size:30px;
    color:white;
    cursor:pointer;
    user-select:none;
}}

.arrow-left{{left:22px;}}
.arrow-right{{right:22px;}}

.hero-dots{{
    position:absolute;
    bottom:22px;
    width:100%;
    display:flex;
    justify-content:center;
    gap:10px;
}}

.hero-dots span{{
    width:12px;
    height:12px;
    border-radius:50%;
    background:rgba(255,255,255,.45);
}}

.hero-dots span:first-child{{
    background:white;
}}

@keyframes slide{{

0%{{transform:translateX(0);}}
30%{{transform:translateX(0);}}
33%{{transform:translateX(-100%);}}
63%{{transform:translateX(-100%);}}
66%{{transform:translateX(-200%);}}
96%{{transform:translateX(-200%);}}
100%{{transform:translateX(0);}}

}}
</style>
<div class="hero-wrapper">
<div class="hero-slider">
<div class="hero-slides">
<div class="hero-slide">
<img src="data:image/png;base64,{hero1}">
<div class="hero-overlay"></div>
<div class="hero-text">
<h2>AI-Powered Malaria Forecasting</h2>
<p>
Predict outbreaks across North-East India using
advanced deep learning and spatio-temporal analytics.
</p>
</div>
</div>
<div class="hero-slide">
<img src="data:image/png;base64,{hero2}">
<div class="hero-overlay"></div>
<div class="hero-text">
<h2>Interactive GeoAI Dashboard</h2>
<p>
Visualize malaria risk across regions with
dynamic spatial intelligence.
</p>
</div>
</div>
<div class="hero-slide">
<img src="data:image/png;base64,{hero3}">
<div class="hero-overlay"></div>
<div class="hero-text">
<h2>Early Warning System</h2>
<p>
Support proactive public health decision-making
using AI-powered forecasting.
</p>
</div>
</div>
</div>
<div class="arrow arrow-left">&#10094;</div>
<div class="arrow arrow-right">&#10095;</div>
<div class="hero-dots">
<span></span>
<span></span>
<span></span>
</div>
</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
# ======================================================
# PAGE CONTENT
# ======================================================
def page_title(title, subtitle=""):

    st.markdown(
        f"""
        <div class="page-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def team_card(
    image_path,
    name,
    designation,
    education,
    email,
    linkedin
):
    col1, col2 = st.columns([1, 4], vertical_alignment="center")
    with col1:
        st.image(image_path, width=180)
    with col2:
        st.markdown(f"## {name}")
        st.markdown(
            f"<p style='margin:0;font-size:22px;font-weight:600;'>{designation}</p>",
            unsafe_allow_html=True,)
        st.markdown(
            f"<p style='margin:4px 0 12px 0;font-size:18px;color:#555;'>{education}</p>",
            unsafe_allow_html=True,)
        st.markdown(f"✉ **Email:** {email}")
        st.markdown(f"👤 **Personal Profile Link:** [{linkedin}]({linkedin})")
    st.markdown("---")

if st.session_state.page == "home":
    page_title("Overview")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    The primary objective of this platform is to develop an AI-driven early warning system for forecasting malaria outbreaks and detecting potential high-risk situations in advance. By combining advanced deep learning forecasting models with intelligent risk assessment techniques, the system enables accurate prediction of future malaria trends and supports proactive public health decision-making. The platform is designed to assist healthcare authorities and researchers in improving outbreak preparedness, optimizing resource allocation, and implementing timely preventive measures through data-driven insights.
    """)

elif st.session_state.page == "about":
    page_title("About Us")
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(
        "Indian Institute of Chemical Technology (IICT), Hyderabad, established in 1944, is a constituent laboratory of the Council of Scientific and Industrial Research (CSIR), New Delhi. With its expertise in chemistry and chemical technology, it provides solutions to challenges faced by Industry, Government Departments and Entrepreneurs through basic and applied research, and process development. The institute is internationally recognized for its contributions to chemistry research and is an ideal place for taking ideas to commercialization through state-of-the-art research and development. CSIR-IICT during its seventy-year journey has made its mark as a dynamic, innovative and result-oriented R&D organization. The clientele spans all comers of the globe. In India, it is CSIR-Indian Institute of Chemical Technology (CSIR-IICT) is one of the oldest National Laboratories the reliable destination of chemical and biotech industries. The reputation that CSIR-IICT could establish amongst the industrial clients as a reliable R&D partner, can be largely attributed to its rich pool of scientists with expertise in broad-ranging research areas and simple and effective business development strategies."
    )
    # ---------------- Vission ---------------- #
    st.markdown("""
    <div class="section-title">
    Vision
    </div>
    """, unsafe_allow_html=True)
    st.write("""
    To Serve society by creating an outstanding knowledge base in chemical and chemical technology
    """)
    # ---------------- Mission ---------------- #
    st.markdown("""
    <div class="section-title">
    Mission
    </div>
    """, unsafe_allow_html=True)
    st.write("""
    CSIR-IICT will Strive towards knowledge intensive translational research in chemistry to meet the country's expectations with novel technologies.
    """)

elif st.session_state.page == "data":
    page_title("About the Data")
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(
        "Malaria remains a major public health challenge in the North-Eastern states of India, which contribute a disproportionately high share of the country’s Plasmodium falciparum burden and represent some of the most persistent transmission zones nationally. The region accounts for about 15% of India’s malaria cases and roughly 12% of the national P. falciparum cases, the most severe form of malaria infection.Transmission in this region is sustained by efficient vector species such as Anopheles minimus and Anopheles baimaii, along with favorable ecological conditions including forested terrain, high humidity, and perennial hill streams that support continuous mosquito breeding. In addition, the North-East serves as a critical epidemiological corridor linking India with Southeast Asia, facilitating the historical introduction of antimalarial drug-resistant parasite strains into the country. Many malaria-affected areas in the region are located in tribal and difficult-to-access locations, which further complicates surveillance and control efforts.In this context, early warning systems based on deep learning–driven forecasting can play an important role by identifying temporal patterns in malaria transmission and supporting evidence-based decision-making for targeted intervention planning. Such predictive approaches align with India’s national malaria elimination strategy (2016–2030) by strengthening preparedness and improving resource prioritization in high-risk transmission settings."
     
    )
    

elif st.session_state.page == "forecasting":
    page_title(
        "Malaria Forecasting"
    )

elif st.session_state.page == "methods":
    page_title("Malaria Outbreak Forecasting & Risk Assessment System")
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("""
    Malaria outbreaks remain a significant public health challenge, particularly in regions with recurring seasonal transmission patterns. Early identification of potential outbreak conditions is essential for timely intervention, resource planning, and effective disease control.

    This system is designed to forecast future malaria incidence and evaluate outbreak risk using deep learning–based time series forecasting and statistical risk analysis. By leveraging historical epidemiological data, the system generates multi-horizon forecasts and identifies emerging high-risk conditions through adaptive risk classification techniques.

    The forecasting framework utilizes N-BEATS models to predict malaria trends over short-term and long-term horizons, enabling proactive public health monitoring and data-driven decision-making.
    """)

# ---------------- Risk Assessment ----------------
    st.markdown("""
    <div class="section-title">
    Risk Assessment
    </div>
    """, unsafe_allow_html=True)
    st.write("""
    In addition to forecasting future malaria cases, the system includes a dynamic risk assessment mechanism to evaluate the potential severity of predicted outbreak conditions. The risk prediction framework analyzes forecasted malaria incidence and categorizes regions into different risk levels based on historical disease patterns.

    Risk classification is performed using a percentile-based statistical approach derived from historical malaria case distributions. This adaptive methodology enables the system to account for regional variations in transmission patterns and disease prevalence, ensuring more reliable and context-aware risk evaluation.
    """)

# ---------------- Risk Levels ----------------
    st.markdown("""
    <div class="section-title">
    Risk Classification Levels
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    - 🟢 **Low Risk**
    - 🟠 **Moderate Risk**
    - 🔴 **High Risk**
    """)
# ---------------- Thresholds ----------------
    st.markdown("""
    <div class="section-title">
    Classification Thresholds
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    - Predicted case values below the **33rd percentile** are classified as **Low Risk**
    - Predicted case values between the **33rd and 66th percentiles** are classified as **Moderate Risk**
    - Predicted case values above the **66th percentile** are classified as **High Risk**
    """)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("""
    By utilizing adaptive statistical thresholds instead of fixed predefined limits, the framework provides a more flexible and region-specific approach to malaria outbreak assessment, improving the effectiveness of early warning and public health monitoring.
    """)

elif st.session_state.page == "team":

    page_title("Our Team")

    team_card(
        image_path="images/image 27.PNG",
        name="Dr.M.Srinivasa Rao",
        designation="Scientist - F ",
        education= " PhD ,Zoology",
        email="msrinivas.iict@csir.res.in",
        linkedin="https://scholar.google.com/citations?hl=en&user=bQg7W7YAAAAJ&view_op=list_works&sortby=pubdate"
    )

    team_card(
        image_path="images/image 28.jpg",
        name="Jangili Shraddha",
        designation="PhD Scholar",
        education= " Mtech CS",
        email="jshraddha888@gmail.com",
        linkedin="linkedin.com/in/jangili-shraddha-545423201"
    )

    team_card(
        image_path="images/image 28.jpg",
        name="hello",
        designation="Mtech",
        education= " PhD ",
        email="rah@iict.res.in",
        linkedin="linkedin.com/in/jangili-shraddha-545423201"
    )

    team_card(
        image_path="images/image 28.jpg",
        name="hello",
        designation="Mtech",
        education= " PhD ",
        email="rah@iict.res.in",
        linkedin="linkedin.com/in/jangili-shraddha-545423201"
    )
    
elif st.session_state.page == "contact":
    page_title("Contact Us")
    st.markdown("""
    ### CSIR – Indian Institute of Chemical Technology (CSIR-IICT)
                
    **Academy of Scientific and Innovative Research (AcSIR)**
                
    ---
                
    **👤 Contact Name:** Dr. D. Srinivasa Reddy
                
    **📍 Contact Address:**  
                
    CSIR – Indian Institute of Chemical Technology (CSIR-IICT)  
    Tarnaka, Hyderabad – 500007  
    Telangana, India
                
    **☎ Contact Phone:** +91-40-27191234
                
    **✉ Email:** director@iict.res.in
    """)

elif st.session_state.page == "admin":
    page_title(
        "Administrator Login",
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if not st.session_state.admin_logged_in:
        with st.container(border=True):
            st.markdown("###  Login")
            username = st.text_input(
                "Username",
                placeholder="Enter username"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password"
            )
            login = st.button(
                "Login",
                use_container_width=True
            )
            if login:
                if username == "admin" and password == "malaria123":
                    st.session_state.admin_logged_in = True
                    st.success("Login Successful")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")
    else:
        st.success(" Welcome Administrator")
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        # ================= DATABASE ================= #
        with col1:

            if st.button(
                " Database",
                use_container_width=True,
                key="database_btn"
            ):
                st.session_state.admin_page = "database"

        # ================= EDIT RECORDS ================= #

        with col2:

            if st.button(
                " Edit Records",
                use_container_width=True,
                key="edit_btn"
            ):
                st.session_state.admin_page = "edit"
  
        # ================= EXPORT ================= #

        with col3:
            if st.button(
                " Export Database",
                use_container_width=True,
                key="export_btn"
            ):
                st.session_state.admin_page = "export"
        st.markdown("<br>", unsafe_allow_html=True)
        # ================= LOGOUT ================= #

        if st.button(
            " Logout",
            use_container_width=True,
            key="logout_btn"
        ):
            st.session_state.admin_logged_in = False
            st.session_state.page = "home"
            st.rerun()
  
    # =====================================================
    # ADMIN DATABASE PAGE
    # =====================================================

    if st.session_state.get("admin_page") == "database":

        st.markdown("---")

        st.subheader("Malaria Database")

        # Connect to SQLite
        conn = sqlite3.connect(
            r"database/malaria.db"
        )

        # Get available states
        states = pd.read_sql_query(
            """
            SELECT DISTINCT state
            FROM malaria_cases
            ORDER BY state
            """,
            conn
        )

        selected_state = st.selectbox(
            "Select State",
            states["state"],
            key="admin_database_state"
        )

        # Load selected state
        df = pd.read_sql_query(
            """
            SELECT
                month AS Month,
                cases AS Cases,
                log_cases AS LogCases
            FROM malaria_cases
            WHERE state=?
            ORDER BY month
            """,
            conn,
            params=(selected_state,)
        )

        conn.close()

        df["Month"] = pd.to_datetime(df["Month"])

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
    # =====================================================
    # EDIT RECORDS PAGE
    # =====================================================

    elif st.session_state.get("admin_page") == "edit":

        page_title(
            "Edit Malaria Records",
            "Modify malaria records for the selected state."
        )

        # --------------------------------------------
        # Connect to Database
        # --------------------------------------------

        conn = sqlite3.connect(DB_PATH)

        # --------------------------------------------
        # Load States
        # --------------------------------------------

        states = pd.read_sql_query(
            """
            SELECT DISTINCT state
            FROM malaria_cases
            ORDER BY state
            """,
            conn
        )
        selected_state = st.selectbox(
            "Select State",
            states["state"],
            key="edit_state"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # --------------------------------------------
        # Load Selected State Data
        # --------------------------------------------

        df = pd.read_sql_query(
            """
            SELECT
                month AS Month,
                cases AS Cases,
                log_cases AS LogCases
            FROM malaria_cases
            WHERE state = ?
            ORDER BY month
            """,
            conn,
            params=(selected_state,)
        )
        df["Month"] = pd.to_datetime(df["Month"])

        # --------------------------------------------
        # Editable Table
        # --------------------------------------------

        edited_df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=["Month", "LogCases"]
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ================= Add Record =============#

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Add New Record")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### DATE")
            new_month = st.date_input(
                label="",
                key="new_month",
                label_visibility="collapsed"
            )
        with col2:
            st.markdown("#####  Cases")
            new_cases = st.number_input(
                label="",
                min_value=0,
                step=1,
                key="new_cases",
                label_visibility="collapsed"
            )

        if st.button(
            "Add Record",
            use_container_width=True
        ):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            month = new_month.strftime("%Y-%m-%d")
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM malaria_cases
                WHERE state = ?
                AND month = ?
                """,
                (selected_state, month)
            )
            exists = cursor.fetchone()[0]
            if exists > 0:
             st.warning("⚠ A record already exists for this month.")
            else:
                cursor.execute(
                    """
                    INSERT INTO malaria_cases
                    (state, month, cases, log_cases)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        selected_state,
                        month,
                        int(new_cases),
                        float(np.log(new_cases))
                    )
                )
                conn.commit()
                st.success("✅ Record added successfully.")
                conn.close()
                st.rerun()

    #=================== Delete Record =================#

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Delete Record")
        delete_month = st.selectbox(
            "Select Month",
            df["Month"],
            key="delete_month"

        )
        confirm_delete = st.checkbox(
            "conform "
        )
        if confirm_delete:
            if st.button(
                " Delete Record",
                use_container_width=True
            ):
                # Open database
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM malaria_cases
                    WHERE state = ?
                    AND month = ?
                    """,
                    (
                        selected_state,
                        delete_month.strftime("%Y-%m-%d")
                    )
                )
                conn.commit()
                conn.close()
                st.success("Record deleted successfully.")
                st.rerun()
    # =====================================================
    # EXPORT DATABASE
    # =====================================================
                 
    elif st.session_state.admin_page == "export":
            
        st.write("Current admin page:", st.session_state.admin_page)
        page_title(
            "Export Database",
            "Download malaria surveillance records from the database."
        )

        conn = sqlite3.connect(DB_PATH)
        export_scope = st.radio(
            "Export Scope",
            [
                "Selected State",
                "Entire Database"
            ]
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if export_scope == "Selected State":
            states = pd.read_sql_query(
                """
                SELECT DISTINCT state
                FROM malaria_cases
                ORDER BY state
                """,
                conn
            )

            selected_state = st.selectbox(
                "Select State",
                states["state"]
            )

        st.markdown("<br>", unsafe_allow_html=True)
        export_format = st.radio(
            "Export Format",
            [
                "CSV",
                "Excel (.xlsx)"
            ]
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if export_scope == "Selected State":
            df = pd.read_sql_query(
                """
                SELECT
                    state AS State,
                    month AS Month,
                    cases AS Cases,
                    log_cases AS LogCases
                FROM malaria_cases
                WHERE state=?
                ORDER BY month
                """,
                conn,
                params=(selected_state,)
            )

        else:
            df = pd.read_sql_query(
                """
                SELECT
                    state AS State,
                    month AS Month,
                    cases AS Cases,
                    log_cases AS LogCases
                FROM malaria_cases
                ORDER BY state, month
                """,
                conn
            )
        conn.close()
        today = datetime.now().strftime("%Y-%m-%d")
        if export_scope == "Selected State":
            filename = (
                selected_state.lower().replace(" ", "_")
                + "_malaria_data_"
                + today
            )
        else:
            filename = "malaria_database_" + today
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=filename + ".csv",
                mime="text/csv",
                use_container_width=True
            )

        else:
            output = BytesIO()
            with pd.ExcelWriter(
                output,
                engine="openpyxl"
            ) as writer:
                df.to_excel(
                    writer,
                    index=False
                )
            st.download_button(
                label="Download Excel",
                data=output.getvalue(),
                file_name=filename + ".xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )           

    # ========== Styling =================== #
            
        st.markdown("""
        <div style="
            height:1px;
            margin:35px 0;
            background:linear-gradient(
                to right,
                transparent,
                rgba(148,163,184,.7),
                transparent
            );
        "></div>
        """, unsafe_allow_html=True)

# ======================================================
# FORECASTING PAGE
# ======================================================

elif st.session_state.page == "forecast":

    page_title(
        "Malaria Forecasting Dashboard"
    )

    # ======================================================
    # STEP 1 — STATE SELECTION
    # ======================================================

    selected_state = st.selectbox(
        "Select North-East State",
        [
            "Assam",
            "Tripura",
            "Meghalaya",
            "Arunachal Pradesh",
        ],
        key="state_select",
    )
    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

    # ======================================================
    # STEP 2 — LOAD DATA FROM DATABASE
    # ======================================================

    try:
        data = load_state_data(selected_state)
    except Exception as e:
        st.error(f"Unable to load data for {selected_state}")
        st.exception(e)
        st.stop()

    # ======================================================
    # STEP 3 — DATA PREPARATION
    # ======================================================

    data = data.dropna(subset=["Month", "LogCases"])
    data = data.sort_values("Month").reset_index(drop=True)
    data["time_idx"] = np.arange(len(data))
    data["series"] = 0

    # ======================================================
    # STEP 4 — DATA PREVIEW
    # ======================================================

    st.markdown(
        f"""
        <div class="section-title">
            Data Preview — {selected_state}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)

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
    else:
        HORIZON = 6
    MODEL_PATH = MODEL_PATHS[horizon_choice]
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
                    showlegend=True,
                    line=dict(
                        color="#004680",
                        width=3
                    ),
                    marker=dict(
                        size=6,
                        color="#004680"
                    ),
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
                    line=dict(
                        color="#ff6435",
                        width=3,
                        dash="dash",
                        shape="spline",
                        smoothing=1.1
                    ),
                    marker=dict(
                        size=5,
                        color="#ff6b35"
                    ),
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
                                y=forecast_y[:i],
                                mode="lines+markers",
                                name="Forecast",
                                line=dict(
                                    color="#ff6b35",
                                    width=3,
                                    dash="dash",
                                    shape="spline",
                                    smoothing=1.1
                                ),
                                marker=dict(
                                    size=5,
                                    color="#ff6b35"
                                ),
                                showlegend=True
                            )
                        ],
                        traces=[1] 
                    )
                )
            fig.frames = frames

            # ================= ANIMATION BUTTON WITH EASING =================

            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",                                                           
                        showactive=False,
                        bgcolor="#004680",          
                        bordercolor="#004680",
                        borderwidth=2,
                        font=dict(color="white", family="Arial",size=13),

                        buttons=[
                            dict(
                                label="▶ Show Forecast",
                                method="animate",
                                args=[
                                    None,
                                    dict(
                                        frame=dict(duration=650, redraw=True),
                                        transition=dict(
                                            duration=450,
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
                        pad=dict(t=8, r=8)
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
                forecast_df.style.map(highlight, subset=["Risk Level"]),
                use_container_width=True
            )

            # ======================================================
            # AI RISK ASSESSMENT
            # ======================================================

            overall_risk = forecast_df["Risk Level"].value_counts().idxmax()
            avg_cases = round(forecast_df["Forecasted Cases"].mean())

            if overall_risk == "Low Risk":
                risk_color = "#2ECC71"
                risk_icon = "🟢"
                status = "Routine Monitoring"

            elif overall_risk == "Medium Risk":
                risk_color = "#F39C12"
                risk_icon = "🟠"
                status = "Enhanced Surveillance"

            else:
                risk_color = "#E74C3C"
                risk_icon = "🔴"
                status = "Immediate Preventive Action"

            st.markdown("## Overall Risk Assessment")

            st.markdown(f"""
            <div style="
            background:white;
            border-left:8px solid {risk_color};
            padding:22px;
            border-radius:16px;
            box-shadow:0 8px 20px rgba(0,0,0,.08);
            margin-bottom:25px;
            ">
            <table style="width:100%;font-size:18px;border-collapse:collapse;">
            <tr>
            <td style="padding:8px;"><b>State</b></td>
            <td>{selected_state}</td>
            </tr>
            <tr>
            <td style="padding:8px;"><b>Forecast Horizon</b></td>
            <td>{HORIZON} Months</td>
            </tr>
            <tr>
            <td style="padding:8px;"><b>Average Forecast Cases</b></td>
            <td>{avg_cases}</td>
            </tr>
            <tr>
            <td style="padding:8px;"><b>Predicted Risk</b></td>
            <td style="color:{risk_color};font-weight:bold;">
            {risk_icon} {overall_risk}
            </td>
            </tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
            # ======================================================
            # RECOMMENDED PUBLIC HEALTH ACTIONS
            # ======================================================

            st.markdown("## Recommended Public Health Actions")

            if overall_risk == "High Risk":

                precautions = [
                    "Intensify malaria surveillance across high-risk areas.",
                    "Conduct indoor residual spraying (IRS) and larval source management.",
                    "Ensure adequate stock of antimalarial drugs and rapid diagnostic kits.",
                    "Strengthen vector monitoring and mosquito breeding site control.",
                    "Launch community awareness campaigns on malaria prevention.",
                    "Increase active fever screening and early case detection.",
                    "Coordinate with district health authorities for rapid response."
                ]

            elif overall_risk == "Medium Risk":

                precautions = [
                    "Maintain routine malaria surveillance.",
                    "Promote the use of insecticide-treated bed nets.",
                    "Remove stagnant water and mosquito breeding sites.",
                    "Increase public awareness on malaria prevention.",
                    "Monitor weekly malaria trends for early warning.",
                    "Ensure availability of diagnostic and treatment facilities."
                ]

            else:

                precautions = [
                    "Continue routine surveillance activities.",
                    "Maintain environmental sanitation and vector control.",
                    "Encourage early diagnosis and prompt treatment.",
                    "Promote continued use of mosquito nets.",
                    "Monitor seasonal changes that may increase malaria transmission."
                ]
            
            actions_html = "<br>".join(
                [f"▸ {item}" for item in precautions]
            )

            st.markdown(f"""
            <div style="
            background:#ffffff;
            border-radius:16px;
            padding:22px;
            box-shadow:0 8px 20px rgba(0,0,0,.08);
            margin-bottom:30px;
            border-left:8px solid {risk_color};
            ">
            {actions_html}

            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
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
            # st.markdown("### GeoAI Spatial Risk Panel")

            # # ================= LOAD GEOJSON =================
            # with open("india_states.geojson", "r", encoding="utf-8") as f:
            #     india_geojson = json.load(f)

            # # ================= DATA =================
            # map_df = pd.DataFrame({
            #     "State": [selected_state],
            #     "Average Forecasted Cases": [cases_value]
            # })

            # # ================= CHOROPLETH MAP =================
            # fig_map = px.choropleth_mapbox(
            #     map_df,
            #     geojson=india_geojson,
            #     locations="State",
            #     featureidkey="properties.NAME_1",
            #     color="Average Forecasted Cases",

            #     color_continuous_scale=[
            #         [0, "#2ECC71"],     # Low Risk
            #         [0.5, "#F39C12"],   # Moderate Risk
            #         [1, "#E74C3C"]      # High Risk
            #     ],

            #     mapbox_style="carto-positron",
            #     zoom=5,
            #     center={"lat": 22.5, "lon": 80},
            #     opacity=0.85
            # )

            # # ================= HIGH RISK VISUAL MARKER =================
            # if risk_level == "High":
            #     fig_map.add_trace(
            #         go.Scattermapbox(
            #             lat=[26.5],
            #             lon=[92.5],
            #             mode="markers",
            #             marker=dict(
            #                 size=30,
            #                 color="rgba(231,76,60,0.35)"
            #             ),
            #             hoverinfo="skip",
            #             showlegend=False
            #         )
            #     )

            # # ================= STATE LABEL (IMPROVED) =================
            # # Extract centroid from GeoJSON for better placement
            # state_lat = None
            # state_lon = None

            # for feature in india_geojson["features"]:
            #     if feature["properties"]["NAME_1"] == selected_state:
            #         coords = feature["geometry"]["coordinates"]

            #         # crude centroid estimation (works for most state polygons)
            #         if feature["geometry"]["type"] == "Polygon":
            #             lon_vals = [p[0] for p in coords[0]]
            #             lat_vals = [p[1] for p in coords[0]]
            #             state_lat = sum(lat_vals) / len(lat_vals)
            #             state_lon = sum(lon_vals) / len(lon_vals)

            #         elif feature["geometry"]["type"] == "MultiPolygon":
            #             lon_vals = []
            #             lat_vals = []
            #             for poly in coords:
            #                 for p in poly[0]:
            #                     lon_vals.append(p[0])
            #                     lat_vals.append(p[1])
            #             state_lat = sum(lat_vals) / len(lat_vals)
            #             state_lon = sum(lon_vals) / len(lon_vals)

            # # Add label if coordinates found
            # if state_lat and state_lon:
            #     fig_map.add_trace(
            #         go.Scattermapbox(
            #         lat=[state_lat],
            #             lon=[state_lon],
            #             mode="text",
            #             text=[selected_state],
            #             textfont=dict(
            #                 size=16,
            #                 color="#111111",
            #                 family="Arial Black"
            #             ),
            #             showlegend=False
            #         )
            #     )

            # # ================= LAYOUT =================
            # fig_map.update_layout(
            #     margin=dict(l=0, r=0, t=40, b=0),
            #     height=600,
            #     coloraxis_colorbar=dict(
            #         title="Average Forecasted Cases",
            #         thickness=18
            #     ),
            #     mapbox={
            #         "layers": []   
            #     }
            # )

            # # ================= DISPLAY =================
            # st.plotly_chart(fig_map, use_container_width=True)
# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<div class="footer" style="text-align:center;">
    <p><strong>CSIR – Indian Institute of Chemical Technology (CSIR-IICT)</strong></p>
    <p>Environmental Information, Awareness, Capacity Building and Livelihood Programme (EIACP)</strong></p>
    <p>Ministry of Environment, Forest and Climate Change (MoEFCC)</strong></p>
</div>
""", unsafe_allow_html=True)   
