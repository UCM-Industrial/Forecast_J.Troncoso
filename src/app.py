import datetime

import streamlit as st

st.set_page_config(
    page_title="RECAST",
    layout="wide",
    page_icon="🔆",
)
st.markdown(
    r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
pages = {
    "About": [
        st.Page(
            "pages/0_home.py",
            title="Home",
            icon=":material/home:",
        ),
        st.Page(
            "pages/9_documentation.py",
            title="Documentation",
            icon=":material/developer_guide:",
        ),
    ],
    "Prepare Data": [
        st.Page(
            "pages/1_preprocessor.py",
            title="Geospatial Processor",
            icon=":material/satellite_alt:",
        ),
        st.Page(
            "pages/2_decomposer.py",
            title="Series Decomposer",
            icon=":material/airware:",
        ),
        st.Page(
            "pages/3_helper_functions.py",
            title="Utility Functions",
            icon=":material/design_services:",
        ),
    ],
    "Modeling": [
        st.Page(
            "pages/4_model.py",
            title="Tuning Suite",
            icon=":material/robot_2:",
        ),
        st.Page(
            "pages/5_forecast.py",
            title="Energy projections",
            icon=":material/electric_bolt:",
        ),
    ],
}


pg = st.navigation(pages)
pg.run()

st.markdown("---")
st.caption(
    f"""
    **RECAST** · Renewable Energy Scenario Forecasting Toolkit  
    Developed by Universidad Catolica del Maule · © {datetime.datetime.now().year}  
    For inquiries, contact: [j.troncoso.morales@proton.me]  
    Version: v1.0.0 — Last updated: [2025-06-29]
    """,
)
