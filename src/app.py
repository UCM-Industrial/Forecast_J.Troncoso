import datetime

import streamlit as st

st.set_page_config(
    page_title="RECAST",
    layout="wide",
    page_icon="🔆",
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
    ],
    "Modeling": [
        st.Page(
            "pages/3_scenario.py",
            title="Scenario Builder",
            icon=":material/design_services:",
        ),
        st.Page(
            "pages/4_model.py",
            title="Forecasting Suit",
            icon=":material/robot_2:",
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
