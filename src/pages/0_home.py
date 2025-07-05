import streamlit as st

st.title(
    "🔆 RECAST: Renewable Energy Scenario Forecasting Toolkit",
)
st.markdown("---")

st.markdown(
    """
    RECAST (Renewable Energy Scenario Forecasting Toolkit ) is a modular platform designed for mid-term and long-term forecasting of renewable energy production in Chile.

    This tool integrates spatially-aware data preprocessing, time series decomposition, scenario simulation, and predictive modeling to support energy analysts, researchers, and decision-makers in forecasting solar and wind energy production under real and synthetic conditions.

    ---
    """,
)

st.subheader("Key Modules")
st.markdown(
    """
    - **Geospatial Processor**: Load and process climate data (e.g., `.grib` files), apply spatial masks, and generate structured inputs for modeling.
    - **Series Decomposer**: Apply STL or MSTL to split production series into trend, seasonal, and residual components.
    - **Forecasting Suit**: Train and evaluate forecasting models (e.g., Holt-Winters, XGBoost), visualize predictions, and export results.
    """,
)

st.markdown("---")
st.subheader("Documentation")
st.markdown(
    "A detailed guide on how to use each module will be available in the **Documentation** section",
)
