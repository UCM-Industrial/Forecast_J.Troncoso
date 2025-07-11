from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast import (
    load_model_bundle,
    read_csv_with_datetime,
    run_etl_pipeline,
)

# --- Configuration ---
# Base directory for temporary data and models
TEMP_DIR = Path().cwd() / "forecast_temp"
TEMP_DIR.mkdir(exist_ok=True)
(TEMP_DIR / "models").mkdir(exist_ok=True)

MODELS_CONFIG = {
    "Wind": {
        "bundle_path": TEMP_DIR / "models" / "eolica_bundle.zip",
        "color": "#2A66C2",
    },
    "Solar": {
        "bundle_path": TEMP_DIR / "models" / "solar_bundle.zip",
        "color": "#EF553B",
    },
}


# --- App Functions ---
@st.cache_data
def run_prediction(model_type: str):
    """Loads the model and data to run the prediction."""
    config = MODELS_CONFIG[model_type]

    model, metadata = load_model_bundle(config["bundle_path"])

    if model is None or metadata is None:
        st.error(f"Missing bundle for {model_type}.")
        st.info(f"Expected path: {config['bundle_path']}")
        return None

    data_path = TEMP_DIR / f"{model_type.lower()}_features.csv"

    if not data_path.exists():
        st.error(f"Data not found in {data_path}")
        return None

    input_data = read_csv_with_datetime(data_path)

    try:
        required_features = metadata["features"]
        data_for_prediction = input_data[required_features]
    except KeyError as e:
        st.error(f"The column {e} requiered for the model was not found.")
        return None

    predictions = model.predict(data_for_prediction)
    results_df = pd.DataFrame(
        predictions,
        index=data_for_prediction.index,
        columns=["prediction"],
    )
    with st.expander("Check model metada"):
        st.json(metadata)
    return results_df


# --- User Interface (UI) ---
st.title("Energy Generation Forecast")

# --- Sidebar ---
st.sidebar.header("Controls")
selected_model_name = st.sidebar.selectbox(
    "Select energy type",
    options=list(MODELS_CONFIG.keys()),
)

if st.sidebar.button("Update Data and Predict"):
    # Runs the full ETL pipeline from the backend
    with st.spinner(
        "Downloading and processing new data... (This may take several minutes)",
    ):
        try:
            # The backend function saves files in TEMP_DIR
            run_etl_pipeline(TEMP_DIR)
            st.sidebar.success("Data updated successfully.")
            # Clear cache to force reloading new data
            st.cache_data.clear()
        except Exception as e:
            st.sidebar.error(f"Error updating data: {e}")


# --- Main Page ---
st.write(TEMP_DIR)
st.header(f"Forecast for model: {selected_model_name}")

config = MODELS_CONFIG[selected_model_name]
results = run_prediction(selected_model_name)

if results is not None:
    with st.expander("Prediction results"):
        st.dataframe(results)

    st.subheader("Forecast Plot")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results["prediction"],
            mode="lines",
            name="Forecast",
            line=dict(color=config["color"]),
        ),
    )
    fig.update_layout(
        title=f"{selected_model_name} Generation Forecast",
        xaxis_title="Date",
        yaxis_title="Generation (MWh)",
        legend_title="Legend",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(
        f"No data available for the {selected_model_name} model. Please update the data from the sidebar.",
    )
