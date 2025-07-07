import json
import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# from forecast import * # Forecast backend

CWD = Path(__file__).parent
DATA_PATH = CWD / "data" / "forecast_input.nc"


@st.cache_data
def load_model_and_config(model_type: str):
    """Loads a model and its configuration file from the 'models' directory."""
    model_dir = CWD / "models" / model_type.lower()
    model_path = model_dir / "model.pkl"
    config_path = model_dir / "config.json"

    if not model_path.exists() or not config_path.exists():
        st.error(
            f"Model files for '{model_type}' not found. Please run `setup.py` first.",
        )
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(config_path) as f:
        config = json.load(f)
    return model, config


@st.cache_data
def load_prediction_data(file_path: Path) -> pd.DataFrame | None:
    """Loads and preprocesses the forecast data from the NetCDF file."""
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)

    return df


def render_forecasting():
    col1, col2 = st.columns(2)

    with col1:
        uploaded_model = st.file_uploader("Upload model", type="pkl")

        if uploaded_model is not None:
            try:
                # Cargar el objeto desde el archivo
                model = pickle.load(uploaded_model)
                st.code(model, language="python")
            except Exception as e:
                st.error(f"Error al cargar el archivo: {e}")

    with col2:
        data = st.file_uploader("Upload Features to Predict", type="csv")

        df: pd.DataFrame = pd.read_csv(data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

    if st.button("Predict"):
        results = pd.DataFrame(
            model.predict(df),
            index=df.index,
            columns=["prediction"],
        )
        st.dataframe(results)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results.iloc[
                    :,
                    0,
                ],
                mode="lines",
                name="Predicción",
            ),
        )

        fig.update_layout(
            title="Eolica Generation prediction",
            xaxis_title="Date",
            yaxis_title="Generation (MWh)",
        )

        st.plotly_chart(fig, use_container_width=True)


def main():
    render_forecasting()


if __name__ == "__main__":
    main()
