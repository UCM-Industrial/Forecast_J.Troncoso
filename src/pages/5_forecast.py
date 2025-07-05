import pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_forecasting():
    col1, col2 = st.columns(2)

    with col1:
        uploaded_model = st.file_uploader("Upload model", type="pkl")

        if uploaded_model is not None:
            try:
                # Cargar el objeto desde el archivo
                model = pickle.load(uploaded_model)
                st.write(model)
            except Exception as e:
                st.error(f"Error al cargar el archivo: {e}")

    with col2:
        data = st.file_uploader("Upload Features to Predict", type="csv")

        df: pd.DataFrame = pd.read_csv(data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        # features = df.drop("seasonal", )
        # target = config["target"]
        # features = config["features"]

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

        st.plotly_chart(fig, use_container_width=True)


def main():
    render_forecasting()


if __name__ == "__main__":
    main()
