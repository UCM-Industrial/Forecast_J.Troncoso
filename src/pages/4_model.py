from typing import Any

import pandas as pd
import streamlit as st

from _util import read_csv_with_datetime
from modeling import (
    ModelFactory,
    TimeSeriesModeler,
    create_forecast_vs_actual,
    plot_feature_importance,
)

st.title("Time Series Forecasting Suite")


def init_session_state() -> None:
    """Initialize session state variables."""
    for key, value in [
        ("data", None),
        ("results", []),
        ("latest_run", None),
        ("config", None),
        ("feature_importance", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data
def load_and_process_data(uploaded_file) -> pd.DataFrame:
    """Load and process the uploaded CSV file."""
    return read_csv_with_datetime(uploaded_file)


def get_model_params(model_key: str) -> dict[str, Any]:
    """Render and return parameters for the selected model."""
    if model_key in ["xgboost", "lightgbm", "catboost"]:
        return {
            "n_estimators": st.slider("N Estimators", 50, 500, 100, 50),
            "max_depth": st.slider("Max Depth", 3, 15, 6),
            "learning_rate": st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01),
        }
    if model_key == "arima":
        return {
            "order": (
                st.slider("p (AR order)", 0, 5, 1),
                st.slider("d (Differencing)", 0, 2, 1),
                st.slider("q (MA order)", 0, 5, 1),
            ),
        }
    if model_key == "holt-winters":
        return {
            "seasonal_periods": st.slider("Seasonal Periods", 2, 24, 12),
            "trend": st.selectbox("Trend", ["add", "mul", None]),
            "seasonal": st.selectbox("Seasonal", ["add", "mul", None]),
        }
    return {}  # For Ridge, Lasso etc. that have no specific UI params


def render_config() -> dict[str, Any] | None:
    data: pd.DataFrame = st.session_state.data
    numeric_cols = data.select_dtypes("number").columns.tolist()

    st.subheader("Select Target & Features")
    target_col = st.selectbox(
        "Target Variable",
        numeric_cols,
        index=len(numeric_cols) - 1,
    )
    feature_cols = st.multiselect(
        "Feature Variables",
        [col for col in numeric_cols if col != target_col],
        default=[col for col in numeric_cols if col != target_col],
    )

    st.subheader("Select Model")
    model_type = st.selectbox(
        "Model Type",
        [
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "Ridge",
            "Lasso",
            "ARIMA",
            "Holt-Winters",
        ],
    )

    st.subheader("Set Parameters")
    params = get_model_params(model_type.lower().replace(" ", "-"))

    # WARNING: Some models does not need Cross validation or test split
    st.subheader("Training Settings")
    cv_folds = st.slider(
        "Cross-Validation Folds",
        min_value=1,
        max_value=5,
        value=2,
    )
    test_size = st.slider(
        "Test size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
    )

    if st.button("Train Model", use_container_width=True):
        if not target_col or not feature_cols:
            st.error("Please select a target and at least one feature.")
            return None
        return {
            "data": data,
            "target": target_col,
            "features": feature_cols,
            "model_type": model_type,
            "params": params,
            "cv_folds": cv_folds,
            "test_size": test_size,
        }
    return None


def render_data_overview(data: pd.DataFrame) -> None:
    """Display data preview and statistical summary."""
    st.subheader("Data Preview")
    st.dataframe(data.head(10), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)


def render_main_content() -> None:
    """Render the main page content using tabs."""
    ...


def render_latest_run() -> None:
    """Display results of the most recent model training session."""
    if not st.session_state.latest_run:
        st.info("Train a model using the sidebar to see results here.")
        return

    run_info = st.session_state.latest_run
    st.subheader(f"Results for: {run_info['model_type']}")

    # Display metrics
    st.metric("CV Mean Absolute Error (MAE)", f"{run_info['cv_results']['mae']:.4f}")
    st.metric("CV R² Score", f"{run_info['cv_results']['r2']:.4f}")

    # Plot predictions
    fig = create_forecast_vs_actual(run_info["df"], "actual", "predicted")
    st.plotly_chart(fig, use_container_width=True)

    fig = plot_feature_importance(st.session_state.feature_importance)
    st.plotly_chart(fig)
    # Feature importance


def render_model_comparison() -> None:
    """Display a table comparing all model runs."""
    st.subheader("Model Performance Comparison")
    if not st.session_state.results:
        st.info("No models have been trained yet.")
        return

    results_df = pd.DataFrame(st.session_state.results)
    # Format and reorder columns for better presentation
    results_df["model_params"] = results_df["model_params"].astype(str)
    st.dataframe(
        results_df[
            ["model_type", "mae", "mse", "r2", "mae_std", "model_params", "cv_folds"]
        ],
        use_container_width=True,
    )

    if st.button("Clear Comparison History", use_container_width=True):
        st.session_state.results = []
        st.rerun()


def run_training_session(config: dict) -> None:
    """Execute the model training and prediction workflow."""
    with st.spinner(f"Training {config['model_type']} model..."):
        try:
            strategy = ModelFactory.create_strategy(
                config["model_type"],
                **config["params"],
            )
            modeler = TimeSeriesModeler(strategy)

            split_point = int(len(config["data"]) * (1 - config["test_size"]))
            train = config["data"].iloc[:split_point]
            test = config["data"].iloc[split_point:]

            X_train = train[config["features"]]
            y_train = train[config["target"]]

            X_test = test[config["features"]]
            y_test = test[config["target"]]

            # Cross-validate and then fit on all data
            cv_results = modeler.cross_validate(X_train, y_train, config["cv_folds"])
            modeler.fit(X_train, y_train)

            predictions = modeler.predict(X_test)

            # Store results for comparison table
            st.session_state.feature_importance = modeler.get_feature_importance()

            run_result = {"model_type": config["model_type"], **cv_results}
            st.session_state.results.append(run_result)

            # Store details of this specific run for the "Latest Run" tab
            st.session_state.latest_run = {
                "model_type": config["model_type"],
                "cv_results": cv_results,
                "df": pd.DataFrame(
                    {"actual": y_test, "predicted": predictions},
                    index=X_test.index,
                ),
            }
            st.success(f"Model {config['model_type']} trained successfully!")

        except (ImportError, ValueError) as e:
            st.error(f"An error occurred: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred during training: {e}")


def main() -> None:
    """Main function to run the Streamlit app."""
    init_session_state()

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload CSV Data", type="csv")
        if uploaded_file:
            st.session_state.data = load_and_process_data(uploaded_file)
            st.success("Data loaded successfully")

        if st.session_state.data is None:
            st.info("Please upload a CSV file to begin.")
            return None

    if st.session_state.data is None:
        st.info("Upload data via the sidebar to get started.")
        return

    tab_list = ["Configuration", "Results", "Performance comparasion"]
    tab1, tab2, tab3 = st.tabs(tab_list)

    with tab1:
        st.session_state.config = render_config()

        if st.session_state.config:
            run_training_session(st.session_state.config)
    with tab2:
        render_latest_run()
    with tab3:
        render_model_comparison()
        # st.rerun()


if __name__ == "__main__":
    main()
