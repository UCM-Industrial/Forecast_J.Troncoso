from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from _util import read_csv_with_datetime
from modeling import ModelFactory, TimeSeriesModeler, create_forecast_vs_actual

st.title("Time Series Forecasting Suite")


def init_session_state() -> None:
    """Initialize session state variables."""
    for key, value in [("data", None), ("results", []), ("latest_run", None)]:
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


def render_config_tab() -> dict[str, Any] | None:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload CSV Data", type="csv")
    if uploaded_file:
        st.session_state.data = load_and_process_data(uploaded_file)

    if st.session_state.data is None:
        st.info("Please upload a CSV file to begin.")
        return None

    data: pd.DataFrame = st.session_state.data
    numeric_cols = data.select_dtypes("number").columns.tolist()

    st.subheader("1. Select Target & Features")
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

    st.subheader("2. Select Model")
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

    st.subheader("3. Set Parameters")
    params = get_model_params(model_type.lower().replace(" ", "-"))

    st.subheader("4. Training Settings")
    cv_folds = st.slider(
        "Cross-Validation Folds",
        min_value=2,
        max_value=10,
        value=5,
    )

    if st.button("Train Model", type="primary", use_container_width=True):
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
    st.title("Time Series Forecasting Suite")

    if st.session_state.data is None:
        st.info("Upload data via the sidebar to get started.")
        return

    tab_list = ["Data Overview", "Latest Model Run", "Model Comparison"]
    tab1, tab2, tab3 = st.tabs(tab_list)

    with tab1:
        render_data_overview(st.session_state.data)
    with tab2:
        render_latest_run()
    with tab3:
        render_model_comparison()


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
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=run_info["df"].index, y=run_info["df"]["actual"], name="Actual"),
    )
    fig.add_trace(
        go.Scatter(
            x=run_info["df"].index,
            y=run_info["df"]["predicted"],
            name="Predicted",
            line=dict(dash="dash"),
        ),
    )
    fig.update_layout(
        title="Forecast vs. Actuals",
        xaxis_title="Date",
        yaxis_title="Value",
    )
    st.plotly_chart(fig, use_container_width=True)


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

            X = config["data"][config["features"]]
            y = config["data"][config["target"]]

            # Cross-validate and then fit on all data
            cv_results = modeler.cross_validate(X, y, config["cv_folds"])
            modeler.fit(X, y)
            predictions = modeler.predict(X)

            # Store results for comparison table
            run_result = {"model_type": config["model_type"], **cv_results}
            st.session_state.results.append(run_result)

            # Store details of this specific run for the "Latest Run" tab
            st.session_state.latest_run = {
                "model_type": config["model_type"],
                "cv_results": cv_results,
                "df": pd.DataFrame(
                    {"actual": y, "predicted": predictions},
                    index=X.index,
                ),
            }
            st.success(f"Model {config['model_type']} trained successfully!")

        except (ImportError, ValueError) as e:
            st.error(f"An error occurred: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred during training: {e}")


def train_model(
    model_type: str,
    params: dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
    cv_folds: int,
) -> TimeSeriesModeler:
    model_type = model_type.lower()

    modeler = TimeSeriesModeler()
    strategy = ModelFactory.create_strategy(model_type=model_type, **params)

    modeler.set_strategy(strategy)
    modeler.cross_validate(X=features, y=target, cv_folds=cv_folds)
    modeler.fit(X=features, y=target)
    return modeler


def render_training_model(data: pd.DataFrame):
    numeric_columns = data.select_dtypes(
        include=[np.number],
    ).columns.tolist()
    target_column: str = st.selectbox(
        "Select target variable",
        numeric_columns,
        help="Chose the variable to predict",
    )

    feature_columns: list[str] = st.multiselect(
        "Select feature variables",
        [col for col in numeric_columns if col != target_column],
        default=[col for col in numeric_columns if col != target_column],
        help="Chose the input features for the model",
    )
    if target_column and feature_columns:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Model Selection")
            model_type = st.selectbox(
                "Choose model type:",
                ["XGBoost", "LightGBM", "CatBoost", "ARIMA", "Holt-Winters"],
                help="Select the machine learning algorithm",
            )

            # Model parameters based on selection
            st.markdown("#### Model Parameters")

            params = None

            if model_type in ["XGBoost", "LightGBM", "CatBoost"]:
                params = render_tree_model_params()
            elif model_type == "ARIMA":
                params = render_arima_params()
            elif model_type == "Holt-Winters":
                params = render_holt_winters_params()

        with col2:
            st.markdown("#### Cross-Validation Settings")
            cv_folds = st.slider("Number of CV folds", 3, 10, 5)
            test_size_pct = st.slider("Test size (%)", 10, 40, 20)

            st.markdown("#### Training Options")
            perform_grid_search = st.checkbox(
                "Perform grid search",
                help="Automatically tune hyperparameters",
            )

            if perform_grid_search and model_type in [
                "XGBoost",
                "LightGBM",
                "CatBoost",
            ]:
                st.markdown("**Grid Search Parameters:**")
                n_est_range = st.slider("N_estimators range", 50, 200)
                depth_range = st.slider("Max_depth range", 3, 8)

            submitted = st.button("Train Model", type="primary")
        if submitted and target_column and feature_columns:
            with st.spinner("Training model..."):
                if not params:
                    raise ValueError("Params is empty")

                modeler = train_model(
                    model_type,
                    params,
                    data[feature_columns],
                    data[target_column],
                    cv_folds,
                )
                st.session_state.model = modeler
                predictions = modeler.predict(data[feature_columns])
                st.session_state.predictions = pd.Series(predictions, index=data.index)

        if st.session_state.model and st.session_state.predictions is not None:
            st.subheader("Predictions")
            pred_df = pd.DataFrame(
                {
                    "Actual": data[target_column],
                    "Predicted": st.session_state.predictions,
                },
            )
            fig = create_forecast_vs_actual(
                pred_df,
                actual_col="Actual",
                predicted_col="Predicted",
            )
            st.plotly_chart(fig)


def main():
    init_session_state()
    st.header("Upload Data")
    uploaded_csv = st.file_uploader(
        "Upload Training Data",
        type="csv",
        help="Upload the time series data with features and target variable",
    )

    if uploaded_csv:
        data = load_and_process_data(uploaded_csv)
        st.session_state.data = data
        st.success("Training data uploaded")

    if st.session_state.data is not None:
        tab1, tab2 = st.tabs(["Data Overview", "Model Training"])

        with tab1:
            render_data_overview(data=st.session_state.data)

        with tab2:
            render_training_model(data=st.session_state.data)


def main_2() -> None:
    """Main function to run the Streamlit app."""
    init_session_state()

    with st.sidebar:
        config = render_config_tab()

    if config:
        run_training_session(config)
        st.rerun()

    render_main_content()


if __name__ == "__main__":
    main_2()
