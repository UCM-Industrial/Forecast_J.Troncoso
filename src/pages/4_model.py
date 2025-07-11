import io
import itertools
import json
import pickle
from datetime import datetime
from typing import Any
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from _util import read_csv_with_datetime
from modeling import (
    ModelFactory,
    TimeSeriesModeler,
    create_forecast_vs_actual,
    plot_feature_importance,
)

st.title("Time Series Forecasting Suit")


def init_session_state() -> None:
    """Initialize session state variables."""
    for key, value in [
        ("data", None),
        ("results", []),
        ("latest_run", None),
        ("config", None),
        ("feature_importance", None),
        ("shap_values", None),
        ("model", None),
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
            "n_estimators": st.slider("N Estimators", 50, 1000, 100, 50),
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
            # "seasonal_periods": st.slider("Seasonal Periods", 2, 24, 12),
            "seasonal_periods": st.number_input("Seasonal Periods", 2, 366, 12),
            "trend": st.selectbox("Trend", ["add", "mul", None]),
            "seasonal": st.selectbox("Seasonal", ["add", "mul", None]),
        }
    return {}  # For Ridge, Lasso etc. that have no specific UI params


def get_basic_param_grid(model_key: str) -> dict[str, list[Any]]:
    """Get basic hyperparameters for grid search - the most important ones."""
    param_grid: dict[str, list[Any]] = {}

    if model_key in ["xgboost", "lightgbm", "catboost"]:
        # Core parameters that have the biggest impact
        n_range = st.slider(
            "N Estimators range",
            min_value=50,
            max_value=1000,
            value=(100, 500),
            step=50,
            help="Number of boosting rounds (trees).",
        )
        param_grid["n_estimators"] = list(range(n_range[0], n_range[1] + 1, 50))

        depth_range = st.slider(
            "Max Depth range",
            min_value=3,
            max_value=16,
            value=(4, 10),
            step=1,
            help="Maximum tree depth for base learners.",
        )
        depth_param = "depth" if model_key == "catboost" else "max_depth"
        param_grid[depth_param] = list(range(depth_range[0], depth_range[1] + 1))

        param_grid["learning_rate"] = [
            float(x.strip())
            for x in st.text_input("Learning Rate", value="0.01, 0.2").split(",")
        ]

    elif model_key == "arima":
        p_range = st.slider("p (AR) range", 0, 5, (0, 3), 1, help="Lag order")
        d_range = st.slider(
            "d (diff) range",
            0,
            2,
            (0, 1),
            1,
            help="Degree of differencing",
        )
        q_range = st.slider(
            "q (MA) range",
            0,
            5,
            (0, 3),
            1,
            help="Moving average order",
        )

        p_list = list(range(p_range[0], p_range[1] + 1))
        d_list = list(range(d_range[0], d_range[1] + 1))
        q_list = list(range(q_range[0], q_range[1] + 1))
        param_grid["order"] = [
            (p, d, q) for p, d, q in itertools.product(p_list, d_list, q_list)
        ]

    elif model_key == "holt-winters":
        sp_range = st.slider(
            "Seasonal Periods range",
            2,
            365,
            (6, 12),
            1,
            help="Periods in full seasonal cycle",
        )
        param_grid["seasonal_periods"] = list(range(sp_range[0], sp_range[1] + 1))

        trend_opts = st.multiselect(
            "Trend options",
            ["add", "mul", None],
            ["add", None],
            help="Trend component type",
        )
        seasonal_opts = st.multiselect(
            "Seasonal options",
            ["add", "mul", None],
            ["add", None],
            help="Seasonal component type",
        )
        param_grid["trend"] = trend_opts
        param_grid["seasonal"] = seasonal_opts

    return param_grid


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

    # st.download_button(
    #     "Download actual CSV",
    #     data[[*feature_cols, target_col]].to_csv(),
    #     file_name=f"{target_col}_master_df.csv",
    #     mime="text/csv",
    # )

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
    st.subheader("Prediction Interval")
    predict_intervals = st.toggle("Calculate Prediction Intervals")
    alpha = None
    if predict_intervals:
        confidence = st.slider(
            "Confidence Level",
            0.80,
            0.99,
            0.95,
            0.01,
            help="The confidence level for the prediction interval.",
        )
        alpha = 1 - confidence  # Alpha is the significance level

    on_grid = st.toggle("Grid search")
    # tab1, tab2 = st.tabs(["Cross Validation", "Grid Search"])

    if not on_grid:
        params = get_model_params(model_type.lower().replace(" ", "-"))
        training_type = "cross_validation"
    elif on_grid:
        params = get_basic_param_grid(model_type.lower().replace(" ", "-"))
        training_type = "grid_search"
    else:
        raise ValueError

    with st.expander("Final Params", expanded=False):
        st.json(params)

    st.subheader("Training Configuration")
    col1, col2 = st.columns(2)

    with col1:
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=1,
            max_value=10,
            value=3,
            help="More folds = better validation but slower training",
        )

    with col2:
        test_size = st.slider(
            "Test Size",
            min_value=0.02,
            max_value=0.4,
            value=0.2,
            # step=0.02,
            help="Fraction of data reserved for final testing",
        )

    if st.button("Train Model", use_container_width=True):
        if not target_col or not feature_cols:
            st.error("Please select a target and at least one feature.")
            return None

        # Add alpha to the params if intervals are enabled
        if alpha is not None:
            if model_type.lower().replace(" ", "-") in [
                "xgboost",
                "lightgbm",
                "catboost",
                "arima",
                "holt-winters",
            ]:
                params["alpha"] = alpha

        return {
            "data": data,
            "target": target_col,
            "features": feature_cols,
            "model_type": model_type,
            "params": params,
            "cv_folds": cv_folds,
            "test_size": test_size,
            "training_type": training_type,
        }

    return None


def render_latest_run() -> None:
    """Display results of the most recent model training session."""
    if not st.session_state.latest_run:
        st.info("Train a model using the sidebar to see results here.")
        return

    run_info = st.session_state.latest_run
    model = st.session_state.model
    st.subheader(f"Results for: {run_info['model_type']}")
    # st.write(run_info)

    col1, col2 = st.columns(2)
    with col1:
        # Display metrics
        st.metric(
            "CV Mean Absolute Error (MAE)",
            f"{run_info['mae']:.4f}",
        )
        st.metric("CV R² Score", f"{run_info['r2']:.4f}")
    with col2:
        metadata = {
            "model_type": run_info["model_type"],
            "training_type": run_info["training_type"],
            "training_timestamp_utc": datetime.utcnow().isoformat(),
            "metrics": {
                "mae": run_info.get("mae"),
                "mse": run_info.get("mse"),
                "r2": run_info.get("r2"),
            },
            "features": run_info["features"],
            # "target": run_info[
            #     "target_col"
            # ],
            "model_params": run_info["params"],
        }

        # 2. Serializar el modelo y los metadatos
        pkl_bytes = pickle.dumps(model)
        json_bytes = json.dumps(metadata, indent=4).encode("utf-8")

        # 3. Crear un archivo ZIP en memoria
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr("model.pkl", pkl_bytes)
            zip_file.writestr("metadata.json", json_bytes)

        # 4. ZIP to download
        st.download_button(
            label="Download Model Bundle (.zip)",
            data=zip_buffer.getvalue(),
            file_name=f"{run_info['model_type']}_bundle.zip",
            mime="application/zip",
        )

    # Plot predictions
    fig = create_forecast_vs_actual(
        df=run_info["predictions_df"],
        actual_col="actual",
        predicted_col="predicted",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    try:
        fig = plot_feature_importance(st.session_state.feature_importance)
        st.plotly_chart(fig)
    except:
        st.warning("This model does not allow Feature importance")

    # SHAP analysis
    st.write(st.session_state.shap_values)
    try:
        st.write("**SHAP values**")
        shap_dict = st.session_state.shap_values
        shap_values = shap_dict["shap_values"]

        # Beeswarm plot
        fig_beeswarm = plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig_beeswarm)

    except:
        st.warning("This model does not allow SHAP Analysis")


def render_model_comparison() -> None:
    """Display a table comparing all model runs."""
    st.subheader("Model Performance Comparison")
    if not st.session_state.results:
        st.info("No models have been trained yet.")
        return

    results_df = pd.DataFrame(st.session_state.results)
    # Format and reorder columns for better presentation
    st.dataframe(
        results_df,
        # results_df[
        #     [
        #         "model_type",
        #         "mae",
        #         "mse",
        #         "r2",
        #         "model_params",
        #         "cv_folds",
        #         "features",
        #         "time",
        #     ]
        # ],
        # use_container_width=True,
    )

    if st.button("Clear Comparison History", use_container_width=True):
        st.session_state.results = []
        st.rerun()


def run_training_session(
    config: dict,
) -> None:
    """Execute the model training and prediction workflow.

    This function handles two types of training:
    1. 'cross_validation': Validates the model with a given set of parameters.
    2. 'grid_search': Performs a grid search to find the best hyperparameters, then trains the model with them.
    """
    with st.spinner(
        f"Running {config['training_type']} for {config['model_type']} model...",
    ):
        try:
            # Note: For grid search, the initial strategy is created with placeholder params.
            # The actual parameter grid is used inside fit_with_grid_search.
            initial_params = (
                {} if config["training_type"] == "grid_search" else config["params"]
            )
            strategy = ModelFactory.create_strategy(
                config["model_type"],
                **initial_params,
            )
            modeler = TimeSeriesModeler(strategy)

            # --- Data Splitting ---
            split_point = int(len(config["data"]) * (1 - config["test_size"]))
            train = config["data"].iloc[:split_point]
            test = config["data"].iloc[split_point:]

            X_train = train[config["features"]]
            y_train = train[config["target"]]
            X_test = test[config["features"]]
            y_test = test[config["target"]]

            run_summary = {}
            cv_results_df = pd.DataFrame()

            # --- Model Training and Validation ---
            if config["training_type"] == "cross_validation":
                # Perform cross-validation with the given parameters
                cv_metrics = modeler.cross_validate(
                    X_train,
                    y_train,
                    config["cv_folds"],
                )
                # Fit the model on the entire training set for predictions
                modeler.fit(X_train, y_train)

                # Prepare results for UI
                cross_validation_summary = {
                    "model_type": config["model_type"],
                    **cv_metrics,
                }
                cv_results_df = pd.DataFrame([cv_metrics])

            elif config["training_type"] == "grid_search":
                # Perform grid search. This method finds the best params and updates
                modeler.fit_with_grid_search(
                    X_train,
                    y_train,
                    param_grid=config["params"],
                    cv=config["cv_folds"],
                )

                # Explicitly fit the model on the full training data.
                modeler.fit(X_train, y_train)

                # Prepare results for UI
                cv_results_df = modeler.cv_results_

                if not cv_results_df.empty:
                    best_result_row = cv_results_df.loc[
                        cv_results_df["rank_test_score"].idxmin()
                    ]
                    grid_search_summary = {
                        "model_type": config["model_type"],
                        "best_cv_score": best_result_row["mean_test_score"],
                        "best_params": str(best_result_row["params"]),
                    }
                else:
                    st.warning("Grid search did not produce results.")

            try:
                shap_values = modeler.get_shap_values(X_test)
            except:
                shap_values = {}
            # --- Prediction and Storing Results ---
            predictions = modeler.predict(X_test)
            metrics = modeler.evaluate_regression_metrics(
                y_true=y_test,
                y_pred=predictions["predicted"],
            )
            run_summary = {
                "model_type": config["model_type"],
                "training_type": config["training_type"],
                "folds": config["cv_folds"],
                "params": modeler.strategy.params,
                "features": config["features"],
                "shap_values": shap_values,
                **metrics,
            }

            if "results" not in st.session_state:
                st.session_state.results = []

            st.session_state.results.append(run_summary)
            st.session_state.latest_run = {
                "predictions_df": pd.concat(
                    [y_test.rename("actual"), predictions],
                    axis=1,
                ),
                **run_summary,
            }
            # st.session_state.latest_run = {
            #     "predictions_df": pd.DataFrame(
            #         {"actual": y_test, "predicted": predictions},
            #         index=X_test.index,
            #     ),
            #     **run_summary,
            # }
            st.write(run_summary)
            st.session_state.feature_importance = modeler.get_feature_importance()
            st.session_state.shap_values = shap_values
            st.session_state.model = modeler.strategy.model

            st.success(
                f"Model {config['model_type']} ({config['training_type']}) trained successfully!",
            )

        except (ImportError, ValueError, TypeError) as e:
            st.error(f"An error occurred: {e}")
            st.write(e)
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

    tab_list = [
        "Configuration",
        "Last results",
        "Performance comparasion",
    ]
    tab1, tab2, tab3 = st.tabs(tab_list)

    with tab1:
        config = render_config()

        if config:
            run_training_session(config)
    with tab2:
        render_latest_run()
    with tab3:
        render_model_comparison()


if __name__ == "__main__":
    main()
