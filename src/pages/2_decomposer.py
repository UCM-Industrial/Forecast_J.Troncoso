import io
import zipfile

import pandas as pd
import streamlit as st

from _util import read_csv_with_datetime
from decomposer import (
    calculate_average_seasonal_pattern,
    calculate_residual_covariance,
    create_decomposition_figure,
    detect_granularity,
    format_granularity_info,
    plot_average_seasonal_pattern,
    plot_covariance_matrix,
    plot_residual_distribution,
    run_mstl,
    run_stl,
    test_residual_properties,
)

# --- Initialize session state ---
if "decomposed_df" not in st.session_state:
    st.session_state.decomposed_df = None
if "decomposition_params" not in st.session_state:
    st.session_state.decomposition_params = {}


@st.cache_data
def load_and_process_data(uploaded_file) -> pd.DataFrame:
    """Load and process the uploaded CSV file."""
    return read_csv_with_datetime(uploaded_file)


@st.cache_data
def cached_run_stl(
    series: pd.Series,
    period: int,
    seasonal_window: int,
) -> pd.DataFrame:
    """Run STL decomposition with caching."""
    return run_stl(series, period=period, seasonal_window=seasonal_window)


@st.cache_data
def cached_run_mstl(series: pd.Series, periods: list[int]) -> pd.DataFrame:
    """Run MSTL decomposition with caching."""
    return run_mstl(series, periods=periods)


def run_multi_column_stl(
    df: pd.DataFrame,
    columns: list[str],
    period: int,
    seasonal_window: int,
) -> pd.DataFrame:
    """Run STL decomposition on multiple columns and combine results."""
    all_results = []
    for col in columns:
        series = df[col]
        result_df = cached_run_stl(series, period, seasonal_window)
        renamed_columns = {
            comp: f"observed_{col}" if comp == "observed" else f"{comp}_{col}"
            for comp in result_df.columns
        }
        all_results.append(result_df.rename(columns=renamed_columns))
    return pd.concat(all_results, axis=1)


def run_multi_column_mstl(
    df: pd.DataFrame,
    columns: list[str],
    periods: list[int],
) -> pd.DataFrame:
    """Run MSTL decomposition on multiple columns and combine results."""
    all_results = []
    for col in columns:
        series = df[col]
        result_df = cached_run_mstl(series, periods)
        renamed_columns = {
            comp: f"observed_{col}" if comp == "observed" else f"{comp}_{col}"
            for comp in result_df.columns
        }
        all_results.append(result_df.rename(columns=renamed_columns))
    return pd.concat(all_results, axis=1)


def parse_periods(periods_str: str) -> list[int] | None:
    """Parse comma-separated periods string into list of integers."""
    try:
        return [int(p.strip()) for p in periods_str.split(",") if p.strip()]
    except ValueError:
        return None


def validate_stl_params(period: int, seasonal_window: int) -> tuple[bool, str]:
    """Validate STL parameters."""
    if period < 2:
        return False, "Seasonal period must be at least 2"
    if seasonal_window < 3:
        return False, "Seasonal window must be at least 3"
    if seasonal_window % 2 == 0:
        return False, "Seasonal window must be odd"
    return True, ""


def render_data_section(df: pd.DataFrame):
    """Render the data upload and preview section."""
    try:
        with st.expander("Data Preview", expanded=False):
            st.dataframe(df.head(10))

        st.subheader("Column Selection")
        multi_column_mode = st.checkbox(
            "Decompose multiple columns",
            value=True,
            help="Enable to select and decompose multiple columns simultaneously",
        )

        if multi_column_mode:
            numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
            if not numeric_columns:
                st.error("No numeric columns found in the dataset.")
                return None, None, None

            selected_columns = st.multiselect(
                "Select columns for decomposition:",
                numeric_columns,
                default=numeric_columns[: min(2, len(numeric_columns))],
                help="Select one or more numeric columns to decompose",
            )

            if not selected_columns:
                st.warning("Please select at least one column for decomposition.")
                return None, None, None

            if selected_columns:
                granularity, confidence, details = detect_granularity(
                    df[selected_columns[0]],
                )
                st.info(
                    f"Granularity (based on '{selected_columns[0]}'): "
                    f"{format_granularity_info(granularity, confidence, details)}",
                )
            return df, selected_columns, True
        else:
            column = st.selectbox("Select column for decomposition:", df.columns)
            if column:
                series = df[column]
                granularity, confidence, details = detect_granularity(series)
                st.info(format_granularity_info(granularity, confidence, details))
                return df, [column], False
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


def render_stl_section(df: pd.DataFrame, columns: list[str], is_multi_mode: bool):
    """Render STL decomposition section."""
    st.header("STL Parameters")
    col1, col2 = st.columns(2)
    period = col1.number_input(
        "Seasonal Period",
        value=24,
        min_value=2,
        help="Observations per cycle (e.g., 24 for hourly data, 7 for daily).",
    )
    seasonal_window = col2.number_input(
        "Seasonal Window",
        value=169,
        min_value=3,
        step=2,
        help="Must be odd. A larger value means a smoother seasonal component.",
    )

    valid, error_msg = validate_stl_params(period, seasonal_window)
    if not valid:
        st.error(error_msg)
        return

    if is_multi_mode and len(columns) > 1:
        st.info(f"Selected columns for decomposition: {', '.join(columns)}")

    if st.button("Run STL", type="primary"):
        st.session_state.decomposition_params = {
            "type": "stl",
            "period": period,
            "columns": columns,
            "is_multi_mode": is_multi_mode,
        }
        run_stl_decomposition(df, columns, period, seasonal_window, is_multi_mode)


def render_mstl_section(df: pd.DataFrame, columns: list[str], is_multi_mode: bool):
    """Render MSTL decomposition section."""
    st.header("MSTL Parameters")
    st.warning(
        "**Performance Notice**: MSTL can be slow on large datasets.",
    )
    periods_str = st.text_input(
        "Seasonal Periods (comma-separated)",
        "24, 168",
        help="e.g., '24, 168' for daily and weekly patterns in hourly data.",
    )

    periods = parse_periods(periods_str)
    if not periods:
        st.error("Invalid periods. Please enter comma-separated integers.")
        return

    st.info(f"Seasonal periods: {periods}")
    if is_multi_mode and len(columns) > 1:
        st.info(f"Selected columns for decomposition: {', '.join(columns)}")

    if st.button("Run MSTL", type="primary"):
        st.session_state.decomposition_params = {
            "type": "mstl",
            "periods": periods,
            "columns": columns,
            "is_multi_mode": is_multi_mode,
        }
        run_mstl_decomposition(df, columns, periods, is_multi_mode)


def run_stl_decomposition(
    df: pd.DataFrame,
    columns: list[str],
    period: int,
    seasonal_window: int,
    is_multi_mode: bool,
):
    """Execute STL decomposition and display results."""
    try:
        with st.spinner(f"Running STL on {len(columns)} column(s)..."):
            result_df = (
                run_multi_column_stl(df, columns, period, seasonal_window)
                if is_multi_mode and len(columns) > 1
                else cached_run_stl(df[columns[0]], period, seasonal_window)
            )
        st.session_state.decomposed_df = result_df
        st.success("STL Decomposition Complete!")
        display_results(result_df, "stl", columns, is_multi_mode)
    except Exception as e:
        st.error(f"STL decomposition failed: {e}")


def run_mstl_decomposition(
    df: pd.DataFrame,
    columns: list[str],
    periods: list[int],
    is_multi_mode: bool,
):
    """Execute MSTL decomposition and display results."""
    try:
        with st.spinner(f"Running MSTL on {len(columns)} column(s)..."):
            result_df = (
                run_multi_column_mstl(df, columns, periods)
                if is_multi_mode and len(columns) > 1
                else cached_run_mstl(df[columns[0]], periods)
            )
        st.session_state.decomposed_df = result_df
        st.success("MSTL Decomposition Complete!")
        display_results(result_df, "mstl", columns, is_multi_mode)
    except Exception as e:
        st.error(f"MSTL decomposition failed: {e}")
        st.write(e)


def display_results(
    result_df: pd.DataFrame,
    decomposition_type: str,
    columns: list[str],
    is_multi_mode: bool,
):
    """Display decomposition results and provide download option."""
    with st.expander("Results Preview", expanded=False):
        st.dataframe(result_df.head(10))

    with st.spinner("Creating visualization..."):
        if is_multi_mode and len(columns) > 1:
            for i, col in enumerate(columns):
                if i >= 2:  # Limit to showing first 2 plots for performance
                    st.info(
                        f"Hiding remaining {len(columns) - i} plots. "
                        "All data is available for download.",
                    )
                    break
                st.subheader(f"Decomposition for: {col}")
                col_columns = [c for c in result_df.columns if c.endswith(f"_{col}")]
                col_df = result_df[col_columns]
                renamed_cols = {
                    c: c.replace(f"_{col}", "")
                    if c.startswith("observed") is False
                    else "observed"
                    for c in col_columns
                }
                fig = create_decomposition_figure(
                    col_df.rename(columns=renamed_cols),
                    decomposition_type,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            fig = create_decomposition_figure(result_df, decomposition_type)
            st.plotly_chart(fig, use_container_width=True)

    st.header("Download Results")
    separate_files = st.toggle("Download components separately (ZIP)", value=True)
    filename_suffix = (
        f"_{'_'.join(columns)}"
        if is_multi_mode and len(columns) > 1
        else f"_{columns[0]}"
    )

    if separate_files:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for component in ["seasonal", "trend", "resid"]:
                comp_df = result_df.filter(like=component)
                zf.writestr(
                    f"{component}{filename_suffix}.csv",
                    comp_df.to_csv(index=True).encode("utf-8"),
                )
        st.download_button(
            "Download Components ZIP",
            zip_buffer,
            f"{decomposition_type}_components{filename_suffix}.zip",
            "application/zip",
        )
    else:
        output_csv = result_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            "Download Combined CSV",
            output_csv,
            f"{decomposition_type}_decomposition{filename_suffix}.csv",
            "text/csv",
        )


def render_validation_tab(decomposed_df: pd.DataFrame, params: dict):
    """Logic for the validation, test and covariance resid decomposition."""
    st.header("Residuals Analysis")
    columns = params.get("columns", [])
    is_multi_mode = params.get("is_multi_mode", False)

    if is_multi_mode and len(columns) > 1:
        selected_col = st.selectbox(
            "Select column to analyze its residuals:",
            options=columns,
        )
        resid_col_name = f"resid_{selected_col}"
        df_to_test = pd.DataFrame({"resid": decomposed_df[resid_col_name]})
    else:
        selected_col = columns[0]
        resid_col_name = "resid"
        df_to_test = decomposed_df

    st.subheader(f"Analysis for: `{selected_col}`")
    if resid_col_name in decomposed_df.columns:
        results = test_residual_properties(df_to_test)
        st.markdown("##### Stationarity Tests")
        col1, col2 = st.columns(2)
        adf = results.get("adf_test", {})
        col1.metric(
            "ADF p-value",
            f"{adf.get('p_value', 99):.4f}",
            adf.get("interpretation", "Error"),
            border=True,
        )
        kpss = results.get("kpss_test", {})
        col2.metric(
            "KPSS p-value",
            f"{kpss.get('p_value', 99):.4f}",
            kpss.get("interpretation", "Error"),
            border=True,
        )

        st.markdown("##### Distribution")
        fig_dist = plot_residual_distribution(df_to_test["resid"])
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.warning(f"No residual data found for '{selected_col}'.")

    if is_multi_mode and len(columns) > 1:
        with st.expander("View Residuals Covariance Matrix"):
            residuals_df = decomposed_df.filter(like="resid")
            residuals_df.columns = [
                c.replace("resid_", "") for c in residuals_df.columns
            ]
            if not residuals_df.empty:
                fig_cov = plot_covariance_matrix(
                    calculate_residual_covariance(residuals_df),
                )
                st.plotly_chart(fig_cov, use_container_width=True)


def render_patterns_tab(decomposed_df: pd.DataFrame, params: dict):
    """Logic for the seasonal patterns exploration."""
    st.header("Average Seasonal Pattern Explorer")
    columns = params.get("columns", [])
    is_multi_mode = params.get("is_multi_mode", False)
    decomp_type = params.get("type")

    selected_col = columns[0]
    smoth = st.toggle("Smoother Pattern")
    if is_multi_mode and len(columns) > 1:
        selected_col = st.selectbox("Select column to view patterns:", options=columns)

    if decomp_type == "stl":
        available_periods = [params.get("period")]
    else:  # mstl
        available_periods = params.get("periods", [])

    if not available_periods or available_periods[0] is None:
        st.warning("No seasonal periods available for analysis.")
        return

    selected_period = st.selectbox("Select seasonal period:", options=available_periods)

    st.subheader(f"Pattern for `{selected_col}` with period `{selected_period}`")

    seasonal_col_name = "seasonal"
    if len(available_periods) > 1:
        seasonal_col_name += f"_{selected_period}"

    if is_multi_mode and len(columns) > 1:
        seasonal_col_name += f"_{selected_col}"

    if seasonal_col_name in decomposed_df:
        avg_pattern = calculate_average_seasonal_pattern(
            decomposed_df[seasonal_col_name],
            selected_period,
            smoothing=smoth,
            # window=7,
        )
        fig_pattern = plot_average_seasonal_pattern(avg_pattern, selected_period)
        st.plotly_chart(fig_pattern, use_container_width=True)

    else:
        st.error(
            f"Could not find data for the selected combination. Column name searched: `{seasonal_col_name}`",
        )

    st.header("Download Patterns")
    if st.button("Prepare all patterns for download"):
        with st.spinner("Calculating all average patterns..."):
            params["smoth_pattern"] = smoth
            all_patterns_df = generate_all_patterns_df(decomposed_df, params)
            if not all_patterns_df.empty:
                csv = all_patterns_df.to_csv(index_label="timestep").encode("utf-8")
                st.download_button(
                    label="Download All Patterns as CSV",
                    data=csv,
                    file_name="average_seasonal_patterns.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No patterns were generated.")


def generate_all_patterns_df(decomposed_df: pd.DataFrame, params: dict):
    """Calculate all the means for the seasonals patterns for each decomposition."""
    all_patterns = {}
    columns = params.get("columns", [])
    is_multi_mode = params.get("is_multi_mode", False)
    smoth_pattern = params.get("smoth_pattern", False)
    periods = (
        params.get("periods")
        if params.get("type") == "mstl"
        else [params.get("period")]
    )

    for col in columns:
        for period in periods:
            seasonal_col = "seasonal"

            if len(periods) > 1:
                seasonal_col += f"_{period}"

            if is_multi_mode and len(columns) > 1:
                seasonal_col += f"_{col}"

            if seasonal_col in decomposed_df:
                pattern = calculate_average_seasonal_pattern(
                    decomposed_df[seasonal_col],
                    period,
                    smoothing=smoth_pattern,
                )
                all_patterns[f"pattern_{col}_period_{period}"] = pattern

    return pd.DataFrame(all_patterns)


def main():
    """Main application function."""
    st.title("Time Series Decomposition")
    st.markdown("Decompose your time series data using STL or MSTL methods.")

    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload your time series CSV", type="csv")
        if not uploaded_file:
            st.info("Please upload a CSV file to begin.")
            return
        df = load_and_process_data(uploaded_file)

    tab1, tab2, tab3 = st.tabs(["Decompose", "Validation", "Patterns"])

    with tab1:
        result = render_data_section(df=df)
        if not result or not result[0] is not None:
            return
        df, columns, is_multi_mode = result

        st.header("Decomposition Method")
        decomp_type = st.radio("Choose method:", ["STL", "MSTL"], horizontal=True)

        if decomp_type == "STL":
            render_stl_section(df, columns, is_multi_mode)
        else:
            render_mstl_section(df, columns, is_multi_mode)

    with tab2:
        if st.session_state.decomposed_df is None:
            st.info("Run a decomposition first to see validation results.")
            return
        else:
            render_validation_tab(
                st.session_state.decomposed_df,
                st.session_state.decomposition_params,
            )
    with tab3:
        if st.session_state.decomposed_df is None:
            st.info("Run a decomposition first to explore patterns.")
        else:
            render_patterns_tab(
                st.session_state.decomposed_df,
                st.session_state.decomposition_params,
            )


if __name__ == "__main__":
    main()
