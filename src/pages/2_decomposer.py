import pandas as pd
import streamlit as st

from _util import read_csv_with_datetime
from decomposer import (
    create_decomposition_figure,
    detect_granularity,
    format_granularity_info,
    run_mstl,
    run_stl,
)

# Configure page
# st.set_page_config(
#     page_title="Time Series Decomposition",
#     layout="wide",
# )


@st.cache_data
def load_and_process_data(uploaded_file) -> pd.DataFrame:
    """Load and process the uploaded CSV file."""
    return read_csv_with_datetime(uploaded_file)


# Cache decomposition results to avoid re-computation
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
    """Run STL decomposition on multiple columns and combine results.

    Args:
        df: DataFrame containing the time series data
        columns: List of column names to decompose
        period: Seasonal period for STL
        seasonal_window: Seasonal window for STL

    Returns:
        Combined DataFrame with all decomposition results
    """
    all_results = []

    for col in columns:
        series = df[col]
        result_df = cached_run_stl(series, period, seasonal_window)

        # Rename columns to include original column name
        renamed_columns = {}
        for component in result_df.columns:
            if component == "observed":
                renamed_columns[component] = f"observed_{col}"
            else:
                renamed_columns[component] = f"{component}_{col}"

        result_df = result_df.rename(columns=renamed_columns)
        all_results.append(result_df)

    # Combine all results
    combined_df = pd.concat(all_results, axis=1)
    return combined_df


def run_multi_column_mstl(
    df: pd.DataFrame,
    columns: list[str],
    periods: list[int],
) -> pd.DataFrame:
    """Run MSTL decomposition on multiple columns and combine results.

    Args:
        df: DataFrame containing the time series data
        columns: List of column names to decompose
        periods: List of seasonal periods for MSTL

    Returns:
        Combined DataFrame with all decomposition results
    """
    all_results = []

    for col in columns:
        series = df[col]
        result_df = cached_run_mstl(series, periods)

        # Rename columns to include original column name
        renamed_columns = {}
        for component in result_df.columns:
            if component == "observed":
                renamed_columns[component] = f"observed_{col}"
            else:
                renamed_columns[component] = f"{component}_{col}"

        result_df = result_df.rename(columns=renamed_columns)
        all_results.append(result_df)

    # Combine all results
    combined_df = pd.concat(all_results, axis=1)
    return combined_df


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

        # Multi-column selection option
        st.subheader("Column Selection")

        # Toggle for single vs multiple columns
        multi_column_mode = st.checkbox(
            "Decompose multiple columns",
            value=True,  # Default to True as requested
            help="Enable to select and decompose multiple columns simultaneously",
        )

        if multi_column_mode:
            # Multi-select for columns
            numeric_columns = df.select_dtypes(
                include=["float64", "int64"],
            ).columns.tolist()
            if not numeric_columns:
                st.error("No numeric columns found in the dataset.")
                return None, None, None

            selected_columns = st.multiselect(
                "Select columns for decomposition:",
                numeric_columns,
                default=numeric_columns[:2]
                if len(numeric_columns) >= 2
                else numeric_columns,
                help="Select one or more numeric columns to decompose",
            )

            if not selected_columns:
                st.warning("Please select at least one column for decomposition.")
                return None, None, None

            # Show granularity info for the first selected column as reference
            if selected_columns:
                reference_series = df[selected_columns[0]]
                granularity, confidence, details = detect_granularity(reference_series)
                granularity_info = format_granularity_info(
                    granularity,
                    confidence,
                    details,
                )
                st.info(
                    f"Granularity info (based on '{selected_columns[0]}'): {granularity_info}",
                )

            return df, selected_columns, True

        else:
            # Single column selection (original behavior)
            column = st.selectbox("Select column for decomposition:", df.columns)

            if column:
                series = df[column]

                if not isinstance(series, pd.Series):
                    raise TypeError(
                        f"Expected a pandas Series but got {type(series).__name__}.",
                    )

                granularity, confidence, details = detect_granularity(series)
                granularity_info = format_granularity_info(
                    granularity,
                    confidence,
                    details,
                )
                st.info(granularity_info)

                return df, [column], False

            return None, None, None

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


def render_stl_section(df: pd.DataFrame, columns: list[str], is_multi_mode: bool):
    """Render STL decomposition section."""
    st.header("STL Parameters")

    col1, col2 = st.columns(2)
    with col1:
        period = st.number_input(
            "Seasonal Period",
            value=24,
            min_value=2,
            help="Number of observations per seasonal cycle. Common values: 24 (daily with hourly data), 12 (yearly with monthly data), 7 (weekly with daily data)",
        )
    with col2:
        seasonal_window = st.number_input(
            "Seasonal Window",
            value=169,
            min_value=3,
            step=2,
        )

    # Validate parameters
    valid, error_msg = validate_stl_params(period, seasonal_window)
    if not valid:
        st.error(error_msg)
        return

    # Show selected columns info
    if is_multi_mode and len(columns) > 1:
        st.info(f"Selected columns for decomposition: {', '.join(columns)}")

    if st.button("Run STL", type="primary"):
        run_stl_decomposition(df, columns, period, seasonal_window, is_multi_mode)


def render_mstl_section(df: pd.DataFrame, columns: list[str], is_multi_mode: bool):
    """Render MSTL decomposition section."""
    st.header("MSTL Parameters")

    st.warning(
        "**Performance Notice**: MSTL decomposition is computationally intensive. "
        "Processing time may be significantly longer for large datasets, "
        "multiple seasonal periods, or high-precision calculations. "
        "Consider testing with smaller samples first.",
    )

    periods_str = st.text_input(
        "Seasonal Periods (comma-separated)",
        value="24, 168",
        help="Enter seasonal periods separated by commas. Examples: '24' (daily), '24,168' (daily+weekly), '12,4' (monthly+quarterly)",
    )

    periods = parse_periods(periods_str)
    if periods is None:
        st.error("Invalid periods format. Please enter comma-separated integers.")
        return

    if len(periods) == 0:
        st.error("Please enter at least one seasonal period.")
        return

    st.info(f"Seasonal periods: {periods}")

    # Show selected columns info
    if is_multi_mode and len(columns) > 1:
        st.info(f"Selected columns for decomposition: {', '.join(columns)}")

    if st.button("Run MSTL", type="primary"):
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
        with st.spinner(f"Running STL decomposition on {len(columns)} column(s)..."):
            if is_multi_mode and len(columns) > 1:
                result_df = run_multi_column_stl(df, columns, period, seasonal_window)
            else:
                # Single column decomposition
                series = df[columns[0]]
                result_df = cached_run_stl(series, period, seasonal_window)

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
        with st.spinner(f"Running MSTL decomposition on {len(columns)} column(s)..."):
            if is_multi_mode and len(columns) > 1:
                result_df = run_multi_column_mstl(df, columns, periods)
            else:
                # Single column decomposition
                series = df[columns[0]]
                result_df = cached_run_mstl(series, periods)

        st.success("MSTL Decomposition Complete!")
        display_results(result_df, "mstl", columns, is_multi_mode)

    except Exception as e:
        st.error(f"MSTL decomposition failed: {e}")


def display_results(
    result_df: pd.DataFrame,
    decomposition_type: str,
    columns: list[str],
    is_multi_mode: bool,
):
    """Display decomposition results and provide download option."""
    # Results preview in expander
    with st.expander("Results Preview", expanded=False):
        st.dataframe(result_df.head(10))

    # Create and display figure
    with st.spinner("Creating visualization..."):
        if is_multi_mode and len(columns) > 1:
            # For multi-column, create separate plots for each column
            for col in columns:
                st.subheader(f"Decomposition for: {col}")

                # Filter columns for this specific original column
                col_columns = [c for c in result_df.columns if c.endswith(f"_{col}")]
                col_df = result_df[col_columns]

                # Rename columns back to standard names for plotting
                renamed_cols = {}
                for c in col_columns:
                    if c.startswith("observed_"):
                        renamed_cols[c] = "observed"
                    else:
                        component = c.replace(f"_{col}", "")
                        renamed_cols[c] = component

                col_df_renamed = col_df.rename(columns=renamed_cols)
                fig = create_decomposition_figure(col_df_renamed, decomposition_type)
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Single column visualization
            fig = create_decomposition_figure(result_df, decomposition_type)
            st.plotly_chart(fig, use_container_width=True)

    # Download section
    st.header("Download Results")

    col1, col2 = st.columns(2)
    with col1:
        output_csv = result_df.to_csv(index=True).encode("utf-8")
        filename_suffix = (
            f"_{'_'.join(columns)}"
            if is_multi_mode and len(columns) > 1
            else f"_{columns[0]}"
        )
        st.download_button(
            "Download CSV",
            output_csv,
            file_name=f"{decomposition_type}_decomposition_results{filename_suffix}.csv",
            mime="text/csv",
        )

    with col2:
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Data Points", len(result_df))
        with metrics_col2:
            st.metric("Columns Processed", len(columns))


def main():
    """Main application function."""
    st.title("Time Series Decomposition")
    st.markdown("Decompose your time series data using STL or MSTL methods.")

    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload your time series CSV", type="csv")

        if not uploaded_file:
            st.info("Please upload a CSV file to begin decomposition.")
            return

        with st.spinner("Loading data..."):
            df = load_and_process_data(uploaded_file)
            st.success("Data loaded successfully")

    # Data upload section
    result = render_data_section(df=df)
    if result[0] is None:
        return

    df, columns, is_multi_mode = result

    # Decomposition type selection
    st.header("Decomposition Method")
    decomp_type = st.radio(
        "Choose decomposition method:",
        ["STL", "MSTL"],
        horizontal=True,
    )

    # Render appropriate section based on selection
    if decomp_type == "STL":
        render_stl_section(df, columns, is_multi_mode)
    else:
        render_mstl_section(df, columns, is_multi_mode)


if __name__ == "__main__":
    main()
