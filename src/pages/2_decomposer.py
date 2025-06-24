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


# Cache data loading to avoid re-reading on every interaction
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


# Cache figure creation
@st.cache_data
def cached_create_figure(result_df: pd.DataFrame, decomposition: str):
    """Create decomposition figure with caching."""
    return create_decomposition_figure(result_df, decomposition=decomposition)


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
        # Data preview in expander to save space
        with st.expander("Data Preview", expanded=False):
            st.dataframe(df.head(10))

        # Column selection
        column = st.selectbox("Select column for decomposition:", df.columns)

        if column:
            series = df[column]

            if not isinstance(series, pd.Series):
                raise TypeError(
                    f"Expected a pandas Series but got {type(series).__name__}.",
                )

            granularity, confience, details = detect_granularity(series)

            granularity_info = format_granularity_info(granularity, confience, details)

            st.info(granularity_info)

            return df, series

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

    return None, None


def render_stl_section(series: pd.Series):
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

    if st.button("Run STL", type="primary"):
        run_stl_decomposition(series, period, seasonal_window)


def render_mstl_section(series: pd.Series):
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

    if st.button("Run MSTL", type="primary"):
        run_mstl_decomposition(series, periods)


def run_stl_decomposition(series: pd.Series, period: int, seasonal_window: int):
    """Execute STL decomposition and display results."""
    try:
        with st.spinner("Running STL decomposition..."):
            result_df = cached_run_stl(series, period, seasonal_window)

        st.success("STL Decomposition Complete!")
        display_results(result_df, "stl")

    except Exception as e:
        st.error(f"STL decomposition failed: {e}")


def run_mstl_decomposition(series: pd.Series, periods: list[int]):
    """Execute MSTL decomposition and display results."""
    try:
        with st.spinner("Running MSTL decomposition..."):
            result_df = cached_run_mstl(series, periods)

        st.success("MSTL Decomposition Complete!")
        display_results(result_df, "mstl")

    except Exception as e:
        st.error(f"MSTL decomposition failed: {e}")


def display_results(result_df: pd.DataFrame, decomposition_type: str):
    """Display decomposition results and provide download option."""
    # Results preview in expander
    with st.expander("Results Preview", expanded=False):
        st.dataframe(result_df.head(10))

    # Create and display figure
    with st.spinner("Creating visualization..."):
        fig = cached_create_figure(result_df, decomposition_type)

    st.plotly_chart(fig, use_container_width=True)

    # Download section
    st.header("Download Results")

    col1, col2 = st.columns(2)
    with col1:
        output_csv = result_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            "Download CSV",
            output_csv,
            file_name=f"{decomposition_type}_decomposition_results.csv",
            mime="text/csv",
        )

    with col2:
        st.metric("Data Points", len(result_df))


def main():
    """Main application function."""
    st.title("Time Series Decomposition")
    st.markdown("Decompose your time series data using STL or MSTL methods.")

    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload your time series CSV", type="csv")

        if not uploaded_file:
            st.info("Please upload a CSV file to begin decomposition.")
            return None, None

        with st.spinner("Loading data..."):
            df = load_and_process_data(uploaded_file)
            st.success("Data loaded successfully")

    # Data upload section
    df, series = render_data_section(df=df)

    if df is None or series is None:
        return

    # Decomposition type selection
    st.header("Decomposition Method")
    decomp_type = st.radio(
        "Choose decomposition method:",
        ["STL", "MSTL"],
        horizontal=True,
    )

    # Render appropriate section based on selection

    if decomp_type == "STL":
        render_stl_section(series)
    else:
        render_mstl_section(series)


if __name__ == "__main__":
    main()
