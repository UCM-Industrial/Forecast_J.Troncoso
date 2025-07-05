import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from _util import read_csv_with_datetime
from scenarios import Scenario, create_scenario_plot

# st.set_page_config("Scenario Builder", layout="wide")


def initialize_session_state():
    """Initialize session state variables."""
    if "df_historic" not in st.session_state:
        st.session_state.df_historic = None
    if "df_scenario" not in st.session_state:
        st.session_state.df_scenario = None
    if "scenario_configs" not in st.session_state:
        st.session_state.scenario_configs = []
    if "current_steps" not in st.session_state:
        st.session_state.current_steps = []


@st.cache_data
def load_and_process_data(uploaded_file: UploadedFile) -> pd.DataFrame:
    """Load and process the uploaded CSV file."""
    return read_csv_with_datetime(uploaded_file)


def render_sidebar():
    """Render the sidebar for data upload and scenario settings."""
    with st.sidebar:
        st.header("Data upload")
        uploaded_file = st.file_uploader(
            "Upload historic CSV data",
            type="csv",
            key="data_uploader",
        )

        st.subheader("Scenario Parameters")
        scenario_name = st.text_input("Scenario Name", "My Energy Scenario")
        future_years = st.number_input(
            "Years to Project",
            min_value=1,
            max_value=50,
            value=5,
        )
        freq = st.selectbox(
            "Data Frequency",
            options=["h", "D"],
            index=0,
            help="Pandas frequency string ('h' for hourly, 'D' for daily).",
        )
        tz = st.text_input(
            "Timezone",
            "America/Santiago",
            help="e.g., 'UTC', 'America/New_York'",
        )

        if uploaded_file:
            df = load_and_process_data(uploaded_file)
            st.session_state.df_historic = df
            if df is not None:
                st.success(
                    f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.",
                )

        return scenario_name, future_years, freq, tz


def render_transformation_form(df: pd.DataFrame):
    """Render the form for adding a new transformation."""
    st.subheader("Add a Transformation")

    cols = df.select_dtypes(include="number").columns.tolist()
    if not cols:
        st.warning("No numeric columns available in the data to apply transformations.")
        return

    selected_cols = st.multiselect("Select Columns", options=cols)

    trans_type = st.selectbox(
        "Transformation Type",
        [
            "Exponential Growth",
            "Linear Trend",
            "Climate Shift",
            "Capacity Steps",
            "Seasonal Adjustment",
            "Add Noise",
        ],
        index=0,
    )

    params = {}
    # Dynamically render parameter inputs based on transformation type
    if trans_type == "Exponential Growth":
        rate = st.number_input(
            "Annual Growth Rate (e.g., 0.05 for 5%)",
            value=0.02,
            format="%.4f",
        )
        params = {"exponential": {"rate": rate}}

    elif trans_type == "Linear Trend":
        rate = st.number_input(
            "Annual Trend Rate (e.g., 0.03 for 3%)",
            value=0.01,
            format="%.4f",
        )
        params = {"trend": {"rate": rate}}

    elif trans_type == "Climate Shift":
        change = st.number_input(
            "Total Change by End of Period (e.g., 0.1 for 10%)",
            value=0.05,
            format="%.4f",
        )
        start_year = st.number_input(
            "Start Year of Change (from beginning of data)",
            value=0,
            min_value=0,
        )
        params = {"climate_shift": {"change": change, "start_year": start_year}}

    elif trans_type == "Add Noise":
        level = st.slider(
            "Noise Level (as fraction of standard deviation)",
            0.0,
            0.5,
            0.05,
        )
        params = {"noise": {"level": level}}

    elif trans_type == "Seasonal Adjustment":
        st.info('Enter factors as JSON. E.g., for months: {"1": 1.1, "7": 0.9}')
        month_factors_str = st.text_area("Monthly Factors (JSON)", "{}")
        hour_factors_str = st.text_area("Hourly Factors (JSON)", "{}")
        try:
            month_factors = eval(month_factors_str) if month_factors_str else {}
            hour_factors = eval(hour_factors_str) if hour_factors_str else {}
            params = {
                "seasonal_adjustment": {
                    "month_factors": month_factors,
                    "hour_factors": hour_factors,
                },
            }
        except Exception:
            st.error("Invalid JSON format for seasonal factors.")
            return

    elif trans_type == "Capacity Steps":
        st.markdown("##### Define Capacity Increases")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            step_date = c1.date_input("Date of Increase")
            step_increase = c2.number_input(
                "Absolute Increase",
                value=10.0,
                format="%.2f",
            )

            if st.button("Add Step", use_container_width=True):
                st.session_state.current_steps.append(
                    {"date": str(step_date), "absolute_increase": step_increase},
                )

        if st.session_state.current_steps:
            st.write("Current Steps:")
            st.json(st.session_state.current_steps)
            if st.button("Clear Steps", type="secondary"):
                st.session_state.current_steps = []

        params = {"capacity_steps": {"steps": st.session_state.current_steps}}

    if st.button(
        "Add Transformation to Scenario",
        type="primary",
        use_container_width=True,
    ):
        if not selected_cols:
            st.error("Please select at least one column.")
        else:
            config = {"cols": selected_cols, "transformations": params}
            st.session_state.scenario_configs.append(config)
            st.session_state.current_steps = []  # Clear steps after adding
            st.success(f"Added '{trans_type}' transformation for {selected_cols}.")
            st.rerun()


def render_scenario_summary_and_execution(scenario: Scenario):
    """Display the current scenario configuration and execution controls."""
    with st.expander("Show Current Scenario Configuration", expanded=True):
        if not scenario.configs:
            st.info("No transformations added yet.")
        else:
            st.text(scenario.summary())
            if st.button("Clear Entire Scenario", use_container_width=True):
                st.session_state.scenario_configs = []
                st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Apply Scenario & Generate Future",
            type="primary",
            use_container_width=True,
            disabled=not scenario.configs,
        ):
            with st.spinner("Generating scenario..."):
                historic_df = st.session_state.df_historic

                # 1. Generate future seasonal data for all numeric columns
                future_dfs = []
                for col in historic_df.select_dtypes(include="number").columns:
                    future_series = scenario.generate_seasonal_like(historic_df[col])
                    future_dfs.append(future_series.to_frame(name=col))

                future_df = pd.concat(future_dfs, axis=1)

                # Combine historic and future data
                combined_df = pd.concat([historic_df, future_df])

                # Apply all configured transformations to the combined dataframe
                scenario_df = scenario.apply(combined_df)

                st.session_state.df_scenario = scenario_df
                st.success("Scenario generated successfully!")


def render_results():
    """Render the results tab with dataframe, plot, and download."""
    st.header("Results & Visualization")

    if st.session_state.get("df_scenario") is None:
        st.info("Apply a scenario from the 'Configuration' tab to see results here.")
        return

    scenario_df = st.session_state.df_scenario
    historic_df = st.session_state.df_historic

    st.subheader("Scenario Data Preview")
    st.dataframe(scenario_df.tail())

    # Create a combined dataframe for plotting with a 'source' column
    historic_df_plot = historic_df.copy()
    historic_df_plot["Source"] = "Historic"

    scenario_df_plot = scenario_df[~scenario_df.index.isin(historic_df.index)].copy()
    scenario_df_plot["Source"] = "Scenario"

    plot_df_combined = pd.concat([historic_df_plot, scenario_df_plot])

    st.subheader("Visualize Scenario")
    numeric_cols = scenario_df.select_dtypes(include="number").columns.tolist()

    # Use session state to persist selections
    if "primary_cols" not in st.session_state:
        st.session_state.primary_cols = []

    primary_cols = st.multiselect(
        "Select columns to plot",
        numeric_cols,
        default=st.session_state.primary_cols,
        key="primary_cols_selector",
    )
    st.session_state.primary_cols = primary_cols

    if primary_cols:
        fig = create_scenario_plot(plot_df_combined, cols=primary_cols)

        # Customize plot for better comparison
        fig.update_layout(
            title=f"Historic vs. Scenario: {', '.join(primary_cols)}",
            yaxis_title=primary_cols[0] if len(primary_cols) == 1 else "Value",
            xaxis_title="Date",
            legend_title="Data Source",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Download Results")
    csv = scenario_df.to_csv().encode("utf-8")
    st.download_button(
        label="Download Scenario Data as CSV",
        data=csv,
        file_name=f"{st.session_state.get('scenario_name', 'scenario')}_data.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_visualization_section(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Primary Axis")
        primary_cols = st.multiselect(
            "Select columns for primary axis",
            numeric_cols,
            key="primary_cols",
        )

    with col2:
        st.subheader("Secondary Axis (Optional)")
        # Filter out already selected primary columns
        available_secondary = [col for col in numeric_cols if col not in primary_cols]
        secondary_cols = st.multiselect(
            "Select columns for secondary axis",
            available_secondary,
            key="secondary_cols",
        )

    # Plot button and generation
    if st.button("Generate Plot"):
        if not primary_cols:
            st.error("Please select at least one column for the primary axis")
            return

        try:
            with st.spinner("Generating plot..."):
                fig = create_scenario_plot(
                    df,
                    cols=primary_cols,
                    second_axis_cols=secondary_cols if secondary_cols else None,
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating plot: {e}")


def main_2():
    """Main function to run the Streamlit app."""
    # st.set_page_config(page_title="Scenario Builder", layout="wide")
    st.title("Renewable Energy Scenario Builder")

    initialize_session_state()

    # --- Sidebar ---
    scenario_name, future_years, freq, tz = render_sidebar()

    # --- Main Content ---
    if st.session_state.df_historic is None:
        st.info("Please CSV file using the sidebar to begin.")
        return

    df_historic = st.session_state.df_historic

    # Instantiate the scenario object
    scenario = Scenario(name=scenario_name, future_years=future_years, freq=freq, tz=tz)
    # Load configs from session state
    scenario.configs = st.session_state.get("scenario_configs", [])
    st.session_state.scenario_name = scenario_name  # Save name for download

    # --- Tabs for workflow ---
    tab1, tab2 = st.tabs(["1. Configuration", "2. Results & Visualization"])

    with tab1:
        st.header("Build scenario")
        render_transformation_form(df_historic)
        st.divider()
        render_scenario_summary_and_execution(scenario)

    with tab2:
        render_results()


def main():
    st.subheader("Upload Data")

    if "dataframes" not in st.session_state:
        st.session_state.dataframe = []

    col1, col2 = st.columns(2)
    with col1:
        uploaded_csv_1 = st.file_uploader("Upload the historic CSV data", type="csv")
        uploaded_csv_2 = st.file_uploader(
            "Upload the historic CSV",
            type="csv",
        )

        if uploaded_csv_1 is None:
            return

        temp_df = load_and_process_data(uploaded_csv_1)

        if uploaded_csv_2:
            temp_df_2 = load_and_process_data(uploaded_csv_2)
            df = temp_df.join(temp_df_2, how="outer")
        else:
            df = temp_df

        # df.index = df.index.tz_convert("UTC")
        # df.index = df.index.tz_convert("America/Santiago")

        st.success(f"Data loaded {df.shape}")

    with col2:
        st.dataframe(df)

    st.subheader("Data visualization")

    render_visualization_section(df)


if __name__ == "__main__":
    main_2()
