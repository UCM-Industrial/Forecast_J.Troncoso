from pathlib import Path

import pandas as pd
import streamlit as st

from _util import (
    create_cyclical_features,
    group_by_datetime_index,
    read_csv_with_datetime,
    truncate_datetime_by_resolution,
)


@st.cache_data
def read_multiple_csv(uploaded_files, datetime_col: str = "datetime"):
    all_columns = set()
    temp_dfs = {}
    for uploaded_file in uploaded_files:
        try:
            # First, try to read a small sample to check if file is valid
            uploaded_file.seek(0)
            sample = pd.read_csv(uploaded_file, nrows=1)

            if sample.empty:
                st.warning(f"File {uploaded_file.name} appears to be empty. Skipping.")
                continue

            # Reset file pointer for actual reading
            uploaded_file.seek(0)

            # Try to read with the datetime utility function
            try:
                df = read_csv_with_datetime(uploaded_file, datetime_col)
            except Exception as datetime_error:
                # If datetime parsing fails, try reading as regular CSV
                st.warning(
                    f"Could not parse datetime in {uploaded_file.name}: {datetime_error}. Reading as regular CSV.",
                )
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)

                # Check if the specified datetime column exists
                if datetime_col in df.columns:
                    try:
                        df[datetime_col] = pd.to_datetime(df[datetime_col])
                        df.set_index(datetime_col, inplace=True)
                    except Exception as dt_convert_error:
                        st.warning(
                            f"Could not convert {datetime_col} to datetime in {uploaded_file.name}: {dt_convert_error}",
                        )
                else:
                    st.info(
                        f"Datetime column '{datetime_col}' not found in {uploaded_file.name}. Using default index.",
                    )

            filename = Path(uploaded_file.name).stem
            temp_dfs[filename] = df
            all_columns.update(df.columns)

        except pd.errors.EmptyDataError:
            st.error(
                f"File {uploaded_file.name} is empty or has no columns to parse. Please check the file format.",
            )
            continue
        except pd.errors.ParserError as pe:
            st.error(
                f"Parsing error in {uploaded_file.name}: {pe}. Please check the CSV format.",
            )
            continue
        except UnicodeDecodeError:
            st.error(
                f"Encoding error in {uploaded_file.name}. Please ensure the file is saved in UTF-8 format.",
            )
            continue
        except Exception as e:
            st.error(f"Unexpected error reading {uploaded_file.name}: {e!s}")
            continue
    return temp_dfs


@st.cache_data
def read_single_csv(uploaded_file):
    return read_csv_with_datetime(uploaded_file)


def find_duplicate_columns(dataframes: dict[str, pd.DataFrame]) -> set[str]:
    """Find columns that appear in multiple dataframes.

    Args:
        dataframes: Dictionary of filename -> DataFrame

    Returns:
        Set of column names that appear in multiple dataframes
    """
    all_columns = set()
    for df in dataframes.values():
        all_columns.update(df.columns)

    duplicate_columns = set()
    for col in all_columns:
        files_with_col = [name for name, df in dataframes.items() if col in df.columns]
        if len(files_with_col) > 1:
            duplicate_columns.add(col)

    return duplicate_columns


def rename_duplicate_columns(
    dataframes: dict[str, pd.DataFrame],
    handle_duplicates: str,
) -> dict[str, pd.DataFrame]:
    """Rename duplicate columns according to the specified strategy.

    Args:
        dataframes: Dictionary of filename -> DataFrame
        handle_duplicates: Strategy for handling duplicates

    Returns:
        Dictionary of dataframes with renamed columns
    """
    if handle_duplicates == "Overwrite":
        return {name: df.copy() for name, df in dataframes.items()}

    duplicate_columns = find_duplicate_columns(dataframes)
    renamed_dataframes = {}

    for filename, df in dataframes.items():
        df_copy = df.copy()

        if duplicate_columns:
            new_columns = []
            for col in df_copy.columns:
                if col in duplicate_columns:
                    if handle_duplicates == "Add filename suffix":
                        new_columns.append(f"{col}_{filename}")
                    elif handle_duplicates == "Add filename prefix":
                        new_columns.append(f"{filename}_{col}")
                else:
                    new_columns.append(col)
            df_copy.columns = new_columns

        renamed_dataframes[filename] = df_copy

    return renamed_dataframes


def prepare_dataframes_for_join(
    dataframes: dict[str, pd.DataFrame],
    datetime_col: str,
    datetime_resolution: str,
) -> dict[str, pd.DataFrame]:
    """Prepare dataframes for joining by setting datetime index with specified resolution.

    Args:
        dataframes: Dictionary of filename -> DataFrame
        datetime_col: Name of datetime column
        datetime_resolution: Resolution for datetime truncation

    Returns:
        Dictionary of dataframes ready for joining
    """
    prepared_dfs = {}

    for filename, df in dataframes.items():
        df_copy = df.copy()
        # WARNING: Provitional fix
        df_copy = df_copy.reset_index()

        # Ensure datetime column exists
        if datetime_col not in df_copy.columns:
            st.error(f"Datetime column '{datetime_col}' not found in {filename}")
            continue

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_copy[datetime_col]):
            try:
                df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
            except Exception as e:
                st.error(
                    f"Could not convert '{datetime_col}' to datetime in {filename}: {e}",
                )
                continue

        # Apply datetime resolution truncation
        df_copy[datetime_col] = truncate_datetime_by_resolution(
            df_copy[datetime_col],
            datetime_resolution,
        )

        # Set as index
        df_copy.set_index(datetime_col, inplace=True)

        prepared_dfs[filename] = df_copy

    return prepared_dfs


def join_dataframes(
    dataframes: dict[str, pd.DataFrame],
    join_type: str,
) -> pd.DataFrame | None:
    """Join multiple dataframes on their datetime index.

    Args:
        dataframes: Dictionary of prepared dataframes
        join_type: Type of join to perform

    Returns:
        Joined dataframe or None if join fails
    """
    if not dataframes:
        return None

    try:
        # Start with the first dataframe
        result_df = next(iter(dataframes.values()))

        # Join with remaining dataframes
        for df in list(dataframes.values())[1:]:
            result_df = result_df.join(df, how=join_type)

        return result_df

    except Exception as e:
        st.error(f"Error joining dataframes: {e}")
        return None


def display_file_previews(dataframes: dict[str, pd.DataFrame]) -> None:
    """Display preview of each dataframe in expandable sections.

    Args:
        dataframes: Dictionary of filename -> DataFrame
    """
    st.subheader("File Preview")
    for filename, df in dataframes.items():
        uploaded_file_name = f"{filename}.csv"
        with st.expander(
            f"Preview: {uploaded_file_name} ({df.shape[0]} rows, {df.shape[1]} cols)",
        ):
            st.dataframe(df.head(), use_container_width=True)


def get_user_configuration(datetime_col: str = "datetime") -> tuple:
    """Get user configuration for joining parameters.

    Args:
        datetime_col: Default datetime column name

    Returns:
        Tuple of (join_type, handle_duplicates, datetime_col_input, datetime_resolution)
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        join_type = st.selectbox(
            "Join Type",
            ["outer", "inner", "left", "right"],
            index=0,
            help="How to join the dataframes on datetime index",
        )

    with col2:
        handle_duplicates = st.selectbox(
            "Handle Duplicate Columns",
            ["Add filename suffix", "Add filename prefix", "Overwrite"],
            index=0,
            help="How to handle columns with the same name",
        )

    with col3:
        datetime_col_input = st.text_input(
            "Datetime Column",
            value=datetime_col,
            help="Name of the datetime column to use as index",
        )

    with col4:
        datetime_resolution = st.selectbox(
            "DateTime Resolution",
            ["exact", "m", "h", "D", "M", "Y"],
            index=0,
            help="Resolution for datetime matching during join",
        )

    return join_type, handle_duplicates, datetime_col_input, datetime_resolution


def csv_multi_joiner(
    uploaded_files,
    datetime_col: str = "datetime",
) -> pd.DataFrame | None:
    """Streamlit module to upload multiple CSVs and join them together.

    Args:
        uploaded_files: List of uploaded file objects
        datetime_col: Default name of datetime column

    Returns:
        Joined dataframe or None if join fails
    """
    # Get user configuration
    join_type, handle_duplicates, datetime_col_input, datetime_resolution = (
        get_user_configuration(datetime_col)
    )

    # Process files using the cached function (assuming this exists)
    temp_dfs = read_multiple_csv(uploaded_files, datetime_col_input)

    if not temp_dfs:
        st.error("No valid CSV files could be processed.")
        return None

    # Rename duplicate columns
    dataframes = rename_duplicate_columns(temp_dfs, handle_duplicates)

    # Display file previews
    display_file_previews(dataframes)

    # Prepare dataframes for joining
    prepared_dfs = prepare_dataframes_for_join(
        dataframes,
        datetime_col_input,
        datetime_resolution,
    )

    if not prepared_dfs:
        st.error("No dataframes could be prepared for joining.")
        return None

    # Join dataframes
    result_df = join_dataframes(prepared_dfs, join_type)

    if result_df is not None:
        st.success(f"Successfully joined {len(prepared_dfs)} files!")
        st.info(
            f"Final dataframe shape: {result_df.shape[0]} rows, {result_df.shape[1]} columns",
        )

    return result_df


def column_operations_tab(uploaded_files, datetime_col="datetime"):
    """Handle column operations in the second tab."""
    # Show loading state
    if uploaded_files:
        with st.spinner("Loading CSV files..."):
            # Get dataframes using cached function
            temp_dfs = read_multiple_csv(uploaded_files, datetime_col)
    else:
        temp_dfs = {}

    if not temp_dfs:
        st.warning("No valid CSV files found. Please check your files and try again.")
        return

    # Show successful file count
    st.success(f"Successfully loaded {len(temp_dfs)} CSV file(s)")

    # Initialize session state for modified dataframes if not exists
    if "modified_dfs" not in st.session_state:
        st.session_state.modified_dfs = {}

    # Select dataframe
    df_names = list(temp_dfs.keys())
    selected_df_name = st.selectbox("Select a Dataframe", df_names)

    if selected_df_name:
        # Get the dataframe (use modified version if exists, otherwise original)
        if selected_df_name in st.session_state.modified_dfs:
            df = st.session_state.modified_dfs[selected_df_name]
            st.info("Showing modified version of the dataframe")
        else:
            df = temp_dfs[selected_df_name].copy()

        # Show current dataframe info
        st.write(
            f"**{selected_df_name}.csv** - Shape: {df.shape[0]} rows, {df.shape[1]} columns",
        )

        # Show dataframe info
        with st.expander("DataFrame Info"):
            st.write("**Columns:**", list(df.columns))
            st.write("**Data Types:**")
            st.dataframe(df.dtypes.to_frame("Type"), use_container_width=True)

        # Column operations
        st.subheader("Sum Columns")

        # Get numeric columns only for summing
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_columns:
            st.warning("No numeric columns found in the selected dataframe.")
            st.info("Available columns and their types:")
            for col, dtype in df.dtypes.items():
                st.write(f"• {col}: {dtype}")
        else:
            st.info(
                f"Found {len(numeric_columns)} numeric columns available for summing",
            )

            columns_to_sum = st.multiselect(
                "Select columns to sum",
                numeric_columns,
                help="Only numeric columns are shown",
            )

            if columns_to_sum:
                # Name for the new combined column
                new_col_name = st.text_input("Name of the new column", "sum_column")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Add Combined Column", type="primary"):
                        if new_col_name and new_col_name.strip():
                            # Create the sum column
                            df[new_col_name.strip()] = df[columns_to_sum].sum(axis=1)

                            # Store modified dataframe in session state
                            st.session_state.modified_dfs[selected_df_name] = df

                            st.success(f"Column '{new_col_name.strip()}' added!")
                            st.rerun()  # Refresh to show updated dataframe
                        else:
                            st.error("Please enter a valid column name.")

                with col2:
                    if st.button("Reset DataFrame"):
                        # Remove from session state to reset to original
                        if selected_df_name in st.session_state.modified_dfs:
                            del st.session_state.modified_dfs[selected_df_name]
                        st.success("DataFrame reset to original!")
                        st.rerun()

        # Show current dataframe
        st.subheader("Current DataFrame")
        st.dataframe(df, use_container_width=True)

        # Download option for modified dataframe
        if selected_df_name in st.session_state.modified_dfs:
            csv_data = df.to_csv()
            st.download_button(
                label=f"Download Modified {selected_df_name}.csv",
                data=csv_data,
                file_name=f"{selected_df_name}_modified.csv",
                mime="text/csv",
            )


def render_multijoin():
    uploaded_files = st.file_uploader(
        "upload csv files",
        type=["csv"],
        accept_multiple_files=True,
        help="select multiple csv files to join together",
    )

    # Add datetime column configuration in sidebar
    datetime_col = st.text_input(
        "Default Datetime Column",
        value="datetime",
        help="Default datetime column name for all operations",
    )

    if not uploaded_files:
        st.info("Please upload at least one CSV file to get started.")
        return None

    master_df = csv_multi_joiner(uploaded_files, datetime_col)
    if master_df is not None:
        output_df = master_df.to_csv(index=True).encode("utf-8")
        st.dataframe(master_df.head(10))
        st.download_button(
            "Download CSV",
            output_df,
            file_name="",
            mime="text/csv",
        )


def render_cyclical_features_section():
    """Render optional cyclical features section."""
    uploaded_file_cyclical = st.file_uploader(
        "upload csv files",
        type=["csv"],
        help="select multiple csv files to join together",
    )

    # Add datetime column configuration in sidebar
    datetime_col = st.text_input(
        "Datetime column",
        value="datetime",
        help="Default datetime column name for all operations",
    )

    if not uploaded_file_cyclical:
        st.info("Please upload at least one CSV file to get started.")
        return None
    st.subheader("Optional: Add Cyclical Features")

    if uploaded_file_cyclical:
        with st.spinner("Loading CSV files..."):
            # Get dataframes using cached function
            temp_dfs = read_single_csv(uploaded_file_cyclical)
    else:
        return

    st.write("Add cyclical encoding for time-based features (useful for ML models)")

    features_options = ["hour", "day", "month", "dayofweek"]
    selected_features = st.multiselect(
        "Select features to encode:",
        features_options,
        default=["hour", "month"],
    )

    if selected_features and st.button("Add Cyclical Features"):
        try:
            df_with_features = create_cyclical_features(
                temp_dfs,
                features=selected_features,
            )
            st.success("✅ Cyclical features added!")

            with st.expander("Preview with cyclical features"):
                st.dataframe(df_with_features.head())

            # Download enhanced data
            enhanced_csv = df_with_features.to_csv().encode("utf-8")
            st.download_button(
                "Download Enhanced CSV",
                enhanced_csv,
                file_name="regional_means_with_cyclical_features.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error adding cyclical features: {e}")


def render_group_by_section():
    st.header("Group by Datetime Index")

    st.write("Upload a CSV file with a datetime index to group it by resolution.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = read_csv_with_datetime(uploaded_file)

            st.subheader("Original DataFrame Head")
            st.write(df.head())

            # Select resolution
            resolution = st.selectbox(
                "Select Grouping Resolution",
                options=["H", "D", "W", "M", "Q", "Y"],
                help="H: Hour, D: Day, W: Week, M: Month, Q: Quarter, Y: Year",
            )

            # Select columns to aggregate
            all_numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
            selected_columns = st.multiselect(
                "Select Columns to Aggregate (leave blank for all numeric)",
                options=all_numeric_columns,
                default=all_numeric_columns,
            )

            # Select aggregation functions
            agg_options = ["sum", "mean", "median", "min", "max", "count", "std"]
            selected_agg_functions = st.multiselect(
                "Select Aggregation Functions",
                options=agg_options,
                default=["sum"],
            )

            if st.button("Group Data"):
                if not selected_columns:
                    st.warning("Please select at least one column to aggregate.")
                elif not selected_agg_functions:
                    st.warning("Please select at least one aggregation function.")
                else:
                    try:
                        # Convert selected_agg_functions to a dictionary if multiple aggregations are chosen
                        if (
                            len(selected_agg_functions) > 1
                            and len(selected_columns) > 0
                        ):
                            agg_dict = {
                                col: selected_agg_functions for col in selected_columns
                            }
                            grouped_df = group_by_datetime_index(
                                df,
                                resolution,
                                agg_functions=agg_dict,
                                columns=selected_columns,
                            )
                        elif len(selected_agg_functions) == 1:
                            grouped_df = group_by_datetime_index(
                                df,
                                resolution,
                                agg_functions=selected_agg_functions[0],
                                columns=selected_columns,
                            )
                        else:  # No columns selected but agg functions are
                            st.warning(
                                "Please select at least one column to aggregate.",
                            )
                            return

                        st.subheader("Grouped DataFrame Head")
                        st.write(grouped_df.head())

                        st.subheader("Download Grouped Data")
                        csv_data = grouped_df.to_csv().encode("utf-8")
                        st.download_button(
                            label="Download Grouped Data as CSV",
                            data=csv_data,
                            file_name="grouped_data.csv",
                            mime="text/csv",
                        )

                    except TypeError as e:
                        st.error(f"Error during grouping: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

        except Exception as e:
            st.error(
                f"Error loading CSV: {e}. Please ensure it's a valid CSV with a datetime index.",
            )


def main():
    st.title("CSV Utility Functions")
    tab1, tab2, tab3 = st.tabs(
        ["CSV Joiner", "Column Operations", "cyclical_encoding"],
    )

    with tab1:
        render_multijoin()

    with tab2:
        render_group_by_section()
    with tab3:
        render_cyclical_features_section()


if __name__ == "__main__":
    main()
