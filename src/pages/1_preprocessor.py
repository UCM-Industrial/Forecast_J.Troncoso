import traceback
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st
import xarray as xr

from preprocessor import (
    create_cyclical_features,
    create_map,
    display_both,  # noqa: F401
    display_climate_data,  # noqa: F401
    display_dataarray,
    display_mask,
    extract_regional_means,
)

warnings.filterwarnings("ignore")
# Configure page
# st.set_page_config(page_title="Geospatial Data Processor", page_icon="🗺️", layout="wide")


# Cache data loading functions
@st.cache_data
def load_shapefile(shapefile_path: str) -> gpd.GeoDataFrame:
    """Load and cache shapefile data."""
    return gpd.read_file(shapefile_path)


@st.cache_data
def load_grib_metadata(grib_path: str) -> dict:
    """Load GRIB file metadata without loading full dataset."""
    try:
        with xr.open_dataset(grib_path, engine="cfgrib") as ds:
            metadata = {
                "data_vars": list(ds.data_vars.keys()),
                "coords": list(ds.coords.keys()),
                "dims": dict(ds.dims),
                "attrs": dict(ds.attrs),
            }
    except Exception as e:
        st.error(f"Error loading GRIB metadata: {e}")
        return {}
    else:
        return metadata


@st.cache_data
def load_grib_dataset(grib_path: str) -> xr.Dataset:
    """Load and cache GRIB dataset."""
    return xr.open_dataset(grib_path, engine="cfgrib")


@st.cache_data
def process_regional_means(
    grib_path: str,
    shapefile_path: str,
    data_variable: str,
    time_coord: str | None,
    chunk_size: dict[str, int],
    column_names: str,
    output_timezone: str,
) -> pd.DataFrame:
    """Process regional means with caching."""
    ds = load_grib_dataset(grib_path)
    gdf = load_shapefile(shapefile_path)

    return extract_regional_means(
        ds,
        gdf,
        data_variable,
        time_coord,
        chunk_size,
        column_names,
        output_timezone,
    )


def validate_file_path(file_path: str, file_type: str) -> tuple[bool, str]:
    """Validate if file path exists and has correct extension."""
    if not file_path.strip():
        return False, f"Please enter a {file_type} file path"

    path = Path(file_path)
    if not path.exists():
        return False, f"File does not exist: {file_path}"

    if file_type == "GRIB" and path.suffix.lower() not in [
        ".grib",
        ".grib2",
        ".grb",
        ".grb2",
    ]:
        return False, "File should have a GRIB extension (.grib, .grib2, .grb, .grb2)"

    if file_type == "Shapefile" and path.suffix.lower() != ".shp":
        return False, "File should have a .shp extension"

    return True, ""


def render_file_input_section() -> tuple[str | None, str | None]:
    """Render file input section and return valid paths."""
    with st.sidebar:
        st.header("Data upload")

        st.subheader("🌐 GRIB File")
        grib_path = st.text_input(
            "GRIB File Path:",
            placeholder="e.g., /path/to/your/data.grib2",
            help="Enter the full path to your GRIB file",
        )

        if grib_path:
            valid, error_msg = validate_file_path(grib_path, "GRIB")
            if valid:
                st.success("✅ GRIB file found")
            else:
                st.error(error_msg)
                grib_path = None

        st.subheader("🗺️ Shapefile")
        shapefile_path = st.text_input(
            "Shapefile Path:",
            placeholder="e.g., /path/to/your/regions.shp",
            help="Enter the full path to your shapefile (.shp)",
        )

        if shapefile_path:
            valid, error_msg = validate_file_path(shapefile_path, "Shapefile")
            if valid:
                st.success("✅ Shapefile found")
            else:
                st.error(error_msg)
                shapefile_path = None

        return grib_path, shapefile_path


def render_grib_info(grib_path: str):
    """Display GRIB file information."""
    st.subheader("📊 GRIB File Information")

    try:
        metadata = load_grib_metadata(grib_path)

        if not metadata:
            return None

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Data Variables:**")
            for var in metadata["data_vars"]:
                st.write(f"• {var}")

            st.write("**Coordinates:**")
            for coord in metadata["coords"]:
                st.write(f"• {coord}")

        with col2:
            st.write("**Dimensions:**")
            for dim, size in metadata["dims"].items():
                st.write(f"• {dim}: {size}")

        return metadata["data_vars"]

    except Exception as e:
        st.error(f"Error loading GRIB file: {e}")
        return None


def render_shapefile_info(shapefile_path: str):
    """Display shapefile information."""
    st.subheader("🗺️ Shapefile Information")

    try:
        gdf = load_shapefile(shapefile_path)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Features", len(gdf))
            st.metric("CRS", str(gdf.crs))

        with col2:
            st.write("**Columns:**")
            for col in gdf.columns:
                if col != "geometry":
                    st.write(f"• {col}")

        # Preview data
        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(gdf.drop("geometry", axis=1).head())

        return gdf.columns.tolist()

    except Exception as e:
        st.error(f"Error loading shapefile: {e}")
        return None


def render_visualization_section(
    grib_path: str,
    shapefile_path: str,
    data_vars: list[str],
):
    """Render data visualization section."""
    st.header("📈 Data Visualization")

    # Variable selection
    selected_var = st.selectbox("Select data variable to visualize:", data_vars)

    if not selected_var:
        return

    # col1, col2 = st.columns(2)

    st.subheader("🗺️ Interactive Leafmap Viewer")
    if st.button("Visualize on Map", key="leafmap_plot"):
        try:
            with st.spinner("Generating interactive map..."):
                ds = load_grib_dataset(grib_path)
                gdf = load_shapefile(shapefile_path)
                da = (
                    ds[selected_var].isel(time=0)
                    if "time" in ds.dims
                    else ds[selected_var]
                )

                m = create_map()
                display_dataarray(m, da)
                display_mask(m, gdf)
                m.to_streamlit(height=600)
        except Exception as e:
            st.error(f"Error displaying map: {e}")


def render_processing_section(
    grib_path: str,
    shapefile_path: str,
    data_vars: list[str],
    shapefile_cols: list[str],
):
    """Render regional mean processing section."""
    st.header("⚙️ Regional Mean Processing")

    # Processing parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        st.session_state.data_variable = st.selectbox("Data Variable:", data_vars)
        column_names = st.selectbox(
            "Region Names Column:",
            [col for col in shapefile_cols if col != "geometry"],
            help="Column in shapefile containing region names",
        )

    with col2:
        time_coord = st.selectbox(
            "Time Coordinate:",
            ["Auto-detect", "time", "valid_time", "datetime"],
            help="Leave as 'Auto-detect' unless you have specific requirements",
        )
        if time_coord == "Auto-detect":
            time_coord = None

        output_timezone = st.selectbox(
            "Output Timezone:",
            ["America/Santiago", "UTC", "America/New_York", "Europe/London"],
        )

    with col3:
        # st.write("**Memory Optimization:**")
        lat_chunk = st.number_input(
            "Latitude Chunk Size:",
            value=50,
            min_value=10,
            max_value=200,
        )
        lon_chunk = st.number_input(
            "Longitude Chunk Size:",
            value=50,
            min_value=10,
            max_value=200,
        )

    chunk_size = {"latitude": lat_chunk, "longitude": lon_chunk}

    # Processing section
    st.subheader("🚀 Run Processing")

    if st.button("🔄 Process Regional Means", type="primary"):
        process_data(
            grib_path,
            shapefile_path,
            st.session_state.data_variable,
            time_coord,
            chunk_size,
            column_names,
            output_timezone,
        )


def process_data(
    grib_path: str,
    shapefile_path: str,
    data_variable: str,
    time_coord: str | None,
    chunk_size: dict[str, int],
    column_names: str,
    output_timezone: str,
):
    """Execute the regional mean processing."""
    try:
        with st.spinner("Processing regional means... This may take several minutes."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Loading datasets...")
            progress_bar.progress(20)

            df = process_regional_means(
                grib_path,
                shapefile_path,
                data_variable,
                time_coord,
                chunk_size,
                column_names,
                output_timezone,
            )

            progress_bar.progress(100)
            status_text.text("Processing complete!")

        st.success("✅ Regional means calculated successfully!")

        # Display results
        st.subheader("📊 Results")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.write(
                f"**Data Shape:** {df.shape[0]} time steps × {df.shape[1]} regions",
            )

            # Preview results
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10))

        with col2:
            st.write("**Summary Statistics:**")
            st.dataframe(df.describe())

        # Time series visualization
        if len(df.columns) <= 17:  # Only plot if reasonable number of regions
            st.subheader("📈 Time Series Visualization")
            fig = px.line(df, title="Regional Mean Time Series")
            fig.update_layout(xaxis_title="Time", yaxis_title=f"{data_variable}")
            st.plotly_chart(fig, use_container_width=True)

        # Download options
        render_download_section(df)

        # Optional cyclical features
        # render_cyclical_features_section(df)

    except Exception as e:
        st.error(f"❌ Processing failed: {e}")
        st.text(traceback.format_exc())
        # st.error("Please check your file paths and parameters.")


def render_download_section(df: pd.DataFrame):
    """Render download options for processed data."""
    st.subheader("💾 Download Results")

    # col1, col2, col3 = st.columns(3)

    # with col1:
    csv_data = df.to_csv().encode("utf-8")
    st.download_button(
        "Download CSV",
        csv_data,
        file_name=f"{st.session_state.data_variable}_means.csv",
        mime="text/csv",
    )

    # with col2:
    #     # Create an in-memory Excel file using BytesIO
    #     df_export = df.copy()
    #
    #     if df_export.index.tz is not None:
    #         df_export.index = df_export.index.tz_localize(None)
    #     # Remove timezone from datetime columns if present
    #     datetime_cols = df_export.select_dtypes(include=["datetime64[ns, UTC]"]).columns
    #
    #     for col in datetime_cols:
    #         df_export[col] = df_export[col].dt.tz_localize(None)
    #     excel_buffer = BytesIO()
    #     with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    #         df.to_excel(writer, sheet_name=f"{st.session_state.data_variable}_Means")
    #
    #     # Important: Reset buffer position to start for reading
    #     excel_buffer.seek(0)
    #
    #     # Provide download button
    #     st.download_button(
    #         label="Download Excel",
    #         data=excel_buffer,
    #         file_name=f"{st.session_state.data_variable}_Means.xlsx",
    #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    #     )

    # with col3:
    #     st.metric("Total Data Points", len(df) * len(df.columns))


def render_cyclical_features_section(df: pd.DataFrame):
    """Render optional cyclical features section."""
    st.subheader("🔄 Optional: Add Cyclical Features")

    with st.expander("Add cyclical time features", expanded=False):
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
                    df,
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


def main():
    """Main application function."""
    st.title("Geospatial Data Processor")
    st.markdown(
        "Process GRIB meteorological data with shapefiles to extract regional means. "
        "Upload your files via file paths and visualize the data interactively.",
    )

    # File input section
    grib_path, shapefile_path = render_file_input_section()

    if not grib_path or not shapefile_path:
        st.info(
            "Please provide valid file paths for both GRIB and shapefile to continue.",
        )
        return

    # File information sections
    col1, col2 = st.columns(2)

    with col1:
        data_vars = render_grib_info(grib_path)

    with col2:
        shapefile_cols = render_shapefile_info(shapefile_path)

    if not data_vars or not shapefile_cols:
        return

    # Add separator
    st.divider()

    # Visualization section
    render_visualization_section(grib_path, shapefile_path, data_vars)

    # Add separator
    st.divider()

    # Processing section
    render_processing_section(grib_path, shapefile_path, data_vars, shapefile_cols)


if __name__ == "__main__":
    main()
