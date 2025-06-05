import logging
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


def convert_to_csv(
    file_path: Path,
    sep: str = "\t",
    output_dir: Path | None = None,
) -> Path:
    """Convert data files to csv."""
    if not output_dir:
        output_dir = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{file_path.stem}.csv"

    file = pd.read_csv(file_path, sep=sep)
    file.to_csv(output_file, index=False)

    return output_file


def iterate_over_files(
    directory: Path,
    files_filter: str,
    mapping_function: Callable,
    **kwargs,
) -> list[Path]:
    """Process multiple files using the provided mapping function."""
    files = sorted(directory.glob(files_filter))
    results = []

    for file in files:
        print(f"Working on {file}")

        try:
            results = mapping_function(file_path=file, **kwargs)
        except Exception as e:
            print(f"Error processing '{file}: {e}'")
    return results


def concat_csv_files(
    csv_files: list[Path],
    output_file: Path | str | None = None,
) -> pd.DataFrame:
    """Concatenate multiple CSV files into a single dataframe and save it."""
    dfs = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    if output_file is not None:
        if isinstance(output_file, str):
            output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_file, index=False)

    return combined_df


def _data_convertion_to_csv():
    """Backup snippet for datasets convertion from tsv to csv."""
    output_directory = Path().parent / "data" / "processed" / "csv_files"
    data_directory = Path().parent / "data" / "raw" / "hourly_real_generation"
    iterate_over_files(
        data_directory,
        files_filter="*.tsv",
        mapping_function=convert_to_csv,
        output_dir=output_directory,
    )


# WARNING: This function is heavy; load 6 to 7 GB of RAM processing the data.
def _data_wrangling_to_pivot(query_names: list[str]):
    """Backup snippet for datasets convertion from tsv to csv."""
    data_directory = Path().parent / "data" / "processed" / "csv_files"
    data_pattern = "*.csv"

    files_to_concat = sorted(data_directory.glob(data_pattern))

    print("Loading data")
    historic_df = concat_csv_files(
        csv_files=files_to_concat,
    )
    print("Pivoting")

    if query_names:
        mask = historic_df["central_nombre"].isin(query_names)
        historic_df[mask].shape

    historic_pivot_df = historic_df.pivot_table(
        index=["fecha_opreal", "hora_opreal"],
        columns="subtipo",
        values="generacion_real_mwh",
        aggfunc="sum",
    )

    print(f"Saving\n{historic_pivot_df.info()}")
    historic_pivot_df.to_csv(
        data_directory.parent / "generacion_real_maule.csv",
    )

    print("Done")


def create_datetime_column(
    df: pd.DataFrame,
    hour_col: str = "hora_opreal",
    date_col: str = "fecha_opreal",
    invalid_hour_value: int | float = 25,
    hour_adjustment: int = -1,
) -> pd.DataFrame:
    """Filters data based on an hour column, adjusts it, and creates a new datetime column by combining a date column and the adjusted hour."""
    processed_df = df.copy()

    # Filter out rows with the specified invalid hour value
    processed_df = processed_df[processed_df[hour_col] != invalid_hour_value]

    # Adjust the hour column and ensure it's integer
    processed_df[hour_col] = processed_df[hour_col].astype(int) + hour_adjustment

    # Ensure the column is datetime type
    processed_df[date_col] = pd.to_datetime(processed_df[date_col])

    # Create the new datetime column by replacing the hour in 'date_col'
    processed_df["datetime"] = processed_df.apply(
        lambda row: row[date_col].replace(hour=row[hour_col]),
        axis=1,
    )
    processed_df = processed_df.drop(columns=[date_col, hour_col])
    if not isinstance(processed_df, pd.DataFrame):
        raise TypeError("The result is not a Pandas DataFrame.")
    return processed_df


def wrangling_plants_data(df: pd.DataFrame | Path | str, output_path: str | Path):
    # Initial aggregation
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)

    wrangling_df_alt = (
        df.groupby(["fecha_opreal", "hora_opreal", "central_nombre"])
        .agg(
            generacion_real_mwh=("generacion_real_mwh", "sum"),
            central_tipo_nemotecnico=("central_tipo_nemotecnico", "first"),
        )
        .reset_index()
    )

    # Calculate active/total counts and percentage in one go
    plant_stats_alt = (
        wrangling_df_alt.groupby(
            ["fecha_opreal", "hora_opreal", "central_tipo_nemotecnico"],
        )
        .agg(
            active_plants_count=("generacion_real_mwh", lambda x: (x > 0).sum()),
            total_plants_count=(
                "central_nombre",
                "nunique",
            ),
        )
        .reset_index()
    )

    # Calculate percentage
    plant_stats_alt["percentage_active"] = np.where(
        plant_stats_alt["total_plants_count"] > 0,
        (
            plant_stats_alt["active_plants_count"]
            / plant_stats_alt["total_plants_count"]
        ),
        0,
    )

    # Pivot the calculated stats to get the desired wide format
    final_pivoted_df_alt = plant_stats_alt.pivot_table(
        index=["fecha_opreal", "hora_opreal"],
        columns="central_tipo_nemotecnico",
        values=["active_plants_count", "total_plants_count", "percentage_active"],
        fill_value=0,
    )

    column_names = []
    for metric, tech in final_pivoted_df_alt.columns:
        if metric == "active_plants_count":
            column_names.append(f"active_plants_{tech}")
        elif metric == "total_plants_count":
            column_names.append(f"total_plants_{tech}")
        elif metric == "percentage_active":
            column_names.append(f"percentage_active_{tech}")

    final_pivoted_df_alt.columns = column_names
    result_df = final_pivoted_df_alt.reset_index()

    if output_path is not None:
        result_df.to_csv(output_path)
    return result_df


def _get_plants_data() -> None:
    csv_files_path = processed_path / "csv_files"
    output_folder_path = processed_path / "plants"

    files = sorted(csv_files_path.glob("*.csv"))

    for file in files:
        logger.debug(f"Working on {file}")

        file_name = f"{file.stem}_plants.csv"

        try:
            wrangling_plants_data(
                df=file,
                output_path=output_folder_path / file_name,
            )
            logger.info(f"File saved in {output_folder_path / file_name}")
        except Exception:
            logger.exception(f"Error processing '{file}'")


if __name__ == "__main__":
    # --- Config ---
    logger = logging.getLogger("example")
    logger.setLevel(logging.INFO)

    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(
        logging.INFO,
    )  # Set the handler's level (optional if same as logger)

    formatter = logging.Formatter("%(levelname)s: %(message)s")
    stdout.setFormatter(formatter)

    logger.addHandler(stdout)

    df_path = Path().cwd().parent / "data" / "raw" / "historic_generation_raw.csv"

    raw_path = Path().cwd().parent / "data" / "raw"
    processed_path = Path().cwd().parent / "data" / "processed"

    # --- CONFIG ---

    data_directory = processed_path / "plants"
    data_pattern = "*.csv"

    files_to_concat = sorted(data_directory.glob(data_pattern))

    logger.info("Concatenating...")
    historic_df = concat_csv_files(
        csv_files=files_to_concat,
    )
    final_df = create_datetime_column(historic_df).drop(columns="Unnamed: 0")

    file_path = processed_path / "plants_data.csv"
    final_df.to_csv(file_path, index=False)
    logger.info(f"Saved in {file_path}")
