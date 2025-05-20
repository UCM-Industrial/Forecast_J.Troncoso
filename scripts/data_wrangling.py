from collections.abc import Callable
from pathlib import Path

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
    output_file: Path | None = None,
) -> pd.DataFrame:
    """Concatenate multiple CSV files into a single dataframe and save it."""
    dfs = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    if output_file is not None:
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


# WARNING: This function is heavy; load 6-7 GB of RAM processing the data.
def _data_wrangling_to_pivot():
    """Backup snippet for datasets convertion from tsv to csv."""
    data_directory = Path().parent / "data" / "processed" / "csv_files"
    data_pattern = "*.csv"

    files_to_concat = sorted(data_directory.glob(data_pattern))

    print("Loading data")
    historic_df = concat_csv_files(
        csv_files=files_to_concat,
    )
    print("Pivoting")
    historic_pivot_df = historic_df.pivot_table(
        index=["fecha_opreal", "hora_opreal"],
        columns="central_tipo_nemotecnico",
        values="generacion_real_mwh",
        aggfunc="sum",
    )

    print(f"Saving\n{historic_pivot_df.info()}")
    historic_pivot_df.to_csv(
        data_directory.parent / "generation_historic_nemotecnico.csv",
    )

    print("Done")


if __name__ == "__main__":
    _data_wrangling_to_pivot()
