import calendar

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Typechecking imports
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas._libs import NaTType
from statsmodels.tsa.seasonal import DecomposeResult


def get_season_dates(year: int) -> list[dict[str, str | pd.Timestamp | NaTType]]:
    """Give the season dates limits for the introduced year."""
    return [
        {
            "name": "Summer",
            "start": pd.Timestamp(f"{year - 1}-12-01"),
            "end": pd.Timestamp(f"{year}-02-29")
            if calendar.isleap(year)
            else pd.Timestamp(f"{year}-02-28"),
        },
        {
            "name": "Autumn",
            "start": pd.Timestamp(f"{year}-03-01"),
            "end": pd.Timestamp(f"{year}-05-31"),
        },
        {
            "name": "Winter",
            "start": pd.Timestamp(f"{year}-06-01"),
            "end": pd.Timestamp(f"{year}-08-31"),
        },
        {
            "name": "Spring",
            "start": pd.Timestamp(f"{year}-09-01"),
            "end": pd.Timestamp(f"{year}-11-30"),
        },
    ]


def draw_season(
    ax,
    start,
    end,
    season_name,
    season_colors,
    start_date,
    end_date,
) -> None:
    if start <= end_date and end >= start_date:
        clipped_start = max(start, start_date)
        clipped_end = min(end, end_date)
        ax.axvspan(
            clipped_start,
            clipped_end,
            alpha=0.3,
            color=season_colors[season_name],
        )


def plot_enhanced_stl_decomposition(
    df: pd.DataFrame,
    target_column: str,
    result: DecomposeResult,
    title="STL Decomposition",
) -> tuple[Figure, Axes]:
    """Plot trend, seasonal and residual with years and seasons."""
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid
    deseasonalized = result.observed - result.seasonal

    # Create figure and subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    # Format the date axis
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    years_fmt = mdates.DateFormatter("%Y")

    # Components to plot
    components = [
        (df[target_column], "Original", axes[0]),
        (trend, "Trend", axes[1]),
        (seasonal, "Seasonal", axes[2]),
        (residual, "Residual", axes[3]),
        (deseasonalized, "Deseasonalized", axes[4]),
    ]

    # Season colors for southern hemisphere
    season_colors = {
        "Summer": "#594141",
        "Autumn": "#59563b",
        "Winter": "#364954",
        "Spring": "#443f57",
    }

    for data, label, ax in components:
        ax.plot(data.index, data, label=label, linewidth=1.2)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.3)

        # Set up date range
        start_date = data.index.min()
        end_date = data.index.max()
        years_range = range(start_date.year, end_date.year + 1)

        # Format x-axis
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)

        # Seasonal shading
        for year in years_range:
            for season in get_season_dates(year):
                draw_season(
                    ax,
                    season["start"],
                    season["end"],
                    season["name"],
                    season_colors,
                    start_date,
                    end_date,
                )

            # December of the last year
            if year == end_date.year and end_date.month >= 12:
                draw_season(
                    ax,
                    pd.Timestamp(f"{year}-12-01"),
                    end_date,
                    "Summer",
                    season_colors,
                    start_date,
                    end_date,
                )

    # Custom legend with season overlays
    season_patches = [
        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.3)
        for color in season_colors.values()
    ]

    handles, labels = axes[0].get_legend_handles_labels()
    handles += season_patches
    labels += list(season_colors.keys())
    axes[0].legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
    )

    # Final layout and annotations
    fig.suptitle(title, fontsize=16, y=0.92)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)

    return fig, axes
