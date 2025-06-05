from calendar import month_abbr, month_name

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_ridgeline_plot(
    data: pd.DataFrame,
    category_col: str,
    value_col: str,
    n_categories: int = 15,
    figsize: tuple[int, int] = (8, 10),
    kde_color: str = "blue",
    bandwidth=0.5,
    max_value=None,
    sort_by: str = "mean",
    show_mean: bool = True,
    chronological_months: bool = False,
    title=None,
    value_unit: str | None = None,
):
    """Create a simple ridgeline plot showing distributions across categories."""
    if max_value is None:
        max_value = data[value_col].max() * 1.1

    # Handle month names chronologically if specified
    if chronological_months:
        # Create a dictionary to map month names to their numerical order
        months_order = {m: i for i, m in enumerate(month_name[1:], 1)}
        months_order.update({m: i for i, m in enumerate(month_abbr[1:], 1)})

        # Check if categories are month names
        unique_cats = data[category_col].unique()
        valid_months = [cat for cat in unique_cats if cat in months_order]

        if valid_months:
            # Sort by month order
            custom_order = sorted(valid_months, key=lambda m: months_order.get(m, 0))

            # Limit to n_categories if needed
            if len(custom_order) > n_categories:
                custom_order = custom_order[:n_categories]

    # Get categories based on sorting method or custom order
    if custom_order:
        top_categories = custom_order
    elif sort_by == "count":
        stats = data.groupby(category_col).size().sort_values(ascending=False)
        top_categories = stats.head(n_categories).index.tolist()
    else:  # mean or median
        agg_func = "mean" if sort_by == "mean" else "median"
        stats = data.groupby(category_col)[value_col].agg(agg_func).sort_values()
        top_categories = stats.head(n_categories).index.tolist()

    # Filter data to keep only selected categories
    filtered_data = data[data[category_col].isin(top_categories)]

    # Create figure and axes
    fig, axs = plt.subplots(nrows=n_categories, ncols=1, figsize=figsize)
    axs = axs.flatten()  # Ensure axes are in a flat array

    # Global reference
    global_reference = data[value_col].mean()

    # Plot each category
    for i, category in enumerate(top_categories):
        # Get data for this category
        subset = filtered_data[filtered_data[category_col] == category]

        # Plot density curve
        sns.kdeplot(
            subset[value_col],
            fill=True,
            bw_adjust=bandwidth,
            ax=axs[i],
            color=kde_color,
        )

        # Add reference line
        axs[i].axvline(global_reference, color="gray", linestyle="--", alpha=0.5)

        # Add category label
        axs[i].text(-max_value * 0.1, 0, category.upper(), ha="left", fontsize=10)

        if show_mean:
            # Add mean value marker
            mean = subset[value_col].mean()
            axs[i].scatter([mean], [0.0001], color="black", s=10)

        # Set plot limits
        axs[i].set_xlim(0, max_value)
        axs[i].set_ylim(0, 0.001)

        # Remove axes
        axs[i].set_axis_off()

        # Add x-axis scale for the last plot
        if i == n_categories - 1:
            step = max_value / 4
            ticks = [step * j for j in range(1, 5)]
            for tick in ticks:
                axs[i].text(
                    tick,
                    -0.0005,
                    f"{int(tick)} {value_unit}",
                    ha="center",
                    fontsize=10,
                )

    if title:
        plt.suptitle(title, fontsize=16, y=0.98)

    plt.tight_layout()
    return fig, axs


def plot_periodogram(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    periods: np.ndarray,
    top_periods_info: dict,
    fs_unit: str = "hour",
    title: str = "Periodogram Analysis",
    max_display_period_val: float
    | None = None,  # New parameter to control max period on plot
) -> None:
    """Plots the periodogram in two forms: spectrum vs frequency and spectrum vs period.

    Parameters:
    - frequencies: array of frequencies.
    - spectrum: array of power spectral density.
    - periods: array of periods.
    - top_periods_info: dictionary with the most significant periods.
    - fs_unit: The time unit implied by fs (e.g., "hour", "day").
    - title: title for the overall plot.
    - max_display_period_val: The maximum period value to display on the x-axis of the
                              period plot. If None, it's determined automatically to
                              include the longest annotated common period.
    """
    plt.figure(figsize=(12, 10))

    # Subplot 1: Spectrum vs. Frequency
    plt.subplot(2, 1, 1)
    plt.semilogy(frequencies, spectrum)
    plt.title(title + " (Frequency Domain)")
    plt.xlabel(f"Frequency [cycles per {fs_unit}]")
    plt.ylabel("Power Spectral Density")
    plt.grid(True)

    # Subplot 2: Spectrum vs. Period
    plt.subplot(2, 1, 2)

    # Define common periods for annotation (base unit is hours)
    common_periods_hours = {
        "1 day": 24,
        "1 week": 24 * 7,
        "1 month": 24 * 30,  # Approx. 30 days
        "3 months": 24 * 90,  # Approx. 90 days
        "6 months": 24 * 182,  # Approx. half a year
        "1 year": 24 * 365,  # Approx. 1 year (covers 12 months)
        "2 years": 24 * 365 * 2,  # Approx. 2 years
        "5 years": 24 * 365 * 5,  # Approx. 5 years
    }

    # Convert common_periods target values to the unit of 'periods'
    common_periods_display = {}
    if fs_unit == "hour":
        common_periods_display = common_periods_hours
    elif fs_unit == "day":
        common_periods_display = {
            label: p_hrs / 24 for label, p_hrs in common_periods_hours.items()
        }
    # Add more conversions if fs_unit can be other things (e.g., "minute", "second")
    # For simplicity, if fs_unit is not hour or day, we assume common_periods_hours values are directly comparable
    elif common_periods_hours:  # check if not empty
        # This case might need specific handling if periods are not in hours or days originally.
        # For now, let's assume if not hour/day, the user wants to use the raw values from common_periods_hours
        # or should provide common_periods in the correct unit directly.
        # As a fallback, we can try to use hour values if fs_unit is not recognized for conversion
        print(
            f"Warning: fs_unit '{fs_unit}' not explicitly handled for common period conversion. Assuming common_periods_hours values are appropriate or adjust logic.",
        )
        common_periods_display = (
            common_periods_hours  # Fallback, may need adjustment by user
        )

    # Determine the maximum period to display on the plot
    current_max_display_period = max_display_period_val
    if current_max_display_period is None:
        if common_periods_display:
            # Default to a bit larger than the longest annotated common period
            current_max_display_period = max(common_periods_display.values()) * 1.1
        elif len(periods) > 0:
            current_max_display_period = max(periods)  # Or show all available periods
        else:
            current_max_display_period = (
                1  # Default if no periods and no common periods
            )

    # Filter periods: include only positive periods up to current_max_display_period
    mask = (periods > 0) & (periods <= current_max_display_period)
    filtered_periods = periods[mask]
    filtered_spectrum = spectrum[mask]

    if len(filtered_periods) > 0:
        plt.semilogy(filtered_periods, filtered_spectrum)

        if common_periods_display:
            max_spec_val_in_view = (
                np.max(filtered_spectrum) if len(filtered_spectrum) > 0 else 1
            )
            min_spec_val_in_view = (
                np.min(filtered_spectrum) if len(filtered_spectrum) > 0 else 0.1
            )
            text_y_position = np.exp(
                np.log(min_spec_val_in_view)
                + (np.log(max_spec_val_in_view) - np.log(min_spec_val_in_view)) * 0.75,
            )

            for label, period_val in common_periods_display.items():
                # Check if the common period is within the plotted range
                if period_val <= max(filtered_periods) and period_val >= min(
                    filtered_periods,
                ):
                    plt.axvline(x=period_val, color="r", linestyle="--", alpha=0.5)
                    plt.text(
                        period_val,
                        text_y_position,  # Position text dynamically
                        label,
                        rotation=90,
                        alpha=0.7,
                        verticalalignment="bottom",
                        horizontalalignment="right"
                        if period_val > np.median(filtered_periods)
                        else "left",
                    )

        plt.title("Periodogram (Period Domain)")
        plt.xlabel(f"Period [{fs_unit}s]")
        plt.ylabel("Power Spectral Density")
        plt.xscale("log")
        # Ensure x-axis limits cover the intended range, especially if filtered_periods is sparse
        if len(filtered_periods) > 0:
            plt.xlim(min(filtered_periods), current_max_display_period)

        plt.grid(True)
    else:
        plt.text(
            0.5,
            0.5,
            f"No period data to display up to {current_max_display_period:.2f} {fs_unit}s.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
        plt.title("Periodogram (Period Domain) - No Data in Range")
        plt.xlabel(f"Period [{fs_unit}s]")
        plt.ylabel("Power Spectral Density")
        if (
            current_max_display_period > 0
        ):  # Avoid issues with log scale if max period is 0 or negative
            plt.xscale("log")  # Still set log scale for consistency
            plt.xlim(
                0.1,
                current_max_display_period,
            )  # Provide a sensible default range

    plt.tight_layout()
    plt.show()

    # Optionally, print the top periods information
    print("Top significant periods:")
    if top_periods_info:
        for period_str, info in top_periods_info.items():
            print(
                f"- {period_str} (value: {info.get('period_value', 'N/A'):.2f} {fs_unit}s) with power {info.get('power', 'N/A'):.2e}",
            )
    else:
        print("No top period information available.")
