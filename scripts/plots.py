from calendar import month_abbr, month_name

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def create_ridgeline_plot(
    data: pd.DataFrame,
    category_col: str,
    value_col: str,
    n_categories: int = 15,
    figsize: tuple[int, int] = (8, 10),
    kde_color: str = "blue",
    colors: dict[str, str],
    bandwidth=0.5,
    max_value=None,
    sort_by: str = "mean",
    show_mean: bool = True,
    show_quantiles: bool = True,
    chronological_months: bool = False,
    title=None,
    value_unit: str | None = None,
):
    """Create a simple ridgeline plot showing distributions across categories."""
    # Set defaults
    # Set defaults
    if colors is None:
        darkgreen = "#9BC184"
        midgreen = "#C2D6A4"
        lowgreen = "#E7E5CB"
        colors = [lowgreen, midgreen, darkgreen, midgreen, lowgreen]

    try:
        n_categories = min(n_categories, data[category_col].nunique())
    except:
        raise ValueError

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
            color=colors[kde_color],
        )

        # Add reference line
        axs[i].axvline(global_reference, color="gray", linestyle="--", alpha=0.5)

        # Add category label
        axs[i].text(-max_value * 0.1, 0, category.upper(), ha="left", fontsize=10)

        if show_quantiles:
            # Compute quantiles (2.5%, 10%, 25%, 75%, 90%, 97.5%)
            quantiles = np.percentile(subset[value_col], [2.5, 10, 25, 75, 90, 97.5])

            # Fill space between each pair of quantiles
            for j in range(len(quantiles) - 1):
                axs[i].fill_between(
                    [quantiles[j], quantiles[j + 1]],  # lower and upper bounds
                    0,  # min y
                    0.0002,  # max y
                    color=colors[j % len(colors)],
                )

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
