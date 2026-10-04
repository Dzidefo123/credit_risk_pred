"""Static vintage curves and transition heatmaps with explicit unknown cells."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from credit_risk.data.validation import STATES


def plot_portfolio(vintages, rolls, directory, is_synthetic):
    prefix = "SYNTHETIC demonstration" if is_synthetic else "Source portfolio"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for cohort, rows in vintages.groupby("origination_month", sort=True):
        axes[0].plot(rows.months_on_book, rows.cumulative_default_rate, label=cohort)
    axes[0].set(
        xlabel="Months on book (booking month = 0)",
        ylabel="Cumulative recorded default / original cohort",
        title="Fully observed cohorts only",
        ylim=(0, 1),
    )
    axes[0].legend(fontsize=8, ncols=2)
    axes[0].grid(alpha=0.2)
    table = vintages.pivot(index="origination_month", columns="months_on_book", values="bad_rate")
    palette = plt.get_cmap("YlOrRd").with_extremes(bad="#dddddd")
    mesh = axes[1].imshow(
        np.ma.masked_invalid(table.to_numpy()), aspect="auto", cmap=palette, vmin=0, vmax=1
    )
    axes[1].set_yticks(range(len(table)), table.index)
    ticks = list(range(0, len(table.columns), 6))
    axes[1].set_xticks(ticks, table.columns[ticks])
    axes[1].set(
        xlabel="Months on book", title="Observed snapshot bad rate: default or threshold DPD"
    )
    fig.colorbar(mesh, ax=axes[1], label="Observed-row bad rate")
    fig.suptitle(prefix + ": vintage performance (gray = unobserved/immature)", fontsize=12)
    fig.savefig(directory / "vintages.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    labels = ["Current", "1-29 DPD", "30-59 DPD", "60-89 DPD", "90+ DPD", "Default"]
    for ax, matrix, title in zip(
        axes,
        [rolls.probability_matrix, rolls.balance_probability_matrix],
        ["Account-weighted", "Origin balance-weighted"],
        strict=True,
    ):
        palette = plt.get_cmap("Blues").with_extremes(bad="#dddddd")
        mesh = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), cmap=palette, vmin=0, vmax=1)
        for i in range(len(STATES)):
            for j in range(len(STATES)):
                value = matrix.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.1%}" if np.isfinite(value) else "NA",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if value > 0.55 else "black",
                )
        ax.set_xticks(range(6), labels, rotation=35, ha="right")
        ax.set_yticks(range(6), labels)
        ax.set(xlabel="Destination at next month-end", ylabel="Origin at month-end", title=title)
        fig.colorbar(mesh, ax=ax, shrink=0.8)
    fig.suptitle(prefix + ": consecutive observed month-end transitions", fontsize=12)
    fig.savefig(directory / "roll_rates.png", dpi=160)
    plt.close(fig)
