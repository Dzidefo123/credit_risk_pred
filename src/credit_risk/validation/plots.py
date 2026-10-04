"""Static reliability and discrimination figures for validation evidence."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_validation(result, labels, selected_predictions, directory):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, (kind, variants) in zip(axes, result["reliability"].items(), strict=True):
        selected = result["selection"]["selected_methods"][kind]
        for method, diagnostic in variants.items():
            bins = [row for row in diagnostic["bins"] if row["rows"]]
            x = [row["mean_probability"] for row in bins]
            y = [row["observed_bad_rate"] for row in bins]
            line = ax.plot(
                x, y, "o-", label=method + (" (selected)" if method == selected else "")
            )[0]
            if method == selected:
                error = np.array(
                    [
                        [row["observed_bad_rate"] - row["wilson_lower"] for row in bins],
                        [row["wilson_upper"] - row["observed_bad_rate"] for row in bins],
                    ]
                )
                ax.errorbar(
                    x, y, yerr=np.maximum(error, 0), fmt="none", capsize=3, color=line.get_color()
                )
        ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
        ax.set(
            xlabel="Mean predicted probability",
            ylabel="Observed event rate",
            title=kind.replace("_", " "),
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Final holdout reliability: inherited two-year delinquency target\n"
        "Bins frozen on calibration data; bars: approximate "
        f"{result['selection']['config']['confidence_level']:.0%} row-binomial Wilson intervals",
        fontsize=11,
    )
    fig.savefig(directory / "calibration.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for kind, p in selected_predictions.items():
        fpr, tpr, _ = roc_curve(labels, p)
        precision, recall, _ = precision_recall_curve(labels, p)
        method = result["selection"]["selected_methods"][kind]
        label = f"{kind} / {method}"
        axes[0].plot(fpr, tpr, label=label)
        axes[1].plot(recall, precision, label=label)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].axhline(np.mean(labels), ls="--", color="gray", label="event prevalence")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle("Final holdout: choices locked using development data", fontsize=11)
    fig.savefig(directory / "discrimination.png", dpi=160)
    plt.close(fig)
