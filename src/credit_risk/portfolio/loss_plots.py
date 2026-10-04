"""Scenario comparisons explicitly separating forward EL and defaulted stock."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss(summary, segments, horizon_months, synthetic, directory):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    names = summary.scenario.to_list()
    x = np.arange(len(names))
    axes[0].bar(x - 0.18, summary.forward_expected_loss, width=0.36, label="Forward nondefault EL")
    axes[0].bar(
        x + 0.18,
        summary.defaulted_loss_assumption,
        width=0.36,
        label="Defaulted-stock residual loss assumption",
    )
    axes[0].set_xticks(x, names)
    axes[0].set(ylabel="Arbitrary source currency units", title="Separate loss components")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.2)
    base = segments.loc[(segments.scenario == "base") & (segments.dimension == "state")]
    labels = base.segment.str.replace("DPD_", "").str.replace("_", " ")
    axes[1].bar(labels, base.forward_expected_loss, label="Forward nondefault EL")
    axes[1].bar(
        labels,
        base.defaulted_loss_assumption,
        bottom=base.forward_expected_loss,
        label="Defaulted-stock residual loss assumption",
    )
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set(ylabel="Arbitrary source currency units", title="Base loss components by state")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.2)
    prefix = "SYNTHETIC benchmark" if synthetic else "Analytical benchmark"
    fig.suptitle(
        f"{prefix}: {horizon_months}-month model PD x assumed LGD x assumed EAD", fontsize=12
    )
    fig.savefig(directory / "expected_loss.png", dpi=160)
    plt.close(fig)
