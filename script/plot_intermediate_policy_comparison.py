import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Plot a CarbonPATH intermediate-memory policy comparison."
    )
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    results = pd.read_csv(args.comparison_csv)
    policies = results["policy"].tolist()
    labels = [policy.replace("_", "\n") for policy in policies]
    colors = ["#9b4a3c", "#c9a227", "#25766c", "#35688f", "#674f8c"]
    x = np.arange(len(results))
    cold_objective = results.loc[results["policy"] == "cold_dram", "objective"].iloc[0]
    objective_improvement = (cold_objective - results["objective"]) / cold_objective * 100

    boundary_columns = sorted(
        column
        for column in results.columns
        if column.startswith("boundary_") and column.endswith("_intermediate_bytes")
    )
    intermediate = results[boundary_columns].sum(axis=1)
    retained = results[
        [column.replace("intermediate_bytes", "retained_bytes") for column in boundary_columns]
    ].sum(axis=1)
    forwarded = results[
        [column.replace("intermediate_bytes", "forwarded_bytes") for column in boundary_columns]
    ].sum(axis=1)
    spilled = results[
        [column.replace("intermediate_bytes", "dram_spilled_bytes") for column in boundary_columns]
    ].sum(axis=1)
    denominator = intermediate.where(intermediate != 0, 1)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#f5f1e8")
    fig.subplots_adjust(hspace=0.48, wspace=0.28, top=0.84)

    objective_ax, latency_ax, energy_ax, movement_ax = axes.flat
    objective_bars = objective_ax.bar(x, objective_improvement, color=colors)
    objective_ax.set_title("A. Objective Improvement", loc="left", weight="bold")
    objective_ax.set_ylabel("Reduction from cold DRAM (%)")
    objective_ax.bar_label(objective_bars, fmt="%.2f%%", padding=3)

    latency_bars = latency_ax.bar(x, results["latency"] / 1e3, color=colors)
    latency_ax.set_title("B. Sequence Latency", loc="left", weight="bold")
    latency_ax.set_ylabel("Latency (us)")
    latency_ax.bar_label(latency_bars, fmt="%.3f", padding=3)

    energy_bars = energy_ax.bar(x, results["energy"] / 1e6, color=colors)
    energy_ax.set_title("C. Sequence Energy", loc="left", weight="bold")
    energy_ax.set_ylabel("Energy (uJ)")
    energy_ax.bar_label(energy_bars, fmt="%.2f", padding=3)

    movement_ax.bar(x, retained / denominator * 100, color="#25766c", label="Retained")
    movement_ax.bar(
        x,
        forwarded / denominator * 100,
        bottom=retained / denominator * 100,
        color="#35688f",
        label="Forwarded",
    )
    movement_ax.bar(
        x,
        spilled / denominator * 100,
        bottom=(retained + forwarded) / denominator * 100,
        color="#9b4a3c",
        label="DRAM spilled",
    )
    movement_ax.set_title("D. Intermediate Placement", loc="left", weight="bold")
    movement_ax.set_ylabel("Intermediate bytes (%)")
    movement_ax.set_ylim(0, 108)
    movement_ax.legend(frameon=False, ncols=3, fontsize=8, loc="upper center")

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Sequential GEMM Intermediate-Memory Policies",
        x=0.07,
        y=0.96,
        ha="left",
        fontsize=19,
        weight="bold",
        color="#17363a",
    )
    fig.text(
        0.07,
        0.91,
        "Fixed architecture, shared cold-memory calibration; lower objective, latency, and energy are better",
        color="#4d5555",
    )
    fig.text(0.07, 0.02, f"Source: {args.comparison_csv}", fontsize=8, color="#666666")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
