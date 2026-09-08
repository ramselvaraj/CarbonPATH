import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def latest_file(directory, pattern):
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {directory}")
    return matches[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Plot expected and actual four-GEMM scaling results."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    metrics_path = latest_file(args.run_dir, "sa_metrics_*.csv")
    architecture_path = latest_file(args.run_dir, "best_arch_*.json")
    metrics = pd.read_csv(metrics_path).dropna(subset=["new_cost", "latency", "energy"])
    selected = metrics.loc[metrics["new_cost"].idxmin()]
    with architecture_path.open() as file:
        architecture = json.load(file)

    gemm_indices = sorted(
        int(column.split("_")[1])
        for column in metrics.columns
        if column.startswith("gemm_") and column.endswith("_latency_ns")
    )
    gemm_count = len(gemm_indices)
    first_latency = selected["gemm_1_latency_ns"]
    first_energy = selected["gemm_1_total_energy_pj"]
    latency_factor = selected["latency"] / first_latency
    energy_factor = selected["energy"] / first_energy

    dram = np.array(
        [selected[f"gemm_{index}_dram_interconnect_energy_pj"] for index in gemm_indices]
    ) / 1e6
    sram = np.array(
        [selected[f"gemm_{index}_sram_energy_pj"] for index in gemm_indices]
    ) / 1e6
    total = np.array(
        [selected[f"gemm_{index}_total_energy_pj"] for index in gemm_indices]
    ) / 1e6
    compute = total - dram - sram
    dram = np.append(dram, dram.sum())
    sram = np.append(sram, sram.sum())
    compute = np.append(compute, compute.sum())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(14, 7.5), facecolor="#f7f5ef")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.45, wspace=0.28)

    scaling_ax = fig.add_subplot(grid[0, 0])
    x = np.arange(2)
    width = 0.34
    scaling_ax.bar(
        x - width / 2,
        [gemm_count, gemm_count],
        width,
        label="Expected",
        color="#b8b2a6",
    )
    actual_bars = scaling_ax.bar(
        x + width / 2,
        [latency_factor, energy_factor],
        width,
        label="Actual",
        color="#0b6e75",
    )
    scaling_ax.set_title("A. Sequential Scaling Validation", loc="left", weight="bold")
    scaling_ax.set_ylabel("Factor relative to one GEMM")
    scaling_ax.set_xticks(x, ["Latency", "Total energy"])
    scaling_ax.set_ylim(0, gemm_count + 0.8)
    scaling_ax.grid(axis="y", alpha=0.2)
    scaling_ax.legend(frameon=False, ncols=2, loc="upper left")
    scaling_ax.bar_label(actual_bars, fmt="%.3fx", padding=3, weight="bold")

    energy_ax = fig.add_subplot(grid[1, 0])
    labels = [f"GEMM {index}" for index in gemm_indices] + ["Sequence"]
    energy_x = np.arange(len(labels))
    energy_ax.bar(energy_x, dram, color="#315c7d", label="DRAM/interconnect")
    energy_ax.bar(energy_x, sram, bottom=dram, color="#d69f3c", label="SRAM")
    energy_ax.bar(
        energy_x,
        compute,
        bottom=dram + sram,
        color="#8c4f66",
        label="Compute",
    )
    energy_ax.set_title("B. Selected-Design Energy Breakdown", loc="left", weight="bold")
    energy_ax.set_ylabel("Energy (uJ)")
    energy_ax.set_xticks(energy_x, labels)
    energy_ax.grid(axis="y", alpha=0.2)
    energy_ax.legend(frameon=False, ncols=3, fontsize=8, loc="upper left")
    for index, value in enumerate(dram + sram + compute):
        energy_ax.text(index, value + 0.6, f"{value:.2f}", ha="center", fontsize=8)

    summary_ax = fig.add_subplot(grid[:, 1])
    summary_ax.axis("off")
    chiplets = [
        (name, values)
        for name, values in architecture.items()
        if name.startswith("Chiplet_")
    ]
    architecture_lines = [
        "SELECTED ARCHITECTURE",
        "Chiplet  Node  Array      SRAM",
    ]
    for name, values in chiplets:
        architecture_lines.append(
            f"C{name.split('_')[1]:<7} {values['tech_node']:>2} nm  "
            f"{values['sys_array_size']:<9} {values['sram_buf']:>4} KB"
        )
    package = architecture["pkg"]
    mapping = architecture["WL_mapping"]["mapping"]
    architecture_lines.extend(
        [
            "",
            f"Package       {package['HI_pkg_type']}",
            f"Memory        {package['mem_pkg_conn']['mem_type'].upper()}",
            f"Protocols     {package['protocol_2.5d']} / {package['protocol_3d']}",
            f"Dataflow      {mapping['dataflow'][0].upper()}",
            f"K splitting   {'enabled' if mapping['if_splitting_k'] else 'disabled'}",
        ]
    )
    metrics_lines = [
        "SELECTED RESULT",
        f"Objective cost       {selected['new_cost']:.3f}",
        f"Sequence latency     {selected['latency'] / 1e3:.3f} us",
        f"Sequence energy      {selected['energy'] / 1e6:.3f} uJ",
        f"Area                 {selected['area']:.3f} mm2",
        f"Power                {selected['power']:.3f} W",
        f"Dollar cost          ${selected['dollar'] / 1e6:.3f} M",
        f"Embodied carbon      {selected['embCarbon'] / 1e3:.3f} tCO2e",
        f"Operational carbon   {selected['opeCarbon']:.3f} kgCO2e",
        "Carbon score weight  0",
        "",
        f"Search candidates    {len(metrics)}",
        f"Accepted moves       {metrics['move_accepted'].fillna(False).astype(bool).sum()}",
    ]
    summary_ax.text(
        0.02,
        0.98,
        "\n".join(metrics_lines),
        va="top",
        family="monospace",
        fontsize=10.5,
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "white", "edgecolor": "#d1ccc0"},
    )
    summary_ax.text(
        0.02,
        0.47,
        "\n".join(architecture_lines),
        va="top",
        family="monospace",
        fontsize=10.5,
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "white", "edgecolor": "#d1ccc0"},
    )

    fig.suptitle(
        "Four-GEMM Cold-Memory Validation",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=18,
        weight="bold",
        color="#17363a",
    )
    fig.text(
        0.055,
        0.935,
        "4 x [128, 128, 128], strict sequential execution; selected from one 45-candidate annealing run",
        fontsize=10,
        color="#4d5555",
    )
    fig.text(
        0.055,
        0.015,
        f"Source: {metrics_path} | Expected scaling = {gemm_count}x",
        fontsize=7.5,
        color="#666666",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
