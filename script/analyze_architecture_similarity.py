"""Architecture similarity analysis for CarbonPATH run sets.

Each "set" is a labelled group of run directories produced by one annealing
schedule/experiment. A run directory is any directory containing both a
``best_arch_*.json`` file and an ``sa_metrics_*.csv`` file.

The script writes a Markdown report plus CSV tables covering:

- objective summary per set (best, worst, relative spread)
- canonical architecture fingerprint groups
- per-field architecture agreement
- pairwise normalized feature distance matrices
- distance-to-best and its correlation with objective

The feature extraction is workload-independent, so the same command works for
any workload by pointing ``--set`` at the relevant run directories.

Example:
    python -m script.analyze_architecture_similarity \
        --workload 6 \
        --set 50=cfg/gen_arch/wl6_10iteration_workload6_50move_convergence_t1_run* \
        --set 75650=cfg/gen_arch/wl6_1iteration_workload6_75650move_convergence_r*_t1 \
        --set 100000=cfg/gen_arch/wl6_1iteration_workload6_100k_convergence_r*_t1 \
        --output reports/architecture_similarity/workload6
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from system.utils.ArchitectureIdentity import (
    canonical_architecture_fingerprint,
    canonicalize_chiplet_labels,
)

MISSING = object()


@dataclass
class RunRecord:
    set_label: str
    run: str
    objective: float
    moves: int
    fingerprint: str
    features: dict
    features_flat: str = ""
    distance_to_best: float = float("nan")

    def __post_init__(self):
        self.features_flat = json.dumps(self.features, sort_keys=True)


@dataclass
class RunSet:
    label: str
    pattern: str
    records: list = field(default_factory=list)


def architecture_features(architecture: dict) -> dict:
    """Flatten a canonicalized architecture into comparable scalar fields."""
    canonical = canonicalize_chiplet_labels(architecture)
    chiplets = sorted(
        (key for key in canonical if key.startswith("Chiplet_")),
        key=lambda key: int(key.split("_", 1)[1]),
    )

    features = {"n_chiplets": len(chiplets)}
    for index, name in enumerate(chiplets, start=1):
        chiplet = canonical[name]
        features[f"chip{index}.tech_node"] = chiplet.get("tech_node")
        features[f"chip{index}.sys_array_size"] = chiplet.get("sys_array_size")
        features[f"chip{index}.sram_buf"] = chiplet.get("sram_buf")

    package = canonical.get("pkg", {})
    features["pkg.HI_pkg_type"] = package.get("HI_pkg_type")
    features["pkg.protocol_3d"] = package.get("protocol_3d")
    features["pkg.protocol_2.5d"] = package.get("protocol_2.5d")

    memory = package.get("mem_pkg_conn", {})
    if isinstance(memory, dict):
        features["pkg.mem_type"] = memory.get("mem_type")
        for index, name in enumerate(chiplets, start=1):
            features[f"pkg.mem_chip{index}"] = memory.get(name)

    connections = package.get("inter_pkg_conn", [])
    if isinstance(connections, list):
        for index, connection in enumerate(connections, start=1):
            if isinstance(connection, dict):
                features[f"pkg.conn{index}"] = "{}->{}:{}@{}".format(
                    connection.get("from"),
                    connection.get("to"),
                    connection.get("connection_type"),
                    connection.get("loc"),
                )

    mapping = canonical.get("WL_mapping", {}).get("mapping", {})
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            features[f"wl.{key}"] = value

    return features


def feature_distance(left: dict, right: dict) -> float:
    """Normalized Hamming distance over the union of field keys."""
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    differences = sum(
        1 for key in keys if left.get(key, MISSING) != right.get(key, MISSING)
    )
    return differences / len(keys)


def _hashable_value(value):
    try:
        hash(value)
        return value
    except TypeError:
        return json.dumps(value, sort_keys=True)


def field_agreement(records: list[RunRecord]) -> pd.DataFrame:
    keys = sorted(set().union(*(set(record.features) for record in records)))
    rows = []
    for key in keys:
        values = Counter(
            _hashable_value(record.features[key])
            for record in records
            if key in record.features
        )
        modal_value, modal_count = values.most_common(1)[0]
        rows.append(
            {
                "field": key,
                "present": f"{sum(values.values())}/{len(records)}",
                "modal_value": modal_value,
                "agreement": modal_count / len(records),
                "distribution": "; ".join(
                    f"{value}={count}" for value, count in values.most_common()
                ),
            }
        )
    return pd.DataFrame(rows)


def load_run_set(label: str, pattern: str) -> RunSet:
    run_set = RunSet(label=label, pattern=pattern)
    for directory in sorted(glob.glob(pattern)):
        path = Path(directory)
        architectures = sorted(path.glob("best_arch_*.json"))
        metrics = sorted(path.glob("sa_metrics_*.csv"))
        if not architectures or not metrics:
            continue
        with architectures[-1].open(encoding="utf-8") as handle:
            architecture = json.load(handle)
        frame = pd.read_csv(metrics[-1])
        run_set.records.append(
            RunRecord(
                set_label=label,
                run=path.name,
                objective=float(frame["best_cost_after"].iloc[-1]),
                moves=int(frame["SA_run_loop"].max()),
                fingerprint=canonical_architecture_fingerprint(architecture),
                features=architecture_features(architecture),
            )
        )
    if not run_set.records:
        raise SystemExit(f"No runs matched set '{label}' with pattern: {pattern}")
    return run_set


def spearman_correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return float("nan")

    def ranks(values):
        order = sorted(range(len(values)), key=lambda index: values[index])
        ranks_out = [0.0] * len(values)
        for rank, index in enumerate(order, start=1):
            ranks_out[index] = rank
        return ranks_out

    left_ranks = ranks(left)
    right_ranks = ranks(right)
    mean_left = sum(left_ranks) / len(left_ranks)
    mean_right = sum(right_ranks) / len(right_ranks)
    covariance = sum(
        (a - mean_left) * (b - mean_right)
        for a, b in zip(left_ranks, right_ranks)
    )
    spread_left = math.sqrt(sum((a - mean_left) ** 2 for a in left_ranks))
    spread_right = math.sqrt(sum((b - mean_right) ** 2 for b in right_ranks))
    if spread_left == 0 or spread_right == 0:
        return float("nan")
    return covariance / (spread_left * spread_right)


def decorate_distances(run_set: RunSet) -> pd.DataFrame:
    """Annotate records with distance-to-best and return a distance matrix."""
    best = min(run_set.records, key=lambda record: record.objective)
    for record in run_set.records:
        record.distance_to_best = feature_distance(record.features, best.features)
    runs = [record.run for record in run_set.records]
    matrix = pd.DataFrame(index=runs, columns=runs, dtype=float)
    for left in run_set.records:
        for right in run_set.records:
            matrix.loc[left.run, right.run] = feature_distance(
                left.features, right.features
            )
    return matrix


def render_report(workload: int, run_sets: list[RunSet], output: Path) -> str:
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Architecture Similarity Report — Workload {workload}",
        "",
        "Generated by `script/analyze_architecture_similarity.py`.",
        "",
    ]

    lines += [
        "## Run sets",
        "",
        "| Set | Runs | Pattern |",
        "|---|---:|---|",
    ]
    for run_set in run_sets:
        lines.append(
            f"| `{run_set.label}` | {len(run_set.records)} | `{run_set.pattern}` |"
        )
    lines.append("")

    lines += [
        "## Objective summary",
        "",
        "| Set | Runs | Moves | Best | Worst | Relative spread | Distinct canonical fingerprints | Largest fingerprint group |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run_set in run_sets:
        objectives = [record.objective for record in run_set.records]
        fingerprints = Counter(record.fingerprint for record in run_set.records)
        moves = {record.moves for record in run_set.records}
        moves_label = str(next(iter(moves))) if len(moves) == 1 else "/".join(
            str(value) for value in sorted(moves)
        )
        spread = (max(objectives) - min(objectives)) / abs(min(objectives))
        lines.append(
            f"| `{run_set.label}` | {len(objectives)} | {moves_label} | "
            f"{min(objectives):.6f} | {max(objectives):.6f} | {spread:.2%} | "
            f"{len(fingerprints)} | {fingerprints.most_common(1)[0][1]} |"
        )
    lines.append("")

    for run_set in run_sets:
        records = run_set.records
        matrix = decorate_distances(run_set)
        lines += [f"## Set `{run_set.label}`", ""]

        lines += ["### Per-run results", ""]
        lines += [
            "| Run | Objective | Moves | Canonical fingerprint | Distance to best |",
            "|---|---:|---:|---|---:|",
        ]
        for record in sorted(records, key=lambda item: item.objective):
            lines.append(
                f"| `{record.run}` | {record.objective:.12g} | {record.moves} | "
                f"`{record.fingerprint}` | {record.distance_to_best:.3f} |"
            )
        lines.append("")

        group_counts = Counter(record.features_flat for record in records)
        lines += [
            "### Exact architecture groups",
            "",
            f"{len(group_counts)} distinct architecture(s) by full canonical feature set.",
            "",
            "| Group size | Runs |",
            "|---:|---|",
        ]
        for signature, count in group_counts.most_common():
            members = [
                record.run for record in records if record.features_flat == signature
            ]
            lines.append(f"| {count} | {', '.join(f'`{run}`' for run in members)} |")
        lines.append("")

        agreement = field_agreement(records)
        varying = agreement[agreement["agreement"] < 1.0].sort_values(
            "agreement", ascending=False
        )
        lines += ["### Per-field agreement", ""]
        if varying.empty:
            lines.append("All fields are identical across runs.")
        else:
            lines += [
                "| Field | Present | Modal value | Agreement | Distribution |",
                "|---|---|---:|---:|---|",
            ]
            for row in varying.itertuples():
                lines.append(
                    f"| `{row.field}` | {row.present} | `{row.modal_value}` | "
                    f"{row.agreement:.0%} | {row.distribution} |"
                )
        lines.append("")

        ordered = [record.run for record in sorted(records, key=lambda r: r.objective)]
        lines += ["### Pairwise feature distance", ""]
        header = "| Run | " + " | ".join(ordered) + " |"
        divider = "|" + "---|" * (len(ordered) + 1)
        lines += [header, divider]
        for run in ordered:
            cells = " | ".join(f"{matrix.loc[run, other]:.2f}" for other in ordered)
            lines.append(f"| `{run}` | {cells} |")
        lines.append("")

        distances = [record.distance_to_best for record in records]
        objectives = [record.objective for record in records]
        correlation = spearman_correlation(distances, objectives)
        mean_distance = statistics.mean(distances)
        lines += [
            "### Distance summary",
            "",
            f"- Mean distance to best: {mean_distance:.3f}",
            f"- Max distance to best: {max(distances):.3f}",
            f"- Spearman correlation (distance to best vs objective): {correlation:.3f}",
            "",
        ]

        csv_frame = pd.DataFrame(
            [
                {
                    "set": record.set_label,
                    "run": record.run,
                    "objective": record.objective,
                    "moves": record.moves,
                    "fingerprint": record.fingerprint,
                    "distance_to_best": record.distance_to_best,
                    **record.features,
                }
                for record in records
            ]
        )
        csv_frame.to_csv(output / f"runs_{run_set.label}.csv", index=False)
        matrix.to_csv(output / f"distance_matrix_{run_set.label}.csv")
        agreement.to_csv(output / f"field_agreement_{run_set.label}.csv", index=False)

    report = "\n".join(lines) + "\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    return report


def parse_set(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--set must be LABEL=PATTERN")
    label, pattern = value.split("=", 1)
    if not label or not pattern:
        raise argparse.ArgumentTypeError("--set must be LABEL=PATTERN")
    return label, pattern


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=int, required=True)
    parser.add_argument(
        "--set",
        dest="sets",
        type=parse_set,
        action="append",
        required=True,
        help="Labelled run-directory glob, e.g. 100000=cfg/gen_arch/wl6_*_t1",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    run_sets = [load_run_set(label, pattern) for label, pattern in args.sets]
    render_report(args.workload, run_sets, output)

    all_runs = pd.concat(
        [pd.read_csv(output / f"runs_{run_set.label}.csv") for run_set in run_sets],
        ignore_index=True,
    )
    all_runs.to_csv(output / "runs.csv", index=False)
    print(f"Wrote {output / 'report.md'}")
    print(f"Wrote {output / 'runs.csv'}")


if __name__ == "__main__":
    main()
