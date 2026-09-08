from pathlib import Path

import pandas as pd

from main import WORKLOAD_CONFIGS, parse_workload_entry
from script.run_full_space_convergence import run_one, summarize_runs
from script.run_workload7_sa_trend import PANEL


OUTPUT_DIR = Path("reports/workload_7_sa_trend")
SCHEDULE = "thorough_2400"
RUNS = 12


def main():
    calibration_path = OUTPUT_DIR / "calibration" / "workload_7" / "calibration.json"
    experiment_cache = OUTPUT_DIR / "simulation_cache.csv"
    if not calibration_path.exists() or not experiment_cache.exists():
        raise FileNotFoundError("The workload-7 trend-test calibration and cache are required")

    workload = parse_workload_entry(7, WORKLOAD_CONFIGS[7])
    rows = []
    for run_index in range(RUNS):
        print(f"[{run_index + 1}/{RUNS}] workload 7 {SCHEDULE}", flush=True)
        rows.append(
            run_one(
                output_dir=OUTPUT_DIR,
                panel=PANEL,
                run_index=run_index,
                schedule_name=SCHEDULE,
                workload=workload,
                calibration_path=calibration_path,
                base_cache=experiment_cache,
            )
        )
        pd.DataFrame(rows).to_csv(
            OUTPUT_DIR / "supplemental_control_2400_runs.csv", index=False
        )

    summary = summarize_runs(pd.DataFrame(rows))
    summary.to_csv(
        OUTPUT_DIR / "supplemental_control_2400_summary.csv", index=False
    )
    print(
        "Summary: "
        f"best={summary.iloc[0]['minimum_cost']:.12g}, "
        f"median={summary.iloc[0]['median_cost']:.12g}, "
        f"maximum={summary.iloc[0]['maximum_cost']:.12g}"
    )


if __name__ == "__main__":
    main()
