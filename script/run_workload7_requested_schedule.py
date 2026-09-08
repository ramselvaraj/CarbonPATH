from pathlib import Path

import pandas as pd

from main import WORKLOAD_CONFIGS, parse_workload_entry
from script.run_full_space_convergence import run_one, summarize_runs
from script.run_workload7_sa_trend import PANEL


OUTPUT_DIR = Path("reports/workload_7_sa_trend")
SCHEDULE = "requested_4000_75650"


def main():
    calibration_path = OUTPUT_DIR / "calibration" / "workload_7" / "calibration.json"
    experiment_cache = OUTPUT_DIR / "simulation_cache.csv"
    if not calibration_path.exists() or not experiment_cache.exists():
        raise FileNotFoundError("The workload-7 trend-test calibration and cache are required")

    workload = parse_workload_entry(7, WORKLOAD_CONFIGS[7])
    result = run_one(
        output_dir=OUTPUT_DIR,
        panel=PANEL,
        run_index=0,
        schedule_name=SCHEDULE,
        workload=workload,
        calibration_path=calibration_path,
        base_cache=experiment_cache,
    )
    runs = pd.DataFrame([result])
    runs.to_csv(OUTPUT_DIR / "requested_4000_75650_run.csv", index=False)
    summary = summarize_runs(runs)
    summary.to_csv(OUTPUT_DIR / "requested_4000_75650_summary.csv", index=False)
    print(
        "Summary: "
        f"best={result['best_cost']:.12g}, "
        f"accepted={result['accepted_moves']}/{result['attempted_moves']}, "
        f"late_improvement={result['late_improvement']}"
    )


if __name__ == "__main__":
    main()
