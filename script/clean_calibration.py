#!/usr/bin/env python3
import os
import glob
import sys

# Add ../ (src/) to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SRC_DIR)


def main():
    # Choose base path depending on knob
    
    base_pattern = "cfg/calibration/calibration_*"

    # Expand for both JSON and CSV
    patterns = [f"{base_pattern}.json", f"{base_pattern}.csv"]

    files_to_remove = []
    for pattern in patterns:
        files_to_remove.extend(glob.glob(pattern))

    if not files_to_remove:
        print("[INFO] No calibration files found.")
        return

    for f in files_to_remove:
        try:
            os.remove(f)
            print(f"[REMOVED] {f}")
        except Exception as e:
            print(f"[ERROR] Could not remove {f}: {e}")

if __name__ == "__main__":
    main()

