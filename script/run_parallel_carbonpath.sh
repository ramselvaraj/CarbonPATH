#!/bin/bash
set -u  # treat unset vars as errors (optional but safer)

# ========== CONFIG ==========
RUN_NAME="run_name_carbon_path"
ITERATION=1
WORKLOADS=(1 2 3 4 5 6)


COST_PROFILES=("t1" "t2" "t3" "t4" )
CHECK_INTERVAL=300        # 5 min
STALL_THRESHOLD=1200       # 20 min (no log growth)

# ========== LOGGING ==========
SUCCESS_LOG="job_success.log"
STALL_LOG="job_stalled.log"
: > "$SUCCESS_LOG"
: > "$STALL_LOG"

# ========== ARRAYS ==========
PIDS=()
JOBS=()
CMDS=()
declare -A LAST_MOD_TIMES   # key: JOB_NAME -> last mtime (epoch)
declare -A STALL_COUNTERS   # key: JOB_NAME -> stalled seconds
declare -A STATES           # key: JOB_NAME -> running|finished|stalled

# ========== HELPERS ==========
safe_kill() {
  local pid="$1"
  if [[ "$pid" =~ ^[0-9]+$ ]] && (( pid > 1 )) && kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid"
  fi
}

log_mtime() {
  local file="$1"
  [[ -f "$file" ]] && stat -c %Y "$file" 2>/dev/null || echo 0
}

echo "[MASTER] Starting job launcher @ $(date)"

# ========== LAUNCH JOBS ==========
for wl in "${WORKLOADS[@]}"; do
  for cost in "${COST_PROFILES[@]}"; do
    JOB="wl${wl}_${cost}_i${ITERATION}"
    CACHE_FILE="cfg/static_cache/static_cache_${wl}.csv"
    LOG_FILE="log_${JOB}.log"

    # ✅ Check if CACHE_FILE exists; if not, copy
    if [[ ! -f "$CACHE_FILE" ]]; then
      echo "[INFO] $CACHE_FILE not found. Creating from cfg/static_cache/static_cache.csv"
      cp cfg/static_cache/static_cache.csv "$CACHE_FILE"
    else
      echo "[INFO] Using existing $CACHE_FILE"
    fi

    CMD="python -m main  --workload ${wl} \
        --iteration ${ITERATION} --run_name ${RUN_NAME} \
        --cache_file ${CACHE_FILE} --cost_profile ${cost}"

    echo "[MASTER] Launching $JOB with command:"
    echo "nohup $CMD > $LOG_FILE 2>&1 &"

    nohup $CMD > "$LOG_FILE" 2>&1 &
    PID=$!

    PIDS+=("$PID")
    JOBS+=("$JOB")
    CMDS+=("$CMD")
    LAST_MOD_TIMES["$JOB"]="$(date +%s)"
    STALL_COUNTERS["$JOB"]=0
    STATES["$JOB"]="running"

    echo "[MASTER] $JOB launched (PID $PID, Log: $LOG_FILE)"
  
  done
done

echo "[MASTER] Monitoring every $((CHECK_INTERVAL/60)) min..."

# ========== MONITOR LOOP ==========
while true; do
  REMAINING=0
  echo "========== Job Status @ $(date) =========="

  for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    JOB="${JOBS[$i]}"
    CMD="${CMDS[$i]}"
    LOG_FILE="log_${JOB}.log"

    # Skip already finished/stalled
    [[ "${STATES[$JOB]}" != "running" ]] && continue

    if [[ "$PID" =~ ^[0-9]+$ ]] && (( PID > 1 )) && kill -0 "$PID" 2>/dev/null; then
      ((REMAINING++))

      # progress via log mtime
      CUR_MTIME="$(log_mtime "$LOG_FILE")"
      LAST_TIME="${LAST_MOD_TIMES[$JOB]}"

      if (( CUR_MTIME > LAST_TIME )); then
        echo "[RUNNING]  $JOB (PID $PID) - log updated"
        LAST_MOD_TIMES["$JOB"]="$CUR_MTIME"
        STALL_COUNTERS["$JOB"]=0
      else
        STALL_COUNTERS["$JOB"]=$(( STALL_COUNTERS["$JOB"] + CHECK_INTERVAL ))
        if (( STALL_COUNTERS["$JOB"] >= STALL_THRESHOLD )); then
          echo "[STALLED]  $JOB (PID $PID) - killing after $((STALL_COUNTERS[$JOB]/60)) min"
          safe_kill "$PID"
          echo "$CMD" >> "$STALL_LOG"
          STATES["$JOB"]="stalled"
          PIDS[$i]=-1
        else
          echo "[RUNNING]  $JOB (PID $PID) - no recent log update"
        fi
      fi

    else
      # Process died; mark finished if not already classified
      echo "[FINISHED] $JOB (PID $PID)"
      echo "$CMD" >> "$SUCCESS_LOG"
      STATES["$JOB"]="finished"
      PIDS[$i]=-1
    fi
  done

  if (( REMAINING == 0 )); then
    echo "[MASTER] ✅ All jobs complete @ $(date)"
    break
  fi

  echo "[MASTER] Sleeping $((CHECK_INTERVAL/60)) minutes..."
  sleep "$CHECK_INTERVAL"
done

echo "[MASTER] Summary:"
echo "Successes:"
cat "$SUCCESS_LOG"
echo
echo "Stalled:"
cat "$STALL_LOG"
