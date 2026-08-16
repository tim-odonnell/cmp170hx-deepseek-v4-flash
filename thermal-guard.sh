#!/usr/bin/env bash
# thermal-guard.sh -- run a GPU workload with an automatic thermal abort.
#
# Kills the workload the moment the card gets too hot, instead of trusting NVIDIA's
# built-in protection. The hardware only throttles at 95C and cuts power at 98C, which
# is far too late for a PASSIVELY COOLED CMP 170HX: by then the leakage-driven runaway
# feedback loop is already underway.
#
# Watches three things:
#   1. GPU core temperature       (default abort 80C, spec max operating is 85C)
#   2. HBM memory temperature     (default abort 88C, spec max operating is 95C)
#   3. RATE OF RISE               (default abort on +8C within a 15s window)
#
# The third is the important one. A steady 75C is fine; 45C climbing to 75C in twelve
# seconds means nothing is carrying the heat away and it will not stop at 80C. Runaway
# is detectable by its slope well before it is detectable by its value.
#
# Usage:
#   ./thermal-guard.sh <command> [args...]
#   ./thermal-guard.sh --selftest              # prove the abort path works, no load
#
# Environment overrides:
#   ABORT_GPU=80  ABORT_MEM=88  WARN_GPU=70  RISE_DELTA=8  RISE_WINDOW=15
#   POLL=2  GPU_INDEX=0  LOGDIR=./thermal-logs
#
# Examples:
#   ./thermal-guard.sh python3 my_benchmark.py
#   ABORT_GPU=70 ./thermal-guard.sh llama-bench -m model.gguf
set -uo pipefail

GPU_INDEX="${GPU_INDEX:-0}"
ABORT_GPU="${ABORT_GPU:-80}"
ABORT_MEM="${ABORT_MEM:-88}"
WARN_GPU="${WARN_GPU:-70}"
RISE_DELTA="${RISE_DELTA:-8}"
RISE_WINDOW="${RISE_WINDOW:-15}"
POLL="${POLL:-2}"
LOGDIR="${LOGDIR:-$(dirname "$0")/thermal-logs}"

RED=$'\e[31m'; YEL=$'\e[33m'; GRN=$'\e[32m'; DIM=$'\e[2m'; RST=$'\e[0m'
[[ -t 1 ]] || { RED=""; YEL=""; GRN=""; DIM=""; RST=""; }

die() { echo "${RED}ERROR:${RST} $*" >&2; exit 2; }
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"

read_temps() {
    # Emits "gpu mem power" -- missing sensors become 0 rather than breaking arithmetic.
    local raw
    raw="$(nvidia-smi -i "${GPU_INDEX}" \
        --query-gpu=temperature.gpu,temperature.memory,power.draw \
        --format=csv,noheader,nounits 2>/dev/null)" || return 1
    echo "${raw}" | awk -F', *' '{
        g=($1+0); m=($2+0); p=($3+0);
        print g, m, p
    }'
}

read_temps >/dev/null 2>&1 || die "cannot read GPU ${GPU_INDEX} -- is the driver loaded?"

# --- self-test: thresholds set below current temperature must abort immediately ---
if [[ "${1:-}" == "--selftest" ]]; then
    cur=$(read_temps | awk '{print $1}')
    echo "Self-test: GPU currently ${cur}C. Setting abort threshold to $((cur - 2))C."
    echo "A correct guard kills the workload within ~${POLL}s."
    ABORT_GPU=$((cur - 2)) exec "$0" sleep 600
fi

[[ $# -gt 0 ]] || die "no command given. Usage: $0 <command> [args...]"

mkdir -p "${LOGDIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
CSV="${LOGDIR}/thermal_${STAMP}.csv"
echo "elapsed_s,gpu_c,mem_c,power_w" > "${CSV}"

echo "${DIM}--------------------------------------------------------------${RST}"
echo " thermal-guard: ${GRN}$*${RST}"
echo " abort:  GPU >= ${ABORT_GPU}C | MEM >= ${ABORT_MEM}C | +${RISE_DELTA}C in ${RISE_WINDOW}s"
echo " warn:   GPU >= ${WARN_GPU}C          poll: ${POLL}s"
echo " log:    ${CSV}"
echo "${DIM}--------------------------------------------------------------${RST}"

# Run the workload in its own process group so we can take down grandchildren too --
# a bare kill on the direct child would orphan anything it spawned, and the orphan
# would happily keep heating the card.
setsid "$@" &
CHILD=$!
PGID=$(ps -o pgid= "${CHILD}" 2>/dev/null | tr -d ' ')
[[ -n "${PGID}" ]] || PGID="${CHILD}"

PEAK_GPU=0; PEAK_MEM=0; PEAK_PWR=0
ABORTED=""; WARNED=0; START=$(date +%s)
HIST_T=(); HIST_V=()

stop_workload() {
    local why="$1"
    ABORTED="${why}"
    echo ""
    echo "${RED}=============================================================${RST}"
    echo "${RED} THERMAL ABORT: ${why}${RST}"
    echo "${RED}=============================================================${RST}"
    kill -TERM "-${PGID}" 2>/dev/null
    for _ in $(seq 1 25); do kill -0 "${CHILD}" 2>/dev/null || break; sleep 0.2; done
    if kill -0 "${CHILD}" 2>/dev/null; then
        echo "${RED} did not exit on SIGTERM -- sending SIGKILL${RST}"
        kill -KILL "-${PGID}" 2>/dev/null
    fi
}

cleanup() { [[ -z "${ABORTED}" ]] && kill -TERM "-${PGID}" 2>/dev/null; }
trap 'echo ""; echo "interrupted -- stopping workload"; cleanup; exit 130' INT TERM

while kill -0 "${CHILD}" 2>/dev/null; do
    if ! vals="$(read_temps)"; then sleep "${POLL}"; continue; fi
    read -r G M P <<< "${vals}"
    NOW=$(date +%s); EL=$((NOW - START))
    echo "${EL},${G},${M},${P}" >> "${CSV}"

    (( G > PEAK_GPU )) && PEAK_GPU=$G
    (( M > PEAK_MEM )) && PEAK_MEM=$M
    (( ${P%.*} > PEAK_PWR )) && PEAK_PWR=${P%.*}

    # rolling window for slope detection
    HIST_T+=("${NOW}"); HIST_V+=("${G}")
    while (( ${#HIST_T[@]} > 1 && NOW - HIST_T[0] > RISE_WINDOW )); do
        HIST_T=("${HIST_T[@]:1}"); HIST_V=("${HIST_V[@]:1}")
    done
    OLDEST="${HIST_V[0]}"; RISE=$(( G - OLDEST ))

    printf "\r [%4ds] GPU %3dC   MEM %3dC   %6sW   rise %+dC/%ds   " \
           "${EL}" "${G}" "${M}" "${P}" "${RISE}" "${RISE_WINDOW}"

    if (( G >= ABORT_GPU )); then
        stop_workload "GPU core hit ${G}C (limit ${ABORT_GPU}C)"; break
    fi
    if (( M > 0 && M >= ABORT_MEM )); then
        stop_workload "HBM memory hit ${M}C (limit ${ABORT_MEM}C)"; break
    fi
    if (( ${#HIST_T[@]} > 1 && NOW - HIST_T[0] >= RISE_WINDOW && RISE >= RISE_DELTA )); then
        stop_workload "runaway slope: +${RISE}C in ${RISE_WINDOW}s (limit +${RISE_DELTA}C)"; break
    fi
    if (( G >= WARN_GPU && WARNED == 0 )); then
        echo ""; echo "${YEL} WARNING: GPU ${G}C -- approaching abort at ${ABORT_GPU}C${RST}"
        WARNED=1
    fi
    sleep "${POLL}"
done

wait "${CHILD}" 2>/dev/null; RC=$?
echo ""
echo "${DIM}--------------------------------------------------------------${RST}"
if [[ -n "${ABORTED}" ]]; then
    echo " result:  ${RED}ABORTED${RST} -- ${ABORTED}"
else
    echo " result:  ${GRN}completed${RST} (exit ${RC})"
fi
echo " peak:    GPU ${PEAK_GPU}C   MEM ${PEAK_MEM}C   ${PEAK_PWR}W"
echo " ran:     $(( $(date +%s) - START ))s"
echo " log:     ${CSV}"
echo "${DIM}--------------------------------------------------------------${RST}"
[[ -n "${ABORTED}" ]] && exit 1
exit "${RC}"
