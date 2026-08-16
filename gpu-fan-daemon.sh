#!/usr/bin/env bash
# gpu-fan-daemon.sh -- drive ASRock Rack BMC fans from the hottest CMP 170HX in the box.
#
# WHY THIS EXISTS
# ---------------
# The BMC's built-in fan curve reads TEMP_CPU / TEMP_MB / TEMP_CARD_SIDE. It has no idea
# the GPU exists. On this machine that is a real hazard: an AI workload drives the GPU to
# 80C while the CPU idles, the BMC sees nothing worth reacting to, and holds the fans at
# 30% while a passively-cooled 250W card cooks.
#
# This daemon closes that loop: read GPU temperature with nvidia-smi, translate it to a
# fan duty, push it to the BMC with ipmitool.
#
# MULTI-GPU
# ---------
# Watches every CMP 170HX in the box, not a fixed index. GPUs are found by PCI device ID
# (WATCH_INDICES=auto, the default) so adding more cards later needs no edit here -- just
# a daemon restart (a reboot does this automatically). Duty is driven by the SINGLE
# HOTTEST reading across all watched cards: since GPU_CURVE/MEM_CURVE are monotonic
# non-decreasing, max(duty(T_i)) == duty(max(T_i)), so taking the hottest temperature
# first and running the curve once is equivalent to computing per-card duties and maxing
# them, for less code. Chassis fans are shared, so the hottest card sets the pace for
# everyone regardless.
#
# SAFETY DESIGN
# -------------
# Taking manual fan control means the BMC stops protecting the machine, so:
#
#  1. Duty is the MAXIMUM of the hottest-GPU curve, hottest-HBM curve and a CPU curve.
#     Manual control can therefore never cool less than automatic mode would have --
#     including for the CPU, whose temperature is still honoured.
#  2. A hard floor (FLOOR_DUTY, default 30%) matches what the BMC already runs at idle.
#     Fans never go below what they were doing before this daemon existed.
#  3. Automatic mode is restored on EVERY exit path: normal stop, Ctrl-C, SIGTERM,
#     crash, or the sensor read failing repeatedly. It fails BACK TO THE BMC, never
#     silent. systemd also re-asserts this via ExecStopPost.
#  4. If nvidia-smi fails MAX_FAILURES times in a row (driver crash, a card fell off the
#     bus), the daemon restores automatic mode and exits rather than holding a stale duty.
#
# BMC NOTE: ASPEED BMCs need one of two incompatible command sets, chosen by AST_GEN
# below (default ast2500). This board (ROMED8-2T) is AST2500: 0x3a 0xd6/0xd7/0xd8/0xda.
# AST2600 boards (e.g. the AM5D4ID-2T) use 0x3a 0xd0 with subcommands instead -- set
# AST_GEN=ast2600 there. Confirmed 2026-08-13 against ASRock Rack's own TSDQA-72 FAQ
# ("How to adjust fan speed with IPMItool commands for ASRockRack AST2500 and AST2600
# Series MotherBoards") and this board's own raw command responses.
#
# Usage:
#   sudo ./gpu-fan-daemon.sh              # run in foreground
#   sudo ./gpu-fan-daemon.sh --once       # apply one update and exit
#   sudo ./gpu-fan-daemon.sh --status     # show sensors and computed duty, change nothing
#   sudo ./gpu-fan-daemon.sh --restore    # force fans back to automatic and exit
set -uo pipefail

WATCH_INDICES="${WATCH_INDICES:-auto}"   # "auto" = discover by DEVICE_IDS, or e.g. "0,1,2,3"
DEVICE_IDS="${DEVICE_IDS:-20c2 2082}"    # CMP 170HX: 20c2 = 8gb card, 2082 = 10gb card
POLL="${POLL:-5}"
FLOOR_DUTY="${FLOOR_DUTY:-30}"
HYSTERESIS="${HYSTERESIS:-3}"      # ignore changes smaller than this, stops hunting
MAX_FAILURES="${MAX_FAILURES:-3}"
LOG="${LOG:-/var/log/gpu-fan-daemon.log}"

# Curves: "temperature:duty" breakpoints, linearly interpolated between them.
# Deliberately aggressive at the top -- with no fan of its own, this card has very
# little thermal mass to buy reaction time.
GPU_CURVE="${GPU_CURVE:-40:30 50:40 60:55 70:75 78:90 82:100}"
MEM_CURVE="${MEM_CURVE:-55:30 65:45 75:65 82:85 88:100}"
CPU_CURVE="${CPU_CURVE:-50:30 65:50 75:70 85:100}"

command -v ipmitool  >/dev/null || { echo "ipmitool not found" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
[[ $EUID -eq 0 ]] || { echo "must run as root (ipmitool needs /dev/ipmi0)" >&2; exit 2; }

AST_GEN="${AST_GEN:-ast2500}"        # ast2500 (this board) or ast2600 (AM5D4ID-2T)

log() { local m="[$(date '+%F %T')] $*"; echo "${m}"; [[ -w "$(dirname "${LOG}")" ]] && echo "${m}" >> "${LOG}" 2>/dev/null; }

# ---------- GPU discovery ----------
discover_watch_list() {                # -> comma-separated nvidia-smi indices, empty if none
    local wanted found=()
    wanted=" $(echo "${DEVICE_IDS}" | tr '[:lower:]' '[:upper:]') "
    while IFS=',' read -r idx pciid; do
        idx="${idx// /}"; pciid="${pciid// /}"
        local did="${pciid:2:4}"     # "0x20C210DE" -> "20C2"
        [[ "${wanted}" == *" ${did} "* ]] && found+=("${idx}")
    done < <(nvidia-smi --query-gpu=index,pci.device_id --format=csv,noheader,nounits 2>/dev/null)
    local IFS=','; echo "${found[*]}"
}

if [[ "${WATCH_INDICES}" == auto ]]; then
    WATCH="$(discover_watch_list)"
else
    WATCH="${WATCH_INDICES}"
fi
[[ -n "${WATCH}" ]] || { echo "no matching GPUs found (DEVICE_IDS='${DEVICE_IDS}')" >&2; exit 2; }

# ---------- BMC primitives ----------
bmc_manual() {
    if [[ "${AST_GEN}" == ast2600 ]]; then
        ipmitool raw 0x3a 0xd0 0x11 0x2 0x2 0x2 0x2 0x2 0x2 0x2 0x2 \
                                    0x2 0x2 0x2 0x2 0x2 0x2 0x2 0x2 >/dev/null 2>&1
    else
        ipmitool raw 0x3a 0xd8 0x1 0x1 0x1 0x1 0x1 0x1 0x1 0x1 \
                                    0x1 0x1 0x1 0x1 0x1 0x1 0x1 0x1 >/dev/null 2>&1
    fi
}
bmc_auto() {
    if [[ "${AST_GEN}" == ast2600 ]]; then
        ipmitool raw 0x3a 0xd0 0x11 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0 \
                                    0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0 >/dev/null 2>&1
    else
        ipmitool raw 0x3a 0xd8 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0 \
                                    0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0 >/dev/null 2>&1
    fi
}
bmc_set_duty() {                     # $1 = duty percent (decimal)
    local h; h=$(printf '0x%02x' "$1"); local d=()
    for _ in $(seq 16); do d+=("${h}"); done
    if [[ "${AST_GEN}" == ast2600 ]]; then
        ipmitool raw 0x3a 0xd0 0x0e "${d[@]}" >/dev/null 2>&1
    else
        ipmitool raw 0x3a 0xd6 "${d[@]}" >/dev/null 2>&1
    fi
}

# ---------- sensors ----------
gpu_temps() {                        # -> "hottest_core hottest_mem" across all watched GPUs; empty on failure
    nvidia-smi -i "${WATCH}" --query-gpu=temperature.gpu,temperature.memory \
        --format=csv,noheader,nounits 2>/dev/null |
        awk -F', *' '{g=($1+0); m=($2+0); if(g>maxg) maxg=g; if(m>maxm) maxm=m}
                     END{if(NR>0) print maxg, maxm}'
}
gpu_temps_detail() {                 # -> one "GPUidx: core=.. mem=.." line per watched GPU
    nvidia-smi -i "${WATCH}" --query-gpu=index,temperature.gpu,temperature.memory \
        --format=csv,noheader,nounits 2>/dev/null |
        awk -F', *' '{printf "  GPU%s: core=%sC mem=%sC\n", $1, $2, $3}'
}
cpu_temp() {                         # -> integer C ; 0 if unavailable
    # NB: the reading line looks like "Sensor Reading : 47 (+/- 0) degrees C".
    # Stripping all non-digits merges the 47 and the tolerance 0 into "470", which
    # pins the fans at 100% forever. Take the first whitespace-delimited field only.
    ipmitool sdr get "CPU Temp" 2>/dev/null |
        awk -F: '/Sensor Reading/{split($2,a," "); print a[1]+0; exit}'
}

# ---------- curve interpolation ----------
curve_duty() {                       # $1 = curve string, $2 = temperature
    local curve="$1" t="$2"
    awk -v t="$t" -v c="$curve" 'BEGIN{
        n=split(c, pts, " ");
        split(pts[1], f, ":"); lo_t=f[1]; lo_d=f[2];
        if (t <= lo_t) { print lo_d; exit }
        for (i=2; i<=n; i++) {
            split(pts[i], f, ":"); hi_t=f[1]; hi_d=f[2];
            if (t <= hi_t) {
                if (hi_t == lo_t) { print hi_d; exit }
                printf "%d\n", lo_d + (hi_d - lo_d) * (t - lo_t) / (hi_t - lo_t); exit
            }
            lo_t=hi_t; lo_d=hi_d;
        }
        print hi_d;
    }'
}

compute_duty() {                     # $1=hottest gpu $2=hottest mem $3=cpu -> duty
    local dg dm dc best
    dg=$(curve_duty "${GPU_CURVE}" "$1")
    dm=$(curve_duty "${MEM_CURVE}" "$2")
    dc=$(curve_duty "${CPU_CURVE}" "$3")
    best=$FLOOR_DUTY
    for v in "$dg" "$dm" "$dc"; do (( v > best )) && best=$v; done
    (( best > 100 )) && best=100
    echo "$best"
}

# ---------- modes that change nothing ----------
if [[ "${1:-}" == "--restore" ]]; then
    bmc_auto; log "fan control restored to BMC AUTOMATIC mode"; exit 0
fi
if [[ "${1:-}" == "--status" ]]; then
    printf "watching GPUs: %s\n" "${WATCH}"
    gpu_temps_detail
    read -r G M <<< "$(gpu_temps)"; C=$(cpu_temp)
    printf "hottest core : %s C  -> %s%%\n" "${G:-?}" "$(curve_duty "${GPU_CURVE}" "${G:-0}")"
    printf "hottest mem  : %s C  -> %s%%\n" "${M:-?}" "$(curve_duty "${MEM_CURVE}" "${M:-0}")"
    printf "CPU          : %s C  -> %s%%\n" "${C:-?}" "$(curve_duty "${CPU_CURVE}" "${C:-0}")"
    printf "floor        : %s%%\n" "${FLOOR_DUTY}"
    printf "WOULD SET    : %s%%\n" "$(compute_duty "${G:-0}" "${M:-0}" "${C:-0}")"
    if [[ "${AST_GEN}" == ast2600 ]]; then
        printf "\nBMC mode : %s\n" "$(ipmitool raw 0x3a 0xd0 0x12 2>/dev/null | awk '{print $1}')  (00=auto 02=manual)"
        printf "BMC duty : %s\n" "$(ipmitool raw 0x3a 0xd0 0x0f 2>/dev/null | awk '{print $1}') hex"
    else
        printf "\nBMC duty setpoints (0x3a 0xd7): %s\n" "$(ipmitool raw 0x3a 0xd7 2>/dev/null)"
        printf "BMC current duty   (0x3a 0xda): %s\n" "$(ipmitool raw 0x3a 0xda 2>/dev/null)"
    fi
    ipmitool sdr type Fan 2>/dev/null | head -4 | awk -F'|' '{gsub(/^ +| +$/,"",$1); print "  "$1": "$5}'
    exit 0
fi

if [[ "${1:-}" == "--once" ]]; then
    read -r G M <<< "$(gpu_temps)"
    [[ -n "${G:-}" ]] || { echo "cannot read GPU temperature" >&2; exit 1; }
    C=$(cpu_temp); C=${C:-0}
    D=$(compute_duty "${G}" "${M:-0}" "${C}")
    bmc_manual && bmc_set_duty "${D}"
    log "--once: watching=${WATCH} hottest_gpu=${G}C hottest_mem=${M:-?}C cpu=${C}C -> duty ${D}% (fans left in MANUAL mode)"
    echo "set ${D}% -- fans remain in MANUAL mode; run --restore to hand back to the BMC"
    exit 0
fi

# ---------- the daemon proper ----------
RESTORED=0
restore_once() {
    (( RESTORED )) && return
    RESTORED=1
    bmc_auto
    log "restored BMC AUTOMATIC fan mode"
}
trap 'restore_once; exit 0' EXIT INT TERM HUP

log "starting: watching GPUs [${WATCH}] floor=${FLOOR_DUTY}% poll=${POLL}s hysteresis=${HYSTERESIS}%"
log "GPU curve: ${GPU_CURVE}"
bmc_manual || { log "FATAL: could not switch BMC to manual mode"; exit 1; }

LAST=-1; FAILS=0
while true; do
    read -r G M <<< "$(gpu_temps)"
    if [[ -z "${G:-}" || "${G}" == "0" ]]; then
        FAILS=$((FAILS + 1))
        log "WARNING: GPU sensor read failed (${FAILS}/${MAX_FAILURES})"
        if (( FAILS >= MAX_FAILURES )); then
            log "FATAL: GPUs unreadable ${MAX_FAILURES}x -- handing fans back to the BMC"
            exit 1     # trap restores automatic mode
        fi
        sleep "${POLL}"; continue
    fi
    FAILS=0
    C=$(cpu_temp); C=${C:-0}
    DUTY=$(compute_duty "${G}" "${M:-0}" "${C}")

    DIFF=$(( DUTY > LAST ? DUTY - LAST : LAST - DUTY ))
    if (( LAST < 0 || DIFF >= HYSTERESIS || DUTY >= 100 )); then
        bmc_set_duty "${DUTY}"
        log "hottest_gpu=${G}C hottest_mem=${M:-?}C cpu=${C}C -> duty ${DUTY}%"
        LAST=${DUTY}
    fi
    sleep "${POLL}"
done
