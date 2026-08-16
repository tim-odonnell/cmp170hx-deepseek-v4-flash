#!/usr/bin/env bash
# cpu-temp-guard.sh -- hard ceiling on CPU temperature by capping max core frequency.
#
# WHY FREQUENCY AND NOT FANS
# --------------------------
# The BMC fans are already saturated on this box: 100% duty for a third of a sustained
# inference run, >=95% for two thirds, with FAN3 at 5600 rpm. There is no airflow left to
# ask for. The only remaining lever that reliably lowers CPU temperature is reducing the
# power the CPU is allowed to draw, and on amd-pstate-epp the accessible way to do that is
# scaling_max_freq.
#
# HOW IT WORKS
# ------------
# Closed loop on k10temp (Tctl and per-CCD Tccd, read straight from hwmon -- far faster and
# finer-grained than IPMI's TEMP_CPU). Above ACT_HIGH it steps the frequency cap down; below
# ACT_LOW it steps back up. Hysteresis between the two stops it oscillating.
#
# It tracks the HOTTEST CCD, not the package average. On this chip the dies can sit 40C
# apart (CCD1 88C while CCD2 idles at 48C) because the inference threads pack onto one die --
# a package-average controller would happily let the hot die cook.
#
# Restores the hardware maximum on every exit path, including crash (systemd ExecStopPost).
#
# Usage:
#   sudo ./cpu-temp-guard.sh              # run in foreground
#   sudo ./cpu-temp-guard.sh --status     # show temps and current cap, change nothing
#   sudo ./cpu-temp-guard.sh --restore    # remove any cap and exit
set -uo pipefail

CEILING="${CEILING:-90}"        # the hard limit you never want exceeded
# v2 (2026-08-06): v1 used ACT_HIGH=87 / 2s poll / symmetric 200MHz steps and BREACHED FIVE
# TIMES, peaking at 95C. Root cause: this CPU jumps ~10C between 2s samples (logged 82C ->
# 92C in one poll), so 3C of margin is nowhere near enough, and stepping back up 200MHz on
# every cool sample kept returning to full clock just in time for the next spike.
# Fix: much bigger margin, faster poll, and a deliberately ASYMMETRIC response --
# drop hard and immediately, recover slowly and only after sustained cool.
ACT_HIGH="${ACT_HIGH:-80}"      # 10C of margin, sized to the observed spike magnitude
ACT_LOW="${ACT_LOW:-72}"
STEP_MHZ="${STEP_MHZ:-400}"     # down-step
UP_STEP_MHZ="${UP_STEP_MHZ:-100}"   # up-step, deliberately 4x smaller
COOL_SAMPLES="${COOL_SAMPLES:-15}"  # consecutive cool polls required before raising at all
PANIC_STEP_MHZ="${PANIC_STEP_MHZ:-1200}"  # ceiling actually breached: back off hard
FLOOR_MHZ="${FLOOR_MHZ:-2800}"  # never throttle below this; the box must stay usable
POLL="${POLL:-1}"
LOG="${LOG:-/var/log/cpu-temp-guard.log}"

[[ $EUID -eq 0 ]] || { echo "must run as root (writes scaling_max_freq)" >&2; exit 2; }

HW=/sys/class/hwmon
POLICIES=(/sys/devices/system/cpu/cpufreq/policy*)
HWMAX=$(cat "${POLICIES[0]}/cpuinfo_max_freq")
HWMIN=$(cat "${POLICIES[0]}/cpuinfo_min_freq")

log(){ local m="[$(date '+%F %T')] $*"; echo "$m"; echo "$m" >> "$LOG" 2>/dev/null; }

k10dir() {
    for d in "$HW"/hwmon*; do
        [[ "$(cat "$d/name" 2>/dev/null)" == "k10temp" ]] && { echo "$d"; return; }
    done
}
K10=$(k10dir)
[[ -n "$K10" ]] || { echo "k10temp hwmon not found" >&2; exit 2; }

hottest() {   # max of Tctl and every Tccd, in whole degrees C
    local max=0 v l
    for f in "$K10"/temp*_input; do
        l=$(cat "${f%_input}_label" 2>/dev/null || echo "")
        [[ "$l" == Tctl || "$l" == Tccd* ]] || continue
        v=$(( $(cat "$f") / 1000 ))
        (( v > max )) && max=$v
    done
    echo "$max"
}
detail() {
    for f in "$K10"/temp*_input; do
        local l; l=$(cat "${f%_input}_label" 2>/dev/null || echo "?")
        printf "%s=%dC " "$l" "$(( $(cat "$f") / 1000 ))"
    done
}
set_cap() {   # $1 = kHz
    local khz=$1
    (( khz > HWMAX )) && khz=$HWMAX
    (( khz < HWMIN )) && khz=$HWMIN
    for p in "${POLICIES[@]}"; do echo "$khz" > "$p/scaling_max_freq" 2>/dev/null; done
}
cur_cap(){ cat "${POLICIES[0]}/scaling_max_freq"; }

if [[ "${1:-}" == "--restore" ]]; then
    set_cap "$HWMAX"; log "restored full frequency ($((HWMAX/1000)) MHz)"; exit 0
fi
if [[ "${1:-}" == "--status" ]]; then
    echo "  sensors : $(detail)"
    echo "  hottest : $(hottest) C   (ceiling ${CEILING}C, act ${ACT_HIGH}/${ACT_LOW})"
    echo "  freq cap: $(( $(cur_cap)/1000 )) MHz of $((HWMAX/1000)) MHz max"
    exit 0
fi

RESTORED=0
restore_once(){ (( RESTORED )) && return; RESTORED=1; set_cap "$HWMAX"; log "restored full frequency"; }
trap 'restore_once; exit 0' EXIT INT TERM HUP

log "starting: ceiling ${CEILING}C  act ${ACT_HIGH}/${ACT_LOW}C  step ${STEP_MHZ}MHz  floor ${FLOOR_MHZ}MHz"
CAP=$HWMAX; set_cap "$CAP"
LAST_LOG=0; COOL=0
while true; do
    D=$(detail); T=$(hottest)
    NEW=$CAP
    if (( T > CEILING )); then
        NEW=$(( CAP - PANIC_STEP_MHZ*1000 )); COOL=0
        log "BREACH ${T}C > ${CEILING}C  [$D] -- panic drop"
    elif (( T >= ACT_HIGH )); then
        NEW=$(( CAP - STEP_MHZ*1000 )); COOL=0
    elif (( T <= ACT_LOW )); then
        COOL=$(( COOL + 1 ))
        (( COOL >= COOL_SAMPLES )) && { NEW=$(( CAP + UP_STEP_MHZ*1000 )); COOL=0; }
    else
        COOL=0
    fi
    (( NEW < FLOOR_MHZ*1000 )) && NEW=$(( FLOOR_MHZ*1000 ))
    (( NEW > HWMAX )) && NEW=$HWMAX
    if (( NEW != CAP )); then
        CAP=$NEW; set_cap "$CAP"
        if (( T >= ACT_HIGH || SECONDS-LAST_LOG > 60 )); then
            log "[$D] hottest ${T}C -> cap $((CAP/1000)) MHz"; LAST_LOG=$SECONDS
        fi
    fi
    sleep "$POLL"
done
