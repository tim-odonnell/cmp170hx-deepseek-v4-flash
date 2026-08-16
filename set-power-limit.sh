#!/usr/bin/env bash
# set-power-limit.sh -- apply a power limit to every CMP 170HX in the box, found by PCI
# device ID rather than a fixed index list, so adding more cards later needs no edit here.
#
# Usage:
#   sudo ./set-power-limit.sh          # apply POWER_LIMIT (default 100) to all found cards
#   POWER_LIMIT=150 sudo ./set-power-limit.sh
set -euo pipefail

POWER_LIMIT="${POWER_LIMIT:-100}"
DEVICE_IDS="${DEVICE_IDS:-20c2 2082}"    # CMP 170HX: 20c2 = 8gb card, 2082 = 10gb card

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }

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

WATCH="$(discover_watch_list)"
[[ -n "${WATCH}" ]] || { echo "no matching GPUs found (DEVICE_IDS='${DEVICE_IDS}')" >&2; exit 2; }

echo "applying ${POWER_LIMIT}W to GPUs [${WATCH}]"
nvidia-smi -i "${WATCH}" -pl "${POWER_LIMIT}"
