#!/bin/bash
# Waits for the soak to finish; runs tool-tests.py only if it PASSED.
cd "$(dirname "$0")/.."
until grep -q -E "SOAK PASS|SOAK FAIL" logs/soak.out; do sleep 30; done
grep -q "SOAK PASS" logs/soak.out || { echo "soak did not pass -- tool tests NOT run"; exit 1; }
echo "soak passed $(date '+%T'), starting tool tests"
python3 scripts/tool-tests.py --port 8099 --model dsv4v --out results/phase6b-tool-tests.json
echo "TOOL TESTS DONE $(date '+%T')"
