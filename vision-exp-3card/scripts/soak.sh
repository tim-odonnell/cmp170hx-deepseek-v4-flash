#!/bin/bash
# Phase 6 soak: mixed traffic against the running Vision-Exp server for DURATION seconds.
# Each loop: full vision-gate suite; a concurrent burst (3 text + 3 image requests at once);
# every 3rd loop a fresh 250k-token deep prefill; every 10th loop a ~630k deep request with the
# gate suite running concurrently. Checks /health and dmesg Xid count after every loop and
# stops at the first failure (so the failure state is preserved for diagnosis).
# usage: nohup ./soak.sh [duration_s=9000] [port=8099] [model=dsv4v] > ../logs/soak.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
DUR=${1:-9000} PORT=${2:-8099} MODEL=${3:-dsv4v}
R=results/soak-$(date +%Y%m%d-%H%M%S); mkdir -p "$R"
XID0=$(sudo dmesg | grep -ci xid); START=$SECONDS; loop=0
echo "SOAK START $(date '+%F %T') duration=${DUR}s results=$R xid_baseline=$XID0"
fail() { echo "SOAK FAIL loop $loop $(date '+%F %T'): $*"; docker logs --tail 80 dsv4-vision-3card > "$R/fail-container-tail.log" 2>&1; exit 1; }
check() {
  curl -sf -m 10 "http://127.0.0.1:$PORT/health" >/dev/null || fail "server unhealthy"
  local x; x=$(sudo dmesg | grep -ci xid); (( x > XID0 )) && fail "new Xid events: $((x - XID0))"
}
burst() {
  python3 - "$PORT" "$MODEL" <<'EOF'
import sys, json, base64, io, concurrent.futures as cf, urllib.request
from PIL import Image
port, model = sys.argv[1], sys.argv[2]
im = Image.new("RGB", (448, 448), (30, 120, 200)); buf = io.BytesIO(); im.save(buf, "PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
def req(i):
    c = f"Write 250 words about pipeline parallelism, variant {i}." if i < 3 else \
        [{"type": "text", "text": f"Describe this image's colour in detail ({i})."}, {"type": "image_url", "image_url": {"url": url}}]
    b = {"model": model, "messages": [{"role": "user", "content": c}], "max_tokens": 300, "temperature": 0.7}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as x: return json.loads(x.read())["usage"]["completion_tokens"]
with cf.ThreadPoolExecutor(6) as ex: toks = list(ex.map(req, range(6)))
print("burst ok", toks)
EOF
}
while (( SECONDS - START < DUR )); do
  loop=$((loop + 1)); t0=$SECONDS
  g=$(python3 scripts/vision-gates.py --port "$PORT" --model "$MODEL" --out "$R/gates-$loop.json" 2>&1 | grep SUMMARY)
  echo "$g" | grep -q false && fail "gate failure: $g"
  check
  b=$(burst 2>&1 | tail -1); [[ $b == burst\ ok* ]] || fail "burst: $b"
  check
  if (( loop % 10 == 0 )); then
    python3 scripts/depth-probe.py --port "$PORT" --model "$MODEL" --tokens 630000 --out "$R/deep630k-$loop.json" > "$R/deep630k-$loop.log" 2>&1 &
    dp=$!; sleep 60
    while kill -0 $dp 2>/dev/null; do python3 scripts/vision-gates.py --port "$PORT" --model "$MODEL" --only G1,G7,G9 2>&1 | grep -q '"G1": true' || fail "gates during 630k"; done
    wait $dp; grep -q '"error": null' "$R/deep630k-$loop.json" || fail "630k deep request failed"
    d="deep630k ok"
  elif (( loop % 3 == 0 )); then
    python3 scripts/depth-probe.py --port "$PORT" --model "$MODEL" --tokens 250000 --out "$R/deep250k-$loop.json" > /dev/null 2>&1 \
      && grep -q '"error": null' "$R/deep250k-$loop.json" || fail "250k deep request failed"
    d="deep250k ok"
  else d="-"; fi
  check
  echo "loop $loop ok $(date '+%T') ($((SECONDS - t0))s) | gates all pass | $b | $d"
done
echo "SOAK PASS $(date '+%F %T') loops=$loop elapsed=$((SECONDS - START))s new_xid=$(( $(sudo dmesg | grep -ci xid) - XID0 ))"
