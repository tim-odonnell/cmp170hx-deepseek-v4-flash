#!/usr/bin/env bash
# Collect every performance/thermal metric from a cowboy-clue soak run into one report,
# formatted for pasting into /home/user/ai-models/LLM-BENCHMARK-RESULTS.md.
#
# Usage: ./collect-bench-metrics.sh [rundir]
set -uo pipefail
R="${1:-/home/user/cowboy-clue benchmark/runs/cowboy-clue-cmp170hx}"
S=$(ls -t "$R"/server_*.log 2>/dev/null | head -1)
T=$(ls -t "$R"/thermal_*.csv 2>/dev/null | head -1)
[[ -f "$S" ]] || { echo "no server log in $R"; exit 1; }

echo "########## STATIC CONFIG ##########"
printf "%-22s %s\n" "GPU" "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
printf "%-22s %s\n" "GPU driver" "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
printf "%-22s %s\n" "GPU power cap" "$(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
printf "%-22s %s\n" "PCIe" "gen$(nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader) x$(nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader)"
printf "%-22s %s\n" "CPU" "$(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2;exit}')"
printf "%-22s %s\n" "RAM" "$(free -g | awk '/^Mem:/{print $2"GiB"}') @ $(sudo dmidecode -t memory 2>/dev/null | awk -F: '/Configured Memory Speed/{gsub(/ /,"",$2);print $2;exit}')"
printf "%-22s %s\n" "llama.cpp build" "$(cd /home/user/llama.cpp-portable && git log -1 --format=%h 2>/dev/null) (sm_80, CUDA 12.8)"
printf "%-22s %s\n" "engine binary" "/home/user/llama.cpp-portable/build-cuda-sm80/bin/llama-server"

echo
echo "########## LOAD ##########"
# Log timestamps are MM.SS.mmm.uuu relative to process start (e.g. 1.17.542.068 = 77.542s),
# NOT H.MM.SS.mmm -- parsing them as hours gives a ~65x overestimate.
awk '/load_model: loading model/{split($1,a,".");s=a[1]*60+a[2]+a[3]/1000}
     /llama_server: model loaded/{split($1,a,".");e=a[1]*60+a[2]+a[3]/1000;
     printf "%-22s %.1f s\n","model load time",e-s; exit}' "$S"
grep -m1 'n_ctx_slot' "$S" | grep -oE 'n_ctx_slot = [0-9]+' | awk '{printf "%-22s %s\n","context",$3}'

echo
echo "########## THROUGHPUT (all tasks in this run) ##########"
python3 - "$S" <<'PY'
import re,sys
txt=open(sys.argv[1],errors='ignore').read()
# NB: llama.cpp pads numbers with runs of spaces inside the parens -- "(  101.64 ms per
# token,     9.84 tokens per second)". The \( must be followed by \s* or nothing matches.
allev=re.findall(r'(prompt )?eval time =\s*[\d.]+ ms /\s*(\d+) tokens \(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)',txt)
dec=[(int(t),float(s)) for p,t,s in allev if not p]
pre=[(int(t),float(s)) for p,t,s in allev if p]
def rep(name,rows):
    if not rows: print(f"  {name}: none"); return
    toks=sum(t for t,_ in rows); sp=[s for _,s in rows]
    wavg=sum(t*s for t,s in rows)/toks if toks else 0
    print(f"  {name:8s} tasks {len(rows):3d}   tokens {toks:7d}   "
          f"min {min(sp):6.2f}  max {max(sp):6.2f}  mean {sum(sp)/len(sp):6.2f}  token-weighted {wavg:6.2f} tok/s")
rep("decode",dec); rep("prefill",pre)
g=re.findall(r'graphs reused =\s*(\d+)',txt)
if g: print(f"  graphs reused: total {sum(int(x) for x in g)}")
tg=[float(x) for x in re.findall(r'tg =\s*([\d.]+) t/s',txt)]
if tg: print(f"  streaming tg samples: n={len(tg)} min {min(tg):.2f} max {max(tg):.2f} mean {sum(tg)/len(tg):.2f} tok/s")

# Wall-clock split. The headline decode figure flatters an agentic workload badly:
# the loop re-sends an accumulating conversation each turn, so prefill dominates.
pt=sum(float(m) for m in re.findall(r'prompt eval time =\s*([\d.]+) ms',txt))/1000
allt=[(bool(p),float(m)) for p,m in re.findall(r'(prompt )?eval time =\s*([\d.]+) ms',txt)]
dt=sum(m for isp,m in allt if not isp)/1000   # ms -> s, same as pt above
ptok=sum(t for t,_ in pre); dtok=sum(t for t,_ in dec)
if pt+dt:
    print(f"\n  TIME SPLIT   prefill {pt:8.1f}s ({100*pt/(pt+dt):4.1f}%)   decode {dt:8.1f}s ({100*dt/(pt+dt):4.1f}%)")
    print(f"  EFFECTIVE    {(ptok+dtok)/(pt+dt):6.2f} tok/s end-to-end ({ptok+dtok} tokens / {pt+dt:.0f}s compute)")

# Speed vs context depth -- does throughput degrade as the conversation grows?
print("\n  per-task (in order; n_past shows context depth):")
tasks=re.findall(r'task (\d+) \| prompt eval time =\s*[\d.]+ ms /\s*(\d+) tokens \(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)[\s\S]{0,200}?\beval time =\s*[\d.]+ ms /\s*(\d+) tokens \(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)',txt)
print(f"    {'task':>6} {'pp_tok':>7} {'pp_t/s':>7} {'tg_tok':>7} {'tg_t/s':>7}")
for t,ppt,pps,tgt,tgs in tasks[-12:]:
    print(f"    {t:>6} {ppt:>7} {pps:>7} {tgt:>7} {tgs:>7}")
PY

echo
echo "########## THERMAL / POWER ##########"
[[ -f "$T" ]] && python3 - "$T" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
def c(n): return [float(r[n]) for r in rows if r.get(n) not in (None,'')]
def pct(v,p):
    s=sorted(v); k=(len(s)-1)*p/100; f=int(k)
    return s[f] if f+1>=len(s) else s[f]+(s[f+1]-s[f])*(k-f)
for k,l,u in [('gpu_c','GPU core','C'),('mem_c','HBM memory','C'),('cpu_c','CPU','C'),
              ('power_w','GPU power','W'),('util_pct','GPU util','%'),
              ('vram_mib','VRAM','MiB'),('fan_duty_pct','fan duty','%'),('fan1_rpm','FAN1','rpm'),('fan2_rpm','FAN2','rpm')]:
    v=c(k)
    if v: print(f"  {l:12s} min {min(v):8.1f}  p50 {pct(v,50):8.1f}  p95 {pct(v,95):8.1f}  max {max(v):8.1f}  avg {sum(v)/len(v):8.1f} {u}")
# Power on this workload is bimodal (decode floor + prefill bursts), so a mean alone
# is misleading -- report how much of the run sits in each band.
p=c('power_w')
if p:
    lo=[x for x in p if x<85]; hi=[x for x in p if x>=85]
    s=f"\n  power bands: <85W {100*len(lo)/len(p):4.1f}% of samples"
    if lo: s+=f" (avg {sum(lo)/len(lo):.1f}W)"
    s+=f"   >=85W {100*len(hi)/len(p):4.1f}%"
    if hi: s+=f" (avg {sum(hi)/len(hi):.1f}W)"
    print(s)
if rows: print(f"\n  duration {int(rows[-1]['elapsed_s'])//60}m {int(rows[-1]['elapsed_s'])%60}s, {len(rows)} samples @2s")
PY

echo
echo "########## OUTPUT ARTIFACT ##########"
for f in "$R"/*.html; do
    [[ -e "$f" ]] || { echo "  (no html produced)"; break; }
    echo "  $(basename "$f"): $(wc -c < "$f") bytes, $(wc -l < "$f") lines"
done

echo
echo "########## COMPARISON BASELINES ##########"
echo "  1x RTX PRO 4000 24GB / DDR5-3600, fp16 KV @65536 (2026-07-27): 9.37 tok/s decode"
echo "  2x RTX PRO 4000 24GB / DDR5-3600, cowboy-clue    (2026-07-31): 9.26 tok/s, 2965 s, 44,092 B / 1,211 lines"
