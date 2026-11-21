#!/bin/bash
# Fault matrix for w-link's tcp_host collectives (COLLECTIVES-DESIGN.md section 7.1): both ranks on THIS box over
# loopback (no USB4 traffic), both on its GPU. Run under the box's GPU lock:
#   flock -w 1800 /var/tmp/halo-gpu.lock tools/dsv41-hip/coll_fault_matrix.sh (COLL_CHECK=path to tf-dsv41-coll-check)
# Each case prints the two ranks' result lines and a PASS/FAIL against the expected errors; the last two are negative
# controls of the word-for-word check. For the proxy transport:
#   TF_COLL_KIND=proxy EXTRA_ARGS="--kit DIR" tools/dsv41-hip/coll_fault_matrix.sh
# For the RDMA proxy (rdma_proxy.zig; both ranks' QPs on this box's one HCA, loopback):
#   TF_COLL_KIND=rdma TF_RDMA_DEVICES=<ibdev> EXTRA_ARGS="--kit DIR" tools/dsv41-hip/coll_fault_matrix.sh
# The ranks run on cores 5 and 6 and their proxy threads on 7 and 4: two spinning threads on one core wait for the
# scheduler tick (4 ms at HZ=250) at every handoff.
set -u
cd "${COLL_DIR:-$(dirname "$0")}"
BIN=${COLL_CHECK:-./tf-dsv41-coll-check}
PORT=29800
export TF_COLL_PROGRESS_MS=2000
export TF_COLL_GPU_WAIT_S=2  # a peer's arrival limit is this + the progress deadline: 4 s, under the 5 s stall and stop
pass=0; fail=0
run_case() {  # name, expect0, expect1, args0, args1, [external action]
  local name=$1 e0=$2 e1=$3 a0=$4 a1=$5 act=${6:-}
  PORT=$((PORT + 40))
  env TF_COLL_PROXY_CPU=7 ${ENV1:-} taskset -c 5 $BIN 1 127.0.0.1 $PORT $a1 ${EXTRA_ARGS:-} > /tmp/w-link-f1.txt 2>&1 &
  local p1=$!
  sleep 0.3
  env TF_COLL_PROXY_CPU=4 ${ENV0:-} taskset -c 6 $BIN 0 127.0.0.1 $PORT $a0 ${EXTRA_ARGS:-} > /tmp/w-link-f0.txt 2>&1 &
  local p0=$!
  if [ -n "$act" ]; then eval "$act"; fi
  wait $p0; local rc0=$?
  wait $p1; local rc1=$?
  local r0=$(grep -o "fault-result:.*" /tmp/w-link-f0.txt | tail -1)
  local r1=$(grep -o "fault-result:.*" /tmp/w-link-f1.txt | tail -1)
  local g0=$(echo "$r0" | sed -n 's/.*error=\([A-Za-z]*\).*/\1/p')
  local g1=$(echo "$r1" | sed -n 's/.*error=\([A-Za-z]*\).*/\1/p')
  [ -z "$g0" ] && g0="exit$rc0" ; [ -z "$g1" ] && g1="exit$rc1"
  if [[ "$g0" =~ ^($e0)$ ]] && [[ "$g1" =~ ^($e1)$ ]]; then verdict=PASS; pass=$((pass+1)); else verdict=FAIL; fail=$((fail+1)); fi
  echo "$verdict $name | rank0: ${r0:-$(tail -2 /tmp/w-link-f0.txt | tr '\n' ' ')} (rc $rc0) | rank1: ${r1:-$(tail -2 /tmp/w-link-f1.txt | tr '\n' ' ')} (rc $rc1)"
}
P="--peer-faults 1"
run_case "kill rank 1 at call 500"        "PeerLost"         "exit137"            "$P"               "--fault kill --at 500"
run_case "kill rank 0 at call 500"        "exit137"          "PeerLost"           "--fault kill --at 500" "$P"
run_case "stall rank 1 (5 s) at call 500" "Timeout"          "stalled"            "$P"               "--fault stall --at 500"
run_case "different site from rank 0"     "Mismatch"         "Mismatch|PeerLost"  "--fault site --at 500" "$P"
run_case "abort on rank 0 from a thread"  "Aborted"          "PeerLost"           "--fault abort --at 0" "$P"
run_case "rank 1 closes its collective"   "PeerLost"         "closed"             "$P"               "--fault close --at 500"
run_case "SIGSTOP rank 1 for 5 s, then SIGCONT" "Timeout"    "PeerLost|Timeout"   "$P" "$P" 'sleep 1; kill -STOP $p1; sleep 5; kill -CONT $p1'
ENV0="TF_COLL_RTO_MIN_US=8000" ENV1="TF_COLL_RTO_MIN_US=12000" run_case "handshake: ranks differ in the RTO floor" "exit1" "exit1" "$P" "$P" ""
# negative controls: one flipped bit in one rank's send pattern must be found exactly once, in that rank's half,
# by both ranks (and once more in the first graph replay when the transport is capturable)
run_control() {  # name, corrupt spec
  local name=$1 spec=$2 g=""
  [[ "${TF_COLL_KIND:-host}" =~ ^(proxy|rdma)$ ]] && g="--graph 2"
  PORT=$((PORT + 40))
  env TF_COLL_PROXY_CPU=7 ${ENV1:-} taskset -c 5 $BIN 1 127.0.0.1 $PORT --corrupt $spec --prefill-chunks 0 $g ${EXTRA_ARGS:-} > /tmp/w-link-f1.txt 2>&1 &
  local p1=$!
  sleep 0.3
  env TF_COLL_PROXY_CPU=4 ${ENV0:-} taskset -c 6 $BIN 0 127.0.0.1 $PORT --corrupt $spec --prefill-chunks 0 $g ${EXTRA_ARGS:-} > /tmp/w-link-f0.txt 2>&1 &
  local p0=$!
  wait $p0; local rc0=$?
  wait $p1; local rc1=$?
  local r0=$(grep -o "negative-control: .*" /tmp/w-link-f0.txt | tail -1)
  local r1=$(grep -o "negative-control: .*" /tmp/w-link-f1.txt | tail -1)
  if [ $rc0 = 0 ] && [ $rc1 = 0 ] && [[ "$r0" == *PASS ]] && [[ "$r1" == *PASS ]]; then verdict=PASS; pass=$((pass+1)); else verdict=FAIL; fail=$((fail+1)); fi
  echo "$verdict $name | rank0: ${r0:-$(tail -2 /tmp/w-link-f0.txt | tr '\n' ' ')} (rc $rc0) | rank1: ${r1:-$(tail -2 /tmp/w-link-f1.txt | tr '\n' ' ')} (rc $rc1)"
}
run_control "negative control: rank 0 flips a bit at call 500 (20 KiB)" "0:500"
run_control "negative control: rank 1 flips a bit at call 92 (16 B Markov)" "1:92"
echo "fault matrix: $pass passed, $fail failed"
rm -f /tmp/w-link-f0.txt /tmp/w-link-f1.txt
