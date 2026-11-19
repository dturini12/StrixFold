#!/usr/bin/env python3
"""USB4 link reliability session: drops, retransmits and RTOs vs busy-poll, MTU, RTO floor and backlog.

Holds /var/tmp/halo-gpu.lock on node0 then node1 for the whole session (all-or-nothing, short waits) and a quiet
window (`~/bin/halo-quiet on|off TAG`, node0 first, refreshed every 5 min because requests expire after 20), changes
only non-persistent settings (MTU, per-route rto_min/quickack, netdev_max_backlog) and restores them in a finally
block; the MTU baseline is 1500 (permanent on both boxes since 2026-10-19). Every run records counter deltas on both boxes and the 1-minute load.
usage: reliab.py SECTION...   sections: alloc base round2 prefill contend pace mtu2 collcheck* gap gpuloop epp rfault soak soakab rtostress overlap mtu route backlog
"""
import subprocess, sys, time, os, json, threading

HOSTS = ("node0-lan", "node1-lan")
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reliab.log")
TB0_IP = {"node0-lan": "10.40.0.1", "node1-lan": "10.40.0.2"}
BASE_MTU = 1500
QUIET_TAG = f"w-link-reliab-{os.getpid()}"

def sh(host, cmd, timeout=900, check=False):
    r = subprocess.run(["ssh", host, cmd], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{host}: {cmd}: rc={r.returncode} {r.stderr.strip()}")
    return r.stdout + (r.stderr if r.returncode else "")

def log(s):
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")

class Lock:
    """GPU/link lock, held by a remote `cat` until we close its stdin."""
    def __init__(self, host, wait_s=1800):
        self.host = host
        self.p = subprocess.Popen(["ssh", host, f"flock -w {wait_s} /var/tmp/halo-gpu.lock "
                                   "sh -c 'echo LOCKED; cat >/dev/null'"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        import select
        ready, _, _ = select.select([self.p.stdout], [], [], wait_s * 2 + 30)
        line = self.p.stdout.readline() if ready else ""
        if "LOCKED" not in line:
            self.release()
            raise TimeoutError(f"{host}: lock not acquired")

    def release(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()
        try:
            self.p.stdout.close()
        except Exception:
            pass

def lock_both():
    """node0 first, then node1, each with a short wait; if either is busy, give everything back and retry, so no box's
    locks are held idle while waiting for the other."""
    t0 = time.time()
    while True:
        a = None
        try:
            # waiting for node0's lock holds nothing, so that wait may be long (LOCK1_WAIT_S); node1's stays short
            a = Lock("node0-lan", wait_s=int(os.environ.get("LOCK1_WAIT_S", "20")))
            b = Lock("node1-lan", wait_s=20)
            return [a, b]
        except TimeoutError as e:
            if a is not None:
                a.release()
            if time.time() - t0 > 4 * 3600:
                raise
            log(f"{time.strftime('%H:%M:%S')} {e}; retrying in 30 s")
            sys.stdout.flush()
            time.sleep(30)

class Quiet:
    """halo-quiet window on both boxes (node0 first); a thread re-requests it every 5 min so it outlives the 20-min expiry."""
    def __init__(self):
        self.stop = threading.Event()
        for h in HOSTS:
            log(f"{h}: " + sh(h, f"~/bin/halo-quiet on {QUIET_TAG}", timeout=60).strip())
        self.t = threading.Thread(target=self._refresh, daemon=True)
        self.t.start()

    def _refresh(self):
        while not self.stop.wait(300):
            for h in HOSTS:
                try:
                    sh(h, f"~/bin/halo-quiet on {QUIET_TAG} > /dev/null", timeout=60)
                except Exception as e:
                    log(f"{h}: quiet refresh failed: {e}")

    def off(self):
        self.stop.set()
        for h in HOSTS:
            try:
                log(f"{h}: " + sh(h, f"~/bin/halo-quiet off {QUIET_TAG}", timeout=60).strip())
            except Exception as e:
                log(f"{h}: quiet off FAILED ({e}); run `~/bin/halo-quiet off {QUIET_TAG}` there")

CNT = r'''
for n in thunderbolt0 thunderbolt1; do d=/sys/class/net/$n/statistics;
 for f in rx_missed_errors rx_errors rx_packets tx_packets rx_bytes tx_bytes tx_dropped; do echo "$n.$f $(cat $d/$f)"; done; done
awk '/^Tcp:/{if(!h){split($0,k);h=1}else{split($0,v);for(i=2;i<=NF;i++) if(k[i]=="RetransSegs"||k[i]=="OutSegs") print "tcp." k[i], v[i]}}' /proc/net/snmp
awk '/^TcpExt:/{if(!h){split($0,k);h=1}else{split($0,v);for(i=2;i<=NF;i++) if(k[i]~/^(TCPTimeouts|TCPLossProbes|TCPSACKReorder|TCPOFOQueue|TCPSpuriousRTOs|TCPDSACKUndo)$/) print "tcp." k[i], v[i]}}' /proc/net/netstat
echo "load $(cut -d' ' -f1 /proc/loadavg)"
'''

def counters(host):
    d = {}
    for line in sh(host, CNT).split("\n"):
        p = line.split()
        if len(p) == 2:
            try:
                d[p[0]] = float(p[1])
            except ValueError:
                pass
    return d

def measure(label, host, cmd, timeout=900):
    b = {h: counters(h) for h in HOSTS}
    t0 = time.time()
    out = sh(host, cmd, timeout)
    dt = time.time() - t0
    a = {h: counters(h) for h in HOSTS}
    parts = []
    for h in HOSTS:
        d = {k: a[h][k] - b[h].get(k, 0) for k in a[h] if k != "load"}
        gb = (d.get("thunderbolt0.rx_bytes", 0) + d.get("thunderbolt1.rx_bytes", 0)) / 1e9
        rxp = d.get("thunderbolt0.rx_packets", 0) + d.get("thunderbolt1.rx_packets", 0)
        missed = d.get("thunderbolt0.rx_missed_errors", 0) + d.get("thunderbolt1.rx_missed_errors", 0)
        outs = d.get("tcp.OutSegs", 0)
        parts.append(f"{h[:5]}: rxGB={gb:.2f} rx_pk={rxp:.0f} missed={missed:.0f} ({100 * missed / max(rxp, 1):.3f}% of rx pk) "
                     f"retrans={d.get('tcp.RetransSegs', 0):.0f} ({1000 * d.get('tcp.RetransSegs', 0) / max(outs, 1):.2f}/1k segs) "
                     f"RTO={d.get('tcp.TCPTimeouts', 0):.0f} TLP={d.get('tcp.TCPLossProbes', 0):.0f} reorder={d.get('tcp.TCPSACKReorder', 0):.0f} "
                     f"load={b[h].get('load', 0):.2f}->{a[h].get('load', 0):.2f}")
    keep = [l for l in out.split("\n") if l.startswith(("size=", "round", "napi", "# label", "[SUM]", "[ ID]", "[  5]", "[  7]", "iperf", "warn", "STDERR", "validation", "dspark", "plain", "prefill", "rc=", "info", "error", "warning", "graph", "soak", "prefill-stress", "overlap")) or "sender" in l or "receiver" in l]
    log(f"### {label} ({dt:.1f}s)  cmd: {cmd}")
    for l in keep:
        log("   " + l.strip()[:220])
    for p in parts:
        log("   " + p)
    return out

def tcplat(label, args):
    return measure(label, "node0-lan", f"cd ~/tfp/link && ./tcplat -c -B 10.40.0.2 {args} -t {label} -o raw")

def iperf(label, extra=""):
    sh("node1-lan", "pkill -x iperf3; setsid -f iperf3 -s -B 10.40.0.2 -p 47130 -1 > /dev/null 2>&1 < /dev/null")
    time.sleep(0.5)
    return measure(label, "node0-lan", f"iperf3 -c 10.40.0.2 -B 10.40.0.1 -p 47130 -t 4 -A 6,6 {extra} | tail -5")

def set_mtu(m):
    for h in HOSTS:
        sh(h, f"sudo ip link set dev thunderbolt0 mtu {m}", check=True)
    time.sleep(1)
    log(f"--- MTU now: " + " ".join(sh(h, "cat /sys/class/net/thunderbolt0/mtu").strip() for h in HOSTS))

def route(opts):
    for h in HOSTS:
        line = sh(h, "ip route show 10.40.0.0/24 dev thunderbolt0").strip()
        metric = line.split("metric")[1].split()[0] if "metric" in line else "100"
        sh(h, f"sudo ip route change 10.40.0.0/24 dev thunderbolt0 proto kernel scope link src {TB0_IP[h]} metric {metric} {opts}", check=True)
    log("--- route now: " + " | ".join(sh(h, "ip route show 10.40.0.0/24 dev thunderbolt0").strip() for h in HOSTS))

def backlog(v):
    for h in HOSTS:
        sh(h, f"echo {v} | sudo tee /proc/sys/net/core/netdev_max_backlog > /dev/null", check=True)
    log("--- netdev_max_backlog now: " + " ".join(sh(h, "cat /proc/sys/net/core/netdev_max_backlog").strip() for h in HOSTS))

D6 = "80x122880,2x307200,1x1548288,11x20480"

def main():
    sections = sys.argv[1:] or ["base"]
    for h in HOSTS:
        if sh(h, "pgrep -x 'ds4|ds4-server|ds4-bench'").strip():
            log(f"{h}: ds4 running before lock; will wait in the lock queue")
    locks = []
    quiet = None
    try:
        locks = lock_both()
        quiet = None if os.environ.get("NOQUIET") else Quiet()  # NOQUIET=1 for functional (non-timing) sessions
        loads = " ".join(h[:5] + " load " + sh(h, "cut -d' ' -f1-3 /proc/loadavg").strip() for h in HOSTS)
        log(f"=== session {time.strftime('%Y-%m-%d %H:%M:%S')} sections={sections} locks held on both boxes; {loads}")
        for h in HOSTS:
            other = sh(h, "pgrep -a -x 'ds4|ds4-server|ds4-bench'; ss -Htn state established | awk '$3 ~ /^10\\.40\\./ || $4 ~ /^10\\.40\\./' | grep -v ':22 '").strip()
            if other:
                log(f"{h}: other link users while locked: {other}")
        if "alloc" in sections:
            outs = {}
            procs = {h: subprocess.Popen(["ssh", h, "cd ~/tfp/link && ./allocgate > logs/allocgate-$(hostname).log 2>&1; cat logs/allocgate-$(hostname).log"],
                                         stdout=subprocess.PIPE, text=True) for h in HOSTS}
            for h, p in procs.items():
                outs[h] = p.communicate(timeout=900)[0]
                log(f"### allocgate {h}\n" + outs[h])
        sh("node1-lan", "pkill -x tcplat; cd ~/tfp/link && setsid -f ./tcplat -l -B 10.40.0.2,10.40.1.2 -p 47110 > logs/server.log 2>&1 < /dev/null")
        time.sleep(0.5)
        if "base" in sections:
            tcplat("rel_ex1m_bp50", "-m ex -P 1048576 -n 3000 -w 300 -C 5 -S 5 -b 50 -y spin")
            tcplat("rel_ex1m_bp0", "-m ex -P 1048576 -n 3000 -w 300 -C 5 -S 5 -b 0 -y spin")
            tcplat("rel_ex1m_poll_bp0", "-m ex -P 1048576 -n 3000 -w 300 -C 5 -S 5 -b 0 -y block")
            iperf("rel_iperf_h1toh2")
            iperf("rel_iperf_bidir", "--bidir")
            tcplat("rel_rD6_default", f"-m ex -P {D6} -n 200 -w 20 -C 5 -S 5 -b 50 -y spin")
            tcplat("rel_rD6_rtomin8ms", f"-m ex -P {D6} -n 200 -w 20 -C 5 -S 5 -b 50 -y spin -R 8000")
            tcplat("rel_ex1m_rtomin8ms", "-m ex -P 1048576 -n 3000 -w 300 -C 5 -S 5 -b 50 -y spin -R 8000")
        if "round2" in sections:
            # DSpark round with the call-site sizes from the dsv41 inventory (1 stream, 5 drafts, R=6 verify rows):
            # 80 x R*5120*4, 2 Engram x R*25600*4, head R*64640*4, 6 drafter x 5*5120*4, 5 Markov x 16 B
            p = "80x122880,2x614400,1x1551360,6x102400,5x16"
            tcplat("rel_rD6v2_tb0", f"-m ex -P {p} -n 200 -w 20 -C 5 -S 5 -b 50 -y spin")
            tcplat("rel_rD6v2_tb0_rto8", f"-m ex -P {p} -n 200 -w 20 -C 5 -S 5 -b 50 -y spin -R 8000")
            p1 = "80x20480,2x102400,1x258560"
            tcplat("rel_rR1v2_tb0", f"-m ex -P {p1} -n 200 -w 20 -C 5 -S 5 -b 50 -y spin")
        if "prefill" in sections:
            # one 2048-row encoder-only prefill chunk: 40 x n*5120*4 (40 MiB) + 2 Engram x n*25600*4 (200 MiB)
            pf = "1x209715200,20x41943040,1x209715200,20x41943040"
            for lab, addrs in (("tb0", "10.40.0.2"), ("striped", "10.40.0.2,10.40.1.2")):
                measure(f"rel_prefill2048_{lab}", "node0-lan",
                        f"cd ~/tfp/link && ./tcplat -c -B {addrs} -m ex -P {pf} -n 3 -w 1 -C 5 -S 5 -b 50 -y spin -t rel_prefill2048_{lab} -o raw")
            for lab, addrs in (("tb0", "10.40.0.2"), ("striped", "10.40.0.2,10.40.1.2")):
                measure(f"rel_prefill512_{lab}", "node0-lan",
                        f"cd ~/tfp/link && ./tcplat -c -B {addrs} -m ex -P 1x52428800,20x10485760,1x52428800,20x10485760 -n 5 -w 1 -C 5 -S 5 -b 50 -y spin -t rel_prefill512_{lab} -o raw")
        if "contend" in sections:
            # decode rounds while a bulk iperf3 flow runs (foreign traffic), with and without the 8 ms RTO floor,
            # at MTU 65520 and 1500 (both cables, both boxes; restored after)
            def set_mtu_c(m):
                for h in HOSTS:
                    sh(h, f"sudo ip link set dev thunderbolt0 mtu {m} && sudo ip link set dev thunderbolt1 mtu {m}", check=True)
                time.sleep(1)
                log(f"--- MTU tb0/tb1 now {m}")
            try:
              for mtu in (65520, 1500):
               set_mtu_c(mtu)
               for lab, ip_s, ip_c in (("tb0", "10.40.0.2", "10.40.0.1"), ("tb1", "10.40.1.2", "10.40.1.1")):
                for rto in ("", "-R 8000"):
                    sh("node1-lan", f"pkill -x iperf3; setsid -f iperf3 -s -B {ip_s} -p 47130 -1 > /dev/null 2>&1 < /dev/null")
                    time.sleep(0.5)
                    sh("node0-lan", f"setsid -f iperf3 -c {ip_s} -B {ip_c} -p 47130 -t 6 -A 7,7 > /tmp/w-link-iperf.txt 2>&1 < /dev/null")
                    time.sleep(1.0)
                    tag = "rto8" if rto else "rtodef"
                    tcplat(f"rel_contend_m{mtu}_{lab}_{tag}", f"-m ex -P {D6} -n 150 -w 10 -C 5 -S 5 -b 50 -y spin {rto}")
                    time.sleep(5)
                    log("   iperf: " + sh("node0-lan", "tail -4 /tmp/w-link-iperf.txt | head -2 | tr -s ' ' | cut -c1-120").replace("\n", " | "))
            finally:
                set_mtu_c(BASE_MTU)
        if "pace" in sections:
            # sender pacing (SO_MAX_PACING_RATE per socket) against RX-ring overruns on bulk symmetric transfers
            sh("node1-lan", "pkill -x tcplat2; cd ~/tfp/link && setsid -f ./tcplat2 -l -B 10.40.0.2,10.40.1.2 -p 47111 > logs/server2.log 2>&1 < /dev/null")
            time.sleep(0.5)
            pf = "1x209715200,20x41943040,1x209715200,20x41943040"
            def t2(label, args):
                return measure(label, "node0-lan", f"cd ~/tfp/link && ./tcplat2 -c -p 47111 {args} -t {label} -o raw")
            for rate in (0, 3000000000, 2500000000, 2000000000, 1500000000):
                t2(f"rel_pace{rate // 100000000}_pf2048_tb0", f"-B 10.40.0.2 -m ex -P {pf} -n 3 -w 1 -C 5 -S 5 -b 50 -y spin -r {rate}")
            for rate in (0, 2000000000, 1500000000):
                t2(f"rel_pace{rate // 100000000}_pf2048_str", f"-B 10.40.0.2,10.40.1.2 -m ex -P {pf} -n 3 -w 1 -C 5 -S 5 -b 50 -y spin -r {rate}")
            t2("rel_pace25_rD6v2", "-B 10.40.0.2 -m ex -P 80x122880,2x614400,1x1551360,6x102400,5x16 -n 200 -w 20 -C 5 -S 5 -b 50 -y spin -r 2500000000")
            sh("node1-lan", "pkill -x tcplat2")
            for fq in ("20G", "16G"):
                iperf(f"rel_iperf_fq{fq}", f"--fq-rate {fq}")
        if "mtu2" in sections:
            # bulk (prefill-size) and bidirectional loss vs MTU, same session for all three MTUs, both cables set alike
            def set_mtu_both(m):
                for h in HOSTS:
                    sh(h, f"sudo ip link set dev thunderbolt0 mtu {m} && sudo ip link set dev thunderbolt1 mtu {m}", check=True)
                time.sleep(1)
                log("--- MTU tb0/tb1 now: " + " ".join(sh(h, "cat /sys/class/net/thunderbolt0/mtu /sys/class/net/thunderbolt1/mtu | paste -sd/").strip() for h in HOSTS))
            pf = "1x209715200,20x41943040,1x209715200,20x41943040"
            try:
                for m in (65520, 9000, 1500):
                    set_mtu_both(m)
                    tcplat(f"rel_m{m}_pf2048_tb0", f"-m ex -P {pf} -n 3 -w 1 -C 5 -S 5 -b 50 -y spin")
                    measure(f"rel_m{m}_pf2048_str", "node0-lan", f"cd ~/tfp/link && ./tcplat -c -B 10.40.0.2,10.40.1.2 -m ex -P {pf} -n 3 -w 1 -C 5 -S 5 -b 50 -y spin -t rel_m{m}_pf2048_str -o raw")
                    tcplat(f"rel_m{m}_rD6v2", "-m ex -P 80x122880,2x614400,1x1551360,6x102400,5x16 -n 200 -w 20 -C 5 -S 5 -b 50 -y spin")
                    tcplat(f"rel_m{m}_rR1v2", "-m ex -P 80x20480,2x102400,1x258560 -n 200 -w 20 -C 5 -S 5 -b 50 -y spin")
                    iperf(f"rel_m{m}_iperf_bidir", "--bidir")
            finally:
                set_mtu_both(BASE_MTU)
        if "collcheck" in sections:
            # w-link's tcp_host collectives through the family's open path, both ranks, device buffers on both ends
            for lab, env in (("1link", ""), ("2links", "TF_TP_STRIPE=10.40.1.1")):
                r1 = subprocess.Popen(["ssh", "node1-lan", f"cd ~/tfp/link && {env} taskset -c 5-7 ./tf-dsv41-coll-check 1 10.40.0.1 29700 --rounds 200 --prefill-chunks 3 --receipt logs/collcheck-{lab}-rank1.txt > logs/collcheck-{lab}-rank1.log 2>&1; echo rc=$?"],
                                      stdout=subprocess.PIPE, text=True)
                measure(f"collcheck_{lab}_rank0", "node0-lan", f"cd ~/tfp/link && {env} taskset -c 5-7 ./tf-dsv41-coll-check 0 10.40.0.1 29700 --rounds 200 --prefill-chunks 3 --receipt logs/collcheck-{lab}-rank0.txt 2>&1 | tail -30; echo rc=$?")
                log("   rank1: " + r1.communicate(timeout=600)[0].strip() + " | " + sh("node1-lan", f"tail -14 ~/tfp/link/logs/collcheck-{lab}-rank1.log").replace("\n", " || "))
        if "collcheck2" in sections:
            # tcp_host staging: blit (mapped coherent + device copies) vs dma, one link and striped, MTU 65520 and 1500
            def mtu_both(m):
                for h in HOSTS:
                    sh(h, f"sudo ip link set dev thunderbolt0 mtu {m} && sudo ip link set dev thunderbolt1 mtu {m}", check=True)
                time.sleep(1)
                log(f"--- MTU tb0/tb1 now {m}")
            try:
                for m in (65520, 1500):
                    mtu_both(m)
                    for lab, env in (("blit_1link", "TF_COLL_STAGING=blit"), ("dma_1link", "TF_COLL_STAGING=dma"), ("blit_2links", "TF_COLL_STAGING=blit TF_TP_STRIPE=10.40.1.1")):
                        lab = f"m{m}_{lab}"
                        r1 = subprocess.Popen(["ssh", "node1-lan", f"cd ~/tfp/link && {env} taskset -c 5-7 ./tf-dsv41-coll-check 1 10.40.0.1 29700 --rounds 200 --prefill-chunks 3 --receipt logs/collcheck-{lab}-rank1.txt > logs/collcheck-{lab}-rank1.log 2>&1; echo rc=$?"], stdout=subprocess.PIPE, text=True)
                        measure(f"collcheck_{lab}_rank0", "node0-lan", f"cd ~/tfp/link && {env} taskset -c 5-7 ./tf-dsv41-coll-check 0 10.40.0.1 29700 --rounds 200 --prefill-chunks 3 --receipt logs/collcheck-{lab}-rank0.txt 2>&1 | tail -30; echo rc=$?")
                        log("   rank1: " + r1.communicate(timeout=600)[0].strip())
            finally:
                mtu_both(BASE_MTU)
        if "collcheck3" in sections:
            # tcp_host chunked pipelining for large gathers, on and off, one link and striped, MTU 1500
            def mtu_both3(m):
                for h in HOSTS:
                    sh(h, f"sudo ip link set dev thunderbolt0 mtu {m} && sudo ip link set dev thunderbolt1 mtu {m}", check=True)
                time.sleep(1)
                log(f"--- MTU tb0/tb1 now {m}")
            try:
                mtu_both3(1500)
                for lab, env in (("pipe_1link", ""), ("nopipe_1link", "TF_COLL_PIPE_MIB=0"), ("pipe_2links", "TF_TP_STRIPE=10.40.1.1"), ("nopipe_2links", "TF_COLL_PIPE_MIB=0 TF_TP_STRIPE=10.40.1.1")):
                    lab = f"m1500_{lab}"
                    r1 = subprocess.Popen(["ssh", "node1-lan", f"cd ~/tfp/link && {env} taskset -c 5-7 ./tf-dsv41-coll-check 1 10.40.0.1 29700 --rounds 100 --prefill-chunks 5 --receipt logs/collcheck-{lab}-rank1.txt > logs/collcheck-{lab}-rank1.log 2>&1; echo rc=$?"], stdout=subprocess.PIPE, text=True)
                    measure(f"collcheck_{lab}_rank0", "node0-lan", f"cd ~/tfp/link && {env} taskset -c 5-7 ./tf-dsv41-coll-check 0 10.40.0.1 29700 --rounds 100 --prefill-chunks 5 --receipt logs/collcheck-{lab}-rank0.txt 2>&1 | tail -30; echo rc=$?")
                    log("   rank1: " + r1.communicate(timeout=600)[0].strip())
            finally:
                mtu_both3(BASE_MTU)
        if "collcheck4" in sections:
            # tcp_proxy (kernels + proxy thread, capturable) incl. a graph replay test; MTU 65520 and 1500
            def mtu_both4(m):
                for h in HOSTS:
                    sh(h, f"sudo ip link set dev thunderbolt0 mtu {m} && sudo ip link set dev thunderbolt1 mtu {m}", check=True)
                time.sleep(1)
                log(f"--- MTU tb0/tb1 now {m}")
            try:
                for m in (65520, 1500):
                    mtu_both4(m)
                    for lab, env in (("proxy_2links", "TF_COLL_KIND=proxy TF_TP_STRIPE=10.40.1.1"),):
                        lab = f"m{m}_{lab}"
                        cmd = f"cd ~/tfp/link && {env} taskset -c 5-7 ./tf-dsv41-coll-check RANK 10.40.0.1 29700 --rounds 200 --prefill-chunks 3 --kit dsv41-kit --graph 100 --receipt logs/collcheck-{lab}-rankRANK.txt"
                        r1 = subprocess.Popen(["ssh", "node1-lan", cmd.replace("RANK", "1") + f" > logs/collcheck-{lab}-rank1.log 2>&1; echo rc=$?"], stdout=subprocess.PIPE, text=True)
                        measure(f"collcheck_{lab}_rank0", "node0-lan", cmd.replace("RANK", "0") + " 2>&1 | tail -30; echo rc=$?")
                        log("   rank1: " + r1.communicate(timeout=900)[0].strip())
            finally:
                mtu_both4(BASE_MTU)
        if "gap" in sections:
            # compute-gap rounds (clean repeat under halo-quiet): a 300 us gap before each exchange, spun or slept,
            # against the no-gap round; spin + busy-poll vs blocking poll()
            R1 = "80x20480,2x51200,1x258048"
            tcplat("q_rD6_nogap_spin", f"-m ex -P {D6} -n 200 -w 20 -C 5 -S 5 -b 50 -y spin")
            tcplat("q_rD6_gap300spin", f"-m ex -P {D6} -n 100 -w 10 -C 5 -S 5 -b 50 -y spin -g 300")
            tcplat("q_rD6_gap300sleep", f"-m ex -P {D6} -n 100 -w 10 -C 5 -S 5 -b 50 -y spin -g 300 -G")
            tcplat("q_rD6_gap300sleep_poll", f"-m ex -P {D6} -n 100 -w 10 -C 5 -S 5 -b 0 -y block -g 300 -G")
            tcplat("q_rR1_nogap_spin", f"-m ex -P {R1} -n 200 -w 20 -C 5 -S 5 -b 50 -y spin")
            tcplat("q_rR1_gap300spin", f"-m ex -P {R1} -n 100 -w 10 -C 5 -S 5 -b 50 -y spin -g 300")
        if "gpuloop" in sections:
            # GPU in the loop (clean repeat under halo-quiet): gpuvis kernel -> mapped slab -> proxy thread -> TCP ->
            # proxy -> slab -> kernel, rank 1 on node1 listening on 10.40.0.2, proxy thread on cpu 5 on both boxes
            R1 = "80x20480,2x51200,1x258048"
            for lab, pat, reps, warm in (("n20k", "20480", 10000, 200), ("n120k", "122880", 10000, 200),
                                         ("rD6", D6, 120, 5), ("rR1", R1, 125, 5)):
                cmd = f"cd ~/tfp/link && ./gpuvis -r RANK -G 1 -a 10.40.0.2 -P {pat} -n {reps} -w {warm} -t q_net_g1_{lab}"
                r1 = subprocess.Popen(["ssh", "node1-lan", cmd.replace("RANK", "1") + f" > logs/q-gpuvis-{lab}-rank1.log 2>&1; echo rc=$?; tail -8 logs/q-gpuvis-{lab}-rank1.log"],
                                      stdout=subprocess.PIPE, text=True)
                time.sleep(1.0)
                measure(f"q_gpuvis_{lab}_rank0", "node0-lan", cmd.replace("RANK", "0") + " 2>&1 | tail -12; echo rc=$?")
                log("   rank1: " + r1.communicate(timeout=900)[0].strip().replace("\n", " | ")[:400])
        if "epp" in sections:
            # EPP A/B: performance (current) vs balance_performance, interleaved twice; per-CPU values saved and restored
            saved = {h: sh(h, "grep -H . /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference") for h in HOSTS}
            def epp(v):
                for h in HOSTS:
                    sh(h, f"echo {v} | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference > /dev/null", check=True)
                log(f"--- EPP now: " + " ".join(h[:5] + " " + sh(h, "cat /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference | sort | uniq -c | tr -s ' ' | paste -sd,").strip() for h in HOSTS))
            try:
                for rep in (1, 2):
                    for v in ("performance", "balance_performance"):
                        epp(v)
                        tag = f"{v[:4]}{rep}"
                        tcplat(f"q_epp_{tag}_pp120k", "-m pp -P 122880 -n 5000 -w 500 -C 5 -S 5 -b 50 -y spin")
                        tcplat(f"q_epp_{tag}_rD6", f"-m ex -P {D6} -n 200 -w 20 -C 5 -S 5 -b 50 -y spin")
                        tcplat(f"q_epp_{tag}_rD6_gap300sleep", f"-m ex -P {D6} -n 100 -w 10 -C 5 -S 5 -b 50 -y spin -g 300 -G")
                        lab = f"epp_{tag}_proxy"
                        cmd = (f"cd ~/tfp/link && TF_COLL_KIND=proxy taskset -c 5-7 ./tf-dsv41-coll-check RANK 10.40.0.1 29700 --rounds 200 "
                               f"--prefill-chunks 1 --kit dsv41-kit --receipt logs/collcheck-{lab}-rankRANK.txt")
                        r1 = subprocess.Popen(["ssh", "node1-lan", cmd.replace("RANK", "1") + f" > logs/collcheck-{lab}-rank1.log 2>&1; echo rc=$?"],
                                              stdout=subprocess.PIPE, text=True)
                        measure(f"q_collcheck_{lab}_rank0", "node0-lan", cmd.replace("RANK", "0") + " 2>&1 | tail -30; echo rc=$?")
                        log("   rank1: " + r1.communicate(timeout=900)[0].strip())
            finally:
                for h in HOSTS:
                    for line in saved[h].split("\n"):
                        if ":" in line:
                            f, v = line.rsplit(":", 1)
                            sh(h, f"echo {v.strip()} | sudo tee {f} > /dev/null")
                log(f"--- EPP restored: " + " ".join(h[:5] + " " + sh(h, "cat /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference | sort | uniq -c | tr -s ' ' | paste -sd,").strip() for h in HOSTS))
        if "rfault" in sections:
            # real-link fault subset (COLLECTIVES-DESIGN 7.2) over both cables, tcp_host and tcp_proxy: internal faults
            # at a decode call, external SIGKILL/SIGSTOP (by saved PID) inside decode rounds and inside prefill chunks
            import re
            FR = re.compile(r"fault-result: rank=(\d) calls=(\d+) error=(\w+) after_ms=(\d+)")
            hosts = {0: "node0-lan", 1: "node1-lan"}
            def rank_cmd(r, env, args, lab):
                return (f"cd ~/tfp/link && bash -c 'echo $$ > /tmp/w-link-rf{r}.pid; exec env {env} taskset -c 5-7 "
                        f"./tf-dsv41-coll-check {r} 10.40.0.1 PORT {args}' > logs/rf-{lab}-rank{r}.log 2>&1; echo rc=$?; "
                        f"grep -h 'fault-result\\|error' logs/rf-{lab}-rank{r}.log | tail -3")
            def act(r, sig, after_s, hold_s, lab):
                h = hosts[r]
                cmd = (f"for i in $(seq 1 600); do grep -q 'info: collectives: tcp_host over' ~/tfp/link/logs/rf-{lab}-rank{r}.log 2>/dev/null && break; sleep 0.05; done; "
                       f"sleep {after_s}; p=$(cat /tmp/w-link-rf{r}.pid); kill -{sig} $p && echo sent {sig} to $p at $(date +%T.%N | cut -c1-12)")
                if sig == "STOP":
                    cmd += f"; sleep {hold_s}; kill -CONT $p; echo CONT"
                return subprocess.Popen(["ssh", h, cmd], stdout=subprocess.PIPE, text=True)
            cases = [  # label, args0, args1, action (rank, signal, after_s, hold_s) or None, expect0, expect1
                ("kill1_decode", "--peer-faults 1", "--fault kill --at 500", None, "PeerLost", "rc137"),
                ("kill0_decode", "--fault kill --at 500", "--peer-faults 1", None, "rc137", "PeerLost"),
                ("stall1_decode", "--peer-faults 1", "--fault stall --at 500", None, "Timeout", "stalled"),
                ("stop1_decode", "--peer-faults 1", "--peer-faults 1", (1, "STOP", 1.0, 5), "Timeout", "PeerLost|Timeout"),
                ("kill1_prefill", "--peer-faults 1 --fault-calls prefill", "--peer-faults 1 --fault-calls prefill", (1, "KILL", 1.0, 0), "PeerLost", "rc137"),
                ("kill0_prefill", "--peer-faults 1 --fault-calls prefill", "--peer-faults 1 --fault-calls prefill", (0, "KILL", 1.0, 0), "rc137", "PeerLost"),
                ("stop1_prefill", "--peer-faults 1 --fault-calls prefill", "--peer-faults 1 --fault-calls prefill", (1, "STOP", 1.0, 5), "Timeout", "PeerLost|Timeout"),
            ]
            port = 29900
            npass = nfail = 0
            for kind, kenv, kargs in (("host", "TF_COLL_KIND=host", ""), ("proxy", "TF_COLL_KIND=proxy", "--kit dsv41-kit")):
                for lab, a0, a1, action, e0, e1 in cases:
                    port += 40
                    lab = f"{kind}_{lab}"
                    # TF_COLL_GPU_WAIT_S=2: a peer's arrival limit (GPU wait + progress) of 4 s, under the 5 s stall/stop
                    env = f"{kenv} TF_TP_STRIPE=10.40.1.1 TF_COLL_PROGRESS_MS=2000 TF_COLL_GPU_WAIT_S=2"
                    c1 = rank_cmd(1, env, f"{a1} {kargs}", lab).replace("PORT", str(port))
                    c0 = rank_cmd(0, env, f"{a0} {kargs}", lab).replace("PORT", str(port))
                    b = {h: counters(h) for h in HOSTS}
                    p1 = subprocess.Popen(["ssh", "node1-lan", c1], stdout=subprocess.PIPE, text=True)
                    time.sleep(0.3)
                    p0 = subprocess.Popen(["ssh", "node0-lan", c0], stdout=subprocess.PIPE, text=True)
                    pa = act(action[0], action[1], action[2], action[3], lab) if action else None
                    outs = {}
                    try:
                        outs[0] = p0.communicate(timeout=180)[0]
                        outs[1] = p1.communicate(timeout=180)[0]
                    except subprocess.TimeoutExpired:
                        for r, h in hosts.items():
                            sh(h, f"kill -9 $(cat /tmp/w-link-rf{r}.pid) 2>/dev/null")
                        outs = {0: p0.communicate()[0], 1: p1.communicate()[0]}
                    acted = pa.communicate(timeout=60)[0].strip().replace("\n", " ") if pa else ""
                    a = {h: counters(h) for h in HOSTS}
                    got = {}
                    for r in (0, 1):
                        m = FR.search(outs[r])
                        rc = re.search(r"rc=(\d+)", outs[r])
                        got[r] = (m.group(3), m.group(2), m.group(4)) if m else (f"rc{rc.group(1) if rc else '?'}", "-", "-")
                    ok = re.fullmatch(e0, got[0][0]) and re.fullmatch(e1, got[1][0])
                    npass += bool(ok)
                    nfail += not ok
                    rto = " ".join(f"{h[:5]} RTO={a[h].get('tcp.TCPTimeouts', 0) - b[h].get('tcp.TCPTimeouts', 0):.0f} retrans={a[h].get('tcp.RetransSegs', 0) - b[h].get('tcp.RetransSegs', 0):.0f}" for h in HOSTS)
                    log(f"{'PASS' if ok else 'FAIL'} rf_{lab}: rank0 {got[0][0]} calls={got[0][1]} after_ms={got[0][2]} | rank1 {got[1][0]} calls={got[1][1]} after_ms={got[1][2]} | {acted} | {rto}")
                    if not ok:
                        log("   rank0: " + outs[0].strip().replace("\n", " | ")[:400])
                        log("   rank1: " + outs[1].strip().replace("\n", " | ")[:400])
                    time.sleep(1)
            log(f"real-link fault subset: {npass} passed, {nfail} failed")
        if "soak" in sections:
            # 15-minute soak (COLLECTIVES-DESIGN 7.2): tcp_proxy with graphs, both cables, mixed decode/graph/prefill
            # traffic, every validated word checked; drop/retransmit/RTO counters over the whole run, load sampled
            secs = int(os.environ.get("SOAK_S", "900"))
            lab = f"soak{secs}_proxy"
            env = "TF_COLL_KIND=proxy TF_TP_STRIPE=10.40.1.1"
            cmd = (f"cd ~/tfp/link && {env} taskset -c 5-7 ./tf-dsv41-coll-check RANK 10.40.0.1 29700 --soak {secs} "
                   f"--kit dsv41-kit --graph 1 --receipt logs/collcheck-{lab}-rankRANK.txt")
            loads = []
            stop = threading.Event()
            def sample():
                while not stop.wait(60):
                    loads.append(" ".join(h[:5] + " " + sh(h, "cut -d' ' -f1 /proc/loadavg", timeout=30).strip() for h in HOSTS))
            ts = threading.Thread(target=sample, daemon=True)
            ts.start()
            r1 = subprocess.Popen(["ssh", "node1-lan", cmd.replace("RANK", "1") + f" > logs/collcheck-{lab}-rank1.log 2>&1; echo rc=$?"], stdout=subprocess.PIPE, text=True)
            measure(f"collcheck_{lab}_rank0", "node0-lan", cmd.replace("RANK", "0") + " 2>&1 | grep -v 'calls, bad words' ; echo rc=$?", timeout=secs + 900)
            log("   rank1: " + r1.communicate(timeout=900)[0].strip() + " | " + sh("node1-lan", f"grep -h 'soak: .*iterations' ~/tfp/link/logs/collcheck-{lab}-rank1.log | tail -1").strip())
            stop.set()
            log("   load per minute: " + " ; ".join(loads))
        if "soakab" in sections:
            # the same mixed traffic for 60 s each: without a quiet window, with one on both boxes, without again
            # (does background work on the boxes slow the collectives, and by how much), with each box's DRAM
            # bandwidth from the data-fabric counters (umcbw) over each leg
            def bw_start(lab):
                return {h: subprocess.Popen(["ssh", h, f"cd ~/tfp/link && sudo -n ./umcbw -i 1000 -d 600 -t {h[:5]}-{lab}"],
                                            stdout=subprocess.PIPE, text=True) for h in HOSTS}
            def bw_stop(procs):
                for h in HOSTS:
                    sh(h, "sudo -n pkill -INT -x umcbw")
                for h, p in procs.items():
                    out = p.communicate(timeout=60)[0]
                    summ = [l for l in out.split("\n") if "summary" in l]
                    log("   " + (summ[0] if summ else f"{h[:5]}: umcbw gave no summary: {out[-200:]}"))
            # scale check of umcbw: a known load (bwload, non-temporal copy on core 7) inside a quiet window
            q = Quiet()
            try:
                procs = {"node0-lan": subprocess.Popen(["ssh", "node0-lan", "cd ~/tfp/link && sudo -n ./umcbw -i 250 -d 14 -t cal"], stdout=subprocess.PIPE, text=True)}
                time.sleep(3)
                log("   " + sh("node0-lan", "cd ~/tfp/link && flock -s -w 5 /var/tmp/halo-quiet.lock taskset -c 7 ./bwload 1024 100").strip())
                out = procs["node0-lan"].communicate(timeout=60)[0]
                log("   umcbw during the scale check: " + " | ".join(l.split(" ", 1)[1] for l in out.strip().split("\n")))
            finally:
                q.off()
            env = "TF_COLL_KIND=proxy TF_TP_STRIPE=10.40.1.1"
            for k, mode in enumerate(("open", "quiet", "open2")):
                q = Quiet() if mode == "quiet" else None
                try:
                    lab = f"soakab_{mode}"
                    cmd = (f"cd ~/tfp/link && {env} taskset -c 5-7 ./tf-dsv41-coll-check RANK 10.40.0.1 {29700 + 40 * k} --soak 60 "
                           f"--kit dsv41-kit --graph 1 --receipt logs/collcheck-{lab}-rankRANK.txt")
                    loads = " ".join(h[:5] + " " + sh(h, "cut -d' ' -f1 /proc/loadavg; cat /proc/pressure/cpu | head -1 | cut -d' ' -f2", timeout=30).strip().replace("\n", " cpu-psi ") for h in HOSTS)
                    log(f"--- {lab}: before {loads}")
                    bw = bw_start(mode)
                    r1 = subprocess.Popen(["ssh", "node1-lan", cmd.replace("RANK", "1") + f" > logs/collcheck-{lab}-rank1.log 2>&1; echo rc=$?"], stdout=subprocess.PIPE, text=True)
                    measure(f"collcheck_{lab}_rank0", "node0-lan", cmd.replace("RANK", "0") + " 2>&1 | grep -v 'calls, bad words' ; echo rc=$?", timeout=900)
                    log("   rank1: " + r1.communicate(timeout=900)[0].strip())
                    bw_stop(bw)
                finally:
                    if q is not None:
                        q.off()
        if "rtostress" in sections:
            # the RTO-floor question (d-decode: RTO 8 ms stalled DwarfStar's 42 MB exchanges at MTU 1500): N 2048-row
            # prefill chunks through tcp_host at TF_COLL_RTO_MIN_US=8000, striped (STRESS_MODE=striped) or one cable
            n = int(os.environ.get("STRESS_N", "200"))
            mode = os.environ.get("STRESS_MODE", "striped")
            rto = os.environ.get("STRESS_RTO_US", "8000")
            env = f"TF_COLL_KIND=host TF_COLL_RTO_MIN_US={rto}" + (" TF_TP_STRIPE=10.40.1.1" if mode == "striped" else "")
            lab = f"rtostress_{mode}_rto{rto}_n{n}"
            cmd = (f"cd ~/tfp/link && {env} taskset -c 5-7 ./tf-dsv41-coll-check RANK 10.40.0.1 29700 --prefill-stress {n} "
                   f"--receipt logs/collcheck-{lab}-rankRANK.txt")
            log(f"--- {lab}: block start {time.strftime('%H:%M:%S')}")
            r1 = subprocess.Popen(["ssh", "node1-lan", cmd.replace("RANK", "1") + f" > logs/collcheck-{lab}-rank1.log 2>&1; echo rc=$?"], stdout=subprocess.PIPE, text=True)
            measure(f"collcheck_{lab}_rank0", "node0-lan", cmd.replace("RANK", "0") + " 2>&1 | tail -20; echo rc=$?", timeout=1800)
            log("   rank1: " + r1.communicate(timeout=900)[0].strip() + " | " + sh("node1-lan", f"grep -h 'prefill-stress' ~/tfp/link/logs/collcheck-{lab}-rank1.log | head -4").strip().replace("\n", " | "))
            log(f"--- {lab}: block end {time.strftime('%H:%M:%S')}")
        if "overlap" in sections:
            # split gathers (coll.Coll.gatherBegin/End) against device work between them: 2048-row prefill chunks as
            # two 1024-row halves, tcp_host striped over both cables; OVERLAP_WORK items: mN = a D2D copy of N MiB per
            # half-step (DRAM-bound), uN = a one-thread kernel spinning N us (no memory traffic)
            n = int(os.environ.get("OVERLAP_N", "10"))
            for work in os.environ.get("OVERLAP_WORK", "m256,m768,m1536").split(","):
                env = "TF_COLL_KIND=host TF_TP_STRIPE=10.40.1.1"
                lab = f"overlap_{work}_n{n}"
                warg = f"--work-mib {work[1:]}" if work[0] == "m" else f"--work-us {work[1:]}"
                cmd = (f"cd ~/tfp/link && {env} taskset -c 5-7 ./tf-dsv41-coll-check RANK 10.40.0.1 29700 --overlap {n} {warg} "
                       f"--receipt logs/collcheck-{lab}-rankRANK.txt")
                r1 = subprocess.Popen(["ssh", "node1-lan", cmd.replace("RANK", "1") + f" > logs/collcheck-{lab}-rank1.log 2>&1; echo rc=$?"], stdout=subprocess.PIPE, text=True)
                measure(f"collcheck_{lab}_rank0", "node0-lan", cmd.replace("RANK", "0") + " 2>&1 | tail -12; echo rc=$?", timeout=1800)
                log("   rank1: " + r1.communicate(timeout=900)[0].strip() + " | " + sh("node1-lan", f"grep -h 'overlap:' ~/tfp/link/logs/collcheck-{lab}-rank1.log | tail -1").strip())
        if "mtu" in sections:
            try:
                for m in (16384, 9000, 1500):
                    set_mtu(m)
                    tcplat(f"rel_mtu{m}_pp120k", "-m pp -P 122880 -n 5000 -w 500 -C 5 -S 5 -b 50 -y spin")
                    tcplat(f"rel_mtu{m}_ex1m", "-m ex -P 1048576 -n 3000 -w 300 -C 5 -S 5 -b 50 -y spin")
                    tcplat(f"rel_mtu{m}_rD6", f"-m ex -P {D6} -n 100 -w 10 -C 5 -S 5 -b 50 -y spin")
                    iperf(f"rel_mtu{m}_iperf")
            finally:
                set_mtu(BASE_MTU)
        if "route" in sections:
            try:
                route("rto_min 8ms quickack 1")
                tcplat("rel_route_rD6", f"-m ex -P {D6} -n 200 -w 20 -C 5 -S 5 -b 50 -y spin")
                tcplat("rel_route_ex1m", "-m ex -P 1048576 -n 3000 -w 300 -C 5 -S 5 -b 50 -y spin")
                tcplat("rel_route_pp120k", "-m pp -P 122880 -n 5000 -w 500 -C 5 -S 5 -b 50 -y spin")
            finally:
                route("")
        if "backlog" in sections:
            try:
                backlog(10000)
                tcplat("rel_backlog10k_ex1m", "-m ex -P 1048576 -n 3000 -w 300 -C 5 -S 5 -b 50 -y spin")
                tcplat("rel_backlog10k_ex1m_bp0", "-m ex -P 1048576 -n 3000 -w 300 -C 5 -S 5 -b 0 -y spin")
            finally:
                backlog(1000)
    finally:
        try:
            sh("node1-lan", "pkill -x tcplat; pkill -x tcplat2; pkill -x iperf3")
            log("=== end; state: " + " | ".join(
                f"{h[:5]} mtu {sh(h, 'cat /sys/class/net/thunderbolt0/mtu /sys/class/net/thunderbolt1/mtu | paste -sd/').strip()} route {sh(h, 'ip route show 10.40.0.0/24 dev thunderbolt0').strip()} "
                f"backlog {sh(h, 'cat /proc/sys/net/core/netdev_max_backlog').strip()}" for h in HOSTS))
        finally:
            if quiet is not None:
                quiet.off()
            for l in reversed(locks):
                l.release()

if __name__ == "__main__":
    main()
