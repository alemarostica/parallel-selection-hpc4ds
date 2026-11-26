#!/usr/bin/env python3

"""Plot benchmark logs. Usage: python3 plot_res.py [results_directory]"""
import os
import re
import sys
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "bash-scripts/results"
OUT = os.path.join(RES,"plots")
MODES = ["mom","rand","fr","frR"]
def parse(name):
    """One dict per run: label, mode, time, rounds, sizes (N after each round)"""
    runs, run = [],None
    with open(os.path.join(RES,name), errors="replace") as f:
        for line in f:
            if(line).startswith("###"):
                m = re.match(r"### (.*?) \| (?:pivot=(\S+) \| )?rep=",line)
                run = {"label": m[1], "mode": m[2], "sizes" : []}
            elif run is None:
                continue
            elif m:= re.match(r"\s+round\s+\d+: N=(\d+) .*-> N=(\d+)",line):
                if not run["sizes"]:
                    run["sizes"].append(int(m[1]))
                run["sizes"].append(int(m[2]))
            elif m:= re.search(r"\((\d+) rounds.*, ([\d.]+) s\)",line):
                run["rounds"],run["time"] = int(m[1]), float(m[2])
                runs.append(run)
    return runs

def stats(runs, mode=None, label=None):
    """(median, below, above) of the selected runs, for error bars"""
    t = [r["time"] for r in runs if(mode is None or r["mode"] == mode) and (label is None or r["label"] == label)]
    assert t, f"no runs for mode={mode} label={label}"
    m = median(t)
    return m,m-min(t),max(t)-m
    


def save(fig,name):
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(os.path.join(OUT,f"{name}.{ext}"),dpi=150)
    plt.close(fig)

def bars(name,title,groups,series,log=False):
    """groups: x labels. series: {legend: [(median,below,above),...]}."""
    fig,ax = plt.subplots(figsize=(9,4.5))
    w = 0.8/len(series)
    for i, (leg,vals) in enumerate(series.items()):
        x = [g + (i- (len(series) -1) / 2)*w for g in range(len(groups))]
        ax.bar(x,[v[0] for v in vals],w,label=leg,capsize=2,yerr=[[v[1] for v in vals],[v[2] for v in vals]])
    ax.set_xticks(range(len(groups)), groups)
    ax.set(ylabel="selection time [s]",title=title)
    if log:
        ax.set_yscale("log")
    ax.legend()
    save(fig,name)

def lines(name, title, xs, series, xlabel,ideal=False):
    fig,ax = plt.subplots(figsize=(7,4.5))
    for leg,vals in series.items():
        ax.errorbar(xs,[v[0] for v in vals],marker="o",capsize=3,label=leg,yerr=[[v[1] for v in vals],[v[2] for v in vals]])

    if ideal:
        for vals in series.values():
            ax.plot(xs,[vals[0][0] * xs[0] / x for x in xs],"k:",lw=0.8)
        ax.plot([],[],"k:",label="ideal")
    ax.set(xlabel=xlabel,ylabel="selection time [s]",title=title,xscale="log",yscale="log")
    if ideal:
        ax.set_xticks(xs,[str(x) for x in xs])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.legend()
    save(fig,name)

def by_label(name,title,filename,labels,ticks,log=True):
    runs=parse(filename)
    bars(name,title,ticks,{m:[stats(runs,m,l) for l in labels] for m in MODES},log)

os.makedirs(OUT,exist_ok=True)

# 1. remaining elements per round (first run of each mode)

runs=parse("cfg_scale_16.txt")
fig,ax= plt.subplots(figsize=(7,4.5))
for mode in MODES[:3]:
    s = next(r for r in runs if r["mode"] == mode)["sizes"]
    ax.plot(range(len(s)), s, marker="o",label=mode)
ax.set(xlabel="round",ylabel="elements still alive",yscale="log",title="Shrinking of the problem (N=1e10,16 ranks x 4 threads)")
ax.legend()
save(fig,"1_rounds")

# 2. time against N (all 16 ranks x 4 threads)

size,s16 = parse("cfg_size.txt"), parse("cfg_scale_16.txt")
lines("2_size", "Time against N (16 ranks x 4 threads)", [2e8,2e9,1e10],{m: [stats(size,m,"N=2e8"),stats(size,m,"N=2e9"),stats(s16,m)] for m in MODES}, "N")

# 3. strong scaling

ranks=[8,16,32]
files= [parse(f"cfg_scale_{r}.txt") for r in ranks]
lines("3_scaling","Strong scaling, N=1e10, 4 threads per rank",ranks,{m: [stats(f,m) for f in files] for m in MODES},"MPI ranks",ideal=True)

# 4. effect of k, 5. input distributions

by_label("4_k","Effect of k (N=2e10)","cfg_k.txt",["k=0","k=N/100","k=N/10","k=N-1"],["0","N/100","N/10","N-1"])
dist= ["sorted","reverse","sorted+rebalance","dup-1e6","dup-10"]
by_label("5_distributions","Input distributions (N=2e10)","cfg_dist.txt",dist,dist)

# 6. sample size

runs= parse("cfg_sample.txt")
labels = ["250000", "500000", "1000000", "2000000", "default","4000000", "8000000"]

ticks = []

for l in labels:
    n = next(r for r in runs if r["label"] == f"sample={l}")["rounds"]
    ticks.append(f"{l}\n({n} rounds)")

bars("6_sample", "FR -R sample size (N=2e10)", ticks, {"fr -R": [stats(runs, "fr", f"sample={l}") for l in labels]})

# 7. ranks x threads

layouts = [parse(f"cfg_threads_{x}.txt") for x in ("1x8","2x4","4x2")]
bars("7_layout", "Ranks x threads, same cores (N=1e10)",["16 x 8", "32 x 4", "64 x 2"],{m: [stats(f, m) for f in layouts] for m in MODES}, log=True)

# 8. switch size

sw = [parse(f) for f in ("cfg_threads_4x2_s500k.txt", "cfg_threads4.txt",
                         "cfg_threads16.txt", "cfg_threads64.txt")]
bars("8_switch", "Switch size -s (N=1e10, 64 ranks x 2 threads)",
     ["500k", "4M", "16M", "64M"],
     {m: [stats(f, m) for f in sw] for m in MODES}, log=True)

# 9. big runs, every repetition as a dot

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
big = {"no warm-up": parse("sel_cmp.txt"),
       "warm-up": parse("sel_cmp_warmup.txt")}
for ax, label in zip(axes, ["N=4e10 ranks=128", "N=2.4e11 ranks=128"]):
    for i, runs in enumerate(big.values()):
        t = [r["time"] for r in runs if r["label"] == label]
        ax.plot([i] * len(t), t, "o")
    ax.set_xticks([0, 1], list(big))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(bottom=0)
    ax.set(title=label.replace(" ranks=", ", ranks="), ylabel="time [s]")
save(fig, "9_big_runs")

print("plots written to", OUT)