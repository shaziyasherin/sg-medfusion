"""Show which (experiment, seed) runs are finished and a rough time remaining.

    python scripts/progress.py --config configs/paths_aimlserver.json --plan minimal --seeds 0 1 2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import EXPERIMENTS, MINIMAL  # noqa: E402
from sgm import config as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--plan", choices=["minimal", "full"], default="minimal")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
a = ap.parse_args()
exps = MINIMAL if a.plan == "minimal" else list(EXPERIMENTS)
root = Path(C.load(a.config)["paths"]["results_dir"])
done, gpu_times, todo_gpu = 0, [], 0
print("%-24s %s" % ("experiment", "   ".join("rep %d" % s for s in a.seeds)))
for e in exps:
    gpu = "_sk" not in EXPERIMENTS[e]
    marks = []
    for s in a.seeds:
        f = root / e / ("seed%d.json" % s)
        if f.exists():
            r = json.load(open(f))
            if gpu:
                gpu_times.append(r.get("wall_time_s", 0))
            marks.append("%.3f" % r["test"]["main"]["auroc"])
            done += 1
        elif (root / e / ("seed%d.ckpt" % s)).exists():
            marks.append("half ")
            todo_gpu += gpu
        else:
            marks.append("  -  ")
            todo_gpu += gpu
    print("%-24s %s%s" % (e, "   ".join(marks), "" if gpu else "   (CPU only)"))
total = len(exps) * len(a.seeds)
print("\n%d of %d runs finished.  Numbers = test AUROC, 'half' = stopped midway (will resume)." % (done, total))
if gpu_times:
    avg = sum(gpu_times) / len(gpu_times)
    print("A GPU run takes about %.0f minutes here -> about %.1f GPU-hours left."
          % (avg / 60, avg * todo_gpu / 3600))
elif done < total:
    print("Time left will be shown after the first GPU run finishes.")
