import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.DirectoryManager import DirectoryManager

def moving_average(x, w=5):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

def read_log(logpath):
    episodes, steps, rewards, eps, buffers = [], [], [], [], []
    with open(logpath, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split(",")
            if len(parts) < 3:
                continue
            try:
                ep = int(parts[0])
                st = int(parts[1])
                rw = float(parts[2])
                epsv = float(parts[3]) if len(parts) > 3 else np.nan
                buf = int(parts[4]) if len(parts) > 4 else 0
            except Exception:
                continue
            episodes.append(ep)
            steps.append(st)
            rewards.append(rw)
            eps.append(epsv)
            buffers.append(buf)
    return np.array(episodes), np.array(steps), np.array(rewards), np.array(eps), np.array(buffers)

def plot(logfile, out=None, smooth=1):
    logfile = Path(logfile)
    if not logfile.exists():
        raise FileNotFoundError(logfile)
    ep, steps, rewards, eps, buffers = read_log(str(logfile))
    if len(rewards) == 0:
        raise ValueError("No data found in log file.")

    plt.figure(figsize=(10,5))
    if smooth > 1:
        ma = moving_average(rewards, smooth)
        x_ma = ep[:len(ma)]
        plt.plot(x_ma, ma, label=f"Reward (MA w={smooth})", color="tab:blue")
        plt.plot(ep, rewards, color="tab:blue", alpha=0.2, label="Reward (raw)")
    else:
        plt.plot(ep, rewards, label="Reward", color="tab:blue")

    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.grid(True)
    plt.title(f"Learning Curve — {logfile.name}")
    plt.legend(loc="upper right")

    # secondary axis: epsilon or buffer size if present
    if eps.size and not np.all(np.isnan(eps)):
        ax2 = plt.twinx()
        ax2.plot(ep, eps, color="tab:orange", linestyle="--", label="epsilon")
        ax2.set_ylabel("Epsilon")
        ax2.legend(loc="lower right")

    if out is None:
        out = logfile.with_suffix(".png")
    else:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved plot to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("timestamp", help="timestamp of the experiment to plot")
    p.add_argument("--smooth", type=int, default=10, help="moving average window (int)")
    args = p.parse_args()

    dm = DirectoryManager(args.timestamp)
    for formula_name in dm.get_formula_names():
        dm.set_formula_name(formula_name)
        for log in dm.get_logs():
            print(f"Plotting log: {log}")
            out = dm.get_plot_folder() + f"plot_{log.stem}.png"
            plot(log, out, smooth=args.smooth)