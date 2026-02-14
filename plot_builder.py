import argparse
import numpy as np
import matplotlib.pyplot as plt

def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = min(window, len(x))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Path to .npz log file")
    ap.add_argument("--window", type=int, default=50, help="Rolling window in EPISODES")
    ap.add_argument("--out", type=str, default="", help="Optional output png path")
    args = ap.parse_args()

    d = np.load(args.path)
    returns = d["returns"].astype(np.float32)
    t_end = d["term_time_steps"].astype(np.int64)

    if len(returns) == 0:
        raise RuntimeError("No episodes in log (returns is empty).")

    w = min(args.window, len(returns))

    y = rolling_mean(returns, w)
    x = t_end if w <= 1 else t_end[w - 1:]  # align with 'valid'

    plt.figure()
    plt.plot(t_end, returns, alpha=0.25, label="episode return (raw)")
    plt.plot(x, y, label=f"rolling mean over {w} episodes")
    plt.xlabel("Step (episode end)")
    plt.ylabel("Episodic return")
    plt.title("Episodic return vs step")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    main()
