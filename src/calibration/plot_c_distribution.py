"""
Plot the C (GDP-sensitivity) distribution used by simulate_market.

C is drawn as:
    n_zero  firms have C = 0
    n_power firms have C = c_max * U^(1/c_exponent),  U ~ Uniform(0, 1)

Note: lowering c_exponent makes 1/c_exponent larger, which pushes mass
TOWARD 0 (concentrating sensitivity in a few firms). c_exponent = 1 is
uniform; c_exponent > 1 pushes mass toward c_max.

Usage
-----
    python src/calibration/plot_c_distribution.py
    python src/calibration/plot_c_distribution.py --c_max 10 --c_exponent 0.5
    python src/calibration/plot_c_distribution.py --c_max 10 --c_exponent 0.1 0.25 0.5 1.0 2.0
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sample_c(n_firms, c_max, c_exponent, zero_fraction, seed=42):
    rng = np.random.default_rng(seed)
    n_zero = int(n_firms * zero_fraction)
    n_power = n_firms - n_zero
    c_values = np.concatenate([
        np.zeros(n_zero),
        c_max * rng.uniform(0, 1, n_power) ** (1.0 / c_exponent),
    ])
    rng.shuffle(c_values)
    return c_values


def plot_one(ax, c_values, c_max, c_exponent, zero_fraction, n_firms):
    bins = np.linspace(0, c_max, 40)
    ax.hist(c_values, bins=bins, color="tab:purple", alpha=0.75, edgecolor="black", lw=0.4)
    ax.axvline(np.mean(c_values), color="red", lw=1.2, ls="--",
               label=f"mean = {np.mean(c_values):.3f}")
    ax.axvline(np.median(c_values), color="orange", lw=1.2, ls=":",
               label=f"median = {np.median(c_values):.3f}")
    ax.set_xlabel("C")
    ax.set_ylabel("count")
    n_pos = int((c_values > 0).sum())
    ax.set_title(
        f"c_max={c_max}, c_exponent={c_exponent}, zero_fraction={zero_fraction}\n"
        f"n_firms={n_firms}  |  zero={n_firms - n_pos}  positive={n_pos}  "
        f"max={c_values.max():.3f}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_firms", type=int, default=250)
    parser.add_argument("--c_max", type=float, default=10.0)
    parser.add_argument("--c_exponent", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--zero_fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/c_distribution.pdf")
    args = parser.parse_args()

    n_panels = len(args.c_exponent)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(8, 3.0 * n_panels),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, alpha in zip(axes, args.c_exponent):
        c_values = sample_c(
            n_firms=args.n_firms,
            c_max=args.c_max,
            c_exponent=alpha,
            zero_fraction=args.zero_fraction,
            seed=args.seed,
        )
        plot_one(ax, c_values, args.c_max, alpha, args.zero_fraction, args.n_firms)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
