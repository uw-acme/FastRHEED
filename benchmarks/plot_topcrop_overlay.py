"""TopCrop boxes of every run of a step, overlaid on each synthetic frame.

For each step and each synthetic frame, draws the 48x48 TopCrop box of every selected FOLO cell
from every complete run of that step (one box outline per pick, translucent so that boxes picked
by many seeds look solid), plus the ground-truth spot centres. Same box geometry as
inference.py's topcrop_out.png: box top-left = 8*cell - 20, minus the 10 px pad for the padded
eval commits (score_hits.UNPADDED lists the unpadded ones).

Writes <out>/<step>/topcrop_out_<sim>.png, and the same figures as one PDF per step with a page
per synthetic frame (SIMS order): <out>/<step>/topcrop_overlay_<step>.pdf.

Usage: plot_topcrop_overlay.py [--alpha 0.15] [--archive DIR] [--steps old 00 ...] [--out DIR]
"""

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from score_hits import PAD, R, SIMS, SYNTH, UNPADDED, run_info

CELL_SIZE = 8
CROP_DIM = 48
BOX_ORIGIN_OFFS = CELL_SIZE // 2 - CROP_DIM // 2  # -20: box top-left = 8*cell + OFFS (top_crop.py)
FRAME_DIM = 300
CMAP = "viridis"
BOX_COLOR = "yellow"
TRUTH_COLOR = "red"


def collect_cells(archive, steps):
    """{step: [(run name, {sim: (k, 2) box top-left corners in frame pixels})]} for complete runs."""
    runs = {s: [] for s in steps}
    for run in sorted(archive.iterdir()):
        if not (run / "run_info.txt").exists():
            continue
        info = run_info(run)
        if info["step"] not in runs:
            continue
        npzs = {s: run / "eval" / f"{s}_0" / "inference.npz" for s in SIMS}
        if not all(p.exists() for p in npzs.values()):
            continue
        shift = 0 if info["eval"] in UNPADDED else -PAD
        corners = {}
        for s, p in npzs.items():
            cells = np.load(p)["folo_cells"]
            corners[s] = CELL_SIZE * cells[cells[:, 0] >= 0] + BOX_ORIGIN_OFFS + shift
        runs[info["step"]].append((run.name, corners))
    return runs


def plot(frame, truth, corners, alpha, title, path, pdf):
    """Save the overlay figure to the PNG at path and append it as a page to pdf (a PdfPages)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
    im = ax.imshow(frame, cmap=CMAP, vmin=0, vmax=frame.max(), interpolation="none")
    for x0, y0 in corners:
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), CROP_DIM, CROP_DIM, fill=False, ec=BOX_COLOR, lw=1, alpha=alpha))
    ax.scatter(truth[:, 0], truth[:, 1], marker="+", c=TRUTH_COLOR, s=40, lw=1, label="ground truth")
    ax.set_xlim(-0.5, FRAME_DIM - 0.5)
    ax.set_ylim(FRAME_DIM - 0.5, -0.5)
    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="counts")
    fig.savefig(path, dpi=150)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=0.15, help="opacity of each box outline")
    ap.add_argument("--archive", type=Path, default=R / "archive/bad_to_good")
    ap.add_argument("--steps", nargs="+", default=["old", "00", "04", "09", "11", "13", "17", "18", "19"])
    ap.add_argument("--out", type=Path, default=None, help="default: <archive>/scores/topcrop_overlay")
    args = ap.parse_args()
    out = args.out or args.archive / "scores" / "topcrop_overlay"

    with h5py.File(SYNTH) as h:
        frames = {s: h[s][0] for s in SIMS}
        truth = {s: h[f"{s}_params"][:, :2] for s in SIMS}

    runs = collect_cells(args.archive, args.steps)
    for step in args.steps:
        if not runs[step]:
            print(f"{step}: no complete runs, skipped")
            continue
        (out / step).mkdir(parents=True, exist_ok=True)
        with PdfPages(out / step / f"topcrop_overlay_{step}.pdf") as pdf:
            for s in SIMS:
                corners = np.concatenate([c[s] for _, c in runs[step]]).reshape(-1, 2)
                title = f"step {step}, {s}: TopCrop boxes of {len(runs[step])} models"
                plot(frames[s], truth[s], corners, args.alpha, title, out / step / f"topcrop_out_{s}.png", pdf)
        print(f"{step}: {len(runs[step])} runs -> {out / step}")


if __name__ == "__main__":
    main()
