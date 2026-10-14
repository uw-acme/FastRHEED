"""Hit rates of the FOLO benchmark runs on the synthetic frames, per step and match radius.

A ground-truth spot (synthetic_ood.h5 sim_<k>_params, frame pixels) is found if a TopCrop-selected
FOLO cell's centre lies within r px of it (nearest pick per spot; picks are not matched one-to-one).
Cell centre in frame pixels = 8*cell + 4, minus the 10 px pad for the 40x40 (padded-input) eval
commits; no shift for the unpadded ones (UNPADDED).

Each run directory (<label>_<folo-hash>[_seed<s>]/ or old[_seed<s>]/, written by run_step.sh)
holds run_info.txt and eval/sim_<k>_0/inference.npz. A run is scored only if it has every sim.

Writes, into --out:
  scores.md  step x radius summary, then one table per radius (stats over seeds, per-sim rates)
  runs.csv   one row per run and radius (found per sim and in total)

Usage: score_hits.py [--archive DIR] [--steps old 00 04 ...] [--radii 4 8 ...] [--out DIR]
"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

R = Path(__file__).resolve().parents[1]
SYNTH = R / "Rheed-Eval/synthetic_RHEED/synthetic_ood.h5"
SIMS = ["sim_3", "sim_4", "sim_5", "sim_5b", "sim_7"]
UNPADDED = {"7dee78e", "49613c2", "old"}  # eval commits whose FOLO input is the unpadded frame
PAD = 10
LABEL_ALIAS = {"00b": "00"}  # 00b = extra step-00 seeds run under another label


def load_truth():
    with h5py.File(SYNTH) as h:
        return {s: h[f"{s}_params"][:, :2] for s in SIMS}  # (mu_x, mu_y) = (col, row)


def run_info(run):
    first = (run / "run_info.txt").read_text().split("\n")[0]
    info = dict(kv.split("=", 1) for kv in first.split())
    info["step"] = LABEL_ALIAS.get(info["label"], info["label"])
    return info


def min_dists(npz, truth, shift):
    """Distance from each true spot to its nearest selected cell centre (inf if none)."""
    cells = np.load(npz)["folo_cells"]  # (x, y) cells, -1 rows = no pick
    centres = 8.0 * cells[cells[:, 0] >= 0] + 4 + shift
    if not len(centres):
        return np.full(len(truth), np.inf)
    return np.linalg.norm(truth[:, None, :] - centres[None, :, :], axis=2).min(axis=1)


def collect(archive, steps, truth):
    """{step: [(run name, seed, {sim: min distances})]} for every complete run of the steps."""
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
        dists = {s: min_dists(npzs[s], truth[s], shift) for s in SIMS}
        runs[info["step"]].append((run.name, info["seed"], dists))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=R / "archive/bad_to_good")
    ap.add_argument("--steps", nargs="+", default=["old", "00", "04", "09", "11", "13", "17", "18", "19"])
    ap.add_argument("--radii", nargs="+", type=float, default=[4, 8, 12, 16, 20])
    ap.add_argument("--out", type=Path, default=None, help="default: <archive>/scores")
    args = ap.parse_args()
    out = args.out or args.archive / "scores"
    out.mkdir(parents=True, exist_ok=True)

    truth = load_truth()
    n_spots = {s: len(truth[s]) for s in SIMS}
    total = sum(n_spots.values())
    runs = collect(args.archive, args.steps, truth)

    # found[step][r] = (n_runs, n_sims) spots found
    found = {
        st: {r: np.array([[int((d[s] <= r).sum()) for s in SIMS] for _, _, d in runs[st]]).reshape(-1, len(SIMS))
             for r in args.radii}
        for st in args.steps
    }

    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "run", "seed", "radius_px", *SIMS, "found", "spots"])
        for st in args.steps:
            for r in args.radii:
                for (name, seed, _), row in zip(runs[st], found[st][r]):
                    w.writerow([st, name, seed, f"{r:g}", *row, row.sum(), total])

    def rad(r):
        return f"{r:g} px"

    lines = [
        "# FOLO synthetic hit rates",
        "",
        f"Archive `{args.archive.relative_to(R) if args.archive.is_relative_to(R) else args.archive}`; "
        f"{total} ground-truth spots over {', '.join(f'{s} ({n_spots[s]})' for s in SIMS)}. "
        "A spot is found if a selected FOLO cell centre is within r px (nearest pick per spot). "
        "Hit rate = spots found / spots, per seed; the stats below are over seeds.",
        "",
        "## Mean hit rate (± std over seeds)",
        "",
        "| step | seeds | " + " | ".join(rad(r) for r in args.radii) + " |",
        "|---|---|" + "---|" * len(args.radii),
    ]
    for st in args.steps:
        cells = []
        for r in args.radii:
            h = found[st][r].sum(axis=1) / total
            cells.append(f"{100 * h.mean():.1f} ± {100 * h.std(ddof=1):.1f}%" if len(h) > 1 else
                         f"{100 * h.mean():.1f}%" if len(h) else "—")
        lines.append(f"| {st} | {len(runs[st])} | " + " | ".join(cells) + " |")

    for r in args.radii:
        lines += [
            "",
            f"## r = {rad(r)}",
            "",
            f"| step | seeds | mean found /{total} | std | median | min | max | all found | "
            + " | ".join(f"{s} hit rate" for s in SIMS) + " |",
            "|---|---|---|---|---|---|---|---|" + "---|" * len(SIMS),
        ]
        for st in args.steps:
            fd = found[st][r]
            if not len(fd):
                lines.append(f"| {st} | 0 |" + " — |" * (6 + len(SIMS)))
                continue
            t = fd.sum(axis=1)
            per_sim = fd.mean(axis=0) / np.array([n_spots[s] for s in SIMS])
            lines.append(
                f"| {st} | {len(t)} | {t.mean():.2f} | {t.std(ddof=1) if len(t) > 1 else 0:.2f} | {np.median(t):g} "
                f"| {t.min()} | {t.max()} | {100 * (t == total).mean():.0f}% | "
                + " | ".join(f"{100 * p:.0f}%" for p in per_sim) + " |"
            )

    (out / "scores.md").write_text("\n".join(lines) + "\n")
    i = lines.index("## Mean hit rate (± std over seeds)")
    print("\n".join(lines[i:i + 4 + len(args.steps)]))
    print(f"\nWrote {out / 'scores.md'} and {out / 'runs.csv'}")


if __name__ == "__main__":
    main()
