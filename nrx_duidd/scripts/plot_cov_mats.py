#!/usr/bin/env python3
"""Heatmaps of HLS space / time / freq covariance matrices.
Author: Nuri Berke Baytekin
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_cov(path: Path) -> np.ndarray:
    mat = np.load(path)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{path}: expected square matrix, got {mat.shape}")
    return mat


def plot_abs(ax, mat, title):
    abs_mat = np.abs(mat)
    vmax = (np.percentile(abs_mat, 99.5) if abs_mat.size > 64
            else float(abs_mat.max()))
    im = ax.imshow(abs_mat, origin="upper", aspect="equal",
                   cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    return im


def plot_log_abs(ax, mat, title, eps=1e-12):
    log_mat = np.log10(np.abs(mat) + eps)
    im = ax.imshow(log_mat, origin="upper", aspect="equal", cmap="magma")
    ax.set_title(title)
    return im


def plot_phase(ax, mat, title):
    phase = np.angle(mat)
    im = ax.imshow(phase, origin="upper", aspect="equal",
                   cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax.set_title(title)
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path("../weights/duidd_12_lmmse_idd_hls"),
        help="Path prefix before _{space,time,freq}_cov_mat.npy",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cov_mat_heatmaps.png"),
        help="Output figure path",
    )
    parser.add_argument(
        "--freq-zoom",
        type=int,
        default=0,
        help="Also plot top-left freq zoom of this size (0 disables)",
    )
    args = parser.parse_args()

    panels = [
        ("space", load_cov(Path(f"{args.prefix}_space_cov_mat.npy"))),
        ("time", load_cov(Path(f"{args.prefix}_time_cov_mat.npy"))),
        ("freq", load_cov(Path(f"{args.prefix}_freq_cov_mat.npy"))),
        ("freq (naive)", load_cov(Path(f"{args.prefix}_freq_cov_mat_naive.npy")))
    ]
    if args.freq_zoom > 0:
        freq_panels = panels[2:4]
        for name, mat in freq_panels:
            z = min(args.freq_zoom, mat.shape[0])
            panels.append((f"{name}[:{z}]", mat[:z, :z]))

    n_cols = len(panels)
    row_specs = (
        ("|C|", plot_abs),
        ("log10|C|", plot_log_abs),
        ("∠C", plot_phase),
    )
    fig, axes = plt.subplots(
        len(row_specs), n_cols,
        figsize=(4.0 * n_cols, 3.8 * len(row_specs)),
        squeeze=False,
    )

    for col, (name, mat) in enumerate(panels):
        for row, (kind, plot_fn) in enumerate(row_specs):
            ax = axes[row, col]
            im = plot_fn(ax, mat, f"{kind}  {name}\nshape={mat.shape}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == len(row_specs) - 1:
                ax.set_xlabel("column")
            if col == 0:
                ax.set_ylabel("row")

    fig.suptitle(f"Covariance heatmaps: {args.prefix.name}", y=1.01)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Saved {args.out}")
    for name, mat in panels[:3]:
        print(f"  {name}: shape={mat.shape}, "
              f"|C| mean={np.mean(np.abs(mat)):.4e}, "
              f"diag mean={np.mean(np.real(np.diag(mat))):.4e}")


if __name__ == "__main__":
    main()
