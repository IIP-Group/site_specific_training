#!/usr/bin/python3
"""Inspect trained DUIDD weight pickles saved by save_weights().

Example:
  python inspect_duidd_weights.py duidd_pixel9pro_j61_2ULL_slot14_mcs10

Author: Nuri Berke Baytekin
"""

import argparse
import os
import pickle
from datetime import datetime

import numpy as np

WEIGHT_NAMES = [
    "alpha",
    "beta",
    "delta",
    "epsilon",
    "eta",
    "gamma",
    "mu",
    "xi",
]


EDGE_WEIGHT_NAMES = [
    "edge_weights_siso_decoder",
    "edge_weights_output_decoder",
]


def resolve_weights_path(path_or_label):
    """Accept a file path or a config label (resolves to ../weights/<label>_weights)."""
    if os.path.isfile(path_or_label):
        return path_or_label

    candidates = [
        path_or_label,
        f"{path_or_label}_weights",
        os.path.join("..", "weights", f"{path_or_label}_weights"),
        os.path.join("weights", f"{path_or_label}_weights"),
        os.path.join("..", "weights", path_or_label),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Weights not found for '{path_or_label}'. Tried: {candidates}"
    )


def default_init(name, shape):
    """Return the initialization used in DUIDDPUSCHReceiver for ``name``."""
    if name in ("alpha", "delta", "gamma"):
        return np.ones(shape, dtype=np.float32)
    if name in ("beta", "epsilon", "mu", "xi"):
        return np.zeros(shape, dtype=np.float32)
    if name == "eta":
        return np.float32(1.0)
    raise ValueError(f"Unknown weight name: {name}")


def format_array(arr, precision=6):
    with np.printoptions(precision=precision, suppress=True, linewidth=120):
        return str(np.asarray(arr))


def split_scalar_and_edge_weights(weights):
    """Split pickle content into DUIDD scalar tensors and LDPC edge weights.

    Edge weights (weighted_bp=True) are large 1-D tensors (one per-edge weight
    per LDPC decoder); DUIDD scalars are small (len I, scalar, or [I, max_bp]).
    """
    scalars, edges = [], []
    for w in weights:
        arr = np.asarray(w)
        if arr.ndim == 1 and arr.size > 1000:
            edges.append(arr)
        else:
            scalars.append(arr)
    return scalars, edges


def summarize_edge_weights(name, arr, precision=6):
    init = 1.0  # Sionna initializes per-edge weights to 1
    delta = arr - init
    frac_changed = float(np.mean(np.abs(delta) > 1e-6))
    print(f"\n=== {name} ===")
    print(f"shape={arr.shape}, dtype={arr.dtype} (init: all {init})")
    with np.printoptions(precision=precision, suppress=True):
        print(f"min={arr.min():.6f}, max={arr.max():.6f}, "
              f"mean={arr.mean():.6f}, std={arr.std():.6f}")
        print(f"fraction changed from init: {frac_changed:.4f}")
        print(f"largest deviations from 1: {np.sort(np.abs(delta))[-5:][::-1]}")


def inspect_weights(weights_path, precision=6):
    with open(weights_path, "rb") as f:
        weights = pickle.load(f)

    scalars, edges = split_scalar_and_edge_weights(weights)

    if len(scalars) != len(WEIGHT_NAMES):
        raise ValueError(
            f"Expected {len(WEIGHT_NAMES)} scalar tensors "
            f"({', '.join(WEIGHT_NAMES)}), got {len(scalars)} "
            f"(+{len(edges)} edge-weight tensors)"
        )
    if edges and len(edges) != len(EDGE_WEIGHT_NAMES):
        raise ValueError(
            f"Expected {len(EDGE_WEIGHT_NAMES)} edge-weight tensors "
            f"with weighted_bp=True, got {len(edges)}"
        )

    print(f"File: {weights_path}")
    print(f"Modified: {datetime.fromtimestamp(os.path.getmtime(weights_path))}")
    print(f"Size bytes: {os.path.getsize(weights_path)}")
    print(f"Num weight tensors: {len(weights)} "
          f"({len(scalars)} DUIDD scalars, {len(edges)} LDPC edge tensors; "
          f"weighted_bp={'True' if edges else 'False'})")

    alpha = scalars[0]
    mu = scalars[6]
    num_idd = int(alpha.shape[0]) if alpha.ndim > 0 else 1
    max_bp = int(mu.shape[1]) if mu.ndim == 2 else None
    print(f"Inferred IDD stages I={num_idd}"
          + (f", max BP iters={max_bp}" if max_bp is not None else ""))

    for i, (name, arr) in enumerate(zip(WEIGHT_NAMES, scalars)):
        init = np.asarray(default_init(name, arr.shape))
        print(f"\n=== [{i}] {name} ===")
        print(f"shape={arr.shape}, dtype={arr.dtype}")
        print("trained:")
        print(format_array(arr, precision=precision))
        print("init:")
        print(format_array(init, precision=precision))

    for name, arr in zip(EDGE_WEIGHT_NAMES, edges):
        summarize_edge_weights(name, arr, precision=precision)


def main():
    parser = argparse.ArgumentParser(
        description="Print trained DUIDD weights from a pickle saved by save_weights()."
    )
    parser.add_argument(
        "weights",
        help="Path to weights pickle, or config label "
             "(e.g. duidd_pixel9pro_j61_2ULL_slot14_mcs10)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Print precision for float values (default: 6)",
    )
    args = parser.parse_args()

    path = resolve_weights_path(args.weights)
    inspect_weights(path, precision=args.precision)


if __name__ == "__main__":
    main()
