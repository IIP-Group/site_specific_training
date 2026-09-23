#!/usr/bin/env python3
"""Estimate PUSCH frequency covariance from direct DM-RS samples.

This script is the companion to ``PUSCHLSChannelEstimatorNoOCC`` from
``pusch_ls_channel_estimator_no_occ.py``.  That estimator skips Sionna 0.19.2's
frequency- and time-domain OCC combining but can retain Sionna's ordinary
interpolation.  This script uses only the original, non-interpolated pilot
positions:

The values on the other subcarrier parity are ignored even when they contain
interpolated channel estimates.  The implementation assumes that ports 0 and
2 are the only active ports and occupy different type-1 DM-RS CDM groups.

Three absolute-scale covariance estimates are supported:

``naive``
    Linearly interpolate each direct pilot observation to all active
    subcarriers, compute its empirical second moment, subtract the covariance
    of noise processed by the same interpolation, and project the result onto
    the positive-semidefinite cone.  No Toeplitz or WSSUS model is imposed.

``lag``
    Estimate the noise-corrected correlation at the directly observable even
    frequency lags, linearly interpolate the unavailable odd lags, and form the
    corresponding Hermitian Toeplitz covariance.  The final unbracketed odd
    lag uses the nearest available even-lag estimate.  This method assumes
    frequency stationarity and a smoothly varying frequency-correlation
    sequence, but it does not assume a PDP model.
    The raw Toeplitz matrix is returned by default.  An optional PSD projection
    is available, although the projected matrix is generally no longer exactly
    Toeplitz.

``pdp``
    Compute the noise-corrected covariance of the direct pilot observations and
    fit a nonnegative, CP-supported PDP under a Toeplitz/WSSUS approximation.
    The fitted PDP then generates the full active-band Toeplitz covariance.

This is research code and it is not optimized for speed, memory or neatness. The code is a mere
demonstration of the proposed method in the accompanying paper.

Author: Nuri Berke Baytekin
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import nnls


Method = Literal["naive", "pdp"]
PublicMethod = Literal[
    "naive", "lag", "pdp", "both", "lag-and-pdp", "all"
]


@dataclass
class DatasetData:
    """Input arrays and metadata for one measurement dataset."""

    name: str
    h_ls: np.ndarray
    layer_mask: Optional[np.ndarray] = None
    ports: Optional[np.ndarray] = None
    noise_ls: Optional[np.ndarray] = None
    err_var: Optional[np.ndarray] = None
    dmrs_symbol_indices: Optional[np.ndarray] = None
    active_subcarrier_indices: Optional[np.ndarray] = None
    weight: float = 1.0


@dataclass
class DatasetStatistics:
    """Second-order statistics for one dataset and one estimation domain."""

    name: str
    channel_covariance: np.ndarray
    noise_covariance: np.ndarray
    num_channel_vectors: int
    num_noise_vectors: int
    mean_pre_occ_err_var: Optional[float]
    port_vector_counts: dict[int, int]


@dataclass
class DirectCombCovarianceEstimate:
    """Outputs of the naive, lag-interpolated, and/or PDP-based estimator."""

    cov_mat_freq_naive: Optional[np.ndarray]
    cov_mat_freq_naive_raw: Optional[np.ndarray]
    cov_mat_freq_lag: Optional[np.ndarray]
    cov_mat_freq_lag_raw: Optional[np.ndarray]
    cov_mat_freq_pdp: Optional[np.ndarray]
    pdp: Optional[np.ndarray]
    frequency_correlation: Optional[np.ndarray]
    frequency_correlation_lag: Optional[np.ndarray]
    observed_even_frequency_correlation: Optional[np.ndarray]
    observed_comb_covariance: Optional[np.ndarray]
    direct_noise_covariance: Optional[np.ndarray]
    model_comb_covariance: Optional[np.ndarray]
    metadata: dict


def _default_ports(num_layers: int) -> np.ndarray:
    if num_layers == 1:
        return np.asarray([0], dtype=np.int64)
    if num_layers == 2:
        return np.asarray([0, 2], dtype=np.int64)
    raise ValueError("Only one- and two-layer datasets are supported.")


def _comb_indices(port: int, n_sc: int) -> np.ndarray:
    """Return the true zero-based pilot positions for port 0 or port 2."""

    if n_sc % 2 != 0:
        raise ValueError("n_sc must be even.")
    if port == 0:
        offset = 0
    elif port == 2:
        offset = 1
    else:
        raise ValueError(
            f"Unsupported DM-RS port {port}; expected port 0 or port 2."
        )
    return np.arange(offset, n_sc, 2, dtype=np.int64)


def _validate_layer_mask(
    layer_mask: Optional[np.ndarray],
    num_slots: int,
    num_layers: int,
    name: str,
) -> np.ndarray:
    if layer_mask is None:
        return np.ones((num_slots, num_layers), dtype=bool)
    mask = np.asarray(layer_mask, dtype=bool)
    if mask.shape != (num_slots, num_layers):
        raise ValueError(
            f"{name}: layer_mask must have shape {(num_slots, num_layers)}, "
            f"received {mask.shape}."
        )
    return mask


def _select_axis_indices(
    x: np.ndarray,
    *,
    axis: int,
    indices: Optional[np.ndarray],
    expected_size: int,
    array_name: str,
) -> np.ndarray:
    """Select an axis or verify that it already has the expected size."""

    if x.shape[axis] == expected_size and indices is None:
        return x
    if indices is None:
        raise ValueError(
            f"{array_name}: axis {axis} has size {x.shape[axis]}; provide "
            f"indices selecting the required {expected_size} entries."
        )
    ind = np.asarray(indices, dtype=np.int64)
    if ind.shape != (expected_size,):
        raise ValueError(
            f"{array_name}: selection indices must have shape "
            f"{(expected_size,)}, received {ind.shape}."
        )
    if np.any(ind < 0) or np.any(ind >= x.shape[axis]):
        raise ValueError(f"{array_name}: selection index is out of range.")
    return np.take(x, ind, axis=axis)


def _reshape_flat_pilot_axis(
    x: np.ndarray,
    *,
    num_dmrs_symbols: int,
    n_sc: int,
    n_comb: int,
    array_name: str,
    allow_single_symbol: bool,
) -> np.ndarray:
    """Split a flattened pilot axis into [time, frequency]."""

    total = x.shape[-1]
    candidates: list[tuple[int, int]] = []
    for num_time in ([num_dmrs_symbols, 1] if allow_single_symbol else [num_dmrs_symbols]):
        for num_freq in (n_sc, n_comb):
            if total == num_time * num_freq:
                candidates.append((num_time, num_freq))
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise ValueError(
            f"{array_name}: flattened final dimension {total} is ambiguous or "
            f"does not match {num_dmrs_symbols} DM-RS symbols with {n_sc} "
            f"dense or {n_comb} compact entries."
        )
    num_time, num_freq = candidates[0]
    return x.reshape(*x.shape[:-1], num_time, num_freq)


def _canonicalize_estimates(
    array: np.ndarray,
    *,
    num_dmrs_symbols: int,
    n_sc: int,
    dmrs_symbol_indices: Optional[np.ndarray],
    active_subcarrier_indices: Optional[np.ndarray],
    array_name: str,
    allow_single_symbol: bool = False,
) -> np.ndarray:
    """Return [slot, rx, layer, DM-RS symbol, dense/compact frequency]."""

    x = np.asarray(array)
    n_comb = n_sc // 2

    # Full interpolated estimator output:
    # [batch, num_rx, num_rx_ant, num_tx, stream, symbol, subcarrier].
    if x.ndim == 7:
        if x.shape[3] != 1:
            raise ValueError(
                f"{array_name}: full Sionna layout requires num_tx=1; "
                f"received shape {x.shape}."
            )
        num_time = x.shape[5]
        if num_time == num_dmrs_symbols and dmrs_symbol_indices is None:
            pass
        elif allow_single_symbol and num_time == 1:
            pass
        else:
            if dmrs_symbol_indices is None:
                raise ValueError(
                    f"{array_name}: provide dmrs_symbol_indices to select the "
                    f"{num_dmrs_symbols} DM-RS symbols from {num_time} symbols."
                )
            dmrs_ind = np.asarray(dmrs_symbol_indices, dtype=np.int64)
            if dmrs_ind.shape != (num_dmrs_symbols,):
                raise ValueError(
                    f"{array_name}: dmrs_symbol_indices must have shape "
                    f"{(num_dmrs_symbols,)}."
                )
            x = np.take(x, dmrs_ind, axis=5)

        x = _select_axis_indices(
            x,
            axis=6,
            indices=active_subcarrier_indices,
            expected_size=n_sc,
            array_name=array_name,
        )
        batch, num_rx, num_rx_ant, _, num_layers, num_time, num_freq = x.shape
        x = x[:, :, :, 0, :, :, :]
        return x.reshape(
            batch, num_rx * num_rx_ant, num_layers, num_time, num_freq
        )

    # Pilot-domain output with interpolation_type=None:
    # [batch, num_rx, num_rx_ant, num_tx, stream, flattened pilots].
    if x.ndim == 6:
        if x.shape[3] != 1:
            raise ValueError(
                f"{array_name}: pilot-domain Sionna layout requires num_tx=1; "
                f"received shape {x.shape}."
            )
        batch, num_rx, num_rx_ant, _, num_layers, _ = x.shape
        x = x[:, :, :, 0, :, :]
        x = x.reshape(batch, num_rx * num_rx_ant, num_layers, x.shape[-1])

    # Flattened canonical layout [slot, rx, layer, time*frequency].
    if x.ndim == 4:
        x = _reshape_flat_pilot_axis(
            x,
            num_dmrs_symbols=num_dmrs_symbols,
            n_sc=n_sc,
            n_comb=n_comb,
            array_name=array_name,
            allow_single_symbol=allow_single_symbol,
        )

    if x.ndim != 5:
        raise ValueError(
            f"{array_name}: expected a supported 4-D, 5-D, 6-D, or 7-D "
            f"layout; received shape {x.shape}."
        )
    if x.shape[-1] not in (n_sc, n_comb):
        raise ValueError(
            f"{array_name}: final dimension must be {n_sc} or {n_comb}; "
            f"received {x.shape[-1]}."
        )
    if not (x.shape[-2] == num_dmrs_symbols or (allow_single_symbol and x.shape[-2] == 1)):
        raise ValueError(
            f"{array_name}: expected {num_dmrs_symbols} DM-RS symbols"
            f"{' or one noise-only symbol' if allow_single_symbol else ''}; "
            f"received {x.shape[-2]}."
        )
    return x


def _canonicalize_err_var(
    err_var: np.ndarray,
    *,
    original_h_ls: np.ndarray,
    canonical_h_shape: tuple[int, ...],
    num_dmrs_symbols: int,
    n_sc: int,
    dmrs_symbol_indices: Optional[np.ndarray],
    active_subcarrier_indices: Optional[np.ndarray],
    array_name: str,
) -> np.ndarray:
    """Canonicalize an error variance, including Sionna-broadcastable input."""

    ev = np.asarray(err_var)
    if ev.ndim == 0:
        value = float(ev)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{array_name}: scalar err_var must be nonnegative.")
        return ev

    # Sionna may return a tensor that is only broadcastable to h_ls.  Expand it
    # in the original layout before applying the same axis selections.
    try:
        ev_raw = np.broadcast_to(ev, np.asarray(original_h_ls).shape)
        ev_can = _canonicalize_estimates(
            ev_raw,
            num_dmrs_symbols=num_dmrs_symbols,
            n_sc=n_sc,
            dmrs_symbol_indices=dmrs_symbol_indices,
            active_subcarrier_indices=active_subcarrier_indices,
            array_name=array_name,
        )
    except ValueError:
        ev_can = _canonicalize_estimates(
            ev,
            num_dmrs_symbols=num_dmrs_symbols,
            n_sc=n_sc,
            dmrs_symbol_indices=dmrs_symbol_indices,
            active_subcarrier_indices=active_subcarrier_indices,
            array_name=array_name,
        )

    try:
        return np.broadcast_to(ev_can, canonical_h_shape)
    except ValueError as exc:
        raise ValueError(
            f"{array_name}: canonical err_var shape {ev_can.shape} is not "
            f"broadcastable to h_ls shape {canonical_h_shape}."
        ) from exc


def _select_direct_comb(x: np.ndarray, *, port: int, n_sc: int) -> np.ndarray:
    """Discard interpolated positions and retain the true pilot parity."""

    n_comb = n_sc // 2
    if x.shape[-1] == n_comb:
        return x
    if x.shape[-1] != n_sc:
        raise ValueError("Unexpected frequency dimension.")
    return x[..., _comb_indices(port, n_sc)]


def _interpolation_matrix(port: int, n_sc: int) -> np.ndarray:
    """Map direct pilot observations to all active subcarriers.

    At the single unbracketed grid boundary, the nearest direct estimate is
    retained, matching the convention in the accompanying derivation.
    """

    n_comb = n_sc // 2
    mat = np.zeros((n_sc, n_comb), dtype=np.float64)
    if port == 0:
        m = np.arange(n_comb)
        mat[2 * m, m] = 1.0
        m_inner = np.arange(n_comb - 1)
        mat[2 * m_inner + 1, m_inner] = 0.5
        mat[2 * m_inner + 1, m_inner + 1] = 0.5
        mat[-1, -1] = 1.0
    elif port == 2:
        m = np.arange(n_comb)
        mat[2 * m + 1, m] = 1.0
        mat[0, 0] = 1.0
        m_inner = np.arange(1, n_comb)
        mat[2 * m_inner, m_inner - 1] = 0.5
        mat[2 * m_inner, m_inner] = 0.5
    else:
        raise ValueError("Interpolation is implemented for ports 0 and 2 only.")
    return mat


def _add_second_moment(
    accumulator: np.ndarray,
    vectors: np.ndarray,
    *,
    transform: Optional[np.ndarray],
    chunk_size: int,
) -> None:
    """Add row-wise vector outer products using R[i,j]=E[x_i conj(x_j)]."""

    for start in range(0, vectors.shape[0], chunk_size):
        x = np.asarray(vectors[start : start + chunk_size], dtype=np.complex128)
        if transform is not None:
            x = x @ transform.T
        accumulator += x.T @ x.conj()


def _add_error_variance_covariance(
    accumulator: np.ndarray,
    variance_sum: np.ndarray,
    *,
    transform: Optional[np.ndarray],
) -> None:
    """Add sum_i J diag(v_i) J^H without forming each diagonal matrix."""

    variance_sum = np.asarray(variance_sum, dtype=np.float64)
    if transform is None:
        accumulator[np.diag_indices_from(accumulator)] += variance_sum
    else:
        accumulator += (transform * variance_sum[None, :]) @ transform.conj().T


def _interpolate_compact_covariance(
    compact_covariance: np.ndarray, *, port: int, n_sc: int
) -> np.ndarray:
    """Compute J C J^T using the sparse interpolation structure.

    Forming a dense W-by-M interpolation matrix and multiplying it with a
    covariance matrix is unnecessarily expensive for W=3276.  Each row of J
    contains at most two nonzero entries, so the same operation can be carried
    out by interpolating the rows and then the columns of C.
    """

    c = np.asarray(compact_covariance, dtype=np.complex128)
    n_comb = n_sc // 2
    if c.shape != (n_comb, n_comb):
        raise ValueError("Compact covariance has an unexpected shape.")

    def interpolate_rows(x: np.ndarray) -> np.ndarray:
        out = np.empty((n_sc, x.shape[1]), dtype=np.complex128)
        if port == 0:
            out[0::2] = x
            out[1:-1:2] = 0.5 * (x[:-1] + x[1:])
            out[-1] = x[-1]
        elif port == 2:
            out[1::2] = x
            out[0] = x[0]
            out[2::2] = 0.5 * (x[:-1] + x[1:])
        else:
            raise ValueError("Interpolation is implemented for ports 0 and 2 only.")
        return out

    rows_interpolated = interpolate_rows(c)
    full = interpolate_rows(rows_interpolated.T).T
    return 0.5 * (full + full.conj().T)


def _prepare_dataset_arrays(
    dataset: DatasetData,
    *,
    n_sc: int,
    num_dmrs_symbols: int,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Canonicalize h_ls, noise_ls, err_var, ports, and layer activity."""

    h = _canonicalize_estimates(
        dataset.h_ls,
        num_dmrs_symbols=num_dmrs_symbols,
        n_sc=n_sc,
        dmrs_symbol_indices=dataset.dmrs_symbol_indices,
        active_subcarrier_indices=dataset.active_subcarrier_indices,
        array_name=f"{dataset.name}.h_ls",
    )
    num_slots, _, num_layers, _, _ = h.shape
    ports = (
        _default_ports(num_layers)
        if dataset.ports is None
        else np.asarray(dataset.ports, dtype=np.int64)
    )
    if ports.shape != (num_layers,):
        raise ValueError(
            f"{dataset.name}: ports must have shape {(num_layers,)}, "
            f"received {ports.shape}."
        )
    if set(ports.tolist()) - {0, 2}:
        raise ValueError(
            f"{dataset.name}: direct-pilot processing requires ports 0 and/or 2."
        )
    layer_mask = _validate_layer_mask(
        dataset.layer_mask, num_slots, num_layers, dataset.name
    )

    noise = None
    if dataset.noise_ls is not None:
        noise = _canonicalize_estimates(
            dataset.noise_ls,
            num_dmrs_symbols=num_dmrs_symbols,
            n_sc=n_sc,
            dmrs_symbol_indices=dataset.dmrs_symbol_indices,
            active_subcarrier_indices=dataset.active_subcarrier_indices,
            array_name=f"{dataset.name}.noise_ls",
            allow_single_symbol=True,
        )
        if noise.shape[0] != num_slots or noise.shape[2] != num_layers:
            raise ValueError(
                f"{dataset.name}: noise_ls must match h_ls in slot and layer "
                f"dimensions; h_ls={h.shape}, noise_ls={noise.shape}."
            )
        if noise.shape[1] == 1 and h.shape[1] != 1:
            noise = np.broadcast_to(
                noise,
                (noise.shape[0], h.shape[1], noise.shape[2], noise.shape[3], noise.shape[4]),
            )
        elif noise.shape[1] != h.shape[1]:
            raise ValueError(
                f"{dataset.name}: noise_ls receiver dimension must be 1 or "
                f"{h.shape[1]}, received {noise.shape[1]}."
            )

    err_var = None
    if dataset.err_var is not None:
        err_var = _canonicalize_err_var(
            dataset.err_var,
            original_h_ls=dataset.h_ls,
            canonical_h_shape=h.shape,
            num_dmrs_symbols=num_dmrs_symbols,
            n_sc=n_sc,
            dmrs_symbol_indices=dataset.dmrs_symbol_indices,
            active_subcarrier_indices=dataset.active_subcarrier_indices,
            array_name=f"{dataset.name}.err_var",
        )

    return h, noise, err_var, ports, layer_mask


def estimate_dataset_statistics(
    dataset: DatasetData,
    *,
    method: Method,
    n_sc: int,
    num_dmrs_symbols: int,
    chunk_size: int,
    drop_exact_zero_vectors: bool,
) -> DatasetStatistics:
    """Estimate one dataset's channel and matched noise covariance."""

    if dataset.weight <= 0:
        raise ValueError(f"{dataset.name}: weight must be positive.")

    h, noise, err_var, ports, layer_mask = _prepare_dataset_arrays(
        dataset, n_sc=n_sc, num_dmrs_symbols=num_dmrs_symbols
    )
    n_comb = n_sc // 2
    compact_shape = (n_comb, n_comb)
    channel_sums = {
        0: np.zeros(compact_shape, dtype=np.complex128),
        2: np.zeros(compact_shape, dtype=np.complex128),
    }
    err_cov_sums = {
        0: np.zeros(compact_shape, dtype=np.complex128),
        2: np.zeros(compact_shape, dtype=np.complex128),
    }
    num_channel_vectors = 0
    port_vector_counts: dict[int, int] = {0: 0, 2: 0}
    pre_occ_variance_sum = 0.0

    scalar_err_var: Optional[float] = None
    if err_var is not None and np.asarray(err_var).ndim == 0:
        scalar_err_var = float(err_var)

    for layer, port_value in enumerate(ports):
        port = int(port_value)
        active_slots = layer_mask[:, layer]
        if not np.any(active_slots):
            continue
        direct = _select_direct_comb(
            h[active_slots, :, layer, :, :], port=port, n_sc=n_sc
        ).reshape(-1, n_comb)
        valid = np.all(np.isfinite(direct), axis=1)
        if drop_exact_zero_vectors:
            valid &= np.any(direct != 0, axis=1)
        direct = direct[valid]
        if direct.shape[0] == 0:
            continue

        _add_second_moment(
            channel_sums[port], direct, transform=None, chunk_size=chunk_size
        )
        count = direct.shape[0]
        num_channel_vectors += count
        port_vector_counts[port] += count

        if err_var is not None:
            if scalar_err_var is not None:
                variance_sum = np.full(n_comb, count * scalar_err_var)
            else:
                ev = _select_direct_comb(
                    err_var[active_slots, :, layer, :, :], port=port, n_sc=n_sc
                ).reshape(-1, n_comb)
                ev = np.asarray(ev[valid], dtype=np.float64)
                if np.any(~np.isfinite(ev)) or np.any(ev < 0):
                    raise ValueError(
                        f"{dataset.name}: err_var contains invalid values at "
                        "the retained direct pilot positions."
                    )
                variance_sum = np.sum(ev, axis=0)
            _add_error_variance_covariance(
                err_cov_sums[port], variance_sum, transform=None
            )
            pre_occ_variance_sum += float(np.sum(variance_sum))

    if num_channel_vectors == 0:
        raise ValueError(f"{dataset.name}: no valid direct-pilot vectors found.")
    if method == "naive":
        channel_sum = sum(
            _interpolate_compact_covariance(value, port=port, n_sc=n_sc)
            for port, value in channel_sums.items()
        )
        err_cov_sum = sum(
            _interpolate_compact_covariance(value, port=port, n_sc=n_sc)
            for port, value in err_cov_sums.items()
        )
    else:
        channel_sum = channel_sums[0] + channel_sums[2]
        err_cov_sum = err_cov_sums[0] + err_cov_sums[2]

    channel_cov = channel_sum / float(num_channel_vectors)
    channel_cov = 0.5 * (channel_cov + channel_cov.conj().T)

    num_noise_vectors = 0
    mean_pre_occ_err_var: Optional[float] = None
    if noise is not None:
        noise_sums = {
            0: np.zeros(compact_shape, dtype=np.complex128),
            2: np.zeros(compact_shape, dtype=np.complex128),
        }
        for layer, port_value in enumerate(ports):
            port = int(port_value)
            active_slots = layer_mask[:, layer]
            if not np.any(active_slots):
                continue
            direct_noise = _select_direct_comb(
                noise[active_slots, :, layer, :, :], port=port, n_sc=n_sc
            ).reshape(-1, n_comb)
            valid = np.all(np.isfinite(direct_noise), axis=1)
            direct_noise = direct_noise[valid]
            _add_second_moment(
                noise_sums[port],
                direct_noise,
                transform=None,
                chunk_size=chunk_size,
            )
            num_noise_vectors += direct_noise.shape[0]
        if num_noise_vectors == 0:
            raise ValueError(f"{dataset.name}: no valid noise vectors found.")
        if method == "naive":
            noise_sum = sum(
                _interpolate_compact_covariance(value, port=port, n_sc=n_sc)
                for port, value in noise_sums.items()
            )
        else:
            noise_sum = noise_sums[0] + noise_sums[2]
        noise_cov = noise_sum / float(num_noise_vectors)
    elif err_var is not None:
        # err_var is already no/abs(pilot)**2 from the modified estimator.  It
        # is pre-OCC and must not be divided by two here.
        noise_cov = err_cov_sum / float(num_channel_vectors)
        mean_pre_occ_err_var = pre_occ_variance_sum / (
            float(num_channel_vectors) * n_comb
        )
    else:
        warnings.warn(
            f"{dataset.name}: neither noise_ls nor err_var was supplied; no "
            "noise correction is applied.",
            RuntimeWarning,
        )
        noise_cov = np.zeros_like(channel_cov)

    noise_cov = 0.5 * (noise_cov + noise_cov.conj().T)
    return DatasetStatistics(
        name=dataset.name,
        channel_covariance=channel_cov,
        noise_covariance=noise_cov,
        num_channel_vectors=num_channel_vectors,
        num_noise_vectors=num_noise_vectors,
        mean_pre_occ_err_var=mean_pre_occ_err_var,
        port_vector_counts={k: v for k, v in port_vector_counts.items() if v},
    )


def _combine_dataset_statistics(
    datasets: Sequence[DatasetData],
    stats: Sequence[DatasetStatistics],
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    if not datasets or len(datasets) != len(stats):
        raise ValueError("Dataset/statistics list mismatch.")
    raw_sum = np.zeros_like(stats[0].channel_covariance)
    noise_sum = np.zeros_like(stats[0].noise_covariance)
    effective_total = 0.0
    vector_total = 0
    details: dict[str, dict] = {}
    for dataset, stat in zip(datasets, stats):
        if stat.channel_covariance.shape != raw_sum.shape:
            raise ValueError("All dataset covariance dimensions must match.")
        effective_count = dataset.weight * stat.num_channel_vectors
        raw_sum += effective_count * stat.channel_covariance
        noise_sum += effective_count * stat.noise_covariance
        effective_total += effective_count
        vector_total += stat.num_channel_vectors
        details[dataset.name] = {
            "weight": float(dataset.weight),
            "num_channel_vectors": int(stat.num_channel_vectors),
            "num_noise_vectors": int(stat.num_noise_vectors),
            "mean_pre_occ_err_var": stat.mean_pre_occ_err_var,
            "port_vector_counts": stat.port_vector_counts,
        }
    if effective_total <= 0:
        raise ValueError("Combined effective observation count is zero.")
    return (
        raw_sum / effective_total,
        noise_sum / effective_total,
        vector_total,
        details,
    )


def _noise_corrected_covariance(
    datasets: Sequence[DatasetData],
    *,
    method: Method,
    n_sc: int,
    num_dmrs_symbols: int,
    chunk_size: int,
    drop_exact_zero_vectors: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict]:
    stats = [
        estimate_dataset_statistics(
            dataset,
            method=method,
            n_sc=n_sc,
            num_dmrs_symbols=num_dmrs_symbols,
            chunk_size=chunk_size,
            drop_exact_zero_vectors=drop_exact_zero_vectors,
        )
        for dataset in datasets
    ]
    raw_cov, noise_cov, total_vectors, details = _combine_dataset_statistics(
        datasets, stats
    )
    channel_cov = raw_cov - noise_cov
    channel_cov = 0.5 * (channel_cov + channel_cov.conj().T)
    return channel_cov, raw_cov, noise_cov, total_vectors, details


def _project_psd(matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    """Project a Hermitian matrix onto the PSD cone without rescaling it."""

    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    clipped = np.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * clipped[None, :]) @ eigenvectors.conj().T
    projected = 0.5 * (projected + projected.conj().T)
    diagnostics = {
        "minimum_raw_eigenvalue": float(eigenvalues[0]),
        "num_negative_raw_eigenvalues": int(np.sum(eigenvalues < 0)),
        "raw_average_diagonal": float(np.real(np.trace(hermitian)) / matrix.shape[0]),
        "projected_average_diagonal": float(
            np.real(np.trace(projected)) / matrix.shape[0]
        ),
    }
    return projected, diagnostics


def _covariance_diagonal_means(covariance: np.ndarray) -> np.ndarray:
    """Average the lower diagonals of a square Hermitian covariance matrix."""

    cov = np.asarray(covariance, dtype=np.complex128)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square.")
    cov = 0.5 * (cov + cov.conj().T)
    return np.asarray(
        [np.mean(np.diag(cov, k=-lag)) for lag in range(cov.shape[0])],
        dtype=np.complex128,
    )


def _build_lag_interpolated_toeplitz_covariance(
    direct_covariance: np.ndarray,
    *,
    n_sc: int,
    output_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate unavailable odd lags and construct a Toeplitz covariance.

    The input covariance contains observations spaced by two active
    subcarriers.  Its q-th lower-diagonal mean therefore estimates full-grid
    lag 2q.  Interior odd lags are the arithmetic means of their neighboring
    even lags.  The maximum lag W-1 has no upper neighbor and retains the
    estimate at W-2, matching the convention in the accompanying derivation.
    No normalization or scale adjustment is applied.
    """

    if n_sc <= 0 or n_sc % 2 != 0:
        raise ValueError("n_sc must be a positive even integer.")
    expected_size = n_sc // 2
    direct = np.asarray(direct_covariance, dtype=np.complex128)
    if direct.shape != (expected_size, expected_size):
        raise ValueError(
            "direct_covariance must have shape "
            f"{(expected_size, expected_size)}, received {direct.shape}."
        )

    even_correlation = _covariance_diagonal_means(direct)
    correlation = np.empty(n_sc, dtype=np.complex128)
    correlation[0::2] = even_correlation
    correlation[1:-1:2] = 0.5 * (
        even_correlation[:-1] + even_correlation[1:]
    )
    correlation[-1] = even_correlation[-1]
    correlation[0] = np.real(correlation[0])

    covariance = toeplitz(correlation, correlation.conj())
    covariance = 0.5 * (covariance + covariance.conj().T)
    return (
        np.asarray(covariance, dtype=output_dtype),
        np.asarray(correlation, dtype=output_dtype),
        np.asarray(even_correlation, dtype=output_dtype),
    )


def _fit_nonnegative_pdp(
    comb_covariance: np.ndarray,
    *,
    n_fft: int,
    pdp_length: int,
    nnls_maxiter: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit p[n]>=0 to C[m+q,m]=sum_n p[n]exp(-j2pi(2q)n/N)."""

    c_hat = np.asarray(comb_covariance, dtype=np.complex128)
    if c_hat.ndim != 2 or c_hat.shape[0] != c_hat.shape[1]:
        raise ValueError("comb_covariance must be square.")
    c_hat = 0.5 * (c_hat + c_hat.conj().T)
    m_size = c_hat.shape[0]
    if not (1 <= pdp_length <= n_fft // 2):
        raise ValueError("pdp_length must lie in [1, n_fft/2].")
    q = np.arange(m_size, dtype=np.float64)
    n = np.arange(pdp_length, dtype=np.float64)
    diagonal_means = _covariance_diagonal_means(c_hat)
    design = np.exp(-1j * 2.0 * np.pi * 2.0 * np.outer(q, n) / n_fft)

    # Multiplicities of the main and conjugate off-diagonals in the full
    # Hermitian Frobenius norm.  Normalizing all weights by one common factor
    # improves conditioning without changing the NNLS minimizer.
    weights = 2.0 * (m_size - q)
    weights[0] = float(m_size)
    weights /= np.max(weights)
    sqrt_w = np.sqrt(weights)
    real_design = np.vstack(
        [sqrt_w[:, None] * design.real, sqrt_w[:, None] * design.imag]
    )
    real_target = np.concatenate(
        [sqrt_w * diagonal_means.real, sqrt_w * diagonal_means.imag]
    )
    pdp, residual = nnls(real_design, real_target, maxiter=nnls_maxiter)
    model_means = design @ pdp
    fit_error = np.linalg.norm(sqrt_w * (diagonal_means - model_means))
    data_norm = np.linalg.norm(sqrt_w * diagonal_means)
    relative_fit_error = float(
        fit_error / max(data_norm, np.finfo(float).eps)
    )
    return pdp, diagonal_means, model_means, float(residual), relative_fit_error


def _build_full_toeplitz_covariance(
    pdp: np.ndarray,
    *,
    n_fft: int,
    n_sc: int,
    output_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, float]:
    p = np.asarray(pdp, dtype=np.float64)
    if np.any(p < 0):
        raise ValueError("PDP must be nonnegative.")
    channel_variance = float(np.sum(p))
    if not np.isfinite(channel_variance) or channel_variance <= 0:
        raise ValueError("Estimated PDP has zero or non-finite total power.")
    d = np.arange(n_sc, dtype=np.float64)
    n = np.arange(p.size, dtype=np.float64)
    correlation = np.exp(-1j * 2.0 * np.pi * np.outer(d, n) / n_fft) @ p
    covariance = toeplitz(correlation, correlation.conj())
    covariance = 0.5 * (covariance + covariance.conj().T)
    return (
        np.asarray(covariance, dtype=output_dtype),
        np.asarray(correlation, dtype=output_dtype),
        channel_variance,
    )


def estimate_direct_comb_frequency_covariances(
    single_layer_dataset: DatasetData,
    dual_layer_dataset: DatasetData,
    *,
    method: PublicMethod = "both",
    n_fft: int = 4096,
    n_sc: int = 3276,
    pdp_length: int = 288,
    num_dmrs_symbols: int = 3,
    chunk_size: int = 512,
    drop_exact_zero_vectors: bool = True,
    project_naive_psd: bool = True,
    project_lag_psd: bool = False,
    output_dtype: np.dtype = np.complex64,
    nnls_maxiter: Optional[int] = None,
) -> DirectCombCovarianceEstimate:
    """Estimate absolute-scale frequency covariances from direct pilots.

    ``both`` retains its original meaning and computes the naive and PDP-based
    estimates.  Use ``lag-and-pdp`` to compare the two Toeplitz methods or
    ``all`` to compute all three estimates.
    """

    valid_methods = ("naive", "lag", "pdp", "both", "lag-and-pdp", "all")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}.")
    run_naive = method in ("naive", "both", "all")
    run_lag = method in ("lag", "lag-and-pdp", "all")
    run_pdp = method in ("pdp", "both", "lag-and-pdp", "all")
    if n_sc % 2 != 0:
        raise ValueError("n_sc must be even.")
    if run_pdp and pdp_length > 288:
        warnings.warn(
            "pdp_length exceeds the 288-sample shortest normal CP assumed "
            "for the 4096-point, 30-kHz-SCS configuration.",
            RuntimeWarning,
        )
    if run_pdp and pdp_length >= n_fft // 2:
        raise ValueError(
            "pdp_length must be smaller than n_fft/2 to avoid the delay "
            "ambiguity caused by observing every second subcarrier."
        )

    datasets = [single_layer_dataset, dual_layer_dataset]
    metadata: dict = {
        "n_fft": int(n_fft),
        "n_sc": int(n_sc),
        "n_direct_comb_samples": int(n_sc // 2),
        "n_direct_pilot_samples": int(n_sc // 2),
        "num_dmrs_symbols": int(num_dmrs_symbols),
        "methods": method,
        "normalized": False,
        "error_variance_note": (
            "err_var is the pre-OCC value no/abs(pilot)**2 returned by "
            "PUSCHLSChannelEstimatorNoOCC. No factor 1/2 is applied."
        ),
        "scale_note": (
            "All output covariance matrices retain the absolute channel-power "
            "scale. No unit-diagonal normalization or post-fit rescaling is "
            "performed."
        ),
    }

    cov_naive = None
    cov_naive_raw = None
    if run_naive:
        (
            naive_raw,
            naive_ls_cov,
            naive_noise_cov,
            naive_vectors,
            naive_details,
        ) = _noise_corrected_covariance(
            datasets,
            method="naive",
            n_sc=n_sc,
            num_dmrs_symbols=num_dmrs_symbols,
            chunk_size=chunk_size,
            drop_exact_zero_vectors=drop_exact_zero_vectors,
        )
        cov_naive_raw = np.asarray(naive_raw, dtype=output_dtype)
        if project_naive_psd:
            naive_projected, psd_diagnostics = _project_psd(naive_raw)
        else:
            naive_projected = naive_raw
            eigenvalues = np.linalg.eigvalsh(naive_raw)
            psd_diagnostics = {
                "minimum_raw_eigenvalue": float(eigenvalues[0]),
                "num_negative_raw_eigenvalues": int(np.sum(eigenvalues < 0)),
                "raw_average_diagonal": float(
                    np.real(np.trace(naive_raw)) / n_sc
                ),
                "projected_average_diagonal": None,
            }
        cov_naive = np.asarray(naive_projected, dtype=output_dtype)
        metadata["naive"] = {
            "total_channel_vectors": int(naive_vectors),
            "dataset_details": naive_details,
            "psd_projection_applied": bool(project_naive_psd),
            "psd_diagnostics": psd_diagnostics,
            "ls_covariance_average_diagonal": float(
                np.real(np.trace(naive_ls_cov)) / n_sc
            ),
            "noise_covariance_average_diagonal": float(
                np.real(np.trace(naive_noise_cov)) / n_sc
            ),
        }

    cov_lag = None
    cov_lag_raw = None
    correlation_lag = None
    observed_even_correlation = None
    cov_pdp = None
    pdp = None
    correlation = None
    observed_comb_cov = None
    model_comb_cov = None
    comb_channel_cov = None
    comb_ls_cov = None
    comb_noise_cov = None
    direct_vectors = 0
    direct_details = None
    if run_lag or run_pdp:
        (
            comb_channel_cov,
            comb_ls_cov,
            comb_noise_cov,
            direct_vectors,
            direct_details,
        ) = _noise_corrected_covariance(
            datasets,
            method="pdp",
            n_sc=n_sc,
            num_dmrs_symbols=num_dmrs_symbols,
            chunk_size=chunk_size,
            drop_exact_zero_vectors=drop_exact_zero_vectors,
        )
        observed_comb_cov = comb_channel_cov

    if run_lag:
        assert comb_channel_cov is not None
        assert comb_ls_cov is not None
        assert comb_noise_cov is not None
        lag_raw_128, lag_corr_128, even_corr_128 = (
            _build_lag_interpolated_toeplitz_covariance(
                comb_channel_cov,
                n_sc=n_sc,
                output_dtype=np.dtype(np.complex128),
            )
        )
        cov_lag_raw = np.asarray(lag_raw_128, dtype=output_dtype)
        if project_lag_psd:
            lag_projected, lag_psd_diagnostics = _project_psd(lag_raw_128)
            cov_lag = np.asarray(lag_projected, dtype=output_dtype)
        else:
            cov_lag = cov_lag_raw
            lag_psd_diagnostics = {
                "eigenvalues_checked": False,
                "raw_average_diagonal": float(
                    np.real(np.trace(lag_raw_128)) / n_sc
                ),
                "projected_average_diagonal": None,
            }
        correlation_lag = np.asarray(lag_corr_128, dtype=output_dtype)
        observed_even_correlation = np.asarray(
            even_corr_128, dtype=output_dtype
        )
        metadata["lag"] = {
            "total_channel_vectors": int(direct_vectors),
            "dataset_details": direct_details,
            "psd_projection_applied": bool(project_lag_psd),
            "psd_diagnostics": lag_psd_diagnostics,
            "direct_zero_lag_after_noise_subtraction": float(
                np.real(even_corr_128[0])
            ),
            "direct_ls_zero_lag": float(
                np.real(np.mean(np.diag(comb_ls_cov)))
            ),
            "direct_noise_zero_lag": float(
                np.real(np.mean(np.diag(comb_noise_cov)))
            ),
            "odd_lag_interpolation": (
                "Interior odd lags are arithmetic means of adjacent even "
                "lags; lag W-1 retains the estimate at W-2."
            ),
        }

    if run_pdp:
        assert comb_channel_cov is not None
        assert comb_ls_cov is not None
        assert comb_noise_cov is not None
        (
            pdp_fit,
            observed_diagonal_means,
            model_diagonal_means,
            nnls_residual,
            relative_fit_error,
        ) = _fit_nonnegative_pdp(
            comb_channel_cov,
            n_fft=n_fft,
            pdp_length=pdp_length,
            nnls_maxiter=nnls_maxiter,
        )
        cov_pdp, correlation, sigma_h2 = _build_full_toeplitz_covariance(
            pdp_fit,
            n_fft=n_fft,
            n_sc=n_sc,
            output_dtype=np.dtype(output_dtype),
        )
        n_comb = n_sc // 2
        m = np.arange(n_comb, dtype=np.float64)
        n = np.arange(pdp_length, dtype=np.float64)
        sensing = np.exp(-1j * 2.0 * np.pi * 2.0 * np.outer(m, n) / n_fft)
        model_comb_cov = (sensing * pdp_fit[None, :]) @ sensing.conj().T
        model_comb_cov = 0.5 * (model_comb_cov + model_comb_cov.conj().T)
        pdp = pdp_fit
        metadata["pdp"] = {
            "pdp_length": int(pdp_length),
            "num_compact_covariance_lags": int(n_comb),
            "total_channel_vectors": int(direct_vectors),
            "dataset_details": direct_details,
            "absolute_channel_variance": float(sigma_h2),
            "observed_comb_zero_lag_after_noise_subtraction": float(
                np.real(np.mean(np.diag(comb_channel_cov)))
            ),
            "model_comb_zero_lag": float(np.sum(pdp_fit)),
            "comb_ls_zero_lag": float(np.real(np.mean(np.diag(comb_ls_cov)))),
            "comb_noise_zero_lag": float(
                np.real(np.mean(np.diag(comb_noise_cov)))
            ),
            "nnls_residual_norm": nnls_residual,
            "relative_weighted_covariance_fit_error": relative_fit_error,
            "observed_diagonal_means_real": observed_diagonal_means.real.tolist(),
            "observed_diagonal_means_imag": observed_diagonal_means.imag.tolist(),
            "model_diagonal_means_real": model_diagonal_means.real.tolist(),
            "model_diagonal_means_imag": model_diagonal_means.imag.tolist(),
        }

    return DirectCombCovarianceEstimate(
        cov_mat_freq_naive=cov_naive,
        cov_mat_freq_naive_raw=cov_naive_raw,
        cov_mat_freq_lag=cov_lag,
        cov_mat_freq_lag_raw=cov_lag_raw,
        cov_mat_freq_pdp=cov_pdp,
        pdp=pdp,
        frequency_correlation=correlation,
        frequency_correlation_lag=correlation_lag,
        observed_even_frequency_correlation=observed_even_correlation,
        observed_comb_covariance=(
            None
            if observed_comb_cov is None
            else np.asarray(observed_comb_cov, dtype=np.complex64)
        ),
        direct_noise_covariance=(
            None
            if comb_noise_cov is None
            else np.asarray(comb_noise_cov, dtype=output_dtype)
        ),
        model_comb_covariance=(
            None
            if model_comb_cov is None
            else np.asarray(model_comb_cov, dtype=np.complex64)
        ),
        metadata=metadata,
    )


def load_dataset_npz(path: Path, *, name: str, weight: float = 1.0) -> DatasetData:
    """Load one input dataset using the schema in the module docstring."""

    with np.load(path, allow_pickle=False) as data:
        if "h_ls" not in data:
            raise KeyError(f"{path} does not contain required array 'h_ls'.")
        return DatasetData(
            name=name,
            h_ls=np.asarray(data["h_ls"]),
            layer_mask=(
                np.asarray(data["layer_mask"], dtype=bool)
                if "layer_mask" in data
                else None
            ),
            ports=(
                np.asarray(data["ports"], dtype=np.int64)
                if "ports" in data
                else None
            ),
            noise_ls=(np.asarray(data["noise_ls"]) if "noise_ls" in data else None),
            err_var=(np.asarray(data["err_var"]) if "err_var" in data else None),
            dmrs_symbol_indices=(
                np.asarray(data["dmrs_symbol_indices"], dtype=np.int64)
                if "dmrs_symbol_indices" in data
                else None
            ),
            active_subcarrier_indices=(
                np.asarray(data["active_subcarrier_indices"], dtype=np.int64)
                if "active_subcarrier_indices" in data
                else None
            ),
            weight=weight,
        )


def save_estimate_npz(path: Path, estimate: DirectCombCovarianceEstimate) -> None:
    """Save all available estimates and diagnostics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(estimate.metadata, indent=2))
    }
    for name in (
        "cov_mat_freq_naive",
        "cov_mat_freq_naive_raw",
        "cov_mat_freq_lag",
        "cov_mat_freq_lag_raw",
        "cov_mat_freq_pdp",
        "pdp",
        "frequency_correlation",
        "frequency_correlation_lag",
        "observed_even_frequency_correlation",
        "observed_comb_covariance",
        "direct_noise_covariance",
        "model_comb_covariance",
    ):
        value = getattr(estimate, name)
        if value is not None:
            arrays[name] = value
    np.savez_compressed(path, **arrays)


def _synthetic_self_test() -> None:
    """Check pilot selection, noise subtraction, scale, and interpolation."""

    rng = np.random.default_rng(17)
    n_fft = 128
    n_sc = 96
    n_comb = n_sc // 2
    pdp_length = 12
    noise_var = 0.035  # Pre-OCC per-RE LS error variance.
    true_pdp = np.exp(-np.arange(pdp_length) / 2.8)
    true_pdp *= 1.9 / np.sum(true_pdp)
    true_variance = float(np.sum(true_pdp))
    dft = np.exp(
        -1j
        * 2.0
        * np.pi
        * np.outer(np.arange(n_sc), np.arange(pdp_length))
        / n_fft
    )

    def draw_dense(num_vectors: int, ports: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        h = np.full((num_vectors, 1, len(ports), 1, n_sc), 50.0 + 20.0j)
        ev = np.full(h.shape, 999.0, dtype=np.float64)
        for layer, port in enumerate(ports):
            taps = (
                rng.standard_normal((num_vectors, pdp_length))
                + 1j * rng.standard_normal((num_vectors, pdp_length))
            ) * np.sqrt(true_pdp[None, :] / 2.0)
            channel = taps @ dft.T
            ind = _comb_indices(port, n_sc)
            noise = (
                rng.standard_normal((num_vectors, n_comb))
                + 1j * rng.standard_normal((num_vectors, n_comb))
            ) * np.sqrt(noise_var / 2.0)
            h[:, 0, layer, 0, ind] = channel[:, ind] + noise
            ev[:, 0, layer, 0, ind] = noise_var
        return h, ev

    def draw_noise(num_vectors: int, ports: Sequence[int]) -> np.ndarray:
        noise_ls = np.full(
            (num_vectors, 1, len(ports), 1, n_sc),
            -80.0 + 40.0j,
            dtype=np.complex128,
        )
        for layer, port in enumerate(ports):
            ind = _comb_indices(port, n_sc)
            direct_noise = (
                rng.standard_normal((num_vectors, n_comb))
                + 1j * rng.standard_normal((num_vectors, n_comb))
            ) * np.sqrt(noise_var / 2.0)
            noise_ls[:, 0, layer, 0, ind] = direct_noise
        return noise_ls

    h_single, ev_single = draw_dense(2200, [0])
    h_dual, ev_dual = draw_dense(1800, [0, 2])
    result = estimate_direct_comb_frequency_covariances(
        DatasetData("synthetic_single", h_single, err_var=ev_single),
        DatasetData(
            "synthetic_dual",
            h_dual,
            ports=np.asarray([0, 2]),
            err_var=ev_dual,
        ),
        method="all",
        n_fft=n_fft,
        n_sc=n_sc,
        pdp_length=pdp_length,
        num_dmrs_symbols=1,
        chunk_size=256,
        output_dtype=np.complex128,
    )
    assert result.cov_mat_freq_pdp is not None
    assert result.cov_mat_freq_lag is not None
    assert result.cov_mat_freq_naive is not None
    true_r = dft @ true_pdp
    true_cov = toeplitz(true_r, true_r.conj())
    pdp_cov_error = np.linalg.norm(result.cov_mat_freq_pdp - true_cov) / np.linalg.norm(
        true_cov
    )
    lag_cov_error = np.linalg.norm(result.cov_mat_freq_lag - true_cov) / np.linalg.norm(
        true_cov
    )
    estimated_variance = result.metadata["pdp"]["absolute_channel_variance"]
    scale_error = abs(estimated_variance - true_variance) / true_variance
    lag_variance = result.metadata["lag"][
        "direct_zero_lag_after_noise_subtraction"
    ]
    lag_scale_error = abs(lag_variance - true_variance) / true_variance
    naive_scale = float(np.real(np.trace(result.cov_mat_freq_naive)) / n_sc)

    print("Direct-pilot synthetic self-test")
    print(f"  PDP covariance relative error: {pdp_cov_error:.4f}")
    print(f"  Lag covariance relative error: {lag_cov_error:.4f}")
    print(f"  PDP scale relative error:      {scale_error:.4f}")
    print(f"  Lag scale relative error:      {lag_scale_error:.4f}")
    print(f"  Naive average diagonal:        {naive_scale:.4f}")
    print(f"  True channel variance:         {true_variance:.4f}")
    if pdp_cov_error > 0.14 or scale_error > 0.08:
        raise RuntimeError("Direct-pilot synthetic self-test failed.")
    if lag_cov_error > 0.14 or lag_scale_error > 0.08:
        raise RuntimeError("Lag-interpolated covariance self-test failed.")
    if not (0.75 * true_variance <= naive_scale <= 1.25 * true_variance):
        raise RuntimeError("Naive covariance scale check failed.")

    # Exercise empirical noise-covariance subtraction independently of the
    # analytical err_var branch. The unused parity again contains large values
    # and must not affect the result.
    empirical_noise_result = estimate_direct_comb_frequency_covariances(
        DatasetData(
            "synthetic_single_noise",
            h_single,
            noise_ls=draw_noise(h_single.shape[0], [0]),
        ),
        DatasetData(
            "synthetic_dual_noise",
            h_dual,
            ports=np.asarray([0, 2]),
            noise_ls=draw_noise(h_dual.shape[0], [0, 2]),
        ),
        method="pdp",
        n_fft=n_fft,
        n_sc=n_sc,
        pdp_length=pdp_length,
        num_dmrs_symbols=1,
        chunk_size=256,
        output_dtype=np.complex128,
    )
    empirical_noise_variance = empirical_noise_result.metadata["pdp"][
        "absolute_channel_variance"
    ]
    empirical_noise_scale_error = (
        abs(empirical_noise_variance - true_variance) / true_variance
    )
    print(
        "  Empirical-noise scale error: "
        f"{empirical_noise_scale_error:.4f}"
    )
    if empirical_noise_scale_error > 0.08:
        raise RuntimeError("Empirical noise-covariance scale check failed.")

    # Exercise the full interpolated Sionna output layout and explicit DM-RS
    # symbol selection.  Values on all non-DM-RS symbols are deliberately large
    # so that accidental inclusion is immediately visible in the scale test.
    num_symbols = 4
    dmrs_symbol = 1
    h_single_full = np.full(
        (h_single.shape[0], 1, 1, 1, 1, num_symbols, n_sc),
        -70.0 + 30.0j,
        dtype=np.complex128,
    )
    ev_single_full = np.full(h_single_full.shape, 777.0, dtype=np.float64)
    h_single_full[:, 0, 0, 0, 0, dmrs_symbol, :] = h_single[:, 0, 0, 0, :]
    ev_single_full[:, 0, 0, 0, 0, dmrs_symbol, :] = ev_single[:, 0, 0, 0, :]

    h_dual_full = np.full(
        (h_dual.shape[0], 1, 1, 1, 2, num_symbols, n_sc),
        -70.0 + 30.0j,
        dtype=np.complex128,
    )
    ev_dual_full = np.full(
        h_dual_full.shape, 777.0, dtype=np.float64
    )
    h_dual_full[:, 0, 0, 0, :, dmrs_symbol, :] = (
        h_dual[:, 0, :, 0, :]
    )
    ev_dual_full[:, 0, 0, 0, :, dmrs_symbol, :] = (
        ev_dual[:, 0, :, 0, :]
    )

    full_layout_result = estimate_direct_comb_frequency_covariances(
        DatasetData(
            "synthetic_single_full",
            h_single_full,
            err_var=ev_single_full,
            dmrs_symbol_indices=np.asarray([dmrs_symbol]),
        ),
        DatasetData(
            "synthetic_dual_full",
            h_dual_full,
            ports=np.asarray([0, 2]),
            err_var=ev_dual_full,
            dmrs_symbol_indices=np.asarray([dmrs_symbol]),
        ),
        method="pdp",
        n_fft=n_fft,
        n_sc=n_sc,
        pdp_length=pdp_length,
        num_dmrs_symbols=1,
        chunk_size=256,
        output_dtype=np.complex128,
    )
    full_layout_variance = full_layout_result.metadata["pdp"][
        "absolute_channel_variance"
    ]
    if abs(full_layout_variance - estimated_variance) > 1e-10:
        raise RuntimeError("Full Sionna-layout extraction test failed.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate frequency covariance from direct PUSCH DM-RS pilots."
    )
    parser.add_argument("--single", type=Path, help="Single-layer dataset NPZ.")
    parser.add_argument(
        "--dual",
        type=Path,
        help="Dual-layer dataset NPZ.",
    )
    parser.add_argument("--output", type=Path, help="Output NPZ path.")
    parser.add_argument(
        "--method",
        choices=["naive", "lag", "pdp", "both", "lag-and-pdp", "all"],
        default="both",
        help=(
            "Estimation method. 'both' retains the original naive+PDP "
            "behavior; 'lag-and-pdp' compares the two Toeplitz methods."
        ),
    )
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--n-sc", type=int, default=3276)
    parser.add_argument("--pdp-length", type=int, default=288)
    parser.add_argument("--num-dmrs-symbols", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--single-weight", type=float, default=1.0)
    parser.add_argument("--dual-weight", type=float, default=1.0)
    parser.add_argument(
        "--skip-naive-psd-projection",
        action="store_true",
        help="Return the raw noise-corrected naive estimate without PSD projection.",
    )
    parser.add_argument(
        "--project-lag-psd",
        action="store_true",
        help=(
            "Project the lag-interpolated covariance onto the PSD cone. "
            "This generally removes its exact Toeplitz structure."
        ),
    )
    parser.add_argument(
        "--keep-zero-vectors",
        action="store_true",
        help="Do not discard exact all-zero channel vectors.",
    )
    parser.add_argument(
        "--complex128-output",
        action="store_true",
        help="Store full covariance matrices as complex128 instead of complex64.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _synthetic_self_test()
        return
    if args.single is None or args.dual is None or args.output is None:
        raise SystemExit(
            "--single, --dual, and --output are required."
        )
    single = load_dataset_npz(
        args.single, name="single_layer", weight=args.single_weight
    )
    dual = load_dataset_npz(
        args.dual,
        name="dual_layer",
        weight=args.dual_weight,
    )
    result = estimate_direct_comb_frequency_covariances(
        single,
        dual,
        method=args.method,
        n_fft=args.n_fft,
        n_sc=args.n_sc,
        pdp_length=args.pdp_length,
        num_dmrs_symbols=args.num_dmrs_symbols,
        chunk_size=args.chunk_size,
        drop_exact_zero_vectors=not args.keep_zero_vectors,
        project_naive_psd=not args.skip_naive_psd_projection,
        project_lag_psd=args.project_lag_psd,
        output_dtype=np.complex128 if args.complex128_output else np.complex64,
    )
    save_estimate_npz(args.output, result)
    print(f"Saved estimate to: {args.output}")
    if "pdp" in result.metadata:
        print(
            "Absolute PDP variance: "
            f"{result.metadata['pdp']['absolute_channel_variance']:.6g}"
        )
        print(
            "Relative PDP fit error: "
            f"{result.metadata['pdp']['relative_weighted_covariance_fit_error']:.6g}"
        )
    if "lag" in result.metadata:
        print(
            "Lag-interpolated zero-lag variance: "
            f"{result.metadata['lag']['direct_zero_lag_after_noise_subtraction']:.6g}"
        )
    if "naive" in result.metadata:
        print(
            "Naive covariance diagonal: "
            f"{result.metadata['naive']['psd_diagnostics']['projected_average_diagonal']}"
        )


if __name__ == "__main__":
    main()
