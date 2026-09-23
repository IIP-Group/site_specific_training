#!/usr/bin/env python3
"""Estimate a 4-by-4 PUSCH spatial covariance from datalake measurements.

The coutput is written as:
    <output-prefix>_space_cov_mat.npy

This is research code and it is not optimized for speed, memory or neatness. The code is a mere
demonstration of the proposed method in the accompanying paper.

Author: Nuri Berke Baytekin
"""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import clickhouse_connect
import numpy as np
import tensorflow as tf

from estimate_pusch_frequency_covariance_from_datalake import (
    DATABASES,
    DMRS_SYMBOLS_EXPECTED,
    N_SC,
    NUM_RX_ANTENNAS,
    REPO_ROOT,
    allocation_power_ratio_db,
    build_no_occ_estimator,
    build_noise_resource_grid,
    build_query,
    create_sionna_transmitter_from_record,
    fetch_fh_row,
    format_ch_datetime64,
    parse_fh,
    pilot_output_to_canonical,
    resolve_timestamp_path,
    rx_slot_to_sionna_y,
    validate_record_configuration,
)


tf.get_logger().setLevel("ERROR")


@dataclass
class SpatialDatasetStatistics:

    name: str
    weight: float
    ls_second_moment_sum: np.ndarray
    num_channel_vectors: int
    noise_second_moment_sum: np.ndarray
    num_noise_vectors: int
    num_records: int
    timestamps: list[str]
    power_ratios_db: np.ndarray
    ports: np.ndarray
    dmrs_symbols: np.ndarray
    pilot_magnitudes: np.ndarray


@dataclass
class SpatialCovarianceEstimate:
    """Spatial covariance and its measured LS/noise terms."""

    covariance: np.ndarray
    corrected_covariance_raw: np.ndarray
    ls_second_moment: np.ndarray
    noise_covariance: np.ndarray
    diagnostics: dict


def direct_comb_indices(port: int) -> np.ndarray:
    """Return the genuine type-1 DM-RS comb positions for one port."""

    if port == 0:
        offset = 0
    elif port == 2:
        offset = 1
    else:
        raise ValueError(f"Unsupported DM-RS port {port}; expected 0 or 2.")
    return np.arange(offset, N_SC, 2, dtype=np.int64)


def select_genuine_pilot_comb(
    values: np.ndarray,
    ports: np.ndarray,
    *,
    description: str,
) -> list[np.ndarray]:
    """Validate zero placeholders and select each genuine pilot comb."""

    array = np.asarray(values)
    if array.ndim < 3 or array.shape[0] != NUM_RX_ANTENNAS:
        raise ValueError(f"Unexpected {description} shape: {array.shape}.")
    if array.shape[-1] != N_SC:
        raise ValueError(
            f"Expected {description} frequency size {N_SC}, "
            f"received {array.shape[-1]}."
        )
    ports = np.asarray(ports, dtype=np.int64)
    if ports.shape != (array.shape[1],):
        raise ValueError(
            f"Expected one port per layer, received ports {ports.shape} "
            f"for {description} shape {array.shape}."
        )

    selected = []
    for layer, port in enumerate(ports):
        valid_indices = direct_comb_indices(int(port))
        valid_mask = np.zeros(N_SC, dtype=bool)
        valid_mask[valid_indices] = True
        layer_values = array[:, layer, ...]
        placeholders = layer_values[..., np.logical_not(valid_mask)]
        nonzero_placeholders = np.count_nonzero(placeholders)
        if nonzero_placeholders:
            raise ValueError(
                f"{description}: layer {layer}, port {port} contains "
                f"{nonzero_placeholders} nonzero values at zero-pilot "
                "placeholder positions. The packed pilot ordering is not "
                "the expected DM-RS-symbol-major layout."
            )
        selected.append(layer_values[..., valid_mask])
    return selected


def _outer_product_sum(
    vectors: np.ndarray,
    *,
    description: str,
) -> tuple[np.ndarray, int]:
    """Return sum(x x^H) and the number of finite spatial vectors."""

    x = np.asarray(vectors)
    if x.ndim != 2 or x.shape[1] != NUM_RX_ANTENNAS:
        raise ValueError(
            f"Expected {description} shape [N, {NUM_RX_ANTENNAS}], "
            f"received {x.shape}."
        )
    valid = np.all(np.isfinite(x), axis=1)
    x = np.asarray(x[valid], dtype=np.complex128)
    if x.shape[0] == 0:
        raise ValueError(f"No finite {description} vectors were found.")
    return x.T @ x.conj(), int(x.shape[0])


def spatial_ls_second_moment_sum(
    h_ls: np.ndarray,
    ports: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Accumulate four-antenna LS vectors over all genuine pilot REs."""

    h = np.asarray(h_ls)
    num_dmrs_symbols = len(DMRS_SYMBOLS_EXPECTED)
    expected_tail = (num_dmrs_symbols, N_SC)
    if h.ndim != 4 or h.shape[0] != NUM_RX_ANTENNAS:
        raise ValueError(f"Unexpected pilot-domain channel shape: {h.shape}.")
    if h.shape[2:] != expected_tail:
        raise ValueError(
            f"Expected channel tail shape {expected_tail}, received {h.shape[2:]}."
        )
    layer_vectors = []
    for direct in select_genuine_pilot_comb(
        h, ports, description="pilot-domain channel"
    ):
        layer_vectors.append(
            np.transpose(direct, (1, 2, 0)).reshape(-1, NUM_RX_ANTENNAS)
        )
    vectors = np.concatenate(layer_vectors, axis=0)
    return _outer_product_sum(vectors, description="spatial LS")


def spatial_noise_second_moment_sum(
    noise_ls: np.ndarray,
    ports: np.ndarray,
    *,
    reference_dmrs_index: int = 0,
) -> tuple[np.ndarray, int]:
    """Accumulate noise vectors from one packed DM-RS block only."""

    noise = np.asarray(noise_ls)
    num_dmrs_symbols = len(DMRS_SYMBOLS_EXPECTED)
    expected_tail = (num_dmrs_symbols, N_SC)
    if noise.ndim != 4 or noise.shape[0] != NUM_RX_ANTENNAS:
        raise ValueError(f"Unexpected pilot-domain noise shape: {noise.shape}.")
    if noise.shape[2:] != expected_tail:
        raise ValueError(
            f"Expected noise tail shape {expected_tail}, "
            f"received {noise.shape[2:]}."
        )
    if not 0 <= reference_dmrs_index < num_dmrs_symbols:
        raise ValueError(f"Invalid reference DM-RS index {reference_dmrs_index}.")

    layer_vectors = []
    reference_noise = noise[:, :, reference_dmrs_index, :]
    for direct in select_genuine_pilot_comb(
        reference_noise, ports, description="pilot-domain noise"
    ):
        layer_vectors.append(direct.T)
    vectors = np.concatenate(layer_vectors, axis=0)
    return _outer_product_sum(vectors, description="spatial noise")


def extract_one_record(
    record,
    y_full: np.ndarray,
    *,
    expected_num_layers: int,
) -> tuple[
    np.ndarray,
    int,
    np.ndarray,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return channel/noise moment sums and configuration for one record."""

    transmitter = create_sionna_transmitter_from_record(record)
    estimator = build_no_occ_estimator(transmitter, record)
    ports, dmrs_symbols, pilot_magnitudes = validate_record_configuration(
        record,
        transmitter,
        estimator,
        expected_num_layers=expected_num_layers,
    )

    y_channel = rx_slot_to_sionna_y(y_full, record)
    num_grid_symbols = int(y_channel.shape[-2])
    y_noise = build_noise_resource_grid(y_full, record, num_grid_symbols)
    no_placeholder = tf.ones(
        [1, 1, int(y_channel.shape[2])], dtype=tf.float32
    )

    h_ls, _ = estimator([tf.constant(y_channel), no_placeholder])
    noise_ls, _ = estimator([tf.constant(y_noise), no_placeholder])
    h_canonical = pilot_output_to_canonical(
        h_ls,
        num_layers=expected_num_layers,
        num_dmrs_symbols=len(dmrs_symbols),
    )
    noise_canonical = pilot_output_to_canonical(
        noise_ls,
        num_layers=expected_num_layers,
        num_dmrs_symbols=len(dmrs_symbols),
    )
    channel_sum, channel_count = spatial_ls_second_moment_sum(
        h_canonical, ports
    )
    noise_sum, noise_count = spatial_noise_second_moment_sum(
        noise_canonical, ports, reference_dmrs_index=0
    )
    return (
        channel_sum,
        channel_count,
        noise_sum,
        noise_count,
        ports,
        dmrs_symbols,
        pilot_magnitudes,
    )


def extract_dataset_statistics(
    client,
    *,
    selector: str,
    name: str,
    limit: int,
    slot: Optional[int],
    mcs_index: Optional[int],
    min_power_ratio_db: float,
    timestamps_path: Optional[Path],
    weight: float,
) -> SpatialDatasetStatistics:
    """Query one dataset and accumulate its spatial statistics online."""

    if selector not in DATABASES:
        raise ValueError(
            f"Unsupported database selector {selector}; known: {sorted(DATABASES)}"
        )
    spec = DATABASES[selector]
    records = client.query_df(
        build_query(
            spec,
            limit=limit,
            slot=slot,
            mcs_index=mcs_index,
            timestamps_path=timestamps_path,
        )
    )
    print(
        f"\n{name}: queried {len(records)} FAPI records from {spec.database} "
        f"(expected layers={spec.num_layers})"
    )
    if records.empty:
        raise RuntimeError(f"{name}: query returned no FAPI records.")

    ls_second_moment_sum = np.zeros(
        (NUM_RX_ANTENNAS, NUM_RX_ANTENNAS), dtype=np.complex128
    )
    noise_second_moment_sum = np.zeros_like(ls_second_moment_sum)
    num_channel_vectors = 0
    num_noise_vectors = 0
    timestamps: list[str] = []
    power_ratios: list[float] = []
    reference_ports: Optional[np.ndarray] = None
    reference_dmrs: Optional[np.ndarray] = None
    reference_pilot_magnitudes: Optional[np.ndarray] = None
    skipped_fh = 0
    skipped_power = 0

    for _, record in records.iterrows():
        fh = fetch_fh_row(client, spec.database, record)
        if len(fh.index) != 1:
            skipped_fh += 1
            continue
        y_full = parse_fh(fh["fhData"].iloc[0])
        ratio_db = allocation_power_ratio_db(y_full, record)
        if ratio_db < min_power_ratio_db:
            skipped_power += 1
            continue

        (
            record_channel_sum,
            record_channel_count,
            record_noise_sum,
            record_noise_count,
            ports,
            dmrs_symbols,
            pilot_magnitudes,
        ) = extract_one_record(
            record,
            y_full,
            expected_num_layers=spec.num_layers,
        )

        if reference_ports is None:
            reference_ports = ports
            reference_dmrs = dmrs_symbols
            reference_pilot_magnitudes = pilot_magnitudes
        else:
            if not np.array_equal(ports, reference_ports):
                raise ValueError(f"{name}: DM-RS ports change across records.")
            if not np.array_equal(dmrs_symbols, reference_dmrs):
                raise ValueError(f"{name}: DM-RS symbols change across records.")
            if not np.allclose(pilot_magnitudes, reference_pilot_magnitudes):
                raise ValueError(f"{name}: pilot magnitudes change across records.")

        ls_second_moment_sum += record_channel_sum
        num_channel_vectors += record_channel_count
        noise_second_moment_sum += record_noise_sum
        num_noise_vectors += record_noise_count
        timestamps.append(format_ch_datetime64(record.TsTaiNs))
        power_ratios.append(ratio_db)

    if not timestamps:
        raise RuntimeError(
            f"{name}: no records survived FH and power filtering "
            f"(missing FH={skipped_fh}, below threshold={skipped_power})."
        )
    assert reference_ports is not None
    assert reference_dmrs is not None
    assert reference_pilot_magnitudes is not None

    ratios = np.asarray(power_ratios, dtype=np.float64)
    fingerprint = hashlib.sha1("|".join(timestamps).encode()).hexdigest()[:12]
    ls_cov = ls_second_moment_sum / float(num_channel_vectors)
    noise_cov = noise_second_moment_sum / float(num_noise_vectors)
    print(
        f"{name}: kept {len(timestamps)}/{len(records)} records "
        f"(missing FH={skipped_fh}, power<{min_power_ratio_db:g} dB="
        f"{skipped_power})"
    )
    print(f"  channel vectors: {num_channel_vectors}")
    print(f"  noise vectors:   {num_noise_vectors}")
    print(f"  ports:           {reference_ports.tolist()}")
    print(f"  DM-RS symbols:   {reference_dmrs.tolist()}")
    print(f"  |pilot|:         {reference_pilot_magnitudes.tolist()}")
    print(
        "  mean diag:       "
        f"R_ls={np.real(np.trace(ls_cov)) / NUM_RX_ANTENNAS:.8g}, "
        f"R_noise={np.real(np.trace(noise_cov)) / NUM_RX_ANTENNAS:.8g}"
    )
    print(
        "  power ratio:     "
        f"min={ratios.min():.2f}, median={np.median(ratios):.2f}, "
        f"max={ratios.max():.2f} dB"
    )
    print(f"  fingerprint:     {fingerprint}")

    return SpatialDatasetStatistics(
        name=name,
        weight=weight,
        ls_second_moment_sum=ls_second_moment_sum,
        num_channel_vectors=num_channel_vectors,
        noise_second_moment_sum=noise_second_moment_sum,
        num_noise_vectors=num_noise_vectors,
        num_records=len(timestamps),
        timestamps=timestamps,
        power_ratios_db=ratios,
        ports=reference_ports,
        dmrs_symbols=reference_dmrs,
        pilot_magnitudes=reference_pilot_magnitudes,
    )


def project_psd(matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    """Project a Hermitian covariance onto the PSD cone without normalization."""

    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    clipped = np.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * clipped[None, :]) @ eigenvectors.conj().T
    projected = 0.5 * (projected + projected.conj().T)
    return projected, {
        "minimum_raw_eigenvalue": float(eigenvalues[0]),
        "num_negative_raw_eigenvalues": int(np.sum(eigenvalues < 0)),
        "raw_average_diagonal": float(
            np.real(np.trace(hermitian)) / NUM_RX_ANTENNAS
        ),
        "projected_average_diagonal": float(
            np.real(np.trace(projected)) / NUM_RX_ANTENNAS
        ),
    }


def estimate_spatial_covariance(
    datasets: Sequence[SpatialDatasetStatistics],
    *,
    project_to_psd: bool,
    output_dtype: np.dtype,
) -> SpatialCovarianceEstimate:
    """Combine datasets and subtract the measured spatial noise covariance."""

    if not datasets:
        raise ValueError("At least one dataset is required.")
    reference_dmrs = datasets[0].dmrs_symbols
    if tuple(reference_dmrs.tolist()) != DMRS_SYMBOLS_EXPECTED:
        raise ValueError(
            f"Expected DM-RS symbols {DMRS_SYMBOLS_EXPECTED}, received "
            f"{tuple(reference_dmrs.tolist())}."
        )

    weighted_ls_sum = np.zeros(
        (NUM_RX_ANTENNAS, NUM_RX_ANTENNAS), dtype=np.complex128
    )
    weighted_noise_sum = np.zeros_like(weighted_ls_sum)
    weighted_channel_count = 0.0
    weighted_noise_count = 0.0
    details: dict[str, dict] = {}

    for dataset in datasets:
        if dataset.weight <= 0:
            raise ValueError(f"{dataset.name}: weight must be positive.")
        if not np.array_equal(dataset.dmrs_symbols, reference_dmrs):
            raise ValueError("DM-RS symbols differ between datasets.")
        weighted_ls_sum += dataset.weight * dataset.ls_second_moment_sum
        weighted_channel_count += dataset.weight * dataset.num_channel_vectors
        weighted_noise_sum += dataset.weight * dataset.noise_second_moment_sum
        weighted_noise_count += dataset.weight * dataset.num_noise_vectors

        dataset_ls = dataset.ls_second_moment_sum / dataset.num_channel_vectors
        dataset_noise = (
            dataset.noise_second_moment_sum / dataset.num_noise_vectors
        )
        details[dataset.name] = {
            "weight": float(dataset.weight),
            "num_records": int(dataset.num_records),
            "num_channel_vectors": int(dataset.num_channel_vectors),
            "num_noise_vectors": int(dataset.num_noise_vectors),
            "ls_average_diagonal": float(
                np.real(np.trace(dataset_ls)) / NUM_RX_ANTENNAS
            ),
            "noise_average_diagonal": float(
                np.real(np.trace(dataset_noise)) / NUM_RX_ANTENNAS
            ),
        }

    if weighted_channel_count <= 0 or weighted_noise_count <= 0:
        raise ValueError("The combined effective sample count is zero.")

    ls_second_moment = weighted_ls_sum / weighted_channel_count
    ls_second_moment = 0.5 * (
        ls_second_moment + ls_second_moment.conj().T
    )
    noise_covariance = weighted_noise_sum / weighted_noise_count
    noise_covariance = 0.5 * (
        noise_covariance + noise_covariance.conj().T
    )
    corrected = ls_second_moment - noise_covariance
    corrected = 0.5 * (corrected + corrected.conj().T)

    if project_to_psd:
        covariance, psd_diagnostics = project_psd(corrected)
    else:
        covariance = corrected
        eigenvalues = np.linalg.eigvalsh(corrected)
        psd_diagnostics = {
            "minimum_raw_eigenvalue": float(eigenvalues[0]),
            "num_negative_raw_eigenvalues": int(np.sum(eigenvalues < 0)),
            "raw_average_diagonal": float(
                np.real(np.trace(corrected)) / NUM_RX_ANTENNAS
            ),
            "projected_average_diagonal": None,
        }

    diagnostics = {
        "datasets": details,
        "psd_projection_applied": bool(project_to_psd),
        "psd": psd_diagnostics,
        "ls_second_moment_average_diagonal": float(
            np.real(np.trace(ls_second_moment)) / NUM_RX_ANTENNAS
        ),
        "noise_covariance_average_diagonal": float(
            np.real(np.trace(noise_covariance)) / NUM_RX_ANTENNAS
        ),
    }
    return SpatialCovarianceEstimate(
        covariance=np.asarray(covariance, dtype=output_dtype),
        corrected_covariance_raw=np.asarray(corrected, dtype=output_dtype),
        ls_second_moment=np.asarray(ls_second_moment, dtype=output_dtype),
        noise_covariance=np.asarray(noise_covariance, dtype=output_dtype),
        diagnostics=diagnostics,
    )


def output_path(prefix: Path) -> Path:
    return prefix.parent / f"{prefix.name}_space_cov_mat.npy"


def noise_output_path(prefix: Path) -> Path:
    return prefix.parent / f"{prefix.name}_space_noise_cov_mat.npy"


def dataset_output_path(prefix: Path, num_layers: int, *, noise: bool) -> Path:
    suffix = "space_noise_cov_mat.npy" if noise else "space_cov_mat.npy"
    return prefix.parent / f"{prefix.name}_{num_layers}ull_{suffix}"


def _exponential_covariance(
    size: int,
    *,
    correlation: float,
    phase: float,
    scale: np.ndarray,
) -> np.ndarray:
    indices = np.arange(size)
    difference = indices[:, None] - indices[None, :]
    correlation_matrix = (
        correlation ** np.abs(difference)
        * np.exp(1j * phase * difference)
    )
    root_scale = np.sqrt(np.asarray(scale, dtype=np.float64))
    return root_scale[:, None] * correlation_matrix * root_scale[None, :]


def synthetic_self_test() -> None:
    """Check full correlated-noise subtraction for four receive antennas."""

    rng = np.random.default_rng(41)
    num_vectors = 100000
    channel_covariance = _exponential_covariance(
        NUM_RX_ANTENNAS,
        correlation=0.72,
        phase=0.17,
        scale=np.asarray([1.8, 1.5, 1.25, 1.0]),
    )
    noise_covariance = _exponential_covariance(
        NUM_RX_ANTENNAS,
        correlation=0.18,
        phase=-0.11,
        scale=np.asarray([0.16, 0.20, 0.14, 0.18]),
    )
    channel_chol = np.linalg.cholesky(channel_covariance)
    noise_chol = np.linalg.cholesky(noise_covariance)
    channel_white = (
        rng.standard_normal((num_vectors, NUM_RX_ANTENNAS))
        + 1j * rng.standard_normal((num_vectors, NUM_RX_ANTENNAS))
    ) / np.sqrt(2.0)
    noise_white = (
        rng.standard_normal((num_vectors, NUM_RX_ANTENNAS))
        + 1j * rng.standard_normal((num_vectors, NUM_RX_ANTENNAS))
    ) / np.sqrt(2.0)
    channel = channel_white @ channel_chol.T
    noise = noise_white @ noise_chol.T
    h_ls = channel + noise

    observed = h_ls.T @ h_ls.conj() / num_vectors
    measured_noise = noise.T @ noise.conj() / num_vectors
    corrected = observed - measured_noise
    relative_error = (
        np.linalg.norm(corrected - channel_covariance)
        / np.linalg.norm(channel_covariance)
    )
    hermitian_error = np.linalg.norm(corrected - corrected.conj().T)
    print("Spatial-covariance synthetic self-test")
    print(f"  relative covariance error: {relative_error:.6f}")
    print(f"  Hermitian error:           {hermitian_error:.3e}")
    print(
        "  measured noise off-diagonal: "
        f"{measured_noise[0, 1]:.6g}"
    )
    if relative_error > 0.03:
        raise RuntimeError("Spatial covariance noise-subtraction test failed.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull single- and dual-layer PUSCH records, run no-OCC LS without "
            "interpolation, and estimate a noise-corrected 4-by-4 spatial "
            "covariance."
        )
    )
    parser.add_argument("--single-db", choices=sorted(DATABASES), default="9")
    parser.add_argument("--dual-db", choices=sorted(DATABASES), default="8")
    parser.add_argument(
        "--limit", type=int, default=100, help="Per-dataset FAPI record limit."
    )
    parser.add_argument("--single-limit", type=int)
    parser.add_argument("--dual-limit", type=int)
    parser.add_argument(
        "--slot", type=int, help="Optional exact Slot refinement."
    )
    parser.add_argument(
        "--mcs-index", type=int, help="Optional exact MCS-index refinement."
    )
    parser.add_argument("--min-power-ratio-db", type=float, default=0.0)
    parser.add_argument("--single-timestamps", type=Path)
    parser.add_argument("--dual-timestamps", type=Path)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=REPO_ROOT / "weights" / "pusch_direct_datalake",
    )
    parser.add_argument("--single-weight", type=float, default=1.0)
    parser.add_argument("--dual-weight", type=float, default=1.0)
    parser.add_argument(
        "--skip-psd-projection",
        action="store_true",
        help="Save the raw noise-corrected spatial covariance.",
    )
    parser.add_argument("--complex128-output", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.limit <= 0:
        parser.error("--limit must be positive.")
    for name in ("single_limit", "dual_limit"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            option_name = name.replace("_", "-")
            parser.error(f"--{option_name} must be positive.")
    if args.single_weight <= 0 or args.dual_weight <= 0:
        parser.error("Dataset weights must be positive.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        synthetic_self_test()
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    single_timestamps = resolve_timestamp_path(args.single_timestamps)
    dual_timestamps = resolve_timestamp_path(args.dual_timestamps)

    client = clickhouse_connect.get_client(host=args.host)
    single = extract_dataset_statistics(
        client,
        selector=args.single_db,
        name="single_layer",
        limit=args.single_limit or args.limit,
        slot=args.slot,
        mcs_index=args.mcs_index,
        min_power_ratio_db=args.min_power_ratio_db,
        timestamps_path=single_timestamps,
        weight=args.single_weight,
    )
    dual = extract_dataset_statistics(
        client,
        selector=args.dual_db,
        name="dual_layer",
        limit=args.dual_limit or args.limit,
        slot=args.slot,
        mcs_index=args.mcs_index,
        min_power_ratio_db=args.min_power_ratio_db,
        timestamps_path=dual_timestamps,
        weight=args.dual_weight,
    )
    if single.ports.size != 1:
        raise ValueError("The --single-db dataset must contain one layer.")
    if dual.ports.size != 2:
        raise ValueError("The --dual-db dataset must contain two layers.")

    output_dtype = np.complex128 if args.complex128_output else np.complex64
    estimate = estimate_spatial_covariance(
        [single, dual],
        project_to_psd=not args.skip_psd_projection,
        output_dtype=np.dtype(output_dtype),
    )
    single_estimate = estimate_spatial_covariance(
        [single],
        project_to_psd=not args.skip_psd_projection,
        output_dtype=np.dtype(output_dtype),
    )
    dual_estimate = estimate_spatial_covariance(
        [dual],
        project_to_psd=not args.skip_psd_projection,
        output_dtype=np.dtype(output_dtype),
    )
    path = output_path(output_prefix)
    noise_path = noise_output_path(output_prefix)
    np.save(path, estimate.covariance)
    np.save(noise_path, estimate.noise_covariance)
    dataset_paths = []
    for num_layers, dataset_estimate in (
        (1, single_estimate),
        (2, dual_estimate),
    ):
        dataset_cov_path = dataset_output_path(
            output_prefix, num_layers, noise=False
        )
        dataset_noise_path = dataset_output_path(
            output_prefix, num_layers, noise=True
        )
        np.save(dataset_cov_path, dataset_estimate.covariance)
        np.save(dataset_noise_path, dataset_estimate.noise_covariance)
        dataset_paths.append((dataset_cov_path, dataset_noise_path))

    diagnostics = estimate.diagnostics
    psd = diagnostics["psd"]
    ls_diagonal = diagnostics["ls_second_moment_average_diagonal"]
    noise_diagonal = diagnostics["noise_covariance_average_diagonal"]
    raw_diagonal = psd["raw_average_diagonal"]
    print("\nCombined spatial covariance")
    print(f"  R_ls mean diagonal:           {ls_diagonal:.8g}")
    print(f"  R_noise mean diagonal:        {noise_diagonal:.8g}")
    print(f"  corrected raw mean diagonal:  {raw_diagonal:.8g}")
    minimum_eigenvalue = psd["minimum_raw_eigenvalue"]
    negative_eigenvalues = psd["num_negative_raw_eigenvalues"]
    print(f"  minimum corrected eigenvalue: {minimum_eigenvalue:.8g}")
    print(f"  negative corrected eigenvalues: {negative_eigenvalues}")
    print("  measured spatial noise covariance:")
    print(np.array2string(estimate.noise_covariance, precision=5))
    print("  raw noise-corrected spatial covariance:")
    print(np.array2string(estimate.corrected_covariance_raw, precision=5))
    print(f"Saved spatial covariance: {path}")
    print(f"Saved spatial noise covariance: {noise_path}")
    for dataset_cov_path, dataset_noise_path in dataset_paths:
        print(f"Saved dataset spatial covariance: {dataset_cov_path}")
        print(f"Saved dataset spatial noise covariance: {dataset_noise_path}")


if __name__ == "__main__":
    main()
