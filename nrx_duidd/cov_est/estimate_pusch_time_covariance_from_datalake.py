#!/usr/bin/env python3
"""Estimate a naive PUSCH time covariance from datalake measurements.

The OFDM symbol immediately following the 13-symbol PUSCH allocation is used
to estimate the LS error variance. It is repeated over the Sionna input grid
only to pass through the same pilot division as the channel estimate.

Otuput is written as:
    <output-prefix>_time_cov_mat.npy

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
    NUM_PUSCH_SYMBOLS,
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


TIME_INTERPOLATION_MATRIX = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.8, 0.2, 0.0],
        [0.6, 0.4, 0.0],
        [0.4, 0.6, 0.0],
        [0.2, 0.8, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.8, 0.2],
        [0.0, 0.6, 0.4],
        [0.0, 0.4, 0.6],
        [0.0, 0.2, 0.8],
        [0.0, 0.0, 1.0],
        [0.0, -0.2, 1.2],
        [0.0, -0.4, 1.4],
    ],
    dtype=np.float64,
)
TIME_INTERPOLATION_MATRIX.setflags(write=False)


@dataclass
class TimeDatasetStatistics:
    """Online temporal statistics for one measurement dataset."""

    name: str
    weight: float
    second_moment_sum: np.ndarray
    num_dmrs_vectors: int
    noise_power_sum: float
    num_noise_power_values: int
    num_records: int
    timestamps: list[str]
    power_ratios_db: np.ndarray
    ports: np.ndarray
    dmrs_symbols: np.ndarray
    pilot_magnitudes: np.ndarray


@dataclass
class TimeCovarianceEstimate:
    """Naive time covariance and the terms used to obtain it."""

    covariance: np.ndarray
    corrected_covariance_raw: np.ndarray
    pilot_ls_second_moment: np.ndarray
    corrected_pilot_covariance_raw: np.ndarray
    interpolated_ls_second_moment: np.ndarray
    error_covariance: np.ndarray
    interpolation_matrix: np.ndarray
    sigma_e2: float
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


def pilot_second_moment_sum(
    h_ls: np.ndarray,
    ports: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return the 3-by-3 moment sum over genuine DM-RS comb entries."""

    h = np.asarray(h_ls)
    num_dmrs_symbols = len(DMRS_SYMBOLS_EXPECTED)
    if h.ndim != 4 or h.shape[2:] != (num_dmrs_symbols, N_SC):
        raise ValueError(f"Unexpected pilot-domain channel shape: {h.shape}.")
    ports = np.asarray(ports, dtype=np.int64)
    if ports.shape != (h.shape[1],):
        raise ValueError(
            f"Expected one port per layer, received ports {ports.shape} "
            f"for channel shape {h.shape}."
        )
    layer_vectors = []
    for layer, port in enumerate(ports):
        direct = h[:, layer, :, :][..., direct_comb_indices(int(port))]
        layer_vectors.append(
            np.transpose(direct, (0, 2, 1)).reshape(-1, num_dmrs_symbols)
        )
    vectors = np.concatenate(layer_vectors, axis=0)
    valid = np.all(np.isfinite(vectors), axis=1)
    vectors = np.asarray(vectors[valid], dtype=np.complex128)
    if vectors.shape[0] == 0:
        raise ValueError("No finite DM-RS-comb channel vectors were found.")
    return vectors.T @ vectors.conj(), int(vectors.shape[0])


def noise_reference_symbol_power(
    noise_ls: np.ndarray,
    ports: np.ndarray,
    *,
    reference_dmrs_index: int = 0,
) -> tuple[float, int]:
    """Estimate LS error power from one DM-RS block and its valid comb."""

    noise = np.asarray(noise_ls)
    num_dmrs_symbols = len(DMRS_SYMBOLS_EXPECTED)
    if noise.ndim != 4 or noise.shape[2:] != (num_dmrs_symbols, N_SC):
        raise ValueError(f"Unexpected pilot-domain noise shape: {noise.shape}.")
    ports = np.asarray(ports, dtype=np.int64)
    if ports.shape != (noise.shape[1],):
        raise ValueError(
            f"Expected one port per layer, received ports {ports.shape} "
            f"for noise shape {noise.shape}."
        )
    if not 0 <= reference_dmrs_index < num_dmrs_symbols:
        raise ValueError(f"Invalid reference DM-RS index {reference_dmrs_index}.")
    layer_values = []
    for layer, port in enumerate(ports):
        indices = direct_comb_indices(int(port))
        layer_values.append(noise[:, layer, reference_dmrs_index, indices])
    values_at_symbol = np.concatenate(layer_values, axis=0)
    finite = np.isfinite(values_at_symbol)
    if not np.any(finite):
        raise ValueError("No finite pilot-domain noise values were found.")
    values = np.asarray(values_at_symbol[finite], dtype=np.complex128)
    power = np.real(values * values.conj())
    return float(np.sum(power, dtype=np.float64)), int(power.size)


def extract_one_record(
    record,
    y_full: np.ndarray,
    *,
    expected_num_layers: int,
) -> tuple[np.ndarray, float, int, np.ndarray, np.ndarray, np.ndarray]:
    """Return pilot-domain LS channel samples and marginal noise power."""

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
    noise_sum, noise_count = noise_reference_symbol_power(
        noise_canonical, ports, reference_dmrs_index=0
    )
    return (
        h_canonical.astype(np.complex64, copy=False),
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
) -> TimeDatasetStatistics:
    """Query one dataset and accumulate its temporal statistics online."""

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

    num_dmrs_symbols = len(DMRS_SYMBOLS_EXPECTED)
    second_moment_sum = np.zeros(
        (num_dmrs_symbols, num_dmrs_symbols), dtype=np.complex128
    )
    num_dmrs_vectors = 0
    noise_power_sum = 0.0
    num_noise_power_values = 0
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
            h_ls,
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
        record_moment, record_vectors = pilot_second_moment_sum(
            h_ls, ports
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

        second_moment_sum += record_moment
        num_dmrs_vectors += record_vectors
        noise_power_sum += record_noise_sum
        num_noise_power_values += record_noise_count
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
    sigma_e2 = noise_power_sum / float(num_noise_power_values)
    fingerprint = hashlib.sha1("|".join(timestamps).encode()).hexdigest()[:12]
    print(
        f"{name}: kept {len(timestamps)}/{len(records)} records "
        f"(missing FH={skipped_fh}, power<{min_power_ratio_db:g} dB="
        f"{skipped_power})"
    )
    print(f"  DM-RS vectors:    {num_dmrs_vectors}")
    print(f"  ports:            {reference_ports.tolist()}")
    print(f"  DM-RS symbols:    {reference_dmrs.tolist()}")
    print(f"  |pilot|:          {reference_pilot_magnitudes.tolist()}")
    print(f"  sigma_e^2:        {sigma_e2:.8g}")
    print(
        "  power ratio:     "
        f"min={ratios.min():.2f}, mean={np.mean(ratios):.2f}, "
        f"max={ratios.max():.2f} dB"
    )
    print(f"  fingerprint:      {fingerprint}")

    return TimeDatasetStatistics(
        name=name,
        weight=weight,
        second_moment_sum=second_moment_sum,
        num_dmrs_vectors=num_dmrs_vectors,
        noise_power_sum=noise_power_sum,
        num_noise_power_values=num_noise_power_values,
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
            np.real(np.trace(hermitian)) / hermitian.shape[0]
        ),
        "projected_average_diagonal": float(
            np.real(np.trace(projected)) / projected.shape[0]
        ),
    }


def estimate_time_covariance(
    datasets: Sequence[TimeDatasetStatistics],
    *,
    project_to_psd: bool,
    output_dtype: np.dtype,
) -> TimeCovarianceEstimate:
    """Correct the 3-by-3 pilot moment, then interpolate it in time."""

    if not datasets:
        raise ValueError("At least one dataset is required.")
    reference_dmrs = datasets[0].dmrs_symbols
    if tuple(reference_dmrs.tolist()) != DMRS_SYMBOLS_EXPECTED:
        raise ValueError(
            "The hardcoded time interpolation matrix requires DM-RS symbols "
            f"{DMRS_SYMBOLS_EXPECTED}, received {tuple(reference_dmrs.tolist())}."
        )
    num_dmrs_symbols = len(DMRS_SYMBOLS_EXPECTED)
    weighted_moment_sum = np.zeros(
        (num_dmrs_symbols, num_dmrs_symbols), dtype=np.complex128
    )
    weighted_vector_count = 0.0
    weighted_noise_sum = 0.0
    weighted_noise_count = 0.0
    details: dict[str, dict] = {}

    for dataset in datasets:
        if dataset.weight <= 0:
            raise ValueError(f"{dataset.name}: weight must be positive.")
        if not np.array_equal(dataset.dmrs_symbols, reference_dmrs):
            raise ValueError("DM-RS symbols differ between datasets.")
        weighted_moment_sum += dataset.weight * dataset.second_moment_sum
        weighted_vector_count += dataset.weight * dataset.num_dmrs_vectors
        weighted_noise_sum += dataset.weight * dataset.noise_power_sum
        weighted_noise_count += dataset.weight * dataset.num_noise_power_values
        details[dataset.name] = {
            "weight": float(dataset.weight),
            "num_records": int(dataset.num_records),
            "num_dmrs_vectors": int(dataset.num_dmrs_vectors),
            "sigma_e2": float(
                dataset.noise_power_sum / dataset.num_noise_power_values
            ),
        }

    if weighted_vector_count <= 0 or weighted_noise_count <= 0:
        raise ValueError("The combined effective sample count is zero.")
    pilot_ls_second_moment = weighted_moment_sum / weighted_vector_count
    pilot_ls_second_moment = 0.5 * (
        pilot_ls_second_moment + pilot_ls_second_moment.conj().T
    )
    sigma_e2 = float(weighted_noise_sum / weighted_noise_count)
    interpolation_matrix = TIME_INTERPOLATION_MATRIX.copy()
    pilot_error_covariance = sigma_e2 * np.eye(
        num_dmrs_symbols, dtype=np.float64
    )
    corrected_pilot = pilot_ls_second_moment - pilot_error_covariance
    corrected_pilot = 0.5 * (corrected_pilot + corrected_pilot.conj().T)

    interpolated_ls_second_moment = (
        interpolation_matrix
        @ pilot_ls_second_moment
        @ interpolation_matrix.T
    )
    error_covariance = sigma_e2 * (
        interpolation_matrix @ interpolation_matrix.T
    )
    corrected = interpolation_matrix @ corrected_pilot @ interpolation_matrix.T
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
                np.real(np.trace(corrected)) / NUM_PUSCH_SYMBOLS
            ),
            "projected_average_diagonal": None,
        }

    diagnostics = {
        "datasets": details,
        "psd_projection_applied": bool(project_to_psd),
        "psd": psd_diagnostics,
        "pilot_ls_second_moment_average_diagonal": float(
            np.real(np.trace(pilot_ls_second_moment)) / num_dmrs_symbols
        ),
        "corrected_pilot_covariance_average_diagonal": float(
            np.real(np.trace(corrected_pilot)) / num_dmrs_symbols
        ),
        "error_covariance_average_diagonal": float(
            np.trace(error_covariance) / NUM_PUSCH_SYMBOLS
        ),
    }
    return TimeCovarianceEstimate(
        covariance=np.asarray(covariance, dtype=output_dtype),
        corrected_covariance_raw=np.asarray(corrected, dtype=output_dtype),
        pilot_ls_second_moment=np.asarray(
            pilot_ls_second_moment, dtype=output_dtype
        ),
        corrected_pilot_covariance_raw=np.asarray(
            corrected_pilot, dtype=output_dtype
        ),
        interpolated_ls_second_moment=np.asarray(
            interpolated_ls_second_moment, dtype=output_dtype
        ),
        error_covariance=np.asarray(error_covariance, dtype=output_dtype),
        interpolation_matrix=interpolation_matrix,
        sigma_e2=sigma_e2,
        diagnostics=diagnostics,
    )


def output_path(prefix: Path) -> Path:
    return prefix.parent / f"{prefix.name}_time_cov_mat.npy"



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull single- and dual-layer PUSCH records, run no-OCC LS without "
            "interpolation, estimate a 3-by-3 DM-RS covariance, "
            "remove noise there, and interpolate it to 13-by-13."
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
        help="Save the raw noise-corrected time covariance.",
    )
    parser.add_argument("--complex128-output", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.limit <= 0:
        parser.error("--limit must be positive.")
    for name in ("single_limit", "dual_limit"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.single_weight <= 0 or args.dual_weight <= 0:
        parser.error("Dataset weights must be positive.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)


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
    estimate = estimate_time_covariance(
        [single, dual],
        project_to_psd=not args.skip_psd_projection,
        output_dtype=np.dtype(output_dtype),
    )
    path = output_path(output_prefix)
    np.save(path, estimate.covariance)

    print("\nCombined time covariance")
    print(f"  sigma_e^2:                    {estimate.sigma_e2:.8g}")
    print(
        "  pilot H_LS mean diagonal:     "
        f"{estimate.diagnostics['pilot_ls_second_moment_average_diagonal']:.8g}"
    )
    print(
        "  corrected pilot mean diag:    "
        f"{estimate.diagnostics['corrected_pilot_covariance_average_diagonal']:.8g}"
    )
    print(
        "  error-covariance mean diag:   "
        f"{estimate.diagnostics['error_covariance_average_diagonal']:.8g}"
    )
    print(
        "  corrected raw mean diag:      "
        f"{estimate.diagnostics['psd']['raw_average_diagonal']:.8g}"
    )
    print(
        "  minimum corrected eigenvalue: "
        f"{estimate.diagnostics['psd']['minimum_raw_eigenvalue']:.8g}"
    )
    print(
        "  negative corrected eigenvalues: "
        f"{estimate.diagnostics['psd']['num_negative_raw_eigenvalues']}"
    )
    print(
        "  diag(E) / sigma_e^2:          "
        f"{np.diag(estimate.error_covariance).real / estimate.sigma_e2}"
    )
    print("  pilot-domain H_LS:")
    print(np.array2string(estimate.pilot_ls_second_moment, precision=5))
    print("  pilot-domain H_LS - sigma_e^2 I:")
    print(
        np.array2string(estimate.corrected_pilot_covariance_raw, precision=5)
    )
    print("  interpolation matrix A:")
    print(np.array2string(estimate.interpolation_matrix, precision=3))
    print(f"Saved time covariance: {path}")


if __name__ == "__main__":
    main()

