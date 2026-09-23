#!/usr/bin/env python3
"""Estimate direct-pilot PUSCH frequency covariances from datalake.

This is the datalake wrapper for
``estimate_pusch_frequency_covariance_direct.py``. It deliberately uses
``PUSCHLSChannelEstimatorNoOCC`` with ``interpolation_type=None`` so that the
stored channel observations are the per-resource-element LS estimates before
Sionna's PUSCH OCC averaging.

The script always estimates both the naive covariance and the lag/Toeplitz covariance.

The lag estimate is not PSD-projected unless ``--project-lag-psd`` is passed.
When projection is enabled, the saved lag matrix is PSD but generally no
longer exactly Toeplitz.

This is research code is not optimized for speed, memory or neatness. The code is a mere
demonstration of the proposed method in the accompanying paper.

Author: Nuri Berke Baytekin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import clickhouse_connect
import numpy as np
import pandas as pd
import tensorflow as tf


COV_EST_DIR = Path(__file__).resolve().parent
REPO_ROOT = COV_EST_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
for module_dir in (COV_EST_DIR, SCRIPTS_DIR, REPO_ROOT):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from estimate_pusch_frequency_covariance_direct import (
    DatasetData,
    estimate_direct_comb_frequency_covariances,
    estimate_dataset_statistics,
)
from pusch_ls_channel_estimator_no_occ import (
    PUSCHLSChannelEstimatorNoOCC,
)
from eval_duidd_from_datalake import (
    create_sionna_transmitter_from_record,
    fapi_dmrs_ports_to_sionna_dmrs_port_set,
    rx_slot_to_sionna_y,
)


tf.get_logger().setLevel("ERROR")

N_SC = 273 * 12
NUM_FH_SYMBOLS = 14
NUM_RX_ANTENNAS = 4
NUM_PUSCH_SYMBOLS = 13
DMRS_SYMBOLS_EXPECTED = (0, 5, 10)


@dataclass(frozen=True)
class DatabaseSpec:
    database: str
    split_time: str
    num_layers: int


DATABASES = {
    '3': DatabaseSpec('quectel_2ULLs_j61_2026_04_27', "2026-04-26 12:48:30.000000000", 2),
    '4': DatabaseSpec('quectel_2ULLs_13dB_j61_2026_04_28', "2026-04-28 12:48:30.000000000", 2),
    '5': DatabaseSpec('quectel_1ULLs_7dB_j61_2026_04_28', "2026-04-28 12:48:30.000000000", 1),
    '6': DatabaseSpec('quectel_2ULLs_13dB_j61_2026_05_12', "2026-05-12 11:53:30", 2),
    '7': DatabaseSpec('quectel_1ULLs_7dB_j61_2026_05_12', "2026-05-12 12:22:00", 1),
    "8": DatabaseSpec(
        "Pixel9Pro_2ULLs_12dB_j61_2026_06_04",
        "2026-06-04 13:27:17.372000000",
        2,
    ),
    "-1": DatabaseSpec("quectel_2ULLs_28dB_jFloor_2026_06_02","2026-06-02 01:00:00",2),
    "-2": DatabaseSpec("quectel_2ULLs_28dB_j61_2026_05_08","2026-06-02 01:00:00",2),
    "9": DatabaseSpec(
        "sgs23_1ULLs_5dB_j61_2026_06_04",
        "2026-06-04 12:59:00.372000000",
        1,
    ),
    "10": DatabaseSpec(
        "Pixel9Pro_2ULLs_12dB_jFloorRealLoop_2026_06_10",
        "2026-06-10 14:47:00",
        2,
    ),
}


@dataclass
class ExtractedDataset:
    dataset: DatasetData
    timestamps: list[str]
    power_ratios_db: np.ndarray
    dmrs_symbol_indices: np.ndarray
    pilot_magnitudes: np.ndarray


def format_ch_datetime64(timestamp) -> str:
    value = pd.Timestamp(timestamp)
    fractional_ns = value.microsecond * 1000 + int(value.nanosecond)
    return f"{value.strftime('%Y-%m-%d %H:%M:%S')}.{fractional_ns:09d}"


def resolve_timestamp_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    if path.exists():
        return path.resolve()
    
    candidate = REPO_ROOT / "eval_timestamps" / path
    if candidate.exists():
        return candidate.resolve()
    
    raise FileNotFoundError(f"Timestamp file not found: {path}")


def resolve_timestamp_request(
    path: Optional[Path],
) -> tuple[Optional[Path], Optional[Path]]:

    if path is None:
        return None, None
    try:
        return resolve_timestamp_path(path), None
    
    except FileNotFoundError:
        if path.is_absolute():
            destination = path
        else:
            destination = REPO_ROOT / "eval_timestamps" / path
        return None, destination.resolve()


def save_timestamps(path: Path, timestamps: Sequence[str]) -> None:
    """Persist randomly selected real datalake timestamps."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite timestamp file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        pickle.dump(list(timestamps), file)
    print(f"Saved {len(timestamps)} random timestamps: {path}")


def load_timestamps(path: Path, limit: int) -> list:
    with path.open("rb") as file:
        timestamps = list(pickle.load(file))
    return timestamps[:limit]


def build_query(
    spec: DatabaseSpec,
    *,
    limit: int,
    slot: Optional[int],
    mcs_index: Optional[int],
    timestamps_path: Optional[Path],
    random_order: bool = False,
) -> str:

    params = [
        "rbSize = 273",
        f"NrOfSymbols = {NUM_PUSCH_SYMBOLS}",
        "mcsIndex > 5",
        "nrOfSymbols = 13",
        "qamModOrder = 4",
        f"nrOfLayers = {spec.num_layers}",
        f"TsTaiNs < toDateTime64('{spec.split_time}', 9)",
    ]
    if slot is not None:
        params.append(f"Slot = {slot}")
    if mcs_index is not None:
        params.append(f"mcsIndex = {mcs_index}")

    if timestamps_path is not None:
        timestamps = load_timestamps(timestamps_path, limit)
        if not timestamps:
            raise ValueError(f"Timestamp file is empty: {timestamps_path}")
        timestamp_literals = ", ".join(
            f"toDateTime64('{format_ch_datetime64(ts)}', 9)"
            for ts in timestamps
        )
        params.append(f"TsTaiNs IN ({timestamp_literals})")

    ordering = "rand()" if random_order else "TsTaiNs, CellId"
    where_clause = "\n                 and ".join(params)

    query = f"""select * from export_fapi_{spec.database}
               where {where_clause}
               order by {ordering}
               limit {limit}"""

    return query


def fetch_fh_row(client, database: str, record):
    """Join one FAPI record to exactly one FH row."""

    timestamp = format_ch_datetime64(record.TsTaiNs)
    return client.query_df(
        f"""select fhData from export_fh_{database}
            where TsTaiNs == toDateTime64('{timestamp}', 9)
              and CellId == {int(record.CellId)}
            limit 1"""
    )


def parse_fh(fh_row):
    real_values = np.asarray(fh_row, dtype=np.int16).view(np.float16)
    real_values = real_values.astype(np.float32)    
    complex_values = real_values.view(np.complex64)

    return np.swapaxes(
        complex_values.reshape(NUM_RX_ANTENNAS, NUM_FH_SYMBOLS, N_SC),
        2,
        0,
    )


def allocation_power_ratio_db(y_full: np.ndarray, record):

    start_symbol = int(record.StartSymbolIndex)
    num_symbols = int(record.NrOfSymbols)
    noise_symbol = start_symbol + num_symbols
    if noise_symbol >= y_full.shape[1]:
        raise ValueError(
            f"No following noise-only symbol is available at index {noise_symbol}."
        )
    
    pusch_power = float(np.mean(np.abs(y_full[:, start_symbol:noise_symbol, :]) ** 2))

    noise_power = float(np.mean(np.abs(y_full[:, noise_symbol, :]) ** 2))
    if noise_power <= 0:
        return float("inf")
    return 10.0 * np.log10(pusch_power / noise_power)


def dmrs_symbol_indices(resource_grid) -> tuple[int, ...]:
    mask = np.asarray(resource_grid.pilot_pattern.mask.numpy())
    symbol_has_pilot = np.any(mask > 0, axis=(0, 1, 3))
    return tuple(int(index) for index in np.flatnonzero(symbol_has_pilot))


def build_no_occ_estimator(transmitter, record):
    return PUSCHLSChannelEstimatorNoOCC(
        resource_grid=transmitter.resource_grid,
        dmrs_length=1,
        dmrs_additional_position=2,
        num_cdm_groups_without_data=int(record.numDmrsCdmGrpsNoData),
        interpolation_type=None,
    )


def validate_record_configuration(
    record,
    transmitter,
    estimator,
    *,
    expected_num_layers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    num_layers = int(record.nrOfLayers)
    if num_layers != expected_num_layers:
        raise ValueError(
            f"Expected {expected_num_layers} layers, received {num_layers}."
        )
    
    ports = np.asarray(
        fapi_dmrs_ports_to_sionna_dmrs_port_set(record.dmrsPorts),
        dtype=np.int64,
    )
    
    

    dmrs_symbols = np.asarray(
        dmrs_symbol_indices(transmitter.resource_grid), dtype=np.int64
    )
    

    pilots = np.asarray(estimator.pilot_symbols.numpy())
    expected_pilot_count = len(DMRS_SYMBOLS_EXPECTED) * N_SC
    
    
    pilot_grid = pilots[0].reshape(num_layers, len(dmrs_symbols), N_SC)

    nonzero_magnitudes = np.abs(pilot_grid[np.abs(pilot_grid) > 0])
    if not np.all(np.isfinite(nonzero_magnitudes)):
        raise ValueError("Pilot tensor contains non-finite magnitudes.")

    return ports, dmrs_symbols, np.unique(nonzero_magnitudes)


def build_noise_resource_grid(y_full: np.ndarray, record, num_symbols: int) -> np.ndarray:
    """Repeat the following noise-only symbol over the Sionna resource grid."""

    start_sc = int(record.rbStart) * 12
    end_sc = start_sc + int(record.rbSize) * 12
    noise_symbol = int(record.StartSymbolIndex) + int(record.NrOfSymbols)
    
    noise_re = y_full[start_sc:end_sc, noise_symbol, :]
    noise_grid = np.repeat(noise_re.T[:, None, :], num_symbols, axis=1)
    return noise_grid[None, None, ...].astype(np.complex64)


def pilot_output_to_canonical(
    value,
    *,
    num_layers: int,
    num_dmrs_symbols: int,
) -> np.ndarray:
    """Convert one pilot-domain output to [rx, layer, time, freq]."""

    array = np.asarray(value.numpy() if hasattr(value, "numpy") else value)
    return array[0, 0, :, 0, :, :].reshape(array.shape[2], num_layers, num_dmrs_symbols, N_SC)


def extract_one_record(record, y_full: np.ndarray, *, expected_num_layers: int):
    """Return canonical no-OCC channel/noise LS arrays for one record."""

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
    no_placeholder = tf.ones([1, 1, int(y_channel.shape[2])], dtype=tf.float32)
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
    return (
        h_canonical.astype(np.complex64, copy=False),
        noise_canonical.astype(np.complex64, copy=False),
        ports,
        dmrs_symbols,
        pilot_magnitudes,
    )


def extract_dataset(
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
    random_order: bool = False,
) -> ExtractedDataset:
    """Query and convert one single- or dual-layer measurement dataset."""

    if selector not in DATABASES:
        raise ValueError(
            f"Unsupported database selector {selector}; known: {sorted(DATABASES)}"
        )
    spec = DATABASES[selector]
    query = build_query(
        spec,
        limit=limit,
        slot=slot,
        mcs_index=mcs_index,
        timestamps_path=timestamps_path,
        random_order=random_order,
    )
    records = client.query_df(query)
    print(
        f"\n{name}: queried {len(records)} FAPI records from {spec.database} "
        f"(expected layers={spec.num_layers})"
    )
    if records.empty:
        raise RuntimeError(f"{name}: query returned no FAPI records.")

    h_values: list[np.ndarray] = []
    noise_values: list[np.ndarray] = []
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

        h_ls, noise_ls, ports, dmrs_symbols, pilot_magnitudes = (
            extract_one_record(record, y_full, expected_num_layers=spec.num_layers)
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

        h_values.append(h_ls)
        noise_values.append(noise_ls)
        timestamps.append(format_ch_datetime64(record.TsTaiNs))
        power_ratios.append(ratio_db)

    if not h_values:
        raise RuntimeError(
            f"{name}: no records survived FH and power filtering "
            f"(missing FH={skipped_fh}, below threshold={skipped_power})."
        )
    
    assert reference_ports is not None
    assert reference_dmrs is not None
    assert reference_pilot_magnitudes is not None

    h_stack = np.stack(h_values, axis=0)
    noise_stack = np.stack(noise_values, axis=0)
    fingerprint = hashlib.sha1("|".join(timestamps).encode()).hexdigest()[:12]
    ratios_array = np.asarray(power_ratios, dtype=np.float64)

    print(
        f"{name}: kept {len(h_values)}/{len(records)} records "
        f"(missing FH={skipped_fh}, power<{min_power_ratio_db:g} dB="
        f"{skipped_power})"
    )
    print(f"  h_ls shape:     {h_stack.shape}")
    print(f"  noise_ls shape: {noise_stack.shape}")
    print(f"  ports:          {reference_ports.tolist()}")
    print(f"  DM-RS symbols:  {reference_dmrs.tolist()}")
    print(f"  |pilot|:        {reference_pilot_magnitudes.tolist()}")
    print(
        "  power ratio:   "
        f"min={ratios_array.min():.2f}, mean={np.mean(ratios_array):.2f}, "
        f"max={ratios_array.max():.2f} dB"
    )
    print(f"  fingerprint:    {fingerprint}")

    dataset = DatasetData(
        name=name,
        h_ls=h_stack,
        ports=reference_ports,
        noise_ls=noise_stack,
        weight=weight,
    )
    return ExtractedDataset(
        dataset=dataset,
        timestamps=timestamps,
        power_ratios_db=ratios_array,
        dmrs_symbol_indices=reference_dmrs,
        pilot_magnitudes=reference_pilot_magnitudes,
    )


def output_path(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}_{suffix}"

def pilot_domain_snr(dataset):
    stats = estimate_dataset_statistics(
        dataset,
        method="pdp",
        n_sc=N_SC,
        num_dmrs_symbols=len(DMRS_SYMBOLS_EXPECTED),
        chunk_size=512,
        drop_exact_zero_vectors=True,
    )

    n_comb = N_SC // 2

    ls_power = np.real(
        np.trace(stats.channel_covariance)
    ) / n_comb

    noise_power = np.real(
        np.trace(stats.noise_covariance)
    ) / n_comb

    channel_power = ls_power - noise_power
    snr_db = 10 * np.log10(channel_power / noise_power)

    return {
        "ls_power": ls_power,
        "channel_power": channel_power,
        "noise_power": noise_power,
        "snr_db": snr_db,
    }


def estimate_and_save(
    *,
    method: str,
    single: DatasetData,
    dual: DatasetData,
    output_prefix: Path,
    chunk_size: int,
    project_lag_psd: bool,
    project_naive_psd: bool,
    output_dtype: np.dtype,
) -> None:
    
    estimate = estimate_direct_comb_frequency_covariances(
        single,
        dual,
        method=method,
        n_fft=4096,
        n_sc=N_SC,
        num_dmrs_symbols=len(DMRS_SYMBOLS_EXPECTED),
        chunk_size=chunk_size,
        project_naive_psd=project_naive_psd,
        project_lag_psd=project_lag_psd,
        output_dtype=output_dtype,
    )
    if method == "naive":
        covariance = estimate.cov_mat_freq_naive
    elif method == "lag":
        covariance = estimate.cov_mat_freq_lag
    else:
        raise ValueError(f"Unsupported method: {method}")
    if covariance is None:
        raise RuntimeError(f"The {method} estimator returned no covariance.")
    covariance_path = output_path(
        output_prefix, f"{method}_freq_cov_mat.npy"
    )
    np.save(covariance_path, covariance)
    print(f"Saved {method} frequency covariance: {covariance_path}")
    if method == "lag":
        noise_covariance = estimate.direct_noise_covariance
        if noise_covariance is None:
            raise RuntimeError("The lag estimator returned no noise covariance.")
        noise_covariance_path = output_path(
            output_prefix, "lag_freq_noise_cov_mat.npy"
        )
        np.save(noise_covariance_path, noise_covariance)
        print(
            "Saved direct-comb frequency noise covariance: "
            f"{noise_covariance_path}"
        )

        lag_metadata = estimate.metadata["lag"]
        diagnostics = {
            "direct_ls_zero_lag": lag_metadata["direct_ls_zero_lag"],
            "direct_noise_zero_lag": lag_metadata["direct_noise_zero_lag"],
            "direct_zero_lag_after_noise_subtraction": lag_metadata[
                "direct_zero_lag_after_noise_subtraction"
            ],
            "psd_projection_applied": lag_metadata["psd_projection_applied"],
            "psd_diagnostics": lag_metadata["psd_diagnostics"],
        }
        diagnostics_path = output_path(
            output_prefix, "lag_freq_cov_diagnostics.json"
        )
        with diagnostics_path.open("w", encoding="utf-8") as file:
            json.dump(diagnostics, file, indent=2, sort_keys=True)
        print(f"Saved lag diagnostics: {diagnostics_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull single- and dual-layer PUSCH records from ClickHouse, "
            "prepare direct no-OCC h_ls/noise_ls, and estimate frequency "
            "covariance matrices."
        )
    )
    parser.add_argument("--single-db", choices=sorted(DATABASES), default="9")
    parser.add_argument(
        "--dual-db", choices=sorted(DATABASES), default="8"
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Per-dataset FAPI record limit."
    )
    parser.add_argument("--single-limit", type=int)
    parser.add_argument("--dual-limit", type=int)
    parser.add_argument(
        "--slot", type=int, help="Optional Slot."
    )
    parser.add_argument(
        "--mcs-index", type=int, help="Optional MCS-index."
    )
    parser.add_argument("--min-power-ratio-db", type=float, default=0.0)
    parser.add_argument(
        "--single-timestamps",
        type=Path,
        help="Single-layer timestamp file; create it randomly if missing.",
    )
    parser.add_argument(
        "--dual-timestamps",
        type=Path,
        help="Dual-layer timestamp file; create it randomly if missing.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=REPO_ROOT / "weights" / "pusch_direct_datalake",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--single-weight", type=float, default=1.0)
    parser.add_argument("--dual-weight", type=float, default=1.0)
    parser.add_argument(
        "--project-lag-psd",
        action="store_true",
        help="Eigenvalue-clip the lag matrix (the result is no longer Toeplitz).",
    )
    parser.add_argument(
        "--skip-naive-psd-projection",
        action="store_true",
        help="Keep the raw noise-corrected naive covariance.",
    )
    parser.add_argument("--complex128-output", action="store_true")
    args = parser.parse_args(argv)

    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    single_timestamps, single_timestamp_output = resolve_timestamp_request(args.single_timestamps)
    dual_timestamps, dual_timestamp_output = resolve_timestamp_request(args.dual_timestamps)

    if single_timestamp_output is not None:
        print(
            "Single-layer timestamp file not found; selecting random records "
            f"and creating {single_timestamp_output}"
        )
    if dual_timestamp_output is not None:
        print(
            "Dual-layer timestamp file not found; selecting random records "
            f"and creating {dual_timestamp_output}"
        )

    single_limit = args.single_limit or args.limit
    dual_limit = args.dual_limit or args.limit

    client = clickhouse_connect.get_client(host=args.host)
    single = extract_dataset(
        client,
        selector=args.single_db,
        name="single_layer",
        limit=single_limit,
        slot=args.slot,
        mcs_index=args.mcs_index,
        min_power_ratio_db=args.min_power_ratio_db,
        timestamps_path=single_timestamps,
        weight=args.single_weight,
        random_order=single_timestamp_output is not None,
    )
    dual = extract_dataset(
        client,
        selector=args.dual_db,
        name="dual_layer",
        limit=dual_limit,
        slot=args.slot,
        mcs_index=args.mcs_index,
        min_power_ratio_db=args.min_power_ratio_db,
        timestamps_path=dual_timestamps,
        weight=args.dual_weight,
        random_order=dual_timestamp_output is not None,
    )

    if single_timestamp_output is not None:
        save_timestamps(single_timestamp_output, single.timestamps)
    if dual_timestamp_output is not None:
        save_timestamps(dual_timestamp_output, dual.timestamps)

    if single.dataset.h_ls.shape[2] != 1:
        raise ValueError("The --single-db dataset must contain one layer.")
    if dual.dataset.h_ls.shape[2] != 2:
        raise ValueError(
            "The --dual-db dataset must contain two layers."
        )

    single_snr = pilot_domain_snr(single.dataset)
    dual_snr = pilot_domain_snr(dual.dataset)
    print("\nSingle-layer pilot-domain SNR:")

    for key, value in single_snr.items():
        print(f"  {key}: {value}")

    print("\nDual-layer pilot-domain SNR:")
    for key, value in dual_snr.items():
        print(f"  {key}: {value}")

    output_dtype = np.complex128 if args.complex128_output else np.complex64

    for method in ("naive", "lag"):
        print(f"\nEstimating {method} frequency covariance...")
        estimate_and_save(
            method=method,
            single=single.dataset,
            dual=dual.dataset,
            output_prefix=output_prefix,
            chunk_size=args.chunk_size,
            project_lag_psd=args.project_lag_psd,
            project_naive_psd=not args.skip_naive_psd_projection,
            output_dtype=np.dtype(output_dtype),
        )

    print("\nDone.")
    if not args.project_lag_psd:
        print(
            "WARNING: the saved lag covariance is the raw Toeplitz estimate; "
            "check that it is PSD before using it with LMMSE interpolation."
        )


if __name__ == "__main__":
    main()
