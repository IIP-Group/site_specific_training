#!/usr/bin/env python3
"""
Evaluate a DUIDD PUSCH receiver using real data from the Aerial Data Lake.

Author: Nuri Berke Baytekin
"""

import os
from os.path import exists

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import argparse
import cupy as cp
import numpy as np
import pandas as pd
import pickle
import clickhouse_connect
from tqdm import tqdm
from collections import defaultdict

from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.algorithms import ChannelEqualizer
from aerial.phy5g.algorithms import NoiseIntfEstimator
from aerial.phy5g.ldpc import LdpcDeRateMatch
from aerial.phy5g.ldpc import LdpcDecoder
from aerial.phy5g.ldpc import CrcChecker
from aerial.phy5g.config import PuschConfig
from aerial.phy5g.config import PuschUeConfig
from aerial.util.cuda import get_cuda_stream
from aerial.util.fapi import dmrs_fapi_to_bit_array

from sionna.nr import (
    PUSCHConfig,
    PUSCHDMRSConfig,
    TBConfig,
    CarrierConfig,
    PUSCHTransmitter,
    TBDecoder,
)
from sionna.ofdm import LMMSEInterpolator

import sys
sys.path.append('../')
from utils import Parameters, load_weights
from utils.duidd import DUIDDPUSCHReceiver

_COV_EST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cov_est")
if _COV_EST_DIR not in sys.path:
    sys.path.insert(0, _COV_EST_DIR)
from pusch_ls_channel_estimator_no_occ import PUSCHLSChannelEstimatorNoOCC

_ = np.seterr(divide='ignore', invalid='ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def fapi_dmrs_ports_to_sionna_dmrs_port_set(dmrs_ports_bmsk):
    """Convert FAPI dmrsPorts bitmap to Sionna dmrs_port_set indices."""
    if dmrs_ports_bmsk is None:
        return [0]
    port_set = [i for i in range(12) if (int(dmrs_ports_bmsk) >> i) & 0x1]
    return port_set if port_set else [0]


def find_successful_retransmission(client, database, current_record, max_lookahead=10):
    """Find pduData from a successful HARQ retransmission (RV sequence 2, 3, 1)."""
    harq_process_id = current_record.harqProcessID
    rnti = current_record.rnti
    current_ts = current_record.TsTaiNs

    query = f"""select TsTaiNs, pduData, tbCrcFail, harqProcessID, rnti, rvIndex
                from export_fapi_{database}
                where rnti = {rnti}
                and TsTaiNs > toDateTime64('{current_ts.timestamp()}', 9)
                order by TsTaiNs
                limit {max_lookahead}"""

    subsequent_transmissions = client.query_df(query)
    if subsequent_transmissions.empty:
        return None, None

    seen_process_ids = {harq_process_id}
    rv_sequence = [2, 3, 1]
    req_rv = rv_sequence[0]
    rv_index = 0

    for _, tx in subsequent_transmissions.iterrows():
        tx_process_id = tx.harqProcessID
        if tx_process_id == harq_process_id:
            if len(seen_process_ids) == 16:
                return None, None
            if tx.rvIndex == req_rv:
                if tx.tbCrcFail == 0 and tx.pduData is not None:
                    return np.array(tx.pduData), tx.TsTaiNs
                rv_index += 1
                if rv_index >= len(rv_sequence):
                    return None, None
                req_rv = rv_sequence[rv_index]
        seen_process_ids.add(tx_process_id)

    return None, None


def create_pusch_config_from_record(pusch_record):
    """Create pyAerial PuschConfig from a datalake FAPI record."""
    pusch_ue_config = PuschUeConfig(
        scid=pusch_record.SCID,
        layers=pusch_record.nrOfLayers,
        dmrs_ports=pusch_record.dmrsPorts,
        rnti=pusch_record.rnti,
        data_scid=pusch_record.dataScramblingId,
        mcs_table=pusch_record.mcsTable,
        mcs_index=pusch_record.mcsIndex,
        code_rate=pusch_record.targetCodeRate,
        mod_order=pusch_record.qamModOrder,
        tb_size=pusch_record.TBSize,
    )
    pusch_config = PuschConfig(
        ue_configs=[pusch_ue_config],
        num_dmrs_cdm_grps_no_data=pusch_record.numDmrsCdmGrpsNoData,
        dmrs_scrm_id=pusch_record.ulDmrsScramblingId,
        start_prb=pusch_record.rbStart,
        num_prbs=pusch_record.rbSize,
        dmrs_syms=dmrs_fapi_to_bit_array(int(pusch_record.ulDmrsSymbPos)),
        dmrs_max_len=1,
        dmrs_add_ln_pos=2,
        start_sym=pusch_record.StartSymbolIndex,
        num_symbols=pusch_record.NrOfSymbols,
    )
    return [pusch_config]


def create_sionna_transmitter_from_record(pusch_record):
    """Create a per-sample Sionna PUSCHTransmitter from a datalake FAPI record."""
    cyclic_prefix = 'normal' if pusch_record.CyclicPrefix == 0 else 'extended'
    config_type = 1 if pusch_record.dmrsConfigType == 0 else 2
    dmrs_port_set = fapi_dmrs_ports_to_sionna_dmrs_port_set(pusch_record.dmrsPorts)

    carrier_config = CarrierConfig(
        n_cell_id=pusch_record.CellId,
        cyclic_prefix=cyclic_prefix,
        subcarrier_spacing=int(30.0),
        n_size_grid=pusch_record.rbSize,
        n_start_grid=pusch_record.rbStart,
        slot_number=int(pusch_record.Slot),
        frame_number=pusch_record.SFN,
    )
    pusch_dmrs_config = PUSCHDMRSConfig(
        config_type=config_type,
        additional_position=2,
        length=1,
        n_id=pusch_record.ulDmrsScramblingId,
        dmrs_port_set=dmrs_port_set,
        n_scid=pusch_record.SCID,
        num_cdm_groups_without_data=pusch_record.numDmrsCdmGrpsNoData,
    )
    tb_config = TBConfig(
        mcs_index=pusch_record.mcsIndex,
        mcs_table=(pusch_record.mcsTable + 1),
        channel_type="PUSCH",
    )
    pc = PUSCHConfig(
        carrier_config=carrier_config,
        pusch_dmrs_config=pusch_dmrs_config,
        tb_config=tb_config,
        num_antenna_ports=pusch_record.nrOfLayers,
        num_layers=pusch_record.nrOfLayers,
        symbol_allocation=[pusch_record.StartSymbolIndex, pusch_record.NrOfSymbols],
        mapping_type='B',
        data_scid=pusch_record.dataScramblingId,
        n_rnti=pusch_record.rnti,
    )
    return PUSCHTransmitter(pc, return_bits=False, output_domain="freq")


def estimate_no_from_pilots(rx_slot, transmitter):
    """Estimate scalar noise variance N0 from pilot residuals."""
    y = tf.cast(rx_slot, tf.complex64)
    y = tf.squeeze(y, axis=0)
    y = tf.squeeze(y, axis=0)

    pilot_mask = transmitter._resource_grid.build_type_grid()[:, 0] == 1
    pilot_mask = pilot_mask[0]
    pilots = tf.cast(transmitter.pilot_pattern.pilots, tf.complex64)

    x_p = tf.transpose(pilots[0])
    active_row = tf.reduce_any(tf.abs(x_p) > 0, axis=1)
    x_p = tf.boolean_mask(x_p, active_row)

    no_ants = []
    for ant in range(int(y.shape[0])):
        y_ant = y[ant]
        y_p = tf.boolean_mask(y_ant, pilot_mask)
        y_p = tf.boolean_mask(y_p, active_row)
        y_p = tf.expand_dims(y_p, axis=1)
        h_bar = tf.linalg.lstsq(x_p, y_p)
        resid = y_p - tf.matmul(x_p, h_bar)
        no_ants.append(tf.reduce_mean(tf.square(tf.abs(resid))))

    return tf.cast(tf.reduce_mean(tf.stack(no_ants)), tf.float32)


def _crop_rx_slot_to_allocation(rx_np, pusch_record):
    start_sc = int(pusch_record.rbStart) * 12
    end_sc = int(pusch_record.rbStart + pusch_record.rbSize) * 12
    start_sym = int(pusch_record.StartSymbolIndex)
    num_symbols = int(pusch_record.NrOfSymbols)
    return rx_np[start_sc:end_sc, start_sym:start_sym + num_symbols, :]


def estimate_no_from_datalake_slot(rx_slot, pusch_record, transmitter=None):
    """Estimate N0 from a datalake FH slot.
    """
    if transmitter is None:
        transmitter = create_sionna_transmitter_from_record(pusch_record)

    rx_np = rx_slot.get() if hasattr(rx_slot, 'get') else np.asarray(rx_slot)
    rx_crop = _crop_rx_slot_to_allocation(rx_np, pusch_record)
    _rx_slot = np.transpose(rx_crop, (2, 1, 0))
    _rx_slot = _rx_slot[np.newaxis, np.newaxis, ...]

    return float(estimate_no_from_pilots(tf.constant(_rx_slot, dtype=tf.complex64), transmitter).numpy())


def rx_slot_to_sionna_y(rx_slot, pusch_record):
 
    rx_np = rx_slot.get() if hasattr(rx_slot, 'get') else np.asarray(rx_slot)
    y = _crop_rx_slot_to_allocation(rx_np, pusch_record)
    y = np.transpose(y, (2, 1, 0))  # [num_rx_ant, num_ofdm_symbols, fft_size]

    return y[np.newaxis, np.newaxis, ...].astype(np.complex64)


def duidd_bits_to_array(b_hat, num_tx_idx=0):
    """Extract hard payload bits from DUIDD output."""
    bits = b_hat[0, num_tx_idx].numpy()
    bits = np.round(bits).astype(np.uint8).flatten()
    return bits


def compare_tb_to_pdu(b_hat, tb_input_np, n_bits, num_tx_idx=0):
    """Compare decoded TB bits against pduData
    """

    bits_hat = duidd_bits_to_array(b_hat, num_tx_idx=num_tx_idx)[:n_bits]
    bits_ref = np.unpackbits(np.asarray(tb_input_np).astype(np.uint8)).flatten()[:n_bits]
    num_bit_errors = int(np.sum(bits_hat != bits_ref))
    tb_error = int(num_bit_errors > 0)

    return tb_error, num_bit_errors, bits_hat, bits_ref


def compare_pdu_bytes(decoded_bytes, tb_input_np, n_bits=None):
    """Compare pyAerial decoded PDU bytes against reference pduData.

    Returns ``(tb_error, num_bit_errors, n_bits)``.
    """
    bits_hat = np.unpackbits(np.asarray(decoded_bytes).astype(np.uint8)).flatten()
    bits_ref = np.unpackbits(np.asarray(tb_input_np).astype(np.uint8)).flatten()
    if n_bits is None:
        n_bits = min(bits_hat.size, bits_ref.size)
    bits_hat = bits_hat[:n_bits]
    bits_ref = bits_ref[:n_bits]
    num_bit_errors = int(np.sum(bits_hat != bits_ref))
    return int(num_bit_errors > 0), num_bit_errors, n_bits


def build_channel_estimator(sys_parameters, transmitter, pusch_record):
    """Build the channel estimator used by DUIDD for a per-sample transmitter."""
    if sys_parameters.system == 'lslin_duidd':
        return PUSCHLSChannelEstimatorNoOCC(
            resource_grid=transmitter.resource_grid,
            dmrs_length=1,
            dmrs_additional_position=2,
            num_cdm_groups_without_data=pusch_record.numDmrsCdmGrpsNoData,
            interpolation_type="lin",
        )
    if sys_parameters.system == 'lmmse_duidd':
        interpolator = LMMSEInterpolator(
            transmitter.resource_grid.pilot_pattern,
            cov_mat_time=sys_parameters.time_cov_mat,
            cov_mat_freq=sys_parameters.freq_cov_mat,
            cov_mat_space=sys_parameters.space_cov_mat,
            order="s-f-t",
        )
        return PUSCHLSChannelEstimatorNoOCC(
            resource_grid=transmitter.resource_grid,
            dmrs_length=1,
            dmrs_additional_position=2,
            num_cdm_groups_without_data=pusch_record.numDmrsCdmGrpsNoData,
            interpolator=interpolator,
        )
    raise ValueError(f"Unsupported DUIDD system: {sys_parameters.system}")


def build_duidd_receiver(sys_parameters, transmitter, pusch_record):
    """Build a per-sample DUIDDPUSCHReceiver from FAPI metadata."""
    channel_estimator = build_channel_estimator(sys_parameters, transmitter, pusch_record)
    
    tb_decoder = TBDecoder(
        transmitter._tb_encoder,
        num_bp_iter=sys_parameters.num_bp_iter,
        cn_type=sys_parameters.cn_type,
    )

    return DUIDDPUSCHReceiver(
        transmitter,
        duidd_schedule=sys_parameters.duidd_schedule,
        channel_estimator=channel_estimator,
        mimo_detector=None,
        demepping_type=sys_parameters.demapping_type,
        tb_decoder=tb_decoder,
        return_tb_crc_status=True,
        low_complexity=sys_parameters.low_complexity_mmse_pic,
        training=False,
        weighted_bp=getattr(sys_parameters, "weighted_bp", False),
        sys_parameters=sys_parameters,
    )


def num_layers_for_db(db):
    """Expected number of MIMO layers for a datalake DB selector."""
    return {
        '0': 1, '1': 1, '2': 1,
        '3': 2, '4': 2, '5': 1,
        '6': 2, '7': 1, '8': 2,
        '9': 1, '-1': 2,
        '10': 2, '11': 1,
    }.get(str(db), 1)

def encoded_bits_matching_llr_ch(tb_input_np, transmitter):
    enc = transmitter._tb_encoder
    n_info = int(enc.k)  # target info-bit length

    tb_bits = np.unpackbits(np.asarray(tb_input_np, dtype=np.uint8)).astype(np.float32)
    tb_bits = tb_bits[:n_info]
    if tb_bits.shape[0] < n_info:
        tb_bits = np.pad(tb_bits, (0, n_info - tb_bits.shape[0]))
    
    # Same layout as map_bits_to_resource_grid / TBEncoder expects
    tb_bits = tf.constant(tb_bits[np.newaxis, np.newaxis, :], dtype=tf.float32)  # [1, 1, k]
    coded = transmitter._tb_encoder(tb_bits)  # [1, 1, n]
    return coded


class PuschRxSeparate:
    """pyAerial PUSCH receiver chain (baseline)."""

    def __init__(self, num_rx_ant, enable_pusch_tdi, eq_coeff_algo=1, ch_est_algo=1):
        self.cuda_stream = get_cuda_stream()
        self.ch_est_algo = ch_est_algo
        self.channel_estimator = ChannelEstimator(
            num_rx_ant=num_rx_ant,
            cuda_stream=self.cuda_stream,
            ch_est_algo=ch_est_algo,
        )
        self.channel_equalizer = ChannelEqualizer(
            num_rx_ant=num_rx_ant,
            enable_pusch_tdi=enable_pusch_tdi,
            eq_coeff_algo=eq_coeff_algo,
            cuda_stream=self.cuda_stream,
        )
        self.noise_intf_estimator = NoiseIntfEstimator(
            num_rx_ant=num_rx_ant,
            eq_coeff_algo=eq_coeff_algo,
            cuda_stream=self.cuda_stream,
        )
        self.derate_match = LdpcDeRateMatch(
            enable_scrambling=True,
            cuda_stream=self.cuda_stream,
        )
        self.decoder = LdpcDecoder(cuda_stream=self.cuda_stream)
        self.crc_checker = CrcChecker(cuda_stream=self.cuda_stream)

    def estimate_no(self, rx_slot, slot, pusch_configs):
        """Return linear noise variance N0 from pyAerial (noise_var_pre_eq is dB)."""
        ch_est = self.channel_estimator.estimate(rx_slot=rx_slot, slot=slot, pusch_configs=pusch_configs)
        
        if self.ch_est_algo == 3:
            ch_est[0] = tf.transpose(ch_est[0].get(), perm=[2, 1, 0, 3])
            ch_est[0] = tf.complex(
                tf.image.resize(tf.math.real(ch_est[0]), (1, 3276), 'nearest'),
                tf.image.resize(tf.math.imag(ch_est[0]), (1, 3276), 'nearest'),
            )

        _, noise_var_pre_eq = self.noise_intf_estimator.estimate(rx_slot=rx_slot, channel_est=ch_est, slot=slot, pusch_configs=pusch_configs)
        noise_db = float(noise_var_pre_eq[0].get())
        return 10 ** (noise_db / 10)

    def run(self, rx_slot, slot, pusch_configs, cell_num):
        ch_est = self.channel_estimator.estimate(rx_slot=rx_slot, slot=slot, pusch_configs=pusch_configs)

        if self.ch_est_algo == 3:
            ch_est[0] = tf.transpose(ch_est[0].get(), perm=[2, 1, 0, 3])
            ch_est[0] = tf.complex(
                tf.image.resize(tf.math.real(ch_est[0]), (1, 3276), 'nearest'),
                tf.image.resize(tf.math.imag(ch_est[0]), (1, 3276), 'nearest'),
            )
            
        lw_inv, noise_var_pre_eq = self.noise_intf_estimator.estimate(rx_slot=rx_slot, channel_est=ch_est, slot=slot, pusch_configs=pusch_configs)
        llrs, _ = self.channel_equalizer.equalize(rx_slot=rx_slot, channel_est=ch_est, lw_inv=lw_inv, noise_var_pre_eq=noise_var_pre_eq, pusch_configs=pusch_configs)
        coded_blocks = self.derate_match.derate_match(input_llrs=llrs, pusch_configs=pusch_configs)
        code_blocks = self.decoder.decode( input_llrs=coded_blocks, pusch_configs=pusch_configs)

        return self.crc_checker.check_crc(input_bits=code_blocks, pusch_configs=pusch_configs)


class DuiddRx:
    """DUIDD / IDD receiver for datalake evaluation."""

    def __init__(self, config_name, idd_only=False, weights_path=None):
        cfg = Parameters(config_name, training=False, system='dummy')
        system = cfg.chest + '_duidd'
        self._sys_parameters = Parameters(config_name, training=False, system=system)
        self._idd_only = idd_only
        self._rx_label = "IDD Rx" if idd_only else "DUIDD Rx"

        default_weights = f"../weights/{self._sys_parameters.label}_weights"
        if idd_only:
            self._weights_path = None
            print("IDD-only mode: using untrained paper defaults "f"(alpha/delta/gamma/eta=1, beta/epsilon/mu/xi=0)")

        elif weights_path is not None:
            self._weights_path = weights_path
            if exists(self._weights_path):
                print(f"Loaded DUIDD weights from {self._weights_path}")
            else:
                raise FileNotFoundError(f"Weights not found at {self._weights_path}")
            
        elif exists(default_weights):
            self._weights_path = default_weights
            print(f"Loaded DUIDD weights from {self._weights_path}")

        else:
            self._weights_path = None
            print(f"Warning: weights not found at {default_weights}; running IDD paper defaults")

    def run(self, rx_slot, pusch_record, no=None, debug=None):
        transmitter = create_sionna_transmitter_from_record(pusch_record)
        dmrs_port_set = fapi_dmrs_ports_to_sionna_dmrs_port_set(pusch_record.dmrsPorts)
        num_layers = int(pusch_record.nrOfLayers)

        if len(dmrs_port_set) != num_layers:
            raise ValueError(
                f"nrOfLayers ({num_layers}) != len(dmrs_port_set) "
                f"({len(dmrs_port_set)}); dmrsPorts={pusch_record.dmrsPorts}")


        receiver = build_duidd_receiver(self._sys_parameters, transmitter, pusch_record)
        if self._weights_path is not None:
            load_weights(receiver, self._weights_path)

        y = rx_slot_to_sionna_y(rx_slot, pusch_record)

        if no is None:
            no = estimate_no_from_datalake_slot(rx_slot, pusch_record, transmitter)

        if debug is not None:
            enc_bits = encoded_bits_matching_llr_ch(debug, transmitter)
            b_hat, tb_crc = receiver([y, tf.constant(no, dtype=tf.float32)], bit_grid=enc_bits)
        else:
            b_hat, tb_crc = receiver([y, tf.constant(no, dtype=tf.float32)])
        return b_hat, tb_crc, transmitter, float(no)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DUIDD PUSCH receiver with datalake data")
    parser.add_argument("--config-name", help="config filename", type=str, required=True)
    parser.add_argument('--limit', type=int, default=10,
                        help='Number of database records to process')
    parser.add_argument('--timestamps', type=str, default=None,
                        help='Timestamps pickle for deterministic replay')
    parser.add_argument('--ue', type=int, default=1,
                        help='0 for Samsung Galaxy S23, 1 for iPhone 14 Pro')
    parser.add_argument('--cell-id', type=int, default=51,
                        help='Cell ID (41, 42, or 51)')
    parser.add_argument('--add-noise-snr', type=float, default=None,
                        help='Add AWGN at given SNR (dB) to real captures')
    parser.add_argument('--db', type=str, default='8',
                        help='Datalake DB selector (same as MDX eval script)')
    parser.add_argument('--no-mode', type=str, default='pyaerial',
                        choices=['pyaerial', 'pilot', 'fixed'],
                        help='Noise variance for Sionna/DUIDD: pyAerial estimate (default), '
                             'Sionna DMRS pilot residual, or fixed --fixed-no')
    parser.add_argument('--fixed-no', type=float, default=0.01,
                        help='Fixed N0 when --no-mode=fixed')
    parser.add_argument('--idd-mode', type=str, default='full',
                        choices=['full', 'first-pass'],
                        help="DUIDD eval mode: 'full' runs the IDD schedule from config "
                             "(MMSE-PIC iterations); 'first-pass' decodes after initial "
                             "LMMSE detection only (skips IDD loop)")
    parser.add_argument('--idd-only', action='store_true',
                        help="Evaluate untrained IDD with paper-default hyperparameters "
                             "(do not load finetuned DUIDD weights)")
    parser.add_argument('--weights-path', type=str, default=None,
                        help="Explicit path to DUIDD weight pickle (overrides config label)")
    args = parser.parse_args()

    assert args.ue in [0, 1]

    num_rx_ant = 4
    num_tx_ant = num_layers_for_db(args.db)

    pusch_rx_separate = PuschRxSeparate(
        num_rx_ant=num_rx_ant,
        enable_pusch_tdi=1,
        eq_coeff_algo=1,
        ch_est_algo=1,
    )
    duidd_rx = DuiddRx(
        config_name=args.config_name,
        idd_only=args.idd_only,
        weights_path=args.weights_path)
    rx_label = duidd_rx._rx_label

    print("=" * 80)
    print("REAL DATA MODE (ClickHouse)")
    print("=" * 80)

    client = clickhouse_connect.get_client(host='localhost')

    load_timestamps_mode = False
    loaded_timestamps = None
    timestamp_dir = f"../eval_timestamps/{args.timestamps}" if args.timestamps else None
    if args.timestamps and exists(timestamp_dir):
        with open(timestamp_dir, 'rb') as f:
            loaded_timestamps = pickle.load(f)
        load_timestamps_mode = True
        print(f"   Loaded {len(loaded_timestamps)} timestamps from {timestamp_dir}")

    ue = args.ue
    match args.db:
        case '0':
            database = "nrx_measurement_2025_11_03"
            split_times = ['2025-11-03 20:48:00.000001', '2025-11-03 20:54:58.960007']
            split_time = split_times[ue]
            num_layers = 1
        case '1':
            database = "nrx_jfloor_1st_2025_12_02"
            split_times = ['2025-12-02 21:54:35.000000000', '2025-12-02 21:59:03.234500000']
            split_time = split_times[ue]
            num_layers = 1
        case '2':
            database = "nrx_drone_2025_12_03"
            split_time = "2025-12-03 12:21:05.202000000"
            num_layers = 1
        case '3':
            database = 'quectel_2ULLs_j61_2026_04_27'
            num_layers = 2
            split_time = "2026-04-26 12:48:30.000000000"
        case '4':
            database = 'quectel_2ULLs_13dB_j61_2026_04_28'
            num_layers = 2
            split_time = "2026-04-28 12:48:30.000000000"
        case '5':
            database = 'quectel_1ULLs_7dB_j61_2026_04_28'
            num_layers = 1
            split_time = "2026-04-28 12:48:30.000000000"
        case '6':
            database = 'quectel_2ULLs_13dB_j61_2026_05_12'
            num_layers = 2
            split_time = '2026-05-12 11:53:30'
        case '7':
            database = 'quectel_1ULLs_7dB_j61_2026_05_12'
            num_layers = 1
            split_time = '2026-05-12 12:22:00'
        case '-1':
            database = 'quectel_2ULLs_28dB_jFloor_2026_06_02'
            num_layers = 2
            split_time = '2026-06-02 01:00:00'
        case '8':
            database = 'Pixel9Pro_2ULLs_12dB_j61_2026_06_04'
            num_layers = 2
            split_time = '2026-06-04 13:22:17.372000000'
        case '9':
            database = 'sgs23_1ULLs_5dB_j61_2026_06_04'
            num_layers = 1
            split_time = '2026-06-04 12:59:00.372000000'
        case '10':
            database = 'Pixel9Pro_2ULLs_12dB_jFloorRealLoop_2026_06_10'
            num_layers = 2
            split_time = '2026-06-10 14:47:00'
        case '11':
            database = 'sgs23_1ULLs_5dB_jFloorRealLoop_2026_06_10'
            num_layers = 1
            split_time = '2026-06-10 15:04:00'
        case _:
            raise ValueError(f"Unknown db selector: {args.db}")

    limit = args.limit

    # and mcsIndex = 10
    # and Slot = 14
    # and rnti = 9304

    if load_timestamps_mode:
        ts_strings = [
            f"toDateTime64('{ts.strftime('%Y-%m-%d %H:%M:%S.%f')}', 9)"
            for ts in loaded_timestamps]
        ts_in_clause = ", ".join(ts_strings)
        query = f"""select * from export_fapi_{database}
                    where TsTaiNs IN ({ts_in_clause})
                    limit {limit}
                    """
    else:
        query = f"""select * from (
                    (select * from export_fapi_{database}
                        where rbSize = 273
                        and mcsIndex > 5
                        and qamModOrder = 4
                        and nrOfLayers = {num_layers}
                        and NrOfSymbols = 13
                        and rvIndex = 0
                        and tbCrcFail = 1
                        and TsTaiNs > toDateTime64('{split_time}', 9)
                        order by rand()
                        limit CEIL({limit} * 0.1)
                    )
                    union all
                    (
                        select * from export_fapi_{database}
                        where rbSize = 273
                        and mcsIndex > 5
                        and qamModOrder = 4
                        and nrOfLayers = {num_layers}
                        and NrOfSymbols = 13
                        and rvIndex = 0
                        and tbCrcFail = 0
                        and TsTaiNs > toDateTime64('{split_time}', 9)
                        order by rand()
                        limit CEIL({limit} * 0.9)
                    )
                ) as combined
                order by rand()
                """

    pusch_records = client.query_df(query)
    print(f"   Found {len(pusch_records)} records to process")

    print("-" * 80)
    print(f"{'Sample':<8} {'TB CRC':<10} {'TB Size':<10} {'PUSCH Rx':<12} {rx_label:<12}")
    print("-" * 80)

    num_samples = 0
    num_tb_errors = defaultdict(int)
    num_bit_errors = defaultdict(int)
    num_bits_compared = defaultdict(int)
    num_skipped = 0
    num_retransmissions = 0
    num_crc_pass = 0
    saved_timestamps = []
    no_values = []

    for _, pusch_record in tqdm(pusch_records.iterrows(), total=len(pusch_records), desc="Processing"):
        query_fh = f"""select TsTaiNs, CellId, fhData from export_fh_{database}
                    where TsTaiNs == toDateTime64('{pusch_record.TsTaiNs.timestamp()}', 9)
                    """
        fh = client.query_df(query_fh)
        if fh.index.size != 1:
            num_skipped += 1
            continue

        if pusch_record.tbCrcFail == 0:
            if pusch_record.pduData is None:
                num_skipped += 1
                continue
            tb_input_np = np.array(pusch_record.pduData)
        else:
            tb_input_np, _ = find_successful_retransmission(client, database, pusch_record)
            num_retransmissions += 1
            if tb_input_np is None:
                num_skipped += 1
                continue

        fh_samp = (np.array(fh['fhData'].iloc[0], dtype=np.int16).view(np.float16)).astype(np.float32)
        rx_slot = np.swapaxes(
            fh_samp.view(np.complex64).reshape(4, 14, 273 * 12), 2, 0)

        if args.add_noise_snr is not None:
            signal_power = np.mean(np.abs(rx_slot) ** 2)
            noise_power = signal_power / (10 ** (args.add_noise_snr / 10))
            noise = np.sqrt(noise_power / 2) * (np.random.randn(*rx_slot.shape) + 1j * np.random.randn(*rx_slot.shape))
            rx_slot = rx_slot + noise.astype(np.complex64)

        rx_tensor = cp.array(rx_slot, dtype=cp.complex64)
        pusch_configs = create_pusch_config_from_record(pusch_record)
        slot_number = int(pusch_record.Slot)

        tbs_pusch, _ = pusch_rx_separate.run(
            rx_slot=rx_tensor,
            slot=slot_number,
            pusch_configs=pusch_configs,
            cell_num=pusch_record.CellId,
        )

        pusch_rx_error, pusch_bit_errors, n_bits_pusch = compare_pdu_bytes(tbs_pusch[0].get(), tb_input_np)
        num_tb_errors["PUSCH Rx"] += pusch_rx_error
        num_bit_errors["PUSCH Rx"] += pusch_bit_errors
        num_bits_compared["PUSCH Rx"] += n_bits_pusch

        if args.no_mode == 'fixed':
            no_est = args.fixed_no
        elif args.no_mode == 'pyaerial':
            no_est = pusch_rx_separate.estimate_no(
                rx_slot=rx_tensor, slot=slot_number, pusch_configs=pusch_configs)
        else:
            no_est = estimate_no_from_datalake_slot(rx_tensor, pusch_record)
            
        b_hat, tb_crc, transmitter, no_used = duidd_rx.run(
            rx_slot=rx_tensor, 
            pusch_record=pusch_record, 
            no=no_est,
            debug=tb_input_np)

        n_bits = int(transmitter._tb_size)
        duidd_error, duidd_bit_errors, _, _ = compare_tb_to_pdu(b_hat, tb_input_np, n_bits)
        num_crc_pass += int(bool(tb_crc.numpy()[0, 0]))
        no_values.append(no_used)

        num_tb_errors[rx_label] += duidd_error
        num_bit_errors[rx_label] += duidd_bit_errors
        num_bits_compared[rx_label] += n_bits
        num_samples += 1

        if args.timestamps and not load_timestamps_mode:
            saved_timestamps.append(pusch_record.TsTaiNs)

    if args.timestamps and not load_timestamps_mode and saved_timestamps:
        with open(timestamp_dir, 'wb') as f:
            pickle.dump(saved_timestamps, f)
        print(f"\nSaved {len(saved_timestamps)} timestamps to {timestamp_dir}")

    results_lines = [
        "-" * 80,
        "\nFinal Results:",
        f"   Mode: REAL DATA (ClickHouse)",
        f"   Config: {args.config_name}",
        f"   Database: {database}",
        f"   Timestamps: {args.timestamps if args.timestamps else 'None'}",
        f"   Total samples processed: {num_samples}",
        f"   Samples skipped: {num_skipped}",
        f"   HARQ Retransmissions: {num_retransmissions}",
        f"   No mode: {args.no_mode}",
        f"   Receiver: {rx_label}",
        f"   IDD-only (paper defaults): {args.idd_only}",
        "",
        "   PUSCH Rx:",
        f"      TB Errors: {num_tb_errors['PUSCH Rx']}",
    ]
    if num_samples > 0:
        results_lines.append(
            f"      BLER: {num_tb_errors['PUSCH Rx'] / num_samples * 100:.2f}%")
        if num_bits_compared["PUSCH Rx"] > 0:
            results_lines.append(
                f"      BER:  {num_bit_errors['PUSCH Rx'] / num_bits_compared['PUSCH Rx']:.6e}"
                f"  ({num_bit_errors['PUSCH Rx']}/{num_bits_compared['PUSCH Rx']} bits)")
    results_lines += [
        "",
        f"   {rx_label}:",
        f"      TB Errors: {num_tb_errors[rx_label]}",
    ]
    if num_samples > 0:
        results_lines.append(
            f"      BLER: {num_tb_errors[rx_label] / num_samples * 100:.2f}%")
        if num_bits_compared[rx_label] > 0:
            results_lines.append(
                f"      BER:  {num_bit_errors[rx_label] / num_bits_compared[rx_label]:.6e}"
                f"  ({num_bit_errors[rx_label]}/{num_bits_compared[rx_label]} bits)")
        results_lines.append(
            f"      TB CRC pass rate: {num_crc_pass / num_samples * 100:.2f}%")
        if no_values:
            results_lines.append(
                f"      N0 (mean/min/max): {np.mean(no_values):.4g} / "
                f"{np.min(no_values):.4g} / {np.max(no_values):.4g}")
    results_lines += ["=" * 80, "Evaluation complete!", "=" * 80]

    for line in results_lines:
        print(line)

    with open("results_duidd_datalake.txt", "a") as f:
        f.write("\n".join(results_lines) + "\n\n")


if __name__ == "__main__":
    main()
