#!/usr/bin/env python3
"""
Evaluate a PUSCH neural receiver using real data from Aerial Data Lake.

This script evaluates a trained neural network-based PUSCH receiver
using real world IQ data from the ClickHouse database.

Author: Nuri Berke Baytekin
"""

import os
from xml.parsers.expat import errors
import tensorflow as tf
# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Silence TensorFlow.
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import argparse
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import onnxruntime as ort
import pickle
import clickhouse_connect
from tqdm import tqdm
from collections import defaultdict

from aerial.phy5g.pusch import PuschRx
from aerial.phy5g.pdsch import PdschTx
from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.algorithms import TrtEngine
from aerial.phy5g.algorithms import TrtTensorPrms
from aerial.phy5g.algorithms import ChannelEqualizer
from aerial.phy5g.algorithms import NoiseIntfEstimator
from aerial.phy5g.ldpc import get_mcs
from aerial.phy5g.ldpc import LdpcDeRateMatch
from aerial.phy5g.ldpc import LdpcDecoder
from aerial.phy5g.ldpc import CrcChecker
from aerial.phy5g.ldpc import random_tb
from aerial.phy5g.ldpc import get_tb_size
from aerial.pycuphy.types import PuschLdpcKernelLaunch
from aerial.phy5g.config import PuschConfig
from aerial.phy5g.config import PuschUeConfig
from aerial.util.cuda import get_cuda_stream
from aerial.util.fapi import dmrs_fapi_to_bit_array

from sionna.nr import PUSCHConfig, PUSCHDMRSConfig, TBConfig, CarrierConfig, PUSCHTransmitter
from sionna.ofdm import LSChannelEstimator

from sionna.channel.tr38901 import PanelArray, UMi, TDL, UMa
from sionna.channel import gen_single_sector_topology



# from utils import E2E_Model, training_loop, Parameters, load_weights


# Sionna imports (for synthetic data generation)
try:
    import sionna
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False

# Hide log10(10) warning
_ = np.seterr(divide='ignore', invalid='ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



def setup_synthetic_channel(num_rx_ant, num_tx_ant, carrier_frequency=2.14e9, 
                           channel_model="Rayleigh", delay_spread=100e-9, speed=0.8333):
    """Set up Sionna channel for synthetic data generation.
    """
    
    # Numerology and frame structure
    num_ofdm_symbols = 14
    fft_size = 4096
    cyclic_prefix_length = 288
    subcarrier_spacing = 30e3
    num_guard_subcarriers = (410, 410)
    num_antenna_ports = 2
    
    # Define the resource grid
    resource_grid = sionna.phy.ofdm.ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=fft_size,
        subcarrier_spacing=subcarrier_spacing,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=cyclic_prefix_length,
        num_guard_carriers=num_guard_subcarriers,
        dc_null=False,
        pilot_pattern=None,
        pilot_ofdm_symbol_indices=None
    )
    resource_grid_mapper = sionna.phy.ofdm.ResourceGridMapper(resource_grid)
    remove_guard_subcarriers = sionna.phy.ofdm.RemoveNulledSubcarriers(resource_grid)

    max_ut_velocity = 0.
    min_ut_velocity = 0.
    
    # Define the antenna arrays
    if channel_model == "Rayleigh":
        ch_model = sionna.phy.channel.RayleighBlockFading(
            num_rx=1,
            num_rx_ant=num_rx_ant,
            num_tx=1,
            num_tx_ant=num_tx_ant
        )
    elif "CDL" in channel_model:
        cdl_model = channel_model[-1]
        ue_array = sionna.phy.channel.tr38901.Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency
        )
        gnb_array = sionna.phy.channel.tr38901.AntennaArray(
            num_rows=1,
            num_cols=int(num_rx_ant/2),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency
        )
        ch_model = sionna.phy.channel.tr38901.CDL(
            cdl_model,
            delay_spread,
            carrier_frequency,
            ue_array,
            gnb_array,
            "uplink",
            min_speed=speed
        )
    elif channel_model == "UMi":

        ch_type = "umi"
        num_cols_per_panel = num_rx_ant//2
        num_rows_per_panel = 1
        polarization = "dual"
        polarization_type = 'cross'

        bs_array = PanelArray(num_rows_per_panel = num_rows_per_panel,
                                  num_cols_per_panel = num_cols_per_panel,
                                  polarization = polarization,
                                  polarization_type  = polarization_type,
                                  antenna_pattern = '38.901',
                                  carrier_frequency = carrier_frequency)

        ut_array = PanelArray(num_rows_per_panel = 1,
                                  num_cols_per_panel = num_antenna_ports,
                                  polarization = 'single',
                                  polarization_type = 'V',
                                  antenna_pattern = 'omni',
                                  carrier_frequency = carrier_frequency)
        
        ch_model = UMi(carrier_frequency=carrier_frequency,
                        o2i_model = 'low',
                        bs_array = bs_array,
                        ut_array = ut_array,
                        direction = 'uplink',
                        enable_pathloss = False,
                        enable_shadow_fading = False)
        
        batch_size = 1
        max_num_tx = 1
        
        topology = gen_single_sector_topology(
                            batch_size,
                            max_num_tx,
                            ch_type,
                            min_ut_velocity=min_ut_velocity,
                            max_ut_velocity=max_ut_velocity,
                            indoor_probability=0.) # disable indoor users
        ch_model.set_topology(*topology)

    elif channel_model == "TDL-B100":
        ch_model = TDL(model="B100",
                    delay_spread=100e-9,
                    carrier_frequency=carrier_frequency,
                    min_speed=min_ut_velocity,
                    max_speed=max_ut_velocity,
                    num_tx_ant=num_antenna_ports,
                    num_rx_ant=num_rx_ant)
    else:
        raise ValueError(f"Invalid channel model {channel_model}!")
    
    channel = sionna.phy.channel.OFDMChannel(
        ch_model,
        resource_grid,
        add_awgn=True,
        normalize_channel=True,
        return_channel=False
    )
    
    def apply_channel(tx_tensor, No):
        """Transmit the Tx tensor through the radio channel."""
        # Add batch and num_tx dimensions that Sionna expects and reshape
        tx_tensor = tf.transpose(tx_tensor, (2, 1, 0))
        tx_tensor = tf.reshape(tx_tensor, (1, -1))[None, None]
        tx_tensor = resource_grid_mapper(tx_tensor)
        rx_tensor = channel(tx_tensor, No)
        rx_tensor = remove_guard_subcarriers(rx_tensor)
        rx_tensor = rx_tensor[0, 0]
        rx_tensor = tf.transpose(rx_tensor, (2, 1, 0))
        return rx_tensor
    
    return channel, apply_channel


def generate_synthetic_samples(num_samples, esno_db_range, pusch_tx, apply_channel_fn, 
                               pusch_configs, num_slots_per_frame=20):
    """Generate synthetic PUSCH samples for evaluation.
    """
    pusch_config = pusch_configs[0]
    ue_config = pusch_config.ue_configs[0]
    
    # Get modulation and coding parameters
    mod_order = ue_config.mod_order
    code_rate = ue_config.code_rate / 10  # Convert back from scaled value
    
    for esno_db in esno_db_range:
        for sample_idx in range(num_samples):
            slot_number = sample_idx % num_slots_per_frame
            
            # Generate random transport block
            tb_input_np = random_tb(
                mod_order=mod_order,
                code_rate=code_rate,
                dmrs_syms=pusch_config.dmrs_syms,
                num_prbs=pusch_config.num_prbs,
                start_sym=pusch_config.start_sym,
                num_symbols=pusch_config.num_symbols,
                num_layers=ue_config.layers
            )
            tb_input = cp.array(tb_input_np, dtype=cp.uint8, order='F')
            
            # Transmit PUSCH (emulated with PDSCH)
            tx_tensor = pusch_tx.run(
                tb_inputs=[tb_input],
                num_ues=1,
                slot=slot_number,
                num_dmrs_cdm_grps_no_data=pusch_config.num_dmrs_cdm_grps_no_data,
                dmrs_scrm_id=pusch_config.dmrs_scrm_id,
                start_prb=pusch_config.start_prb,
                num_prbs=pusch_config.num_prbs,
                dmrs_syms=pusch_config.dmrs_syms,
                start_sym=pusch_config.start_sym,
                num_symbols=pusch_config.num_symbols,
                scids=[ue_config.scid],
                layers=[ue_config.layers],
                dmrs_ports=[ue_config.dmrs_ports],
                rntis=[ue_config.rnti],
                data_scids=[ue_config.data_scid],
                code_rates=[ue_config.code_rate],
                mod_orders=[mod_order]
            )
            
            # Apply channel
            tx_tensor = tf.experimental.dlpack.from_dlpack(tx_tensor.toDlpack())
            No = pow(10., -esno_db / 10.)
            rx_tensor = apply_channel_fn(tx_tensor, No)
            rx_tensor = tf.experimental.dlpack.to_dlpack(rx_tensor)
            rx_tensor = cp.from_dlpack(rx_tensor)
            
            yield rx_tensor, slot_number, tb_input_np, esno_db


def find_successful_retransmission(client, database, current_record, max_lookahead=10):
    """Find the next successful retransmission with the same HARQ process ID.
    
    HARQ process IDs cycle from 0-15. When a transmission fails, it gets retransmitted
    with the same process ID. This function searches for retransmissions following the
    RV sequence: 2 - 3 - 1. If all fail, returns None.
    
    Args:
        client: ClickHouse client connection
        current_record: Current PUSCH record with tbCrcFail = 1
        max_lookahead: Maximum number of subsequent transmissions to check
        
    Returns:
        Tuple of (pduData, retransmission_TsTaiNs) from successful retransmission, 
        or (None, None) if not found
    """
    harq_process_id = current_record.harqProcessID
    rnti = current_record.rnti
    current_ts = current_record.TsTaiNs
    cell_id = current_record.CellId
    
    query = f"""select TsTaiNs, pduData, tbCrcFail, harqProcessID, rnti, rvIndex 
                from export_fapi_{database}
                where rnti = {rnti}
                and CellId = {cell_id}
                and TsTaiNs > toDateTime64('{current_ts.timestamp()}', 9)
                and NrOfSymbols = 13
                order by TsTaiNs
                limit {max_lookahead}"""
    
    subsequent_transmissions = client.query_df(query)
    
    if subsequent_transmissions.empty:
        return None, None
    
    seen_process_ids = {harq_process_id}
    
    rv_sequence = [2, 3, 1]
    req_rv = rv_sequence[0]  # Start with RV=2
    rv_index = 0
    
    # Look for retransmissions with matching HARQ process ID and required RV
    for _, tx in subsequent_transmissions.iterrows():
        tx_process_id = tx.harqProcessID
        
   
        if tx_process_id == harq_process_id:
            if len(seen_process_ids) == 16:
                return None, None
            
            if tx.rvIndex == req_rv:
                if tx.tbCrcFail == 0 and tx.pduData is not None:
                    return np.array(tx.pduData), tx.TsTaiNs
                else:
                    rv_index += 1
                    if rv_index >= len(rv_sequence):
                        return None, None
                    req_rv = rv_sequence[rv_index]
        
        # Track seen process IDs (after checking for match)
        seen_process_ids.add(tx_process_id)
    
    return None, None


def create_pusch_config_from_record(pusch_record):
    """Create PUSCHConfig and PUSCHUeConfig from a database record.
    
    Args:
        pusch_record: A pandas Series containing a single PUSCH record from the database
        
    Returns:
        List containing a single PUSCHConfig object
    """
    
    tb_size = pusch_record.TBSize
    
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
        tb_size=pusch_record.TBSize
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
        num_symbols=13 # pusch_record.NrOfSymbols
    )
    
    return [pusch_config]

def fapi_dmrs_ports_to_sionna_dmrs_port_set(dmrs_ports_bmsk: int) -> list[int]:
    """Convert FAPI dmrsPorts bitmap to Sionna dmrs_port_set indices.

    Examples:
      dmrsPorts=1 (0b1)  -> [0]
      dmrsPorts=3 (0b11) -> [0, 1]
    """
    if dmrs_ports_bmsk is None:
        return [0]
    b = int(dmrs_ports_bmsk)
    port_set = [i for i in range(12) if (b >> i) & 0x1]
    return port_set if port_set else [0]

def create_sionna_transmitter_from_record(pusch_record):
    cyclic_prefix = 'normal' if pusch_record.CyclicPrefix == 0 else 'extended'
    config_type = 1 if pusch_record.dmrsConfigType == 0 else 2 # Careful with Sionna
    dmrs_port_set = fapi_dmrs_ports_to_sionna_dmrs_port_set(pusch_record.dmrsPorts)


    carrier_config = CarrierConfig(
                n_cell_id=pusch_record.CellId, # same as DMRS scrambling ID
                cyclic_prefix=cyclic_prefix,
                subcarrier_spacing=int(30.0), # in kHz
                n_size_grid=pusch_record.rbSize,
                n_start_grid=pusch_record.rbStart,
                slot_number=int(pusch_record.Slot),
                frame_number=pusch_record.SFN)

    pusch_dmrs_config = PUSCHDMRSConfig(
                        config_type=config_type,
                        additional_position=2,
                        length=1,
                        n_id = pusch_record.ulDmrsScramblingId, # defaults to n_cell_id
                        dmrs_port_set=dmrs_port_set,
                        n_scid=pusch_record.SCID, #scid = 0
                        num_cdm_groups_without_data=pusch_record.numDmrsCdmGrpsNoData)

    tb_config = TBConfig(
                mcs_index=pusch_record.mcsIndex,
                mcs_table=(pusch_record.mcsTable + 1), # Careful with Sionna
                channel_type="PUSCH")

    pc = PUSCHConfig(
        carrier_config=carrier_config,
        pusch_dmrs_config=pusch_dmrs_config,
        tb_config=tb_config,
        num_antenna_ports=pusch_record.nrOfLayers,
        num_layers=pusch_record.nrOfLayers,
        symbol_allocation=[pusch_record.StartSymbolIndex, pusch_record.NrOfSymbols],
        mapping_type='B',
        data_scid=pusch_record.dataScramblingId, # same as n_cell_id
        n_rnti=pusch_record.rnti,
        )
    
    transmitter = PUSCHTransmitter(pc, return_bits=False, output_domain="freq")

    return transmitter

def map_bits_to_resource_grid(tb_input, transmitter):
    """Map transport block bits to resource grid shape matching rx_tensor."""
    
    tb_encoder = transmitter._tb_encoder
    layer_mapper = transmitter._layer_mapper
    resource_grid_mapper = transmitter._resource_grid_mapper
    mapper = transmitter._mapper

    mod_order = transmitter._num_bits_per_symbol

    tb_bits = np.unpackbits(tb_input)[None, :]
    tb_bits = tf.expand_dims(tb_bits, axis=0)
    
    
    tb_encoded_bits = tb_encoder(tb_bits)
    
    # Reshape to symbol bits (each symbol has mod_order bits)
    n_bits = tf.shape(tb_encoded_bits)[-1]
    n_symbols = n_bits // mod_order

    new_shape = tf.concat([tf.shape(tb_encoded_bits)[:-1], [n_symbols, mod_order]], axis=0)
    symbol_bits = tf.reshape(tb_encoded_bits, new_shape)
    weights = tf.pow(2, tf.range(mod_order - 1, -1, -1, dtype=tf.float32)) # for 16QAM convert 4 bits to int 0-15
    symbol_numbers = tf.reduce_sum(symbol_bits * weights, axis=-1)
    symbol_numbers = tf.cast(symbol_numbers, tf.complex64)
    
    layer_mapped_bits = layer_mapper(symbol_numbers)
    
    symbol_number_grid = resource_grid_mapper(layer_mapped_bits)

    symbol_number_grid = tf.cast(symbol_number_grid, tf.int32)

    # Convert symbol numbers to bits in the last dimension
    bit_grid = tf.bitwise.right_shift(tf.expand_dims(symbol_number_grid, axis=-1), tf.range(mod_order - 1, -1, -1, dtype=tf.int32)) & 1
    bit_grid = tf.cast(bit_grid, tf.float32)
    bit_grid = tf.squeeze(bit_grid, axis=1)

    # mask pilots
    pilot_position = transmitter._resource_grid.build_type_grid()[:,0] == 1
    pilot_position = tf.cast(pilot_position, tf.float32)
    pilot_position = tf.expand_dims(pilot_position, axis=-1)
    pilot_position = tf.broadcast_to(pilot_position, tf.shape(bit_grid))
    bit_grid = tf.where(pilot_position==1, tf.constant(-1., bit_grid.dtype), bit_grid)
    
    return bit_grid


# Default tensor geometry for 273 PRB, 13 data symbols, 3 DMRS OFDM symbols.
_NRX_NUM_SUBCARRIERS = 3276
_NRX_NUM_SYMBOLS = 13
_NRX_NUM_PILOTS = 4914
_NRX_NUM_DMRS_SYMBOLS = 3
_NRX_NUM_DMRS_SUBCARRIERS = 6
_NRX_NUM_BITS_PER_SYMBOL = 8


def _dmrs_trt_tensor_dims(num_layers, num_dmrs_symbols, num_dmrs_subcarriers):
    if num_layers == 1:
        return (num_layers,), (num_dmrs_symbols,), (num_dmrs_subcarriers,)
    return (
        (num_layers,),
        (num_layers, num_dmrs_symbols),
        (num_layers, num_dmrs_subcarriers),
    )


def pack_nrx_dmrs_input_tensors(dmrs_syms, num_layers):

    dmrs_ofdm_pos_1 = np.where(np.array(dmrs_syms))[0].astype(np.int32)
    dmrs_subcarrier_pos_1 = np.array([0, 2, 4, 6, 8, 10], dtype=np.int32)
    active_dmrs_ports = np.ones((1, num_layers), dtype=np.float32)

    if num_layers == 1:
        dmrs_ofdm_pos = dmrs_ofdm_pos_1[None, ...]
        dmrs_subcarrier_pos = dmrs_subcarrier_pos_1[None, ...]
    else:
        dmrs_ofdm_pos = np.tile(dmrs_ofdm_pos_1[None, :], (num_layers, 1))[None, ...]
        dmrs_subcarrier_pos = np.tile(
            dmrs_subcarrier_pos_1[None, :], (num_layers, 1)
        )[None, ...]

    return active_dmrs_ports, dmrs_ofdm_pos, dmrs_subcarrier_pos


def build_nrx_trt_tensor_params(
    num_rx_ant,
    num_layers,
    num_subcarriers=_NRX_NUM_SUBCARRIERS,
    num_symbols=_NRX_NUM_SYMBOLS,
    num_pilots=_NRX_NUM_PILOTS,
    num_dmrs_symbols=_NRX_NUM_DMRS_SYMBOLS,
    num_dmrs_subcarriers=_NRX_NUM_DMRS_SUBCARRIERS,
    num_bits_per_symbol=_NRX_NUM_BITS_PER_SYMBOL,
):

    active_dims, dmrs_ofdm_dims, dmrs_subcarrier_dims = _dmrs_trt_tensor_dims(num_layers, num_dmrs_symbols, num_dmrs_subcarriers)

    return {
        "input_tensors": [
            TrtTensorPrms('rx_slot_real', (num_subcarriers, num_symbols, num_rx_ant), np.float32),
            TrtTensorPrms('rx_slot_imag', (num_subcarriers, num_symbols, num_rx_ant), np.float32),
            TrtTensorPrms('h_hat_real', (num_pilots, num_layers, num_rx_ant), np.float32),
            TrtTensorPrms('h_hat_imag', (num_pilots, num_layers, num_rx_ant), np.float32),
            TrtTensorPrms('active_dmrs_ports', active_dims, np.float32),
            TrtTensorPrms('dmrs_ofdm_pos', dmrs_ofdm_dims, np.int32),
            TrtTensorPrms('dmrs_subcarrier_pos', dmrs_subcarrier_dims, np.int32),
        ],
        "output_tensors": [
            TrtTensorPrms('output_1', (num_bits_per_symbol, 1, num_subcarriers, num_symbols), np.float32),
            TrtTensorPrms('output_2', (1, num_subcarriers, num_symbols, num_bits_per_symbol), np.float32),
        ],
    }


def num_layers_for_db(db):
    return {
        '0': 1, '1': 1, '2': 1,
        '3': 2, '4': 2, '5': 1, '6': 2,
        '7': 1, '8': 2, '9': 1, '10': 2,
        '11': 1
    }.get(str(db), 1)


def _pusch_allocation_dmrs_mask(pusch_config):
    return np.array(pusch_config.dmrs_syms[pusch_config.start_sym:pusch_config.start_sym + pusch_config.num_symbols],dtype=np.int8) == 1



def pusch_ldpc_rate_match_length_bits(pusch_config):
    """Number of LLRs (bits) expected by LDPC for this PUSCH (one codeword / UE[0]).
    """
    dmrs_sym = _pusch_allocation_dmrs_mask(pusch_config)
    n_non_dmrs = int(np.sum(~dmrs_sym))
    re_per_layer = (n_non_dmrs * 12) * pusch_config.num_prbs
    ue = pusch_config.ue_configs[0]
    return int(re_per_layer * ue.mod_order * ue.layers)


def ldpc_derate_match_explicit_kwargs(pusch_config, rate_match_lengths):
    ue = pusch_config.ue_configs[0]
    return dict(
        pusch_configs=None,
        tb_sizes=[ue.tb_size * 8],
        code_rates=[ue.code_rate / 10240.0],
        rate_match_lengths=list(rate_match_lengths),
        mod_orders=[ue.mod_order],
        num_layers=[ue.layers],
        redundancy_versions=[ue.rv],
        ndis=[ue.ndi],
        cinits=[(ue.rnti << 15) + ue.data_scid],
        ue_grp_idx=[0],
    )



class NeuralRx:
    """PUSCH neural receiver class.
    
    This class encapsulates the PUSCH neural receiver chain built using
    pyAerial components.
    """

    def __init__(self, num_rx_ant, trt_model_file, num_layers=1):
        """Initialize the neural receiver.

        Args:
            num_rx_ant: Number of receiver antennas.
            trt_model_file: Path to the TensorRT engine file.
            num_layers: Number of MIMO layers (must match the TRT model and data).
        """
        self.cuda_stream = get_cuda_stream()
        self.num_rx_ant = num_rx_ant
        self.num_layers = int(num_layers)
        self.ch_est_algo = 3

        # Build the components of the receiver. The channel estimator outputs just the LS
        # channel estimates.
        self.channel_estimator = ChannelEstimator(
            num_rx_ant=num_rx_ant,
            ch_est_algo=self.ch_est_algo,
            cuda_stream=self.cuda_stream
        )

        trt_tensors = build_nrx_trt_tensor_params(num_rx_ant, self.num_layers)

        # Create the pyAerial TRT engine object
        self.trt_engine = TrtEngine(
            trt_model_file=trt_model_file,
            max_batch_size=1,
            input_tensors=trt_tensors["input_tensors"],
            output_tensors=trt_tensors["output_tensors"],
        )

        # LDPC (de)rate matching and decoding
        self.derate_match = LdpcDeRateMatch(
            enable_scrambling=True,
            cuda_stream=self.cuda_stream
        )
        self.decoder = LdpcDecoder(cuda_stream=self.cuda_stream)
        self.crc_checker = CrcChecker(cuda_stream=self.cuda_stream)



    
    def run(self, rx_slot, slot, pusch_configs, debug_info=None):
    

        # Channel estimation shape (1638, 1, 4, 3) for ch_est_algo=3
        # (4, 1, 3276, 3) for ch_est_algo=1
        ch_est = self.channel_estimator.estimate(
            rx_slot=rx_slot,
            slot=slot,
            pusch_configs=pusch_configs
        )

        if self.ch_est_algo == 1:
            # Reshape channel estimate to match ch_est_algo=3 output
            # remove interpolation by removing every second subcarrier
            ch_est[0] = ch_est[0][:, :, ::2, :].transpose((2, 1, 0, 3))

        # y = tf.transpose(rx_slot[:,:-1,:].get(), perm=[2, 1, 0])
        # y = y[None, None, ...]
        # h_hat, _ = self.sionna_chest(y, -1.0)
        # h = tf.squeeze(h_hat, axis=(1, 3, 4))
        # h = tf.transpose(h, perm=[3, 0, 1, 2])
        # h = h[::2,:,:,:]
        # h = tf.gather(h, [0, 5, 10,], axis=3)

        # Neural receiver part - outputs LLRs for all symbols
        num_layers = int(pusch_configs[0].ue_configs[0].layers)
        if num_layers != self.num_layers:
            raise ValueError(
                f"PUSCH config has {num_layers} layer(s) but NeuralRx was initialized "
                f"for {self.num_layers} layer(s). Recreate NeuralRx with matching "
                f"num_layers and TRT model."
            )
        active_dmrs_ports, dmrs_ofdm_pos, dmrs_subcarrier_pos = pack_nrx_dmrs_input_tensors(
            pusch_configs[0].dmrs_syms, num_layers
        )
        rx_slot_in = rx_slot[None, :, pusch_configs[0].start_sym:pusch_configs[0].start_sym+pusch_configs[0].num_symbols, :]
        # rx_slot_in = rx_slot[None, :, pusch_configs[0].start_sym:pusch_configs[0].start_sym+14, :]
        ch_est_in = np.transpose(ch_est[0], (3, 0, 1, 2)).reshape(ch_est[0].shape[0] * ch_est[0].shape[3], ch_est[0].shape[1], ch_est[0].shape[2])
        ch_est_in = ch_est_in[None, ...]
        
        
        input_tensors = {
            "rx_slot_real": np.real(rx_slot_in).astype(np.float32),
            "rx_slot_imag": np.imag(rx_slot_in).astype(np.float32),
            "h_hat_real": np.real(ch_est_in).astype(np.float32),
            "h_hat_imag": np.imag(ch_est_in).astype(np.float32),
            "active_dmrs_ports": active_dmrs_ports.astype(np.float32),
            "dmrs_ofdm_pos": dmrs_ofdm_pos.astype(np.int32),
            "dmrs_subcarrier_pos": dmrs_subcarrier_pos.astype(np.int32)
        }
        outputs = self.trt_engine.run(input_tensors)
        raw_llr = outputs["output_1"][0, ...]

        data_syms = np.array(pusch_configs[0].dmrs_syms[pusch_configs[0].start_sym:pusch_configs[0].start_sym+ pusch_configs[0].num_symbols]) == 0
        llrs = np.take(raw_llr, np.where(data_syms)[0], axis=3)
        coded_blocks = self.derate_match.derate_match(
            input_llrs=[llrs],
            pusch_configs=pusch_configs,
        )
        # print(llrs.shape)
        # debugging BER
        if debug_info is not None:
            b_full = tf.squeeze(debug_info, axis=(0, 1))
            
            l = tf.gather(tf.squeeze(llrs, axis=1), [0, 1, 2, 3], axis=0)
            l = tf.transpose(l, perm=[2, 1, 0])
            dm_sl = np.array(
                pusch_configs[0].dmrs_syms[
                    pusch_configs[0].start_sym:pusch_configs[0].start_sym
                    + pusch_configs[0].num_symbols
                ]
            )
            keep_syms = np.where(dm_sl == 0)[0].astype(np.int32)
            b = tf.gather(b_full, keep_syms, axis=0)
            b_hat = tf.cast(l > 0, b.dtype)
            errors = tf.not_equal(b, b_hat)
            ber = tf.reduce_mean(tf.cast(errors, tf.float32))

        code_blocks = self.decoder.decode(
            input_llrs=coded_blocks,
            pusch_configs=pusch_configs
        )

        decoded_tbs, crc = self.crc_checker.check_crc(
            input_bits=code_blocks,
            pusch_configs=pusch_configs
        )

        return decoded_tbs, crc, llrs

class NeuralRxOnnx:
    """Neural receiver using ONNX Runtime"""

    def __init__(self, num_rx_ant, onnx_model_file, num_layers=1, transmitter=None):
        self.cuda_stream = get_cuda_stream()
        self.num_rx_ant = num_rx_ant
        self.num_layers = int(num_layers)
        self.ch_est_algo = 3

        self.channel_estimator = ChannelEstimator(
            num_rx_ant=num_rx_ant,
            ch_est_algo=self.ch_est_algo,
            cuda_stream=self.cuda_stream,
        )

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
            if 'CUDAExecutionProvider' in ort.get_available_providers() \
            else ['CPUExecutionProvider']
        self.onnx_session = ort.InferenceSession(onnx_model_file, providers=providers)
        self.output_names = [o.name for o in self.onnx_session.get_outputs()]

        self.derate_match = LdpcDeRateMatch(
            enable_scrambling=True,
            cuda_stream=self.cuda_stream,
        )
        self.decoder = LdpcDecoder(cuda_stream=self.cuda_stream)
        self.crc_checker = CrcChecker(cuda_stream=self.cuda_stream)

    def run(self, rx_slot, slot, pusch_configs, debug_info=None):
        ch_est = self.channel_estimator.estimate(
            rx_slot=rx_slot,
            slot=slot,
            pusch_configs=pusch_configs,
        )
        if self.ch_est_algo == 1:
            ch_est[0] = ch_est[0][:, :, ::2, :].transpose((2, 1, 0, 3))

        num_layers = int(pusch_configs[0].ue_configs[0].layers)
        active_dmrs_ports, dmrs_ofdm_pos, dmrs_subcarrier_pos = pack_nrx_dmrs_input_tensors(
            pusch_configs[0].dmrs_syms, num_layers
        )
        if num_layers > 1:
            dmrs_ofdm_pos = dmrs_ofdm_pos[0]
            dmrs_subcarrier_pos = dmrs_subcarrier_pos[0]

        rx_slot_in = rx_slot[
            None, :,
            pusch_configs[0].start_sym:pusch_configs[0].start_sym + pusch_configs[0].num_symbols,
            :,
        ]
        ch_est_in = np.transpose(ch_est[0], (3, 0, 1, 2)).reshape(
            ch_est[0].shape[0] * ch_est[0].shape[3],
            ch_est[0].shape[1],
            ch_est[0].shape[2],
        )
        ch_est_in = ch_est_in[None, ...]

        def _np_f32(x):
            x = np.asarray(x, dtype=np.float32)
            return x.get() if hasattr(x, 'get') else x
        
        input_feed = {
            "rx_slot_real": np.real(rx_slot_in).astype(np.float32).get(),
            "rx_slot_imag": np.imag(rx_slot_in).astype(np.float32).get(),
            "h_hat_real": np.real(ch_est_in).astype(np.float32).get(),
            "h_hat_imag": np.imag(ch_est_in).astype(np.float32).get(),
            "active_dmrs_ports": active_dmrs_ports.astype(np.float32),
            "dmrs_ofdm_pos": dmrs_ofdm_pos.astype(np.int32),
            "dmrs_subcarrier_pos": dmrs_subcarrier_pos.astype(np.int32),
        }
        ort_out = self.onnx_session.run(self.output_names, input_feed)
        outputs = dict(zip(self.output_names, ort_out))

        raw_llr = outputs[self.output_names[0]][0, ...]
        n_cdm = int(pusch_configs[0].num_dmrs_cdm_grps_no_data)

        data_syms = np.array(
            pusch_configs[0].dmrs_syms[
                pusch_configs[0].start_sym:pusch_configs[0].start_sym
                + pusch_configs[0].num_symbols
            ]
        ) == 0
        llrs = np.take(raw_llr, np.where(data_syms)[0], axis=3)
        coded_blocks = self.derate_match.derate_match(
            input_llrs=[llrs],
            pusch_configs=pusch_configs,
        )

        if debug_info is not None:
            # b_full = tf.squeeze(debug_info, axis=(0, 1))
            b_full = tf.squeeze(debug_info, axis=0) # remove batch

            # l = tf.gather(tf.squeeze(llrs, axis=1), [0, 1, 2, 3], axis=0)
            # l = tf.transpose(l, perm=[2, 1, 0])
            l = tf.gather(llrs, [0, 1, 2, 3], axis=0)
            l = tf.transpose(l, perm=[1, 3, 2, 0]) # layer x num_symbols x num_subcarriers x bits_per_symbol
            dm_sl = np.array(
                pusch_configs[0].dmrs_syms[
                    pusch_configs[0].start_sym:pusch_configs[0].start_sym
                    + pusch_configs[0].num_symbols
                ]
            )
            keep_syms = np.where(dm_sl == 0)[0].astype(np.int32)
            b = tf.gather(b_full, keep_syms, axis=1)
            b_hat = tf.cast(l > 0, b.dtype)
            ber = tf.reduce_mean(tf.cast(tf.not_equal(b, b_hat), tf.float32))
            bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=b, logits=-l))

        code_blocks = self.decoder.decode(
            input_llrs=coded_blocks,
            pusch_configs=pusch_configs,
        )
        decoded_tbs, crc = self.crc_checker.check_crc(
            input_bits=code_blocks,
            pusch_configs=pusch_configs,
        )
        return decoded_tbs, crc, llrs

class PuschRxSeparate:
    """PUSCH receiver class.
    
    This class encapsulates the whole PUSCH receiver chain built using
    pyAerial components.    
    """

    def __init__(self,
                 num_rx_ant,
                 enable_pusch_tdi,
                 eq_coeff_algo=1,
                 ch_est_algo=1):
        """Initialize the PUSCH receiver."""
        self.cuda_stream = get_cuda_stream()

        self.ch_est_algo = ch_est_algo # because algo 1 interpolates while 3 does not

        # Build the components of the receiver.
        self.channel_estimator = ChannelEstimator(
            num_rx_ant=num_rx_ant,
            cuda_stream=self.cuda_stream,
            ch_est_algo=ch_est_algo)
        self.channel_equalizer = ChannelEqualizer(
            num_rx_ant=num_rx_ant,
            enable_pusch_tdi=enable_pusch_tdi,
            eq_coeff_algo=eq_coeff_algo,
            cuda_stream=self.cuda_stream)
        self.noise_intf_estimator = NoiseIntfEstimator(
            num_rx_ant=num_rx_ant,
            eq_coeff_algo=eq_coeff_algo,
            cuda_stream=self.cuda_stream)
        self.derate_match = LdpcDeRateMatch(
            enable_scrambling=True,
            cuda_stream=self.cuda_stream)
        self.decoder = LdpcDecoder(cuda_stream=self.cuda_stream)
        self.crc_checker = CrcChecker(cuda_stream=self.cuda_stream)

    def run(
        self,
        rx_slot,
        slot,
        pusch_configs,
        cell_num,
        debug_info=None
    ):
        """Run the receiver."""
        # Channel estimation.
        ch_est = self.channel_estimator.estimate(
            rx_slot=rx_slot,
            slot=slot,
            pusch_configs=pusch_configs
        ) 
        # ch_est[0].shape (4, 1, 3276, 3) -> is good (algo 1)

        num_layers = int(pusch_configs[0].ue_configs[0].layers)

        # Broken
        if self.ch_est_algo == 3: # need to reshape and use nn interpolation
            # use NN to interpolate channel estimate to full size
            ch_est[0] = tf.transpose(ch_est[0].get(), perm=[2, 1, 0, 3])
            ch_est[0] = tf.complex(tf.image.resize(tf.math.real(ch_est[0]), (num_layers,3276), 'nearest'),
                  tf.image.resize(tf.math.imag(ch_est[0]), (num_layers,3276), 'nearest'))


        # Noise and interference estimation.
        lw_inv, noise_var_pre_eq = self.noise_intf_estimator.estimate(
            rx_slot=rx_slot,
            channel_est=ch_est,
            slot=slot,
            pusch_configs=pusch_configs
        )

        llrs, sym = self.channel_equalizer.equalize(
            rx_slot=rx_slot,
            channel_est=ch_est,
            lw_inv=lw_inv,
            noise_var_pre_eq=noise_var_pre_eq,
            pusch_configs=pusch_configs
        )

        coded_blocks = self.derate_match.derate_match(
            input_llrs=llrs,
            pusch_configs=pusch_configs
        )
    
        code_blocks = self.decoder.decode(
            input_llrs=coded_blocks,
            pusch_configs=pusch_configs
        )

        decoded_tbs, crc = self.crc_checker.check_crc(
            input_bits=code_blocks,
            pusch_configs=pusch_configs
        )

        return decoded_tbs, crc

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate PUSCH neural receiver with real or synthetic data"
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data instead of ClickHouse database data'
    )
    parser.add_argument(
        '--num-synthetic-samples',
        type=int,
        default=100,
        help='Number of synthetic samples to generate per SNR point (default: 1000)'
    )
    parser.add_argument(
        '--esno-range-start',
        type=float,
        default=1.0,
        help='Start of Es/No range in dB for synthetic data (default: -4.0)'
    )
    parser.add_argument(
        '--esno-range-end',
        type=float,
        default=1.1,
        help='End of Es/No range in dB for synthetic data (default: -2.8)'
    )
    parser.add_argument(
        '--esno-range-step',
        type=float,
        default=0.2,
        help='Step size for Es/No range in dB for synthetic data (default: 0.2)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default="nrx_rt_3dmrs_13sym1",
        help='Model name to use for evaluation (default: nrx_rt_3dmrs_13sym1)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10000,
        help='Number of database records to process (default: all records)'
    )
    parser.add_argument(
        '--timestamps',
        type=str,
        default=None,
        help='Path to timestamps pickle file. If file exists, load and use those timestamps for deterministic replay. If not, save timestamps to this file.'
    )
    parser.add_argument(
        '--synthetic-data',
        type=str,
        default=None,
        help='Path to synthetic data pickle file. If file exists, load data instead of generating. If not, save generated data to this file.'
    )
    parser.add_argument(
        '--ue',
        type=int,
        default=1,
        help='0 for Samsung Galaxy S23, 1 for iPhone 14 Pro.'
    )
    parser.add_argument(
        '--cell-id',
        type=int,
        default=51,
        help='O-RU to USE; right now we have cell 51 (planar array); cell 41 and 42 have ULA antennas'
    )
    parser.add_argument(
        '--add-noise-snr',
        type=float,
        default=None,
        help='Add Gaussian noise to real data with specified SNR (dB) relative to signal power. If not set, no noise is added.'
    )
    parser.add_argument(
        '--bce',
        action='store_true',
        help='Compute, display and save BCE values'
    )
    parser.add_argument(
        '--db',
        type=str,
        default=1,
        help='Which database to use (0: lab, 1: j_floor 1st, 2: drone)'
    )
    parser.add_argument(
        '--use-onnx',
        action='store_true',
        help='Run neural receiver with ONNX Runtime (.onnx) instead of TensorRT (.plan)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Debug mode (create bit grid)'
    )
    args = parser.parse_args()

    assert args.ue in [0, 1,], "UE must be 0 (Samsung Galaxy S23) or 1 (iPhone 14 Pro)"
    assert args.cell_id in [41, 42, 51], "Cell ID must be 41, 42, or 51"
    if args.synthetic:
        assert not args.bce, "BCE computation is not supported for synthetic data"
    nrx_trt_file = f"../onnx_models/{args.model}"
    assert os.path.exists(nrx_trt_file), f"Model file {nrx_trt_file} does not exist!"
    # assert args.synthetic or args.db in ['0', '1', '2'], "Database must be '0' (lab) or '1' (j_floor 1st) or '2' (drone)"

    num_rx_ant = 4 
    num_tx_ant = 1
    cell_id = args.cell_id
    enable_pusch_tdi = 1
    eq_coeff_algo = 1
    rx_ch_est_algo = 1

    if not args.synthetic and args.db in ['3', '4', '5', '6', '7', '8', '9']:
        num_tx_ant = 2
    
    pusch_rx = PuschRx(
        cell_id=cell_id,
        num_rx_ant=num_rx_ant,
        num_tx_ant=num_tx_ant,
        enable_pusch_tdi=enable_pusch_tdi,
        eq_coeff_algo=eq_coeff_algo,
        ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL
    )

    pusch_rx_separate = PuschRxSeparate(
        num_rx_ant=num_rx_ant,
        enable_pusch_tdi=enable_pusch_tdi,
        eq_coeff_algo=eq_coeff_algo,
        ch_est_algo=rx_ch_est_algo
    )

    # trt = args.model
    # nrx_trt_file = f"../onnx_models/{trt}"
    if args.synthetic:
        num_layers = 1
    else:
        num_layers = num_layers_for_db(args.db)

    if args.use_onnx:
        neural_rx = NeuralRxOnnx(
            num_rx_ant=num_rx_ant,
            onnx_model_file=nrx_trt_file,
            num_layers=num_layers,
        )
    else:
        neural_rx = NeuralRx(
            num_rx_ant=num_rx_ant,
            trt_model_file=nrx_trt_file,
            num_layers=num_layers,
        )
    backend = 'onnx' if args.use_onnx else 'trt'
    print(f"   NeuralRx: {num_layers} layer(s), backend={backend}, model={nrx_trt_file}")
   
    # # Create the neural receiver
    # neural_rx = NeuralRx(
    #     num_rx_ant=num_rx_ant,
    #     trt_model_file=nrx_trt_file,
    #     num_layers=num_layers,
    # )
    # print(f"   NeuralRx configured for {num_layers} layer(s)")
    
    if args.synthetic:
        print("=" * 80)
        print("SYNTHETIC DATA MODE")
        print("=" * 80)
        
        # Check if synthetic data pickle file exists for deterministic replay
        load_synthetic_mode = False
        synthetic_data_list = None
        timestamp_dir = f"../eval_timestamps/{args.synthetic_data}"
        if args.synthetic_data and os.path.exists(args.timestamp_dir):
            with open(timestamp_dir, 'rb') as f:
                synthetic_data_list = pickle.load(f)
            load_synthetic_mode = True
            print(f"   Loaded {len(synthetic_data_list)} synthetic samples from {timestamp_dir}")
        
        # Synthetic data parameters
        rnti = 1234
        scid = 0
        data_scid = 0
        layers = 1
        mcs_index = 14
        mcs_table = 0
        dmrs_ports = 1
        start_prb = 0
        num_prbs = 273
        start_sym = 0
        num_symbols = 13  # 12 for synthetic (notebook uses this)
        dmrs_scrm_id = 41  # Use 41 for synthetic
        cell_id = 41  # Use 41 for synthetic to match notebook
        dmrs_syms = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
        dmrs_max_len = 1
        dmrs_add_ln_pos = 2
        num_dmrs_cdm_grps_no_data = 2
        
        mod_order, code_rate = get_mcs(mcs_index, mcs_table + 1)
        tb_size = get_tb_size(
            mod_order=mod_order,
            code_rate=code_rate,
            dmrs_syms=dmrs_syms,
            num_prbs=num_prbs,
            start_sym=start_sym,
            num_symbols=num_symbols,
            num_layers=layers
        )
        
        pusch_ue_config = PuschUeConfig(
            scid=scid,
            layers=layers,
            dmrs_ports=dmrs_ports,
            rnti=rnti,
            data_scid=data_scid,
            mcs_table=mcs_table,
            mcs_index=mcs_index,
            code_rate=int(code_rate * 10),
            mod_order=mod_order,
            tb_size=tb_size // 8
        )
        
        pusch_configs = [PuschConfig(
            ue_configs=[pusch_ue_config],
            num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
            dmrs_scrm_id=dmrs_scrm_id,
            start_prb=start_prb,
            num_prbs=num_prbs,
            dmrs_syms=dmrs_syms,
            dmrs_max_len=dmrs_max_len,
            dmrs_add_ln_pos=dmrs_add_ln_pos,
            start_sym=start_sym,
            num_symbols=num_symbols
        )]
        
        channel_model = "UMi"
        # channel_model = "TDL-B100"
        if load_synthetic_mode:
            # Use loaded data
            total_samples = len(synthetic_data_list)
            print(f"   Using {total_samples} pre-generated samples from pickle file")
        else:
            # Generate new data
            pusch_tx = PdschTx(
                cell_id=cell_id,
                num_rx_ant=num_tx_ant,
                num_tx_ant=num_tx_ant,
            )
            
            _, apply_channel = setup_synthetic_channel(
                num_rx_ant=num_rx_ant,
                num_tx_ant=num_tx_ant,
                carrier_frequency=2.14e9, #3.5e9
                channel_model=channel_model,
                delay_spread=100e-9,
                speed=0.8333
            )
            
            esno_db_range = np.arange(
                args.esno_range_start,
                args.esno_range_end,
                args.esno_range_step
            )
            
            # Create synthetic data generator
            data_generator = generate_synthetic_samples(
                num_samples=args.limit,
                esno_db_range=esno_db_range,
                pusch_tx=pusch_tx,
                apply_channel_fn=apply_channel,
                pusch_configs=pusch_configs
            )
            
            total_samples = len(esno_db_range) * args.limit
            print(f"   Generating {args.limit} samples per SNR point")
            print(f"   Es/No range: {args.esno_range_start} to {args.esno_range_end} dB (step: {args.esno_range_step})")
            print(f"   Total samples: {total_samples}")
        
    else:
        print("=" * 80)
        print("REAL DATA MODE (ClickHouse)")
        print("=" * 80)
        
        client = clickhouse_connect.get_client(host='localhost')
        
        # Check if timestamps pickle file exists for deterministic replay
        load_timestamps_mode = False
        loaded_timestamps = None
        timestamp_dir = f"../eval_timestamps/{args.timestamps}"
        if args.timestamps and os.path.exists(timestamp_dir):
            with open(timestamp_dir, 'rb') as f:
                loaded_timestamps = pickle.load(f)
            load_timestamps_mode = True
            print(f"   Loaded {len(loaded_timestamps)} timestamps from {timestamp_dir}")

        ue = args.ue
        match args.db:
            case '0':
                database = "nrx_measurement_2025_11_03"
                
                target_rntis = [0x43df, 0x3035]  # RNTIs Samsung Galaxy S23 and iPhone 14 Pro
                target_rnti = target_rntis[ue]
                split_times = ['2025-11-03 20:48:00.000001', '2025-11-03 20:54:58.960007']
                split_time = split_times[ue]
                num_layers = 1
                # target_rnti = 0x43df  # 17375 in decimal (Samsung Galaxy S23)
                # target_rnti = 0x3035 # 12341 in decimal (iphone 14 pro)
                # split_time = '2025-11-03 20:48:00.000001'  # Validation split starts here
                # split_time = '2025-11-03 20:54:58.960007' # for iphone 14 pro
                if args.timestamps is None:
                    # If not loading timestamps, we are generating them for the first time
                    # So we set the split time to a very early time to include all data
                    args.timestamps = "timestamps1.pkl"
            case '1':
                database = "nrx_jfloor_1st_2025_12_02"
                target_rntis = [0xd9a1, 0xb1c3]
                target_rnti = target_rntis[ue]
                split_times = ['2025-12-02 21:54:35.000000000',  '2025-12-02 21:59:03.234500000']
                split_time = split_times[ue]
                num_layers = 1
                if args.timestamps is None:
                    args.timestamps = "timestamps_jfloor1_ue1_1000.pkl"

            case '2':
                database = "nrx_drone_2025_12_03"
                target_rnti = 10217
                split_time = "2025-12-03 12:21:05.202000000"
                num_layers = 1
                if args.timestamps is None:
                    args.timestamps = "timestamps_drone3.pkl"
            case '3': # 2 layers dummy dataset
                database = 'quectel_2ULLs_j61_2026_04_27' # broken params
                target_rnti = 65248 #61815
                num_layers = 2
                split_time = "2026-04-26 12:48:30.000000000" # dummy
                limit = 1000 # dummy
            case '4': # 2 layers
                database = 'quectel_2ULLs_13dB_j61_2026_04_28'
                target_rnti = 0xf177 #61815
                num_layers = 2
                split_time = "2026-04-28 12:48:30.000000000" # 14245 samples out of 31187 before split (1656 fail)
                limit = 1000 # dummy
            case '5': # 1 layer
                database = 'quectel_1ULLs_7dB_j61_2026_04_28'
                target_rnti = 19995
                num_layers = 1
                split_time = "2026-04-28 12:48:30.000000000" # 14245 samples out of 31187 before split (1656 fail)
                limit = 1000 # dummy
            case '6':
                database = 'quectel_2ULLs_13dB_j61_2026_05_12'
                num_layers = 2
                split_time = '2026-05-12 11:53:30' # 55470 samples out of 94800 before split (3707 fail)
                limit = 1000 # dummy
            case '7':
                database = 'quectel_1ULLs_7dB_j61_2026_05_12'
                num_layers = 1
                split_time = '2026-05-12 12:22:00' # 55191 samples out of 112800 before split (4015 fail)
                limit = 1000 # dummy
            case '-1': # dummy high snr dataset with 2 layers
                database = 'quectel_2ULLs_28dB_jFloor_2026_06_02'
                num_layers = 2
                split_time = '2026-06-02 01:00:00' #
                limit = 1000
            case '8': # 2 layer port 0, 2 cdmnodata = 2
                database = 'Pixel9Pro_2ULLs_12dB_j61_2026_06_04'
                num_layers = 2
                split_time = '2026-06-04 13:27:17.372000000' # 20021 samples out of 51549 (1454 fail)
                limit = 1000 # dummy
            case '9':
                database = 'sgs23_1ULLs_5dB_j61_2026_06_04'
                num_layers = 1
                split_time = '2026-06-04 12:59:00.372000000' # 9978 samples out of 60691 (467 fail)
                limit = 1000 # dummy
            case '10': # 2 layer jFloor
                database = 'Pixel9Pro_2ULLs_12dB_jFloorRealLoop_2026_06_10'
                num_layers = 2
                split_time = '2026-06-10 14:47:00' # 30746 samples out of 77084 (2823 fail)
                limit = 1000 # dummy
            case '11': # 1 layer jFloor
                database = 'sgs23_1ULLs_5dB_jFloorRealLoop_2026_06_10'
                num_layers = 1
                split_time = '2026-06-10 15:04:00' # 27369 samples out of 97335 (1734 fail)
                limit = 1000 # dummy

        limit = args.limit  # Number of samples to evaluate

        if load_timestamps_mode:
            # Deterministic replay: query only records with specific timestamps
            # Format timestamps for ClickHouse IN clause
            ts_strings = [f"toDateTime64('{ts.strftime('%Y-%m-%d %H:%M:%S.%f')}', 9)" for ts in loaded_timestamps]
            ts_in_clause = ", ".join(ts_strings)
            query = f"""select * from export_fapi_{database}
                        where TsTaiNs IN ({ts_in_clause})
                        and CellId = 51
                        limit {limit}
                        """
        else:
            # Balanced sampling: 10% failed, 90% successful
            query = f"""select * from (
                        (select * from export_fapi_{database}
                            where rbSize = 273
                            and mcsIndex > 5
                            and qamModOrder = 4
                            and CellId = 51
                            and nrOfLayers = {num_layers}
                            and rvIndex = 0
                            and tbCrcFail = 1
                            and NrOfSymbols = 13
                            and TsTaiNs > toDateTime64('{split_time}', 9)
                            order by rand()
                            limit CEIL({limit} * 0.1)
                        )
                        union all
                        (
                            select * 
                            from export_fapi_{database}
                            where rbSize = 273
                            and mcsIndex > 5
                            and qamModOrder = 4
                            and CellId = 51
                            and nrOfLayers = {num_layers}
                            and rvIndex = 0
                            and tbCrcFail = 0
                            and NrOfSymbols = 13
                            and TsTaiNs > toDateTime64('{split_time}', 9)
                            order by rand()
                            limit CEIL({limit} * 0.9)
                        )
                    ) as combined
                    order by rand()
                    """
        # query = f"""select * from export_fapi_nrx_measurement_2025_11_03
        #         where nUEs > 1
        #         and rbSize = 273
        #         and mcsIndex > 5
        #         and qamModOrder = 4
        #         and rnti = {target_rnti}
        #         and rvIndex = 0
        #         and tbCrcFail = 0
        #         and TsTaiNs < toDateTime64('{split_time}', 9)
        #         order by TsTaiNs
        #         limit 100
        #         """

        pusch_records = client.query_df(query)
        print(f"   Found {len(pusch_records)} records to process")
        total_samples = len(pusch_records)
    
    print("-" * 80)
    print(f"{'Sample':<8} {'TB CRC':<10} {'TB Size':<10} {'PUSCH Rx':<12} {'Neural Rx':<12}")
    print("-" * 80)
    
    num_samples = 0
    num_tb_errors = defaultdict(int)
    num_skipped = 0
    num_retransmissions = 0
    saved_timestamps = []  # List to collect timestamps for deterministic replay
    bce_values = []  # List to collect BCE values for CDF computation
    first_sample_bce_per_bit = None
    
    snr_stats = defaultdict(lambda: {'total': 0, 'pusch_errors': 0, 'neural_errors': 0})
    
    if args.synthetic:
        # SYNTHETIC DATA MODE
        snr_sample_count = {}
        saved_synthetic_data = []  # List to save generated data for deterministic replay
        
        # Choose data source: loaded pickle or generator
        if load_synthetic_mode:
            data_source = synthetic_data_list
        else:
            data_source = data_generator
        
        for sample_data in tqdm(data_source, total=total_samples, desc="Processing"):
            if load_synthetic_mode:
                # Unpack from saved data (numpy arrays)
                rx_tensor_np, slot_number, tb_input_np, esno_db = sample_data
                rx_tensor = cp.array(rx_tensor_np, dtype=cp.complex64)
            else:
                # Unpack from generator (already cupy array)
                rx_tensor, slot_number, tb_input_np, esno_db = sample_data
                # Save for deterministic replay (convert cupy to numpy)
                if args.synthetic_data:
                    saved_synthetic_data.append((rx_tensor.get(), slot_number, tb_input_np, esno_db))
            
            if esno_db not in snr_sample_count:
                snr_sample_count[esno_db] = 0

            
            tbs_pusch, tb_crcs_pusch = pusch_rx_separate.run(
                rx_slot=rx_tensor,
                slot=slot_number,
                pusch_configs=pusch_configs,
                cell_num=cell_id
            )
            
            pusch_rx_error = int(not np.array_equal(tbs_pusch[0].get(), tb_input_np))
            num_tb_errors["PUSCH Rx"] += pusch_rx_error
            
            
            tbs_nrx, crc_nrx, llr = neural_rx.run(
                rx_slot=rx_tensor,
                slot=slot_number,
                pusch_configs=pusch_configs,
                debug_info=bit_grid if args.debug else None
            )
            neural_rx_error = int(not np.array_equal(tbs_nrx[0], tb_input_np))
            num_tb_errors["Neural Rx"] += neural_rx_error
            
            # Track per-SNR statistics
            snr_stats[esno_db]['total'] += 1
            snr_stats[esno_db]['pusch_errors'] += pusch_rx_error
            snr_stats[esno_db]['neural_errors'] += neural_rx_error
            
            snr_sample_count[esno_db] += 1
            num_samples += 1
        
        # Save synthetic data to pickle file
        if args.synthetic_data and not load_synthetic_mode and saved_synthetic_data:
            with open(args.synthetic_data, 'wb') as f:
                pickle.dump(saved_synthetic_data, f)
            print(f"\nSaved {len(saved_synthetic_data)} synthetic samples to {args.synthetic_data}")
    else:
        # REAL DATA MODE
        for index, pusch_record in tqdm(pusch_records.iterrows(), total=len(pusch_records), desc="Processing"):
            query_fh = f"""select TsTaiNs, CellId, fhData from export_fh_{database} 
                        where TsTaiNs == toDateTime64('{pusch_record.TsTaiNs.timestamp()}',9)
                        and CellId == {cell_id}
                        """
            fh = client.query_df(query_fh)
            
            if fh.index.size != 1:
                num_skipped += 1
                continue
            cellId = pusch_record.CellId

            if pusch_record.tbCrcFail == 0:
                if pusch_record.pduData is None:
                    num_skipped += 1
                    continue
                tb_input_np = np.array(pusch_record.pduData)
            else:
                tb_input_np, retransmission_ts = find_successful_retransmission(client, database, pusch_record)
                num_retransmissions += 1
                if tb_input_np is None:
                    num_skipped += 1
                    continue
            
            fh_samp = (np.array(fh['fhData'].iloc[0], dtype=np.int16).view(np.float16)).astype(np.float32)
            rx_slot = np.swapaxes(fh_samp.view(np.complex64).reshape(4, 14, 273 * 12), 2, 0)
            
            # Add Gaussian noise if SNR is specified
            if args.add_noise_snr is not None:
                signal_power = np.mean(np.abs(rx_slot) ** 2)
                noise_power = signal_power / (10 ** (args.add_noise_snr / 10))
                # Complex Gaussian noise: variance split equally between real and imaginary
                noise = np.sqrt(noise_power / 2) * (np.random.randn(*rx_slot.shape) + 1j * np.random.randn(*rx_slot.shape))
                rx_slot = rx_slot + noise.astype(np.complex64)
            
            rx_tensor = cp.array(rx_slot, dtype=cp.complex64)
            
            pusch_configs = create_pusch_config_from_record(pusch_record)
            if args.bce or args.debug:
                transmitter = create_sionna_transmitter_from_record(pusch_record)
                bit_grid = map_bits_to_resource_grid(tb_input_np, transmitter)
            else:
                bit_grid = None

            
            slot_number = int(pusch_record.Slot)
            
            tbs_pusch, tb_crcs_pusch = pusch_rx_separate.run(
                rx_slot=rx_tensor,
                slot=slot_number,
                pusch_configs=pusch_configs,
                cell_num=cellId,
            )

            # tb_crcs_pusch, tbs_pusch = pusch_rx.run(
            #     rx_slot=rx_tensor,
            #     slot=slot_number,
            #     pusch_configs=pusch_configs,
            # )
            
            pusch_rx_error = int(not np.array_equal(tbs_pusch[0].get(), tb_input_np))
            num_tb_errors["PUSCH Rx"] += pusch_rx_error
            pusch_rx_result = "FAIL" if pusch_rx_error else "PASS"
            
            tbs_nrx, crc_nrx, llrs = neural_rx.run(
                rx_slot=rx_tensor,
                slot=slot_number,
                pusch_configs=pusch_configs,
                debug_info=bit_grid if args.debug else None
            )
            neural_rx_error = int(not np.array_equal(tbs_nrx[0], tb_input_np))
            num_tb_errors["Neural Rx"] += neural_rx_error
            
            # Compute BCE between LLRs and bit_grid
            if args.bce and bit_grid is not None:
                pc0 = pusch_configs[0]
                
                # Process llrs: (8, 1, 3276, n_sym) -> (4, 3276, n_sym) -> (n_sym, 3276, 4)
                llrs_proc = llrs[:4, 0, :, :]
                llrs_proc = np.transpose(llrs_proc, (2, 1, 0))
                bits_proc = tf.squeeze(bit_grid, axis=(0, 1)).numpy()
                dm_sl = np.array(pc0.dmrs_syms[pc0.start_sym:pc0.start_sym + pc0.num_symbols])
                keep_syms = np.where(dm_sl == 0)[0]
                bits_proc = bits_proc[keep_syms, :, :]

                llrs_proc = -llrs_proc  # Invert LLRs to match bit mapping convention
                # sample_bce = np.mean(np.maximum(llrs_proc, 0) - llrs_proc * bits_proc + np.log1p(np.exp(-np.abs(llrs_proc))))
                
                # # Flatten for BCE computation
                llrs_flat = llrs_proc.flatten().astype(np.float32)
                bits_flat = bits_proc.flatten().astype(np.float32)
                
                # # Compute BCE: sigmoid_cross_entropy_with_logits
                # # BCE = max(llr, 0) - llr * label + log(1 + exp(-|llr|))
                bce_per_bit = np.maximum(llrs_flat, 0) - llrs_flat * bits_flat + np.log1p(np.exp(-np.abs(llrs_flat)))
                
                # Save per-bit BCE for the first sample (for plotting)
                if first_sample_bce_per_bit is None:
                    first_sample_bce_per_bit = bce_per_bit.copy()
                
                # # Mean BCE for this sample
                sample_bce = np.mean(bce_per_bit)
                bce_values.append(bce_per_bit)
            
            num_samples += 1
            
            # Save timestamp for deterministic replay (only if not in load mode)
            if args.timestamps and not load_timestamps_mode:
                saved_timestamps.append(pusch_record.TsTaiNs)
    
    # Save timestamps to pickle file if requested (real data mode only, and not loading)
    if not args.synthetic and args.timestamps and not load_timestamps_mode and saved_timestamps:
        with open(timestamp_dir, 'wb') as f:
            pickle.dump(saved_timestamps, f)
        print(f"\nSaved {len(saved_timestamps)} timestamps to {timestamp_dir}")
    
    # Print final results and save to file
    ue_names = ["Samsung S23", "iPhone 14 Pro"]

    ###### PRINT RESULTS ######
    results_lines = []
    results_lines.append("-" * 80)
    results_lines.append(f"\nFinal Results:")
    results_lines.append(f"   Mode: {f'SYNTHETIC {channel_model}' if args.synthetic else 'REAL DATA (ClickHouse)'}")
    results_lines.append(f"   Model: {args.model}")
    if not args.synthetic:
        results_lines.append(f"   Database: {database}")
        results_lines.append(f"   UE: {ue_names[ue]}")
        results_lines.append(f"   CellId: {cell_id}")
        if args.add_noise_snr is not None:
            results_lines.append(f"   Added Noise SNR: {args.add_noise_snr:.1f} dB")
        else:
            results_lines.append(f"   Added Noise: None")
    results_lines.append(f"   Total samples processed: {num_samples}")
    if not args.synthetic:
        results_lines.append(f"   Samples skipped: {num_skipped}")
        results_lines.append(f"   HARQ Retransmissions: {num_retransmissions}")
    results_lines.append(f"   ")
    results_lines.append(f"   PUSCH Rx:")
    results_lines.append(f"      TB Errors: {num_tb_errors['PUSCH Rx']}")
    if num_samples > 0:
        results_lines.append(f"      BLER: {num_tb_errors['PUSCH Rx'] / num_samples * 100:.2f}%")
    results_lines.append(f"   ")
    results_lines.append(f"   Neural Rx:")
    results_lines.append(f"      TB Errors: {num_tb_errors['Neural Rx']}")
    if num_samples > 0:
        results_lines.append(f"      BLER: {num_tb_errors['Neural Rx'] / num_samples * 100:.2f}%")
    
    # BCE statistics (for real data mode)
    # bce_values is a list of per-bit BCE arrays, concatenate all
    if not args.synthetic and bce_values:
        with open("bce_values_base.npy", "wb") as f:
            np.save(f, np.concatenate(bce_values))
        bce_all = np.concatenate(bce_values)
        results_lines.append(f"      BCE Statistics (all bits from {len(bce_values)} samples):")
        results_lines.append(f"         Total bits: {len(bce_all)}")
        results_lines.append(f"         Mean: {np.mean(bce_all):.4f}")
        results_lines.append(f"         Std:  {np.std(bce_all):.4f}")
        results_lines.append(f"         Min:  {np.min(bce_all):.4f}")
        results_lines.append(f"         Max:  {np.max(bce_all):.4f}")
        # CDF percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        results_lines.append(f"      BCE CDF Percentiles:")
        for p in percentiles:
            results_lines.append(f"         {p}th: {np.percentile(bce_all, p):.4f}")
    results_lines.append(f"   ")
    
    # Print per-SNR breakdown for synthetic data
    if args.synthetic and snr_stats:
        results_lines.append("=" * 80)
        results_lines.append("PER-SNR BREAKDOWN (Synthetic Data)")
        results_lines.append("=" * 80)
        results_lines.append(f"{'Es/No (dB)':<12} {'Samples':<10} {'PUSCH BLER':<15} {'Neural BLER':<15}")
        results_lines.append("-" * 80)
        
        # Sort by SNR value
        for esno_db in sorted(snr_stats.keys()):
            stats = snr_stats[esno_db]
            total = stats['total']
            pusch_bler = (stats['pusch_errors'] / total * 100) if total > 0 else 0
            neural_bler = (stats['neural_errors'] / total * 100) if total > 0 else 0
            
            results_lines.append(f"{esno_db:<12.1f} {total:<10} {pusch_bler:>6.2f}%        {neural_bler:>6.2f}%")
        
        results_lines.append("-" * 80)
    
    results_lines.append("=" * 80)
    results_lines.append("Evaluation complete!")
    results_lines.append("=" * 80)
    
    # Print to console
    for line in results_lines:
        print(line)
    
    # Append to results.txt
    with open("../results/nrx_evaluation_results.txt", "a") as f:
        f.write("\n".join(results_lines) + "\n\n")
    
    # Plot normalized CDF of all per-bit BCE values
    if not args.synthetic and args.bce and bce_values:
        with open(f"bce_values{args.model}.npy", "wb") as f:
            np.save(f, np.concatenate(bce_values))
        bce_all = np.concatenate(bce_values)
        sorted_bce = np.sort(bce_all)
        # Normalized CDF: y-axis is cumulative probability (0 to 1)
        cdf = np.linspace(0, 1, len(sorted_bce))
        
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_bce, cdf, linewidth=1)
        plt.xlabel('BCE')
        plt.ylabel('Cumulative Probability')
        plt.title(f'Normalized CDF of Per-bit BCE ({len(bce_values)} samples, {len(bce_all)} bits)')
        plt.grid(True, alpha=0.3)
        plt.xlim(left=0)
        plt.ylim(0, 1)
        plt.savefig('plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved BCE CDF plot to plot.png ({len(bce_all)} total bits)")


if __name__ == "__main__":
    main()

