#!/usr/bin/env python3
"""
Create TFRecord from Aerial Data Lake data
This script reads 5G NR PUSCH data from Aerial Data Lake and creates TFRecord files with 
IQ samples, channel estimates, and bit labels. For failed transmissions (tbCrcFail = 1),
the script finds the next successful retransmission with the same HARQ process ID to obtain
the correct labels.

Author: Nuri Berke Baytekin
"""

import os
import argparse
import json
import sys
from pathlib import Path
import gc
from itertools import cycle

DATABASE_NAMES = {
    0: "nrx_measurement_2025_11_03",
    1: "nrx_jfloor_1st_2025_12_02",
    2: "nrx_drone_2025_12_03",
    3: "quectel_2ULLs_j61_2026_04_27",
    4: "quectel_1ULLs_7dB_j61_2026_04_28",
    5: "quectel_2ULLs_13dB_j61_2026_04_28",
    6: "quectel_2ULLs_13dB_j61_2026_05_12",
    7: "quectel_1ULLs_7dB_j61_2026_05_12",
    8: "Pixel9Pro_2ULLs_12dB_j61_2026_06_04",
    9: "sgs23_1ULLs_5dB_j61_2026_06_04",
    10: "Pixel9Pro_2ULLs_12dB_jFloorRealLoop_2026_06_10",
    11: "sgs23_1ULLs_5dB_jFloorRealLoop_2026_06_10",
}


def get_database_metadata(db_index):
    """Return the database and TFRecord names associated with ``--db``."""

    try:
        database = DATABASE_NAMES[db_index]
    except KeyError as error:
        raise ValueError(
            f"Unknown database index {db_index}; choose one of {sorted(DATABASE_NAMES)}"
        ) from error
    
    return {
        "db_index": db_index,
        "database": database,
        "tfrecord_filename": f"{database}.tfrecord",
    }


def build_argument_parser():
    parser = argparse.ArgumentParser(description='Create TFRecord from Aerial Data Lake data')
    parser.add_argument('--limit', type=int, default=5000, help='Number of records to process')
    parser.add_argument('--cell_id', type=int, default=51, help='Cell ID to use for FH data')
    parser.add_argument('--db', type=int, default=4, choices=sorted(DATABASE_NAMES), help='Database index to use')
    parser.add_argument('--describe-db', action='store_true', help='Print database metadata as JSON and exit')
    return parser


if __name__ == '__main__' and '--describe-db' in sys.argv:
    metadata_args = build_argument_parser().parse_args()
    print(json.dumps(get_database_metadata(metadata_args.db)))
    raise SystemExit(0)
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import cupy as cp

# Connecting to clickhouse on remote server
import clickhouse_connect

from sionna.nr import PUSCHConfig, PUSCHDMRSConfig, TBConfig, CarrierConfig, PUSCHTransmitter

from sionna.ofdm import LSChannelEstimator

# pyAerial: used to compute an accurate per-example noise variance N0 that
# matches the evaluation pipeline (eval_duidd_from_datalake.py).
from aerial.phy5g.algorithms import ChannelEstimator, NoiseIntfEstimator
from aerial.phy5g.config import PuschConfig, PuschUeConfig
from aerial.util.cuda import get_cuda_stream
from aerial.util.fapi import dmrs_fapi_to_bit_array

from tqdm import tqdm

# Hide log10(10) warning
_ = np.seterr(divide='ignore', invalid='ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

np.set_printoptions(threshold=100)  # Control the number of elements to display
np.set_printoptions(edgeitems=100)  # Control the number of edge items to display
np.set_printoptions(linewidth=200) # Control the width of the display


def find_successful_retransmission(client, database, current_record, max_lookahead=10):
    """Find the next successful retransmission with the same HARQ process ID.
    
    HARQ process IDs cycle from 0-15. When a transmission fails, it gets retransmitted
    with the same process ID. This function searches for retransmissions following the
    RV sequence: 2 -> 3 -> 1. If all fail, returns None.
    
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
    allowed_retransmission_time = 30  # milliseconds
    
    # Query for subsequent transmissions with same RNTI, ordered by time
    # We need to see all transmissions to track the cycle
    query = f"""select TsTaiNs, pduData, tbCrcFail, harqProcessID, rnti, rvIndex 
                from export_fapi_{database} 
                where rnti = {rnti}
                and TsTaiNs > toDateTime64('{current_ts.timestamp()}', 9)
                and TsTaiNs <= toDateTime64('{current_ts.timestamp()}', 9) + toIntervalMillisecond({allowed_retransmission_time})
                and CellId = {cell_id}
                and NrOfSymbols = 13
                order by TsTaiNs
                limit {max_lookahead}"""
    
    subsequent_transmissions = client.query_df(query)
    

    seen_process_ids = {harq_process_id}
    
    rv_sequence = [2, 3, 1]
    req_rv = rv_sequence[0]  # Start with RV=2
    rv_index = 0
    
    # Look for retransmissions with matching HARQ process ID and required RV
    for _, tx in subsequent_transmissions.iterrows():
        tx_process_id = tx.harqProcessID
        
        if tx_process_id == harq_process_id:
            if len(seen_process_ids) >= 16:
                return None, None
            
            if tx.rvIndex == req_rv:
                if tx.tbCrcFail == 0 and tx.pduData is not None:
                    return np.array(tx.pduData), tx.TsTaiNs
                else:
                    rv_index += 1
                    if rv_index >= len(rv_sequence):
                        return None, None
                    req_rv = rv_sequence[rv_index]
        
        seen_process_ids.add(tx_process_id)
    
    return None, None


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

    # Reshape to group bits into symbols
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

    # mask pilots to -1
    pilot_position = transmitter._resource_grid.build_type_grid()[:,0] == 1
    pilot_position = tf.cast(pilot_position, tf.float32)
    pilot_position = tf.expand_dims(pilot_position, axis=-1)
    pilot_position = tf.broadcast_to(pilot_position, tf.shape(bit_grid))
    bit_grid = tf.where(pilot_position==1, tf.constant(-1., bit_grid.dtype), bit_grid)

    # # make it 14 OFDM symbols
    # new_entry = -tf.ones_like(bit_grid[:, :, :1, :, :])
    # bit_grid = tf.concat([bit_grid, new_entry], axis=2)
    
    return bit_grid


def serialize_example(y, h, b, no, b_info):
    def _bytes_feature(value):
        if isinstance(value, tf.Tensor):
            value = value.numpy()  # convert EagerTensor to bytes
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    feature = {
        "y": _bytes_feature(tf.io.serialize_tensor(y)),
        "h": _bytes_feature(tf.io.serialize_tensor(h)),
        "b": _bytes_feature(tf.io.serialize_tensor(tf.cast(b, tf.int8))),
        "b_info": _bytes_feature(tf.io.serialize_tensor(tf.cast(b_info, tf.int8))),
        "no": _bytes_feature(tf.io.serialize_tensor(tf.cast(no, tf.float32))),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def estimate_no_from_pilots(rx_slot, transmitter):
    y = tf.cast(rx_slot, tf.complex64)
    y = tf.squeeze(y, axis=0)  # [num_rx, num_rx_ant, num_ofdm, num_sc]
    y = tf.squeeze(y, axis=0)  # [num_rx_ant, num_ofdm, num_sc]

    pilot_mask = transmitter._resource_grid.build_type_grid()[:, 0] == 1  # [num_sc, num_ofdm]
    pilot_mask = pilot_mask[0] # remove batch
    pilots = tf.cast(transmitter.pilot_pattern.pilots, tf.complex64)

    # x_p = tf.reshape(pilots, [-1])
    x_p = pilots[0]
    x_p = tf.transpose(x_p)
    # active = tf.abs(x_p) > 0
    active_row = tf.reduce_any(tf.abs(x_p) > 0, axis=1)
    x_p = tf.boolean_mask(x_p, active_row)

    # Estimate N0 per RX antenna, then average
    no_ants = []
    for ant in range(int(y.shape[0])):
        y_ant = y[ant]
        y_p = tf.boolean_mask(y_ant, pilot_mask)
        y_p = tf.boolean_mask(y_p, active_row)
        y_p = tf.expand_dims(y_p, axis = 1)
        # y_p = tf.boolean_mask(y_p, active[-1])
        # h_bar = tf.reduce_mean(y_p / x_p)
        # h_bar = tf.reduce_sum(tf.math.conj(x_p) * y_p) / tf.cast(tf.reduce_sum(tf.abs(x_p)**2), tf.complex64)
        h_bar = tf.linalg.lstsq(x_p, y_p)
        resid = y_p - tf.matmul(x_p, h_bar)
        no_ants.append(tf.reduce_mean(tf.square(tf.abs(resid))))

    no_est = tf.reduce_mean(tf.stack(no_ants))
    return tf.cast(no_est, tf.float32)


def create_aerial_pusch_config_from_record(pusch_record):
    """Build a pyAerial PuschConfig from a datalake FAPI record.
    """
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


class AerialNoiseEstimator:

    def __init__(self, num_rx_ant, eq_coeff_algo=1, ch_est_algo=1):
        self.cuda_stream = get_cuda_stream()
        self.ch_est_algo = ch_est_algo
        self.channel_estimator = ChannelEstimator(
            num_rx_ant=num_rx_ant,
            cuda_stream=self.cuda_stream,
            ch_est_algo=ch_est_algo,
        )
        self.noise_intf_estimator = NoiseIntfEstimator(
            num_rx_ant=num_rx_ant,
            eq_coeff_algo=eq_coeff_algo,
            cuda_stream=self.cuda_stream,
        )

    def estimate_no(self, rx_slot, slot, pusch_configs):
        """Return linear noise variance N0"""
        ch_est = self.channel_estimator.estimate(rx_slot=rx_slot, slot=slot, pusch_configs=pusch_configs)
        _, noise_var_pre_eq = self.noise_intf_estimator.estimate(rx_slot=rx_slot, channel_est=ch_est, slot=slot,pusch_configs=pusch_configs)

        noise_db = float(noise_var_pre_eq[0].get())
        no = 10 ** (noise_db / 10)
        return no

def fapi_dmrs_ports_to_sionna_dmrs_port_set(dmrs_ports_bmsk: int) -> list[int]:
    """Convert FAPI dmrsPorts bitmap to Sionna dmrs_port_set indices.
        Sionna uses 0-based indices for dmrs_port_set, while FAPI uses a bitmap representation.

    Examples:
      dmrsPorts=1 (0b1)  -> [0]
      dmrsPorts=3 (0b11) -> [0, 1]
    """
    if dmrs_ports_bmsk is None:
        return [0]
    b = int(dmrs_ports_bmsk)
    port_set = [i for i in range(12) if (b >> i) & 0x1]
    return port_set if port_set else [0]


def main():
    args = build_argument_parser().parse_args()
    database_metadata = get_database_metadata(args.db)
    database = database_metadata["database"]
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth (allocate as needed)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    print(f"Connecting to ClickHouse at localhost...")
    client = clickhouse_connect.get_client(host='localhost')

    match args.db:
        case 0: # nrx_measurement_2025_11_03
            # Filter criteria
            # RNTI: 0x43df = 17375 (Samsung Galaxy S23)
            target_rnti = 0x43df  # 17375 in decimal
            # Time range: from 20:42:37.960007 to 20:54:26.960008
            num_layers = 1

            # Train split until 2025-11-03 20:48:00.000000: 950 failed (will be split to 4 prbs)
            # Test split from 2025-11-03 20:48:00.000001: 1357 failed (will be used as full prbs)
            split_time = '2025-11-03 20:48:00.000000'
            limit = 950 # failed transmissions
        case 1: # jfloor_1st_2025_12_02
            target_rnti = 0xd9a1  # 55713 in decimal (Samsung Galaxy S23)
            num_layers = 1
            # Train split until 2025-12-02 21:54:35.000000: 970 failed
            # Test split from 2025-12-02 21:54:35.000001: 238 failed
            split_time = '2025-12-02 21:54:35.000000000' 
            limit = 970 # failed transmissions
        case 2: # drone
            target_rnti =  18601 #18601: with 2235 failures, 10217: with 3870 failures )
            num_layers = 1
            split_time = '2025-12-03 12:21:05.202000000' # 18601 is before the split_time
            limit = 2235 # failed transmissions
        case 3: # 2 layers (dummy)
            target_rnti = 0xfee0 #65248
            num_layers = 2
            split_time = "2026-04-30 12:21:05.202000000" # dummy
            limit = 1000 # dummy
        case 4:
            target_rnti = 0x4e1b #19995
            num_layers = 1
            split_time = "2026-04-28 13:03:00.000000000" # 24665 samples out of 49199 before split (2458 fail)
            limit = 1000 # dummy
        case 5:
            target_rnti = 0xf177 #61815
            num_layers = 2
            split_time = "2026-04-28 12:48:30.000000000" # 14245 samples out of 31187 before split (1656 fail)
            limit = 1000 # dummy
        case 6: # j61s 2 layers 2026-05-12
            num_layers = 2
            split_time = '2026-05-12 11:53:30' # 55470 samples out of 94800 before split (3707 fail)
            limit = 1000 # dummy
        case 7: # j61s 1 layer 2026-05-12
            num_layers = 1
            split_time = '2026-05-12 12:22:00' # 55191 samples out of 112800 before split (4015 fail)
            limit = 1000 # dummy
        case 8: # 2 layer port 0, 2 cdmnodata = 2
            num_layers = 2
            split_time = '2026-06-04 13:27:17.372000000' # 35578 samples out of 55600 (2597 fail)
            limit = 1000 # dummy
        case 9:
            num_layers = 1
            split_time = '2026-06-04 12:59:00.372000000' # 50712 samples out of 60691 (2505 fail)
            limit = 1000 # dummy
        case 10: # 2 layer jFloor
            num_layers = 2
            split_time = '2026-06-10 14:47:00' # 46338 samples out of 77084 (4360 fail)
            limit = 1000 # dummy
        case 11: # 1 layer jFloor
            num_layers = 1
            split_time = '2026-06-10 15:04:00' # 69966 samples out of 97335 (4612 fail)
            limit = 1000 # dummy

    # limit = int(10*np.min(limit, int(np.ceil(args.limit * 0.1))))
    # Balanced sampling: 10% failed, 90% successful
    query = f"""select * from (
                (select * from export_fapi_{database}
                    where rbSize = 273
                    and NrOfSymbols = 13
                    and mcsIndex > 5
                    and qamModOrder = 4
                    and CellId = 51
                    and nrOfLayers = {num_layers}
                    and NrOfSymbols = 13
                    and rvIndex = 0
                    and tbCrcFail = 1
                    and TsTaiNs < toDateTime64('{split_time}', 9)
                    order by rand()
                    limit CEIL({args.limit} * 0.1)
                )
                union all
                (
                    select * 
                    from export_fapi_{database}
                    where rbSize = 273
                    and NrOfSymbols = 13
                    and qamModOrder = 4
                    and CellId = 51
                    and nrOfLayers = {num_layers}
                    and NrOfSymbols = 13
                    and rvIndex = 0
                    and tbCrcFail = 0
                    and TsTaiNs < toDateTime64('{split_time}', 9)
                    order by rand()
                    limit CEIL({args.limit} * 0.9)
                )
            ) as combined
            order by rand()
            """



    print(f"Querying database: {query}")
    pusch_records = client.query_df(query)
    
    print(f"Found {len(pusch_records)} records to process")

    num_prbs = 273
    num_rx_ant = 4
    output_path = f"../../finetuning_datasets/{database_metadata['tfrecord_filename']}"
    no_mode = 'pyAerial'

    # Counters for summary
    total_processed = 0
    harq_retransmissions_used = 0
    skipped_no_retransmission = 0

    noise_estimator = AerialNoiseEstimator(num_rx_ant=num_rx_ant, eq_coeff_algo=1, ch_est_algo=1)

    # Cycle through records repeatedly until we reach the desired limit
    with tf.io.TFRecordWriter(str(output_path)) as writer:
    # for _ in range(10): # placeholder for writer
        record_iterator = cycle(pusch_records.iterrows())
        pbar = tqdm(total=args.limit, desc="Processing records")

        for index, pusch_record in record_iterator:
            if total_processed >= args.limit:
                break

            # Query for FH data
            query = f"""select TsTaiNs,CellId,fhData from export_fh_{database} 
                        where TsTaiNs == toDateTime64('{pusch_record.TsTaiNs.timestamp()}',9)
                        and CellId == {args.cell_id}"""
            fh = client.query_df(query)
            if fh.empty:
                print(f"Warning: No FH data found for record {index} at time {pusch_record.TsTaiNs}. Skipping.")
                continue
            numCells = fh.index.size

            if numCells > 1: # hopefully only single cell
                print(f"Warning: More than one FH cell found for record {index} at time {pusch_record.TsTaiNs}. Continuing.")
                # del fh
                continue
            cellNum = 0

            retransmission_ts = None
            if pusch_record.tbCrcFail == 0: # transmission successful
                if pusch_record.pduData is None:
                    print(f"Skipping record {index}: pduData is None for successful transmission")
                    continue
                tb_input = np.array(pusch_record.pduData)
            else: # transmission failed, need to find successful retransmission
                tb_input, retransmission_ts = find_successful_retransmission(client, database, pusch_record)
                if tb_input is None:
                    skipped_no_retransmission += 1
                    continue
                harq_retransmissions_used += 1

            ###### SIONNA TRANSMITTER TO CREATE GRID ##############
            cyclic_prefix = 'normal' if pusch_record.CyclicPrefix == 0 else 'extended'
            config_type = 1 if pusch_record.dmrsConfigType == 0 else 2 # Careful with Sionna

            carrier_config = CarrierConfig(
                        n_cell_id=pusch_record.CellId, # same as DMRS scrambling ID if n_id is not provided on DMRS Config
                        cyclic_prefix=cyclic_prefix,
                        subcarrier_spacing=int(30.0), # in kHz
                        n_size_grid=pusch_record.rbSize,
                        n_start_grid=pusch_record.rbStart,
                        slot_number=int(pusch_record.Slot),
                        frame_number=pusch_record.SFN)

            dmrs_port_set = fapi_dmrs_ports_to_sionna_dmrs_port_set(pusch_record.dmrsPorts)
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
                # num_antenna_ports=1,
                num_layers=pusch_record.nrOfLayers,
                symbol_allocation=[pusch_record.StartSymbolIndex, pusch_record.NrOfSymbols],
                mapping_type='B',
                data_scid=pusch_record.dataScramblingId, # same as n_cell_id
                n_rnti=pusch_record.rnti,
                )
            
            transmitter = PUSCHTransmitter(pc, return_bits=False, output_domain="freq")

            rg = transmitter._resource_grid
            chest = LSChannelEstimator(rg, interpolation_type='lin')

            if (transmitter._tb_size != 8*(pusch_record.TBSize)):
                print(f"Warning: TB size mismatch for record {index}: expected {transmitter._tb_size}, got {pusch_record.TBSize}")
                # del fh
                continue
            
            fh_samp = (np.array(fh['fhData'].iloc[cellNum], dtype=np.int16).view(np.float16)).astype(np.float32)
            # Full-slot FH grid [num_sc, 14, num_rx_ant]
            rx_slot_full = np.swapaxes(fh_samp.view(np.complex64).reshape(4, 14, 273 * 12), 2, 0)
            rx_slot = rx_slot_full[:,:pusch_record.NrOfSymbols,:]

            # channel_estimator input: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size]
            _rx_slot = tf.expand_dims(tf.transpose(rx_slot, perm=[2,1,0]), axis=0)  # add batch dim
            _rx_slot = tf.expand_dims(_rx_slot, axis=1)  # add num_rx dim
            # create dummy no of shape  [batch_size, num_rx, num_rx_ant]
            no_dummy = tf.ones((_rx_slot.shape[0], _rx_slot.shape[1], _rx_slot.shape[2]), dtype=tf.float32)
            num_tx = int(pusch_record.nrOfLayers)
            if len(dmrs_port_set) != num_tx:
                print(
                    f"Skipping record {index}: nrOfLayers ({num_tx}) != "
                    f"len(dmrs_port_set) ({len(dmrs_port_set)}); dmrsPorts={pusch_record.dmrsPorts}"
                )
                continue
            h_hat, err_var = chest([_rx_slot, no_dummy])

            if no_mode == 'Sionna':
                no_est = estimate_no_from_pilots(_rx_slot, transmitter)
            elif no_mode == 'pyAerial':
                aerial_pusch_configs = create_aerial_pusch_config_from_record(pusch_record)
                no_est = tf.constant(
                    noise_estimator.estimate_no(
                        cp.asarray(rx_slot_full, dtype=cp.complex64),
                        int(pusch_record.Slot),
                        aerial_pusch_configs),
                    dtype=tf.float32)
            else:
                print(f"Warning: Invalid no_mode value: {no_mode}")
                continue
            
            # [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
            h_hat = h_hat[:,0,:,0,:num_tx]
            h_hat = tf.transpose(h_hat, [0, 2, 4, 3, 1])
            h_hat = tf.concat([tf.math.real(h_hat), tf.math.imag(h_hat)], axis=-1)


            # Map bits to resource grid
            bit_grid = map_bits_to_resource_grid(tb_input, transmitter)
            tb_bits = np.unpackbits(np.asarray(tb_input, dtype=np.uint8)).astype(np.int8)
            n_tb = int(transmitter._tb_size)
            if tb_bits.size != n_tb:
                print(f"Warning: TB size mismatch for record {index}: expected {n_tb}, got {tb_bits.size}")
                continue
            b_info = tb_bits.reshape(1, 1, n_tb)  # [batch, num_tx, tb_size]

            # full PRB -> no-op
            subcarriers_per_prb = 12
            chunk_size = num_prbs * subcarriers_per_prb
            max_start_prb = 273 - num_prbs  
            start_prb = np.random.randint(0, max_start_prb + 1)
            start_sc = start_prb * subcarriers_per_prb
            end_sc = start_sc + chunk_size

            # Slice tensors along subcarrier dimension
            _rx_slot = _rx_slot[:, :, :, :, start_sc:end_sc]  # [batch, num_rx, num_rx_ant, ofdm_symbols, 48]
            h_hat = h_hat[:, :, start_sc:end_sc, :, :]  # [batch, num_tx, 48, ofdm_symbols, 2*num_rx_ant]
            bit_grid = bit_grid[:, :, :, start_sc:end_sc, :]  # [batch, num_tx, 48, ofdm_symbols, bits_per_symbol]

            tf_example = serialize_example(
                y=_rx_slot, h=h_hat, b=bit_grid, no=no_est, b_info=b_info)
            writer.write(tf_example)
            total_processed += 1
            pbar.update(1)
                
        pbar.close()

    print(f"\nProcessing complete. TFRecord saved to: {output_path}")
    print(f"Summary:")
    print(f"  Total records processed: {total_processed}")
    print(f"  HARQ retransmissions used: {harq_retransmissions_used}")
    print(f"  Records skipped (no successful retransmission): {skipped_no_retransmission}")


if __name__ == "__main__":
    main()
