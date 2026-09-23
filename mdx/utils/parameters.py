
# Modified to support MDX & Extra features
# Mahdi Abdollahpour (mahdi.abdollahpour@unibo.it)
# 2025



# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# reads in the configuration file and initialized all relevant system components
# This allows to train and evaluate different system configurations on the same # server and simplifies logging of the training process.

import numpy as np
import configparser
import tensorflow as tf
from os.path import exists
from sionna.nr import PUSCHConfig, PUSCHDMRSConfig, TBConfig, CarrierConfig, PUSCHTransmitter, PUSCHPilotPattern
from sionna.channel.tr38901 import PanelArray, UMi, TDL, UMa
from sionna.mimo import StreamManagement
from sionna.channel import OFDMChannel, AWGN
from .channel_models import DoubleTDLChannel, DatasetChannel, NTDLChannel, DataLakeChannel


class Parameters:
    r"""
    Simulation parameters

    Parameters
    ----------
    config_name : str
        name of the config file.

    system : str
        Receiver algorithm to be used.Must be one of:
        * "mdx" : Model driven
        * "nrx" : Neural receiver
        * "baseline_lmmse_kbest" : LMMSE estimation and K-Best detection
        * "baseline_perf_csi_kbest" : perfect CSI and K-Best detection
        * "baseline_lmmse_lmmse" : LMMSE estimation and LMMSE equalization
        * "baseline_lsnn_lmmse" : LS estimation/nn interpolation and LMMSE equalization
        * "baseline_lsnn_kbest" : LS estimation/nn interpolation and K-Best equalization
        * "dummy" : stops after parameter import. Can be used only to parse the
        config.

    training: bool, False,
        If True, training parameters are loaded. Otherwise, the evaluation
        parameters are used.

    verbose: bool, False
        If True, additional information is printed during init.

    compute_cov: bool, False
        If True, the UMi channel model is loaded automatically to avoid
        overfitting to TDL models.

    num_tx_eval: int or None
        If provided, the max number of users is limited to ``num_tx_eval``.
        For this, the first DMRS ports are selected.
    """
    def __init__(self, config_name, system="dummy", training=False, verbose=False, compute_cov=False, num_tx_eval=None):

        # check input for consistency
        assert isinstance(verbose, bool), "verbose must be bool."
        assert isinstance(training, bool), "training must be bool."
        assert isinstance(config_name, str), "config_name must be str."
        assert isinstance(system, str), "system must be str."
        assert isinstance(compute_cov, bool), "compute_cov must be bool."

        self.system = system

        ###################################
        ##### Load configuration file #####
        ###################################

        # create parser object and read config file
        fn = f'../config/{config_name}'
        if exists(fn):
            config = configparser.RawConfigParser()
            # automatically add fileformat if needed
            config_name.replace(".cfg","") + ".cfg"
            config.read(fn)
        else:
            raise FileNotFoundError("Unknown config file.")

        # and import all parameters as attributes
        self.config_str = ""
        for section in config.sections():
            s = f"\n---- {section} ----- "
            self.config_str += s + "<br />" # add linebreak for Tensorboard
            if verbose:
                print(s)
            for option in config.options(section):
                setattr(self, f"{option}", eval(config.get(section,option)))
                s = f"{option}: {eval(config.get(section,option))}"
                self.config_str += s + "<br />" # add linebreak for Tensorboard
                if verbose:
                    print(s)


       
        self._training = training
        self._verbose = verbose
        self._compute_cov = compute_cov
        self._num_tx_eval = num_tx_eval
        self.re_init(training, verbose, compute_cov, num_tx_eval)

    def re_init(self,training=None, verbose=None, compute_cov=None, num_tx_eval=None,
                n_size_bwp_eval=None, batch_size_eval=None, batch_size_eval_small=None,
                max_ut_velocity_eval=None, min_ut_velocity_eval=None,
                channel_norm_eval=None,
                channel_type_eval=None, tdl_models=None): # Just a fast solution



        if training is None:
            training = self._training
        if verbose is None:
            verbose = self._verbose
        if compute_cov is None:
            compute_cov = self._compute_cov
        if num_tx_eval is None:
            num_tx_eval = self._num_tx_eval

        if n_size_bwp_eval is not None:
            self.n_size_bwp_eval = n_size_bwp_eval
        if batch_size_eval is not None:
            self.batch_size_eval = batch_size_eval
        if batch_size_eval_small is not None:
            self.batch_size_eval_small = batch_size_eval_small            
        if max_ut_velocity_eval is not None:
            self.max_ut_velocity_eval = max_ut_velocity_eval
        if min_ut_velocity_eval is not None:
            self.min_ut_velocity_eval = min_ut_velocity_eval
        if channel_norm_eval is not None:
            self.channel_norm_eval = channel_norm_eval


        # Overwrite channel and PRBs in inference mode with "eval" parameters
        # This allows to configure different parameters during training and
        # evaluation.



        if not training:
            self.channel_type = self.channel_type_eval
            self.n_size_bwp = self.n_size_bwp_eval
            self.max_ut_velocity = self.max_ut_velocity_eval
            self.min_ut_velocity = self.min_ut_velocity_eval
            self.channel_norm = self.channel_norm_eval
            self.cfo_offset_ppm = self.cfo_offset_ppm_eval
            self.tfrecord_filename = self.tfrecord_filename_eval
            if self.channel_type == "Dataset":
                self.random_subsampling = self.random_subsampling_eval

        # load only config parameters and return without initializing the rest
        # of the system
        if self.system == "dummy":
            return

        #####################################
        ##### Init PUSCH configurations #####
        #####################################

        # init PUSCHConfig
        carrier_config = CarrierConfig(
                n_cell_id=self.n_cell_id,
                cyclic_prefix=self.cyclic_prefix,
                subcarrier_spacing=int(self.subcarrier_spacing/1e3), # in kHz
                n_size_grid=self.n_size_bwp,
                n_start_grid=self.n_start_grid,
                slot_number=self.slot_number,
                frame_number=self.frame_number)

        # init DMRSConfig
        pusch_dmrs_config=PUSCHDMRSConfig(
                config_type=self.dmrs_config_type,
                type_a_position=self.dmrs_type_a_position,
                additional_position=self.dmrs_additional_position,
                length=self.dmrs_length,
                dmrs_port_set=self.dmrs_port_sets[0], # first user
                n_scid=self.n_scid,
            num_cdm_groups_without_data=self.num_cdm_groups_without_data)

        mcs_list = self.mcs_index
        # generate pusch configs for all MCSs
        self.pusch_configs = []   # self.pusch_configs[MCS_CONFIG][N_UE]
        for mcs_list_idx in range(len(mcs_list)):
            self.pusch_configs.append([])
            mcs_index = mcs_list[mcs_list_idx]
            # init TBConfig
            tb_config = TBConfig(
                    mcs_index=mcs_index,
                    mcs_table=self.mcs_table,
                    channel_type="PUSCH")
                    #n_id=self.n_ids[0])
            if self.num_antenna_ports >1:
                # first user PUSCH config
                pc = PUSCHConfig(
                        carrier_config=carrier_config,
                        pusch_dmrs_config=pusch_dmrs_config,
                        tb_config=tb_config,
                        num_antenna_ports = self.num_antenna_ports,
                        precoding = self.precoding,
                        symbol_allocation = self.symbol_allocation,
                        tpmi = self.tpmi,
                        mapping_type=self.dmrs_mapping_type,)
            else:
                # first user PUSCH config
                pc = PUSCHConfig(
                        carrier_config=carrier_config,
                        pusch_dmrs_config=pusch_dmrs_config,
                        tb_config=tb_config,
                        num_antenna_ports = self.num_antenna_ports,
                        # precoding = self.precoding,
                        symbol_allocation = self.symbol_allocation,
                        # tpmi = self.tpmi,
                        mapping_type=self.dmrs_mapping_type,)
                        
            # clone new PUSCHConfig for each additional user
            for idx,_ in enumerate(self.dmrs_port_sets):
                p = pc.clone() # generate new PUSCHConfig
                # set user specific parts
                p.dmrs.dmrs_port_set = self.dmrs_port_sets[idx]
                # The following parameters are derived from default.
                # Comment lines if specific configuration is not required.
                # p.n_id = self.n_ids[idx]
                # p.dmrs.n_id = self.dmrs_nid[idx]
                # p.n_rnti = self.n_rntis[idx]
                self.pusch_configs[mcs_list_idx].append(p)

        ##############################
        ##### Consistency checks #####
        ##############################

        # after training we can only reduce the number of iterations
        assert self.num_nrx_iter_eval<=self.num_nrx_iter, \
            "num_nrx_iter_eval must be smaller or equal num_nrx_iter."

        # for the evaluation, only activate num_tx_eval configs
        if not training:
                # overwrite num_tx_eval if explicitly provided:
            if num_tx_eval is not None:
                num_tx_eval = num_tx_eval
            else: # if not provided use all available port sets
                num_tx_eval = len(self.dmrs_port_sets)
            self.max_num_tx = num_tx_eval # non-varying users for evaluation
            self.min_num_tx = num_tx_eval # non-varying users for evaluation

        for mcs_list_idx in range(len(mcs_list)):
            self.pusch_configs[mcs_list_idx] = self.pusch_configs[mcs_list_idx][:self.max_num_tx]
        print(f"Evaluating the first {self.max_num_tx} port sets.")

        ##################################
        ##### Configure Transmitters #####
        ##################################

        # Generate and store DMRS for all slot numbers
        self.pilots = []
        for slot_num in range(carrier_config.num_slots_per_frame):
            for pcs in self.pusch_configs:
                for pc in pcs:
                    pc.carrier.slot_number = slot_num
            # only generate pilot pattern for first MCS's PUSCH config, as
            # pilots are independent from MCS index
            pilot_pattern = PUSCHPilotPattern(self.pusch_configs[0])
            self.pilots.append(pilot_pattern.pilots)
        self.pilots = tf.stack(self.pilots, axis=0)
        self.pilots = tf.constant(self.pilots)
        for pcs in self.pusch_configs:
            for pc in pcs:
                pc.carrier.slot_number = self.slot_number

        # transmitter is a list of PUSCHTransmitters, one for each MCS
        self.transmitters = []
        for mcs_list_idx in range(len(mcs_list)):
            # and init transmitter
            self.transmitters.append(
                PUSCHTransmitter(
                            self.pusch_configs[mcs_list_idx],
                            return_bits=False,
                            output_domain="freq",
                            verbose=self.verbose))

            # support end-to-end learning / custom constellations
            # see https://arxiv.org/pdf/2009.05261 for details
            if self.custom_constellation: # trainable constellations
                print("Activating trainable custom constellations.")
                self.transmitters[mcs_list_idx]._mapper.constellation.trainable = True
            # Center constellations. This could be also deactivated for more
            # degrees of freedom.
            self.transmitters[mcs_list_idx]._mapper.constellation.center = True

        # chest will fail if we use explicit masking of pilots.
        if self.mask_pilots and self.initial_chest in ("ls", "nn"):
            print("Warning: initial_chest will fail with masked pilots.")

        # StreamManagement required for KBestDetector
        self.sm = StreamManagement(np.ones([1, self.max_num_tx], int), 1)

        num_mcs = len(mcs_list)
        self.points_list = [self.transmitters[i]._mapper.constellation.points for i in range(num_mcs)]  # List of tensors for each transmitter's points

        # ##############################
        # ##### Initialize Channel #####
        # ##############################
        self.tdl_models = ["A"]
        self._pc = pc

        

        if tdl_models is not None:
            self.tdl_models = tdl_models
        if channel_type_eval is not None:
            self.initialize_channel(training=training, compute_cov=compute_cov, channel_type_eval=channel_type_eval,
                                    tdl_models=tdl_models)
        else:
            self.initialize_channel(training=training, compute_cov=compute_cov)

        ##############################
        ##### Positional Encoding #####
        ##############################

        if hasattr(self, 'pe_type'):
            if self.system=="mdx":
                num_ofdm_symbols = self.transmitters[0].resource_grid.num_ofdm_symbols
                if self.pe_type == 0: # NRX default
                    self.pe_d = 2
                if self.pe_type == 1: # sin coding
                    seq_len = 12*num_ofdm_symbols
                    d = self.pe_d
                    n = self.pe_n

                    PE = np.zeros((seq_len, d))
                    for k in range(seq_len):
                        for i in np.arange(int(d/2)):
                            denominator = np.power(n, 2*i/d)
                            PE[k, 2*i] = np.sin(k/denominator)
                            PE[k, 2*i+1] = np.cos(k/denominator)
                    
                    PE = tf.convert_to_tensor(PE, dtype=tf.float32)
                    self.PE = tf.reshape(PE,[12,num_ofdm_symbols,d])
                if self.pe_type==2 or self.pe_type==3: # PRB coding or NRX default + PRB coding
                    rows = tf.cast(tf.range(12, dtype=tf.float32) / 11.0, dtype=tf.float32)  # 0 to 11, normalized to [0, 1]
                    cols = tf.cast(tf.range(num_ofdm_symbols, dtype=tf.float32) / (num_ofdm_symbols - 1.0), dtype=tf.float32)  # 0 to 13, normalized to [0, 1]

                    A_0 = tf.tile(tf.expand_dims(rows, axis=1), [1, num_ofdm_symbols])  
                    A_1 = tf.tile(tf.expand_dims(cols, axis=0), [12, 1]) 

                    # pe [12, num_ofdm_symbols, 2]
                    self.PE = tf.stack([A_0, A_1], axis=-1)
                    self.pe_d = 2
                if self.pe_type==3: # NRX default + PRB coding
                    self.pe_d = self.pe_d + 2

        ################
        ##### MISC #####
        ################

        # Load covariance matrices
        if self.system in ("baseline_lmmse_kbest", "baseline_lmmse_lmmse"):

            # test if files exist
            fn = f'../weights/{self.label}_time_cov_mat.npy'
            if not exists(fn):
                raise FileNotFoundError("time_cov_mat.npy not found. " \
                    "Please run compute_cov_mat.py for given config first.")

            self.space_cov_mat = tf.cast(np.load(
                        f'../weights/{self.label}_space_cov_mat.npy'),
                                                tf.complex64)
            self.time_cov_mat = tf.cast(np.load(
                        f'../weights/{self.label}_time_cov_mat.npy'),
                                            tf.complex64)
            self.freq_cov_mat = tf.cast(np.load(
                        f'../weights/{self.label}_freq_cov_mat.npy'),
                                            tf.complex64)
        
    def initialize_channel(self, training=False, channel_type_eval=None, tdl_models=None, compute_cov=False,
                                delay_spread_min=10,   # in nano seconds
                                delay_spread_max=300,  # in nano seconds
                                doppler_shift_max=325  # Hz
                                ):
        ##############################
        ##### Initialize Channel #####
        ##############################
        if channel_type_eval is not None:
            self.channel_type = channel_type_eval
        if tdl_models is not None:
            self.tdl_models = tdl_models
        # always use UMi to calculate covariance matrix
        if compute_cov:
            if not self.channel_type in ("UMi", "UMa"): # use UMa if selected
                print("Setting channel type to UMi for covariance computation.")
                self.channel_type = "UMi"

        # Sanity check
        if self.channel_type in ("DoubleTDLlow","DoubleTDLmedium",
                                 "DoubleTDLhigh") and self.max_num_tx==1:
                print("Warning: SelectedDoubleTDL model only defined for 2 "\
                      "users. Selecting TDL-B100 instead.")
                self.channel_type = "TDL-B100"

        # Initialize channel
        # Remark: new channel models can be added here
        if self.channel_type in ("UMi", "UMa"):
            if self.num_rx_antennas==1: # ignore polarization for single antenna
                print("Using vertical polarization for single antenna setup.")
                num_cols_per_panel = 1
                num_rows_per_panel = 1
                polarization = "single"
                polarization_type = 'V'
            else:
                # we use a ULA array to be aligned with TDL models
                num_cols_per_panel = self.num_rx_antennas//2
                num_rows_per_panel = 1
                polarization = "dual"
                polarization_type = 'cross'

            bs_array = PanelArray(num_rows_per_panel = num_rows_per_panel,
                                  num_cols_per_panel = num_cols_per_panel,
                                  polarization = polarization,
                                  polarization_type  = polarization_type,
                                  antenna_pattern = '38.901',
                                  carrier_frequency = self.carrier_frequency)

            ut_array = PanelArray(num_rows_per_panel = 1,
                                  num_cols_per_panel = self._pc.num_antenna_ports,
                                  polarization = 'single',
                                  polarization_type = 'V',
                                  antenna_pattern = 'omni',
                                  carrier_frequency = self.carrier_frequency)

            if self.channel_type == "UMi":
                self.channel_model = UMi(
                                carrier_frequency=self.carrier_frequency,
                                o2i_model = 'low',
                                bs_array = bs_array,
                                ut_array = ut_array,
                                direction = 'uplink',
                                enable_pathloss = False,
                                enable_shadow_fading = False)
            else: # UMa
                self.channel_model = UMa(
                                carrier_frequency=self.carrier_frequency,
                                o2i_model = 'low',
                                bs_array = bs_array,
                                ut_array = ut_array,
                                direction = 'uplink',
                                enable_pathloss = False,
                                enable_shadow_fading = False)

            self.channel = OFDMChannel(
                    channel_model=self.channel_model,
                    resource_grid=self.transmitters[0]._resource_grid,          # resource grid is independent of MCS
                    add_awgn=True,
                    normalize_channel=self.channel_norm,
                    return_channel=True)

        elif self.channel_type == "TDL-B100":
            tdl = TDL(model="B100",
                      delay_spread=100e-9,
                      carrier_frequency=self.carrier_frequency,
                      min_speed=self.min_ut_velocity,
                      max_speed=self.max_ut_velocity,
                      num_tx_ant=self._pc.num_antenna_ports,
                      num_rx_ant=self.num_rx_antennas)
            self.channel = OFDMChannel(tdl,
                                       self.transmitters[0].resource_grid,      # resource grid is independent of MCS
                                       add_awgn=True,
                                       normalize_channel=self.channel_norm,
                                       return_channel=True)
        elif self.channel_type == "TDL-C300":
            tdl = TDL(model="C300",
                      delay_spread=300e-9,
                      carrier_frequency=self.carrier_frequency,
                      min_speed=self.min_ut_velocity,
                      max_speed=self.max_ut_velocity,
                      num_tx_ant=self._pc.num_antenna_ports,
                      num_rx_ant=self.num_rx_antennas)
            self.channel = OFDMChannel(tdl,
                                       self.transmitters[0].resource_grid,      # resource grid is independent of MCS
                                       add_awgn=True,
                                       normalize_channel=self.channel_norm,
                                       return_channel=True)
        # DoubleTDL for evaluation
        elif self.channel_type == "DoubleTDLlow":
            self.channel = DoubleTDLChannel(self.carrier_frequency,
                                    self.transmitters[0].resource_grid,         # resource grid is independent of MCS
                                    correlation="low",
                                    num_tx_ant=self._pc.num_antenna_ports,
                                    num_rx_ant = self.num_rx_antennas,
                                    norm_channel=self.channel_norm)
        # DoubleTDL for evaluation
        elif self.channel_type == "DoubleTDLmedium":
            self.channel= DoubleTDLChannel(self.carrier_frequency,
                                    self.transmitters[0].resource_grid,         # resource grid is independent of MCS
                                    correlation="medium",
                                    num_tx_ant=self._pc.num_antenna_ports,
                                    norm_channel=self.channel_norm)
        # DoubleTDL for evaluation
        elif self.channel_type == "DoubleTDLhigh":
            self.channel = DoubleTDLChannel(self.carrier_frequency,
                                    self.transmitters[0].resource_grid,         # resource grid is independent of MCS
                                    correlation="high",
                                    num_tx_ant=self._pc.num_antenna_ports,
                                    norm_channel=self.channel_norm)
        # NTDL for evaluation
        elif self.channel_type == "NTDLlow":
            # tdl_models=["A"]
            # if hasattr(self, 'tdl_models'):
            #     tdl_models = self.tdl_models
            self.channel = NTDLChannel(carrier_frequency=self.carrier_frequency,
                                    resource_grid=self.transmitters[0].resource_grid,        # resource grid is independent of MCS
                                    correlation="low",
                                    num_tx_ant=self._pc.num_antenna_ports,
                                    num_rx_ant = self.num_rx_antennas,
                                    max_num_tx = self.max_num_tx,
                                    norm_channel=self.channel_norm,
                                    tdl_models=self.tdl_models,
                                    delay_spread_min=delay_spread_min,   # in nano seconds
                                    delay_spread_max=delay_spread_max,  # in nano seconds
                                    doppler_shift_max=doppler_shift_max  # Hz
                                    )


        elif self.channel_type == "AWGN":
            self.channel = AWGN()


        elif self.channel_type == "Dataset":
            channel_model = DatasetChannel("../data/" + self.tfrecord_filename,
                                    max_num_examples=-1, # loads entire dataset
                                    training=training,
                                    num_tx=self.max_num_tx,
                                    random_subsampling=self.random_subsampling,
                                    )
            self.channel = OFDMChannel(channel_model,
                                       self.transmitters[0].resource_grid,      # resource grid is independent of MCS
                                       add_awgn=True,
                                       normalize_channel=self.channel_norm,
                                       return_channel=True)

        elif self.channel_type == "Datalake":
            # DataLake channel - no channel model needed as data is pre-loaded
            self.channel = None  # No channel model for DataLake
            
            # Create DataLake channel instance
            # Resource grid shape: [num_subcarriers, num_ofdm_symbols]
            num_subcarriers = self.n_size_bwp * 12  # PRBs * 12 subcarriers per PRB
            num_ofdm_symbols = 13  # From symbol_allocation = [0, 13]
            resource_grid_shape = (num_subcarriers, num_ofdm_symbols)
            
            self.datalake_channel = DataLakeChannel(
                tf_fn=self.datalake_tf_fn,
                n_size_bwp = self.n_size_bwp,
                max_num_tx=self.max_num_tx,
                resource_grid_shape=resource_grid_shape,
                num_rx_ant=self.num_rx_antennas,
                training=training,
            )

        else:
            raise ValueError("Unknown Channel type.")
        
        # Hardware impairments
        assert self.cfo_offset_ppm==0
        self.frequency_offset = None
