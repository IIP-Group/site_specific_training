#!/usr/bin/python3

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

# evaluate BLER of NRX and baseline systems
# results are saved in files and can be visualized with the corresponding
# jupyter notebooks

####################################################################
# Parse args
####################################################################

import argparse

parser = argparse.ArgumentParser()

# the config defines the sys parameters
parser.add_argument("-config_name", help="config filename", type=str)
# limits the number of target of block errors during the simulation
parser.add_argument("-num_target_block_errors",
                    help="Number of target block errors", type=int, default=500)
parser.add_argument("-max_mc_iter",
                    help="Maximum Monte Carlo iterations",
                    type=int, default=500)
parser.add_argument("-target_bler",
                help="Early stop BLER simulations at a specific target BLER",
                type=float, default=0.001)
parser.add_argument("-num_cov_samples",
                    help="Number of samples for covariance generation", type=int, default=1000000)
parser.add_argument("-gpu", help="GPU to use", type=int, default=0)
parser.add_argument("-num_tx_eval",
                    help="Number of active users",
                    type=int, nargs='+', default=-1)
parser.add_argument("-mcs_arr_eval_idx",
                    help="Select the MCS array index for evaluation. Use -1 to evaluate all MCSs.", type=int, default=-1)
parser.add_argument("-eval_nrx_only", help="Only evaluate the NN",
                    action="store_true", default=False)
parser.add_argument("-debug", help="Set debugging configuration", action="store_true", default=False)

parser.add_argument("-all_gpus", help="distribution on all GPUs", action="store_true", default=False)

parser.add_argument("-dont_load", help="will not load previous evaluation.", action="store_true", default=False)

parser.add_argument("-forced_transfer", help="Transfer weights specified in cfg file.", action="store_true", default=False)


# Eval
parser.add_argument("-snr_db_eval_min",type=float, default=-5.)
parser.add_argument("-snr_db_eval_max",type=float, default=5.)
parser.add_argument("-snr_db_eval_stepsize",type=float, default=1.)
parser.add_argument("-max_ut_velocity_eval",type=float, default=34.)
parser.add_argument("-channel_type_eval",type=str, default="NTDLlow")
parser.add_argument("-tdl_models",help="-tdl_models A B C ...",type=str, nargs='+', default=["A"])
parser.add_argument("-n_size_bwp_eval",type=int, default=132)
parser.add_argument("-batch_size_eval",type=int, default=30)
parser.add_argument("-batch_size_eval_small",type=int, default=3)
parser.add_argument("-dir",type=str, help="directory to save results.", default="../results/")
parser.add_argument("-name_suffix",type=str, help="add to results name", default="")
parser.add_argument("-mcs_index",help="-mcs_index 9 14 19",type=int, nargs='+', default=[-1])

parser.add_argument("-methods",  nargs='+', help="methods list: mdx, nrx, baseline_lslin_lmmse, baseline_lslin_kbest, baseline_lmmse_kbest, baseline_perf_csi_lmmse, baseline_lmmse_lmmse, baseline_perf_csi_kbest",type=str, default="baseline_lslin_lmmse")

parser.add_argument("-snr_dbs",  nargs='+', help="List of SNR values in dB (e.g., -5 -4 0 10)",type=float, default=[])


# Parse all arguments
args = parser.parse_args()

config_name = args.config_name
max_mc_iter = args.max_mc_iter
num_target_block_errors = args.num_target_block_errors
eval_nrx_only = args.eval_nrx_only
num_cov_samples = args.num_cov_samples
gpu = args.gpu
target_bler = args.target_bler
num_tx_eval = args.num_tx_eval
mcs_arr_eval_idx = args.mcs_arr_eval_idx
dont_load = args.dont_load
methods = args.methods
res_dir = args.dir
name_suffix = args.name_suffix
mcs_index = args.mcs_index
snr_dbs = args.snr_dbs
forced_transfer = args.forced_transfer

if not args.all_gpus:
    distribute = None # use "all" to distribute over multiple GPUs
else:
    distribute = "all" # use "all" to distribute over multiple GPUs
####################################################################
# Imports and GPU configuration
####################################################################

import os
# Avoid warnings from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')

if distribute != "all":
    try:
        tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
        print('Only GPU number', args.gpu, 'used.')
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
    except RuntimeError as e:
        print(f"error\n:{e}")


import sys
sys.path.append('../')

import sionna as sn
# from sionna.utils import sim_ber
from utils import E2E_Model, Parameters, load_weights
import numpy as np
import pickle
from os.path import exists

from utils import transfer_weights_from_h5

if args.debug:
    tf.config.run_functions_eagerly(True)

from sim_ber import sim_ber

##############
# Save Results
#############
import time

def save_results(results_filename, data, max_retries=3, wait_time=5):
    """
    Tries to save data to a file using pickle with retries.

    Args:
        results_filename (str): The filename to save the data.
        data (any): The data to be pickled and saved.
        max_retries (int, optional): Maximum number of retries. Default is 3.
        wait_time (int, optional): Time to wait before retrying (in seconds). Default is 5.
    """
    for attempt in range(max_retries):
        try:
            with open(results_filename, "wb") as f:
                pickle.dump(data, f)
            time.sleep(5)
            print("File saved successfully.")
            return True  # Success
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("All attempts failed.")
                return False  # Failure



##################################################################
# Run evaluations
##################################################################
if os.path.isdir(res_dir):
    print(f"Directory '{res_dir}' exists")
else:
    print(f"Directory '{res_dir}' does not exist")

# dummy parameters to access filename and to load results
sys_parameters = Parameters(config_name,
                            training=True,
                            system='dummy') # dummy system only to load config


sys_parameters.snr_db_eval_min = args.snr_db_eval_min
sys_parameters.snr_db_eval_max = args.snr_db_eval_max
sys_parameters.snr_db_eval_stepsize = args.snr_db_eval_stepsize

# Reset evaluation properties
def set_eval_params(sys_parameters,args):
    print(f"setting the evaluation channel model to:{args.channel_type_eval} {args.tdl_models}")
    print(f"setting n_size_bwp_eval to {args.n_size_bwp_eval}")

    sys_parameters.re_init(n_size_bwp_eval=args.n_size_bwp_eval,
                           batch_size_eval=args.batch_size_eval,
                           batch_size_eval_small=args.batch_size_eval_small,
                           max_ut_velocity_eval=args.max_ut_velocity_eval,
                           channel_type_eval=args.channel_type_eval,tdl_models=args.tdl_models)

    return sys_parameters

sys_parameters = set_eval_params(sys_parameters,args)

# two different batch sizes can be configured
# the small one is used for the highly complex K-best-based receivers
# otherwise OOM errors occur
batch_size = sys_parameters.batch_size_eval
batch_size_small = sys_parameters.batch_size_eval_small

# results are directly saved in files
results_filename = f"{sys_parameters.label}{name_suffix}_results"
results_filename = res_dir + results_filename

if not dont_load and exists(results_filename):
    print(f"### File '{results_filename}' found. " \
          "It will be updated with the new results.")
    with open(results_filename, 'rb') as f:
        data = pickle.load(f)
        if len(data) == 3:
            ebno_db_, BERs, BLERs = data

            BIT_ERRORs = {} 
            BLOCK_ERRORs = {} 
            NB_BITs = {} 
            NB_BLOCKs = {}
            SNRs = {}
        if len(data) == 7:
            ebno_db_, BERs, BLERs, BIT_ERRORs, BLOCK_ERRORs, NB_BITs, NB_BLOCKs = data
            SNRs = {}
        if len(data) == 8:
            ebno_db_, BERs, BLERs, BIT_ERRORs, BLOCK_ERRORs, NB_BITs, NB_BLOCKs, SNRs = data
        

else:
    print(f"### No file '{results_filename}' found or user decided not to load. One will be created.")

    BERs = {}
    BLERs = {}
    BIT_ERRORs = {} 
    BLOCK_ERRORs = {} 
    NB_BITs = {} 
    NB_BLOCKs = {}
    SNRs = {}
    ebno_db_ = np.arange(sys_parameters.snr_db_eval_min,
                        sys_parameters.snr_db_eval_max,
                        sys_parameters.snr_db_eval_stepsize)

if len(snr_dbs)>0: # replace snr values if provided
    ebno_db = snr_dbs
    print(f"ebno_db set to:{ebno_db}")
else:
    ebno_db = np.arange(sys_parameters.snr_db_eval_min,
                        sys_parameters.snr_db_eval_max,
                        sys_parameters.snr_db_eval_stepsize)
# evaluate for different number of active transmitters
if num_tx_eval == -1:
    num_tx_evals = np.arange(sys_parameters.min_num_tx,
                             sys_parameters.max_num_tx+1, 1)
else:
    if isinstance(num_tx_eval, int):
        num_tx_evals = [num_tx_eval]
    elif isinstance(num_tx_eval, (list, tuple)):
        num_tx_evals = num_tx_eval
    else:
        raise ValueError("num_tx_eval must be int or list of ints.")

if mcs_arr_eval_idx == -1:
    mcs_arr_eval_idxs = list(range(len(sys_parameters.mcs_index)))
else:
    if isinstance(mcs_arr_eval_idx, int):
        mcs_arr_eval_idxs = [mcs_arr_eval_idx]
    elif isinstance(mcs_arr_eval_idx, (list, tuple)):
        mcs_arr_eval_idxs = mcs_arr_eval_idx
    else:
        raise ValueError("mcs_arr_eval_idx must be int or list of ints.")

print(f"Evaluating for {num_tx_evals} active users and mcs_index elements {mcs_arr_eval_idxs}.")

# the evaluation can loop over multiple number of active DMRS ports / users
# num_tx_evals = [num_tx_evals[0]]# Loops results are wrong
for num_tx_eval in num_tx_evals:

    # --------------------------------------------------------------------
    # Generate covariance matrices for LMMSE-based baselines
    # if not eval_nrx_only:
    if "baseline_lmmse_kbest" in methods or "baseline_lmmse_lmmse" in methods:
        print("Generating cov matrix.")
        os.system(f"python compute_cov_mat.py -config_name {config_name} -gpu {gpu} -num_samples {num_cov_samples} -num_tx_eval {num_tx_eval} -n_size_bwp_eval {args.n_size_bwp_eval}")
    #-------------------------------------------



    # -------------------- Loop over all evaluation MCS indices -------------------
    # mcs_arr_eval_idxs = [mcs_arr_eval_idxs[0]]# Loops results are wrong
    for mcs_arr_eval_idx in mcs_arr_eval_idxs: 


 # --------------------------------------------------------------------
        # MDX receiver
        #
        if "mdx" in methods:
            sn.config.xla_compat = True
            sys_parameters = Parameters(config_name,
                                        training=False,
                                        num_tx_eval=num_tx_eval,
                                        system='mdx')

            sys_parameters = set_eval_params(sys_parameters,args)

            # check channel types for consistency
            if sys_parameters.channel_type == 'TDL-B100':
                assert num_tx_eval == 1,\
                        "Channel model 'TDL-B100' only works with one transmitter"
            elif sys_parameters.channel_type in ("DoubleTDLlow", "DoubleTDLmedium",
                                                "DoubleTDLhigh"):
                assert num_tx_eval == 2,\
                    "Channel model 'DoubleTDL' only works with two transmitters exactly"
            e2e_nn = E2E_Model(sys_parameters, training=False, mcs_arr_eval_idx=mcs_arr_eval_idx)

            print("\nRunning: " + sys_parameters.system)
            #  Run once and load the weights
            e2e_nn(1, 1.)
            filename = f'../weights/{sys_parameters.label}_weights.h5'
            
            if forced_transfer:
                if exists(sys_parameters.transfer_weights_path):
                    transfer_weights_from_h5(e2e_nn, sys_parameters.transfer_weights_path,
                    start_token="neural_pusch_receiver/cgnnofdm", verbose=False)
                    print(f"weights transfered from:\n{sys_parameters.transfer_weights_path}")
                else:
                    print("Transfer weights do not exist.")
            else:
                if exists(filename):
                    load_weights(e2e_nn, filename)
                    print(f"weights loaded from:\n{filename}")
                elif exists(sys_parameters.transfer_weights_path):
                    transfer_weights_from_h5(e2e_nn, sys_parameters.transfer_weights_path,
                    start_token="neural_pusch_receiver/cgnnofdm", verbose=False)
                    print(f"weights transfered from:\n{sys_parameters.transfer_weights_path}")
                else:
                    print("weights do not exist.")


            # and set number iterations for evaluation
            e2e_nn._receiver._neural_rx.num_it = sys_parameters.num_nrx_iter_eval

            # Start sim
            # ber, bler = sim_ber(e2e_nn,
            ber, bler, bit_errors, block_errors, nb_bits, nb_blocks= sim_ber(e2e_nn,
                                graph_mode="xla",
                                ebno_dbs=ebno_db,
                                max_mc_iter=max_mc_iter,
                                num_target_block_errors=num_target_block_errors,
                                batch_size=batch_size,
                                distribute=distribute,
                                target_bler=target_bler,
                                early_stop=True,
                                forward_keyboard_interrupt=True)
            BERs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
            BLERs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler
            
            BIT_ERRORs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = bit_errors
            BLOCK_ERRORs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = block_errors
            NB_BITs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_bits
            NB_BLOCKs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_blocks

            # with open(results_filename, "wb") as f:
            #     pickle.dump([ebno_db, BERs, BLERs], f)

            # data = [ebno_db, BERs, BLERs]
            SNRs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = ebno_db
            data = [ebno_db_, BERs, BLERs, BIT_ERRORs, BLOCK_ERRORs, NB_BITs, NB_BLOCKs, SNRs]
            save_results(results_filename, data)
            
            tf.keras.backend.clear_session()
            del e2e_nn 
            
            sn.config.xla_compat = False
            # End Neural Receiver
        else:
            print("skipping MDX")


        # --------------------------------------------------------------------
        # Neural receiver
        #
        if "nrx" in methods:
            sn.config.xla_compat = True
            sys_parameters = Parameters(config_name,
                                        training=False,
                                        num_tx_eval=num_tx_eval,
                                        system='nrx')
            sys_parameters = set_eval_params(sys_parameters,args)

            # check channel types for consistency
            if sys_parameters.channel_type == 'TDL-B100':
                assert num_tx_eval == 1,\
                        "Channel model 'TDL-B100' only works with one transmitter"
            elif sys_parameters.channel_type in ("DoubleTDLlow", "DoubleTDLmedium",
                                                "DoubleTDLhigh"):
                assert num_tx_eval == 2,\
                    "Channel model 'DoubleTDL' only works with two transmitters exactly"
            e2e_nn = E2E_Model(sys_parameters, training=False, mcs_arr_eval_idx=mcs_arr_eval_idx)

            print("\nRunning: " + sys_parameters.system)
            #  Run once and load the weights
            e2e_nn(1, 1.)
            # filename = f'../weights/{sys_parameters.label}_weights.h5'

            # filename_with_extension = f'../weights/{sys_parameters.label}_weights.h5'
            # filename_without_extension = f'../weights/{sys_parameters.label}_weights'

            # if os.path.exists(filename_with_extension):
            #     filename = filename_with_extension
            # elif os.path.exists(filename_without_extension):
            #     filename = filename_without_extension
            # else:
            #     filename = None 

            filename = f'../weights/{sys_parameters.label}_weights'
            if os.path.exists(filename):
                print(f"Found weights: {filename}")
                load_weights(e2e_nn, filename)
            else:
                print("Did not found weights")
                


            # and set number iterations for evaluation
            e2e_nn._receiver._neural_rx.num_it = sys_parameters.num_nrx_iter_eval

            # Start sim
            # ber, bler = sim_ber(e2e_nn,
            ber, bler, bit_errors, block_errors, nb_bits, nb_blocks= sim_ber(e2e_nn,
                                graph_mode="xla",
                                ebno_dbs=ebno_db,
                                max_mc_iter=max_mc_iter,
                                num_target_block_errors=num_target_block_errors,
                                batch_size=batch_size,
                                distribute=distribute,
                                target_bler=target_bler,
                                early_stop=True,
                                forward_keyboard_interrupt=True)
            BERs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
            BLERs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler

            BIT_ERRORs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = bit_errors
            BLOCK_ERRORs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = block_errors
            NB_BITs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_bits
            NB_BLOCKs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_blocks

            # with open(results_filename, "wb") as f:
            #     pickle.dump([ebno_db, BERs, BLERs], f)
            
            # data = [ebno_db, BERs, BLERs]
            SNRs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = ebno_db
            data = [ebno_db_, BERs, BLERs, BIT_ERRORs, BLOCK_ERRORs, NB_BITs, NB_BLOCKs, SNRs]
            save_results(results_filename, data)

            sn.config.xla_compat = False
            # End Neural Receiver
        else:
            print("skipping NRX")

        # --------------------------------------------------------------------
        # Baseline: LS estimation/lin interpolation + LMMSE detection
        #
        if "baseline_lslin_lmmse" in methods:
            sn.config.xla_compat = True
            sys_parameters = Parameters(config_name,
                                        training=False,
                                        num_tx_eval=num_tx_eval,
                                        system='baseline_lslin_lmmse')

            sys_parameters = set_eval_params(sys_parameters,args)

            e2e_baseline = E2E_Model(sys_parameters, training=False,
                                     mcs_arr_eval_idx=mcs_arr_eval_idx)

            print("\nRunning: " + sys_parameters.system)
            # ber, bler = sim_ber(e2e_baseline,
            ber, bler, bit_errors, block_errors, nb_bits, nb_blocks= sim_ber(e2e_baseline,
                            graph_mode="xla",
                            ebno_dbs=ebno_db,
                            max_mc_iter=max_mc_iter,
                            num_target_block_errors=num_target_block_errors,
                            target_bler=target_bler,
                            batch_size=batch_size,
                            distribute=distribute,
                            early_stop=True,
                            forward_keyboard_interrupt=True)
            BERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
            BLERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler

            BIT_ERRORs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bit_errors
            BLOCK_ERRORs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = block_errors
            NB_BITs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_bits
            NB_BLOCKs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_blocks

            # with open(results_filename, "wb") as f:
            #     pickle.dump([ebno_db, BERs, BLERs], f)
            
            # data = [ebno_db, BERs, BLERs]
            
            SNRs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ebno_db
            data = [ebno_db_, BERs, BLERs, BIT_ERRORs, BLOCK_ERRORs, NB_BITs, NB_BLOCKs, SNRs]            
            save_results(results_filename, data)

            sn.config.xla_compat = False
        else:
            print("skipping LSlin & LMMSE")

        # --------------------------------------------------------------------
        # Baseline: LS estimation/lin interpolation + K-Best detection
        #
        if "baseline_lslin_kbest" in methods:
            sn.config.xla_compat = True
            sys_parameters = Parameters(config_name,
                                        training=False,
                                        num_tx_eval=num_tx_eval,
                                        system='baseline_lslin_kbest')

            sys_parameters = set_eval_params(sys_parameters,args)

            e2e_baseline = E2E_Model(sys_parameters, training=False,
                                     mcs_arr_eval_idx=mcs_arr_eval_idx)

            print("\nRunning: " + sys_parameters.system)
        #     ber, bler = sim_ber(e2e_baseline,
            ber, bler, bit_errors, block_errors, nb_bits, nb_blocks= sim_ber(e2e_baseline,
                            graph_mode="xla",
                            ebno_dbs=ebno_db,
                            max_mc_iter=max_mc_iter,
                            num_target_block_errors=num_target_block_errors,
                            target_bler=target_bler,
                            batch_size=batch_size,
                            distribute=distribute,
                            early_stop=True,
                            forward_keyboard_interrupt=True)
            BERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
            BLERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler

            BIT_ERRORs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bit_errors
            BLOCK_ERRORs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = block_errors
            NB_BITs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_bits
            NB_BLOCKs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_blocks

            # with open(results_filename, "wb") as f:
            #     pickle.dump([ebno_db, BERs, BLERs], f)
            
            # data = [ebno_db, BERs, BLERs]
            SNRs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ebno_db
            data = [ebno_db_, BERs, BLERs, BIT_ERRORs, BLOCK_ERRORs, NB_BITs, NB_BLOCKs, SNRs]     
            save_results(results_filename, data)

            sn.config.xla_compat = False
        else:
            print("skipping LSlin & K-Best")

        # --------------------------------------------------------------------
        # Baseline: LMMSE estimation/interpolation + K-Best detection
        #
        if "baseline_lmmse_kbest" in methods:
            sn.config.xla_compat = False
            sys_parameters = Parameters(config_name,
                                        training=False,
                                        num_tx_eval=num_tx_eval,
                                        system = 'baseline_lmmse_kbest')

            sys_parameters = set_eval_params(sys_parameters,args)

            e2e_baseline = E2E_Model(sys_parameters, training=False,
                                     mcs_arr_eval_idx=mcs_arr_eval_idx)

            print("\nRunning: " + sys_parameters.system)
        #     ber, bler = sim_ber(e2e_baseline,
            ber, bler, bit_errors, block_errors, nb_bits, nb_blocks= sim_ber(e2e_baseline,
                            graph_mode="graph",
                            ebno_dbs=ebno_db,
                            max_mc_iter=max_mc_iter,
                            num_target_block_errors=num_target_block_errors,
                            target_bler=target_bler,
                            batch_size=batch_size_small, # must be small for large PRBs
                            #distribute=distribute, # somehow does not compile
                            early_stop=True,
                            forward_keyboard_interrupt=True)
            BERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
            BLERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler

            BIT_ERRORs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bit_errors
            BLOCK_ERRORs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = block_errors
            NB_BITs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_bits
            NB_BLOCKs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_blocks

            # with open(results_filename, "wb") as f:
            #     pickle.dump([ebno_db, BERs, BLERs], f)
            
            # data = [ebno_db, BERs, BLERs]
            SNRs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ebno_db
            data = [ebno_db_, BERs, BLERs, BIT_ERRORs, BLOCK_ERRORs, NB_BITs, NB_BLOCKs, SNRs]     
            save_results(results_filename, data)


            sn.config.xla_compat = False
        else:
            print("skipping LMMSE & KBest")


        # --------------------------------------------------------------------
        # Baseline: Perfect CSI + LMMSE
        #
        # currently not evaluated
        if "baseline_perf_csi_lmmse" in methods:
            sys_parameters = Parameters(config_name,
                                        training=False,
                                        num_tx_eval=num_tx_eval,
                                        system='baseline_perf_csi_lmmse')

            sys_parameters = set_eval_params(sys_parameters,args)

            e2e_baseline = E2E_Model(sys_parameters, training=False, mcs_arr_eval_idx=mcs_arr_eval_idx)

            print("\nRunning: " + sys_parameters.system)
        #     ber, bler = sim_ber(e2e_baseline,
            ber, bler, bit_errors, block_errors, nb_bits, nb_blocks= sim_ber(e2e_baseline,
                            graph_mode="graph",
                            ebno_dbs=ebno_db,
                            max_mc_iter=max_mc_iter, # account for reduced bs
                            num_target_block_errors=num_target_block_errors,
                            batch_size=batch_size, # must be small due to TF bug in K-best
                            early_stop=True,
                            forward_keyboard_interrupt=True)
            BERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
            BLERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler

            BIT_ERRORs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bit_errors
            BLOCK_ERRORs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = block_errors
            NB_BITs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_bits
            NB_BLOCKs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_blocks

            # with open(results_filename, "wb") as f:
            #     pickle.dump([ebno_db, BERs, BLERs], f)
            
            # data = [ebno_db, BERs, BLERs]
            SNRs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ebno_db
            data = [ebno_db_, BERs, BLERs, BIT_ERRORs, BLOCK_ERRORs, NB_BITs, NB_BLOCKs, SNRs]     
            save_results(results_filename, data)


        else:
            print("skipping Perfect CSI & LMMSE")

        # --------------------------------------------------------------------
        # Baseline: LMMSE estimation/interpolation + LMMSE detection
        #
        if "baseline_lmmse_lmmse" in methods:
            sn.config.xla_compat = False
            sys_parameters = Parameters(config_name,
                                        training=False,
                                        num_tx_eval=num_tx_eval,
                                        system='baseline_lmmse_lmmse')

            sys_parameters = set_eval_params(sys_parameters,args)

            e2e_baseline = E2E_Model(sys_parameters, training=False, mcs_arr_eval_idx=mcs_arr_eval_idx)

            print("Running: " + sys_parameters.system)
        #     ber, bler = sim_ber(e2e_baseline,
            ber, bler, bit_errors, block_errors, nb_bits, nb_blocks= sim_ber(e2e_baseline,
                            graph_mode="graph",
                            ebno_dbs=ebno_db,
                            max_mc_iter=max_mc_iter, # account for reduced bs
                            num_target_block_errors=num_target_block_errors,
                            #target_bler=target_bler,
                            batch_size=batch_size_small, # must be small due to TF bug in K-best
                            #distribute=distribute,
                            early_stop=True,
                            forward_keyboard_interrupt=True)
            BERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
            BLERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler


            BIT_ERRORs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bit_errors
            BLOCK_ERRORs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = block_errors
            NB_BITs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_bits
            NB_BLOCKs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_blocks

            # with open(results_filename, "wb") as f:
            #     pickle.dump([ebno_db, BERs, BLERs], f)
            
            # data = [ebno_db, BERs, BLERs]
            SNRs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ebno_db
            data = [ebno_db_, BERs, BLERs, BIT_ERRORs, BLOCK_ERRORs, NB_BITs, NB_BLOCKs, SNRs]     
            save_results(results_filename, data)


            sn.config.xla_compat = False
        else:
            print("skipping LMMSE")
            sys_name = f"Baseline - LMMSE+LMMSE"

        # --------------------------------------------------------------------
        # Baseline: Perfect CSI + K-Best detection
        #
        if "baseline_perf_csi_kbest" in methods:
            sn.config.xla_compat = False
            sys_parameters = Parameters(config_name,
                                        training=False,
                                        num_tx_eval=num_tx_eval,
                                        system='baseline_perf_csi_kbest')

            sys_parameters = set_eval_params(sys_parameters,args)

            e2e_baseline = E2E_Model(sys_parameters, training=False,
                                     mcs_arr_eval_idx=mcs_arr_eval_idx)

            print("\nRunning: " + sys_parameters.system)
            # ber, bler = sim_ber(e2e_baseline,
            ber, bler, bit_errors, block_errors, nb_bits, nb_blocks= sim_ber(e2e_baseline,
                            graph_mode="graph",
                            ebno_dbs=ebno_db,
                            max_mc_iter=max_mc_iter, # account for reduced bs
                            num_target_block_errors=num_target_block_errors,
                            target_bler=target_bler,
                            batch_size=batch_size_small, # must be small due to TF bug in K-best
                            distribute=distribute,
                            early_stop=True,
                            forward_keyboard_interrupt=True)
            BERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
            BLERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler


            BIT_ERRORs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bit_errors
            BLOCK_ERRORs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = block_errors
            NB_BITs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_bits
            NB_BLOCKs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = nb_blocks

            # with open(results_filename, "wb") as f:
            #     pickle.dump([ebno_db, BERs, BLERs], f)
            
            # data = [ebno_db, BERs, BLERs]
            SNRs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ebno_db
            data = [ebno_db_, BERs, BLERs, BIT_ERRORs, BLOCK_ERRORs, NB_BITs, NB_BLOCKs, SNRs]     
            save_results(results_filename, data)



            sn.config.xla_compat = False
        else:
            print("skipping Perfect CSI & K-Best")


