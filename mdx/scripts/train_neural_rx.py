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

# training of the neural receiver for a given configuration file
# the training loop can be found in utils.training_loop

####################################################################
# Parse args
####################################################################

import argparse
from os.path import exists

parser = argparse.ArgumentParser()
# the config defines the sys parameters
parser.add_argument("-config_name", help="config filename", type=str)
# GPU to use
parser.add_argument("-gpu", help="GPU to use", type=int, default=0)
# Easier debugging with breakpoints when running the code eagerly
parser.add_argument("-debug", help="Set debugging configuration", action="store_true", default=False)
# seed
parser.add_argument("-seed", help="Set seed of training", type=int, default=43)
parser.add_argument("-system", help="Set system of training", type=str, default="nrx")

# Parse all arguments
args = parser.parse_args()

####################################################################
# Imports and GPU configuration
####################################################################

# Avoid warnings from TensorFlow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Enable all TensorFlow logs
# os.environ["XLA_FLAGS"] = f"--xla_dump_to=/scratch2/mabdollahpo/neural_rxm/logs/dump --xla_dump_hlo_as_text"  # Dump XLA com




import tensorflow as tf
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
try:
    print('Only GPU number', args.gpu, 'used.')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except RuntimeError as e:
    print(e)

import sys
sys.path.append('../')

from utils import E2E_Model, training_loop, Parameters, load_weights, compute_lr_multipliers, save_weights
from utils import transfer_weights_from_h5
##################################################################
# Training parameters
##################################################################

# all relevant parameters are defined in the config_file
config_name = args.config_name

# initialize system parameters
sys_parameters = Parameters(config_name,
                            system=args.system,
                            training=True)


label = f'{sys_parameters.label}'
filename = '../weights/'+ label + '_weights'
save_format='pkl'
if args.system=="mdx":
    filename = '../weights/'+ label + '_weights.h5'
    save_format='h5'


training_logdir = '../logs' # use TensorBoard to visualize

import numpy as np
random_seed = np.random.randint(0, 2**32)
training_seed = random_seed

if args.debug:
    tf.config.run_functions_eagerly(True)
    training_logdir = training_logdir + "/debug"

#################################################################
# Start training
#################################################################

sys_training = E2E_Model(sys_parameters, training=True)
sys_training(1, 1.) # run once to init weights in TensorFlow
sys_training.summary()

# load weights if the exists already

transfer_loaded = False
if hasattr(sys_parameters, 'transfer_weights_path'):
    if exists(sys_parameters.transfer_weights_path):
        print("\nTransfer Weights exist - loading transfered weights.")
        # load_weights(sys_training, sys_parameters.transfer_weights_path)
        transfer_weights_from_h5(sys_training, sys_parameters.transfer_weights_path,
         start_token="neural_pusch_receiver/cgnnofdm", verbose=False)
        print(f"weights transfered from:/n{sys_parameters.transfer_weights_path}")
        transfer_loaded = True
    else:
        print(f"\nTransfer weights path specified but does not exist. The specified filesname:\n{sys_parameters.transfer_weights_path}")

if not transfer_loaded:
    if exists(filename):
        print("\nWeights exist already - loading stored weights.",end="..")
        load_weights(sys_training, filename)
        print(f"\b\bweights loaded from:\n{filename}")
    elif exists(f"{filename}.h5"):
        file_name_=f"{filename}.h5"
        print("\nWeights exist with h5 format - loading stored weights.",end="..")
        load_weights(sys_training, file_name_, skip_mismatch=False, by_name=False)
        print(f"\nweights loaded from:\n{file_name_}")
    else:
        print(f"weights do not exist! specified filename:\n{filename}\n",flush=True)

if hasattr(sys_parameters, 'mcs_training_snr_db_offset'):
    mcs_training_snr_db_offset = sys_parameters.mcs_training_snr_db_offset
else:
    mcs_training_snr_db_offset = None

if hasattr(sys_parameters, 'mcs_training_probs'):
    mcs_training_probs = sys_parameters.mcs_training_probs
else:
    mcs_training_probs = None


# save_weights(sys_training, filename, save_format)


# run the training / weights are automatically saved
# UEs' MCSs will be drawn randomly
training_loop(sys_training,
              label=label,
              filename=filename,
              training_logdir=training_logdir,
              training_seed=training_seed,
              training_schedule=sys_parameters.training_schedule,
              eval_ebno_db_arr=sys_parameters.eval_ebno_db_arr,
              min_num_tx=sys_parameters.min_num_tx,
              max_num_tx=sys_parameters.max_num_tx,
              sys_parameters=sys_parameters,
              mcs_arr_training_idx=list(range(len(sys_parameters.mcs_index))), # train with all supported MCSs
              mcs_training_snr_db_offset=mcs_training_snr_db_offset,
              mcs_training_probs=mcs_training_probs,
              transfer_loaded = transfer_loaded,
              xla=sys_parameters.xla,
              save_format=save_format)


print("Training finished.")

