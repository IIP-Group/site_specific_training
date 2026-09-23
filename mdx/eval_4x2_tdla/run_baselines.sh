#!/bin/bash

# Mahdi Abdollahpour (mahdi.abdollahpour@unibo.it)
# 2025


config_name="baselines_4x2.cfg"

max_mc_iter=2000
num_target_block_errors=500
gpu=0
target_bler=0.0001

snr_db_eval_min=-12
snr_db_eval_max=15
snr_db_eval_stepsize=1
max_ut_velocity_eval=56.
channel_type_eval="NTDLlow"
tdl_models="A"
n_size_bwp_eval=273
batch_size_eval_small=3
batch_size_eval=30
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/
echo "eval@: ${dir}"

mcs_index=(-1 -1 -1)
name_suffix="_tdla"
# methods="mdx nrx baseline_lslin_lmmse, baseline_lslin_kbest, baseline_lmmse_kbest, baseline_perf_csi_lmmse, baseline_lmmse_lmmse, baseline_perf_csi_kbest"

# -----------------------------------------------------------------------
# #---------------- baseline_lslin_lmmse --------------------------------
# -----------------------------------------------------------------------


# --------- mcs 0
python3 evaluate.py -config_name="${config_name}" -gpu="${gpu}" -mcs_arr_eval_idx 0 -num_tx_eval 2 \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" -mcs_index "${mcs_index[@]}" \
-methods baseline_lslin_lmmse -name_suffix="${name_suffix}"


# --------- mcs 1
python3 evaluate.py -config_name="${config_name}" -gpu="${gpu}" -mcs_arr_eval_idx 1 -num_tx_eval 2 \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" -mcs_index "${mcs_index[@]}" \
-methods baseline_lslin_lmmse -name_suffix="${name_suffix}"


# --------- mcs 2
python3 evaluate.py -config_name="${config_name}" -gpu="${gpu}" -mcs_arr_eval_idx 2 -num_tx_eval 2 \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" -mcs_index "${mcs_index[@]}" \
-methods baseline_lslin_lmmse -name_suffix="${name_suffix}"

# -----------------------------------------------------------------------
#------------------ baseline_perf_csi_kbest -----------------------------
# -----------------------------------------------------------------------

# ------- mcs0
python3 evaluate.py -config_name="${config_name}" -gpu="${gpu}" -mcs_arr_eval_idx 0 -num_tx_eval 2 \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" -mcs_index "${mcs_index[@]}" \
-methods baseline_perf_csi_kbest -name_suffix="${name_suffix}" #-snr_dbs "${snr_dbs[@]}"


# ------- mcs1
python3 evaluate.py -config_name="${config_name}" -gpu="${gpu}" -mcs_arr_eval_idx 1 -num_tx_eval 2 \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" -mcs_index "${mcs_index[@]}" \
-methods baseline_perf_csi_kbest -name_suffix="${name_suffix}" #-snr_dbs "${snr_dbs[@]}"

# ------- mcs2
max_mc_iter=5000
num_target_block_errors=1000

python3 evaluate.py -config_name="${config_name}" -gpu="${gpu}" -mcs_arr_eval_idx 2 -num_tx_eval 2 \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" -mcs_index "${mcs_index[@]}" \
-methods baseline_perf_csi_kbest -name_suffix="${name_suffix}" #-snr_dbs "${snr_dbs[@]}"


# -----------------------------------------------------------------------
#  -------------------------- baseline_lmmse_kbest ---------------------
# -----------------------------------------------------------------------


python3 evaluate.py -config_name="${config_name}" -gpu="${gpu}" -mcs_arr_eval_idx 0 -num_tx_eval 2 \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" -mcs_index "${mcs_index[@]}" \
-methods baseline_lmmse_kbest -name_suffix="${name_suffix}" #-snr_dbs "${snr_dbs[@]}"

python3 evaluate.py -config_name="${config_name}" -gpu="${gpu}" -mcs_arr_eval_idx 1 -num_tx_eval 2 \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" -mcs_index "${mcs_index[@]}" \
-methods baseline_lmmse_kbest -name_suffix="${name_suffix}" #-snr_dbs "${snr_dbs[@]}"

python3 evaluate.py -config_name="${config_name}" -gpu="${gpu}" -mcs_arr_eval_idx 2 -num_tx_eval 2 \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" -mcs_index "${mcs_index[@]}" \
-methods baseline_lmmse_kbest -name_suffix="${name_suffix}" #-snr_dbs "${snr_dbs[@]}"