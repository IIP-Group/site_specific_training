#!/bin/bash

# Mahdi Abdollahpour (mahdi.abdollahpour@unibo.it)
# 2025


config_name="mdx_res_blocks2_var_mcs_it1_ext.cfg"
name_suffix="_tdla"



max_mc_iter=5000
num_target_block_errors=500
gpu=0
target_bler=0.0001



snr_db_eval_min=-4
snr_db_eval_max=13
snr_db_eval_stepsize=1
max_ut_velocity_eval=56.
channel_type_eval="NTDLlow"
tdl_models="A"
n_size_bwp_eval=273
batch_size_eval_small=3
batch_size_eval=30
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/
echo "eval@: ${dir}"
# methods="mdx nrx baseline_lslin_lmmse, baseline_lslin_kbest, baseline_lmmse_kbest, baseline_perf_csi_lmmse, baseline_lmmse_lmmse, baseline_perf_csi_kbest"


python3 evaluate.py -num_tx_eval 2 -mcs_arr_eval_idx 0 -config_name="${config_name}" -gpu="${gpu}" \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" \
-methods mdx -name_suffix="${name_suffix}"

python3 evaluate.py -num_tx_eval 2 -mcs_arr_eval_idx 1 -config_name="${config_name}" -gpu="${gpu}" \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" \
-methods mdx -name_suffix="${name_suffix}"

python3 evaluate.py -num_tx_eval 2 -mcs_arr_eval_idx 2 -config_name="${config_name}" -gpu="${gpu}" \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" \
-methods mdx -name_suffix="${name_suffix}"