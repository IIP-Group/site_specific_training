#!/bin/bash

# Mahdi Abdollahpour (mahdi.abdollahpour@unibo.it)
# 2025


config_name="baselines_16x4.cfg"

max_mc_iter=1000
num_target_block_errors=500
gpu=0
target_bler=0.0001

snr_db_eval_min=-12
snr_db_eval_max=10
snr_db_eval_stepsize=0.5
max_ut_velocity_eval=34.
channel_type_eval="NTDLlow"
tdl_models="A"
n_size_bwp_eval=273
batch_size_eval_small=3
batch_size_eval=30
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/
echo "eval@: ${dir}"

mcs_index=(-1 -1 -1)
name_suffix=""
# methods="mdx nrx baseline_lslin_lmmse, baseline_lslin_kbest, baseline_lmmse_kbest, baseline_perf_csi_lmmse, baseline_lmmse_lmmse, baseline_perf_csi_kbest"


# -----------------------------------------------------------------------
#  -------------------------- baseline_lmmse_kbest ---------------------
# -----------------------------------------------------------------------

max_mc_iter=10000

# --------------------  mcs 0
name_suffix="_g3"
gpu=3



batch_size_eval_small=1
config_name="mdx_var_mcs_BSL_LMMSE_16x4_mcs0.cfg"

python3 evaluate.py -config_name="${config_name}" -gpu="${gpu}" -mcs_arr_eval_idx 0 -num_tx_eval 4 \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" -mcs_index "${mcs_index[@]}" \
-methods baseline_lmmse_kbest -batch_size_eval_small="${batch_size_eval_small}" -name_suffix="${name_suffix}" #-snr_dbs "${snr_dbs[@]}"


batch_size_eval_small=1
config_name="mdx_var_mcs_BSL_LMMSE_16x4_mcs1.cfg"
python3 evaluate.py -config_name="${config_name}" -gpu="${gpu}" -mcs_arr_eval_idx 1 -num_tx_eval 4 \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" -mcs_index "${mcs_index[@]}" \
-methods baseline_lmmse_kbest -batch_size_eval_small="${batch_size_eval_small}" -name_suffix="${name_suffix}" #-snr_dbs "${snr_dbs[@]}"

batch_size_eval_small=1
config_name="mdx_var_mcs_BSL_LMMSE_16x4_mcs2.cfg"
python3 evaluate.py -config_name="${config_name}" -gpu="${gpu}" -mcs_arr_eval_idx 2 -num_tx_eval 4 \
-num_target_block_errors="${num_target_block_errors}" -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
-snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
-max_ut_velocity_eval="${max_ut_velocity_eval}" -channel_type_eval="${channel_type_eval}" -tdl_models="${tdl_models}" \
-n_size_bwp_eval="${n_size_bwp_eval}" -batch_size_eval="${batch_size_eval}" -dir="${dir}" -mcs_index "${mcs_index[@]}" \
-methods baseline_lmmse_kbest   -batch_size_eval_small="${batch_size_eval_small}" -name_suffix="${name_suffix}"


echo "Finished running all LMMSE GPU:0."
exit 3