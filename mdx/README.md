# Site-Specific Training of a Model-Driven Neural Receiver (MDX)

The MDX described in [1] and implemented in [Mahdi-Abdollahpour/mdx](https://github.com/Mahdi-Abdollahpour/mdx) is a low-complexity, model-driven MIMO detector for 5G NR multi-user MIMO PUSCH reception. It combines conventional receiver algorithms with a compact neural network to perform joint channel estimation and data detection. Pilot-aided LS channel estimates, LMMSE equalization, and data-aided LS channel refinement are followed by a four-block ResNet and final LMMSE max-log soft-output detection.

This codebase is a fork of [Mahdi-Abdollahpour/mdx](https://github.com/Mahdi-Abdollahpour/mdx). It extends the original synthetic-channel workflow with testbed-compatible pretraining configurations, a Data Lake channel for site-specific finetuning, and evaluation on real-world measurements using the workflow in [2].

## Requirements

The code was tested with Python 3.10.12. Install the Python dependencies listed in [`requirements.txt`](requirements.txt).

Real-data finetuning and evaluation additionally require:

- NVIDIA pyAerial 2025.2.dev1
- NVIDIA Aerial Data Lake release 25-2
- Access to a database containing the `fapi` and `fh` tables
- CUDA toolkit 12.9

## Setup and Contributions

Experiments are configured through files in [`config/`](config/). The [`mdx_res_blocks2_var_mcs_it1_ext_aerial_nomcs.cfg`](config/mdx_res_blocks2_var_mcs_it1_ext_aerial_nomcs.cfg) pretraining configuration uses randomized 3GPP UMi channels and matches the 13-symbol resource grid used for the measured PUSCH data.

This fork adds the following components to the original workflow:

- [`train_neural_rx.py`](scripts/train_neural_rx.py) supports site-specific finetuning from pretrained weights when `channel_type = 'Datalake'` and `datalake_tf_fn` identifies the corresponding TFRecord.
- An additional Data Lake channel model loads received IQ samples, coded-bit labels, LS channel estimates, and noise-variance estimates from the TFRecord for finetuning.
- [`eval_mdx_from_datalake_TF.py`](scripts/eval_mdx_from_datalake_TF.py) uses real-world measurements to evaluate the MDX and the MMSE Reference Rx, which corresponds to the pyAerial PUSCH Rx presented in [3]. It reports dataset BLER, and timestamp files enable deterministic replay.

## Workflow

A tutorial for the complete workflow is provided in the [MDX site-specific training tutorial notebook](notebooks/site_specific_training_tutorial.ipynb). Refer to the [Technical Configurations](#db) section for detailed information with regards to the dataset and configuration files.

The workflow consists of:

0. **Pretrain on synthetic channels.** Train the MDX using [`train_neural_rx.py`](scripts/train_neural_rx.py) with `-system mdx` and the provided pretraining configuration, which uses randomized 3GPP UMi channels. Weights use the configuration's `global.label` and are stored as `weights/<label>_weights.h5`, with training logs under [`logs/`](logs/).
1. **Get real-world measurements.** Ensure that the real-world measurement database is stored in NVIDIA Aerial Data Lake.
2. **Prepare the finetuning data.** Place the matching pre-extracted TFRecord to [`/finetuning_datasets/`](../finetuning_datasets/) and make sure that the `datalake_tf_fn` field in the relevant configuration file points to the right filepath. The TFRecords are produced by the shared Data Lake extraction pipeline and contain the features required by the MDX Data Lake channel.
3. **Perform offline finetuning.** Run [`train_neural_rx.py`](scripts/train_neural_rx.py) with `-system mdx` and a configuration that sets `channel_type = 'Datalake'`. The configuration's `transfer_weights_path` selects the pretrained checkpoint, while `global.label` determines the name for the finetuned `.h5` weights.
4. **Evaluate performance.** Run [`eval_mdx_from_datalake_TF.py`](scripts/eval_mdx_from_datalake_TF.py) with the matching configuration, Data Lake database, and optional timestamp file. The evaluator loads the label-matched weights and compares the MDX with the MMSE Reference Rx on the test slot samples.

The supported measured-data workflow uses single-layer transmission only.

## <a id="db"></a>Technical Configurations

The measurement campaigns, Data Lake database selectors, finetuning TFRecords, and finetuning / test splits used in the experiments are documented in [`datasets.md`](../datasets.md).

The tables below list the MDX configurations, TFRecords, and timestamp files used for the documented experiments.

#### Pretraining

| Configuration | Channel Model |
| --- | --- |
| [`mdx_res_blocks2_var_mcs_it1_ext_aerial_nomcs.cfg`](config/mdx_res_blocks2_var_mcs_it1_ext_aerial_nomcs.cfg) | 3GPP UMi |

#### Finetuning and Evaluation

| Dataset | Finetuning Configuration | Finetuning TFRecord | Evaluation Timestamps |
| --- | --- | --- | --- |
| Small Laboratory Jun.&nbsp;2026&nbsp;&nbsp;(1&nbsp;ULL) | [`mdx_datalake_j61_1ULL.cfg`](/mdx/config/mdx_datalake_j61_1ULL.cfg) | [`sgs23_1ULLs_5dB_j61_2026_06_04_no.tfrecord`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/tfrecord/sgs23_1ULLs_5dB_j61_2026_06_04_no.tfrecord) | [`timestamps_junj61_1ULL.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/timestamps/timestamps_junj61_1ULL.pkl) |
| Large&nbsp;Office&nbsp;Floor Jun.&nbsp;2026&nbsp;&nbsp;(1&nbsp;ULL) | [`mdx_datalake_jfloor_1ULL.cfg`](/mdx/config/mdx_datalake_jfloor_1ULL.cfg) | [`sgs23_1ULLs_5dB_jFloorRealLoop_2026_06_10_no.tfrecord`]((https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/tfrecord/sgs23_1ULLs_5dB_jFloorRealLoop_2026_06_10_no.tfrecord)) | [`timestamps_junjfloor_1ULL.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_floor/jun_26/timestamps/timestamps_junjfloor_1ULL.pkl.pkl) |

## Repository Map

- [`config/`](config/): configurations used for synthetic pretraining, site-specific finetuning, and experimental model export.
- [`scripts/`](scripts/): scripts for training, synthetic evaluation, experimental model export, and real-world data evaluation.
- [`utils/`](utils/): MDX model definitions, channel providers, parameter handling, and export helpers.
- [`notebooks/`](notebooks/): tutorial and analysis notebooks, including the [site-specific training tutorial](notebooks/site_specific_training_tutorial.ipynb).
- [`weights/`](weights/): pretrained and finetuned MDX `.h5` checkpoints.
- [`eval_timestamps/`](eval_timestamps/): timestamp files used for deterministic evaluation.
- [`logs/`](logs/) and [`results/`](results/): training logs and evaluation outputs.

## References

<a id="ref-1"></a>[1] M. Abdollahpour, M. Bertuletti, Y. Zhang, Y. Li, L. Benini, and A. Vanelli-Coralli, “A Compute&Memory Efficient Model-Driven Neural 5G Receiver for Edge AI-assisted RAN,” in *Proc. IEEE Global Communications Conference (GLOBECOM)*, 2025, pp. 5248–5253. Available: https://arxiv.org/abs/2508.12892

<a id="ref-2"></a>[2] R. Wiesmayr, N. B. Baytekin, C. Dick, and C. Studer, “On the Impact of Site-Specific Training for a Real-World 5G NR System,” in *Proc. Asilomar Conference on Signals, Systems, and Computers*, 2026. Available: https://arxiv.org/abs/2609.04004

<a id="ref-3"></a>[3] NVIDIA Corporation, “Aerial CUDA-Accelerated RAN,” release 25-2. Available: https://docs.nvidia.com/aerial/cuda-accelerated-ran/25-2/index.html

