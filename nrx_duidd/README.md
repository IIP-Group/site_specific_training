# Site-Specific Training of a Model-Based Receiver (DUIDD)

The DUIDD receiver described in [1] is a model-based receiver derived from classical iterative detection and decoding (IDD). Classical IDD exchanges soft information between a soft-input soft-output MMSE-PIC detector and an LDPC message-passing decoder. DUIDD interleaves data detection and channel decoding and uses deep unfolding to make the soft-information exchange, decoder message damping, and decoder state forwarding tunable.

The DUIDD architecture uses flooding min-sum check-node updates. Its number of tunable parameters depends on the number of MMSE-PIC detection stages:

| Schedule | MMSE-PIC Stages (`I`) | Tunable Parameters |
| --- | ---: | ---: |
| `[12]` | 1 | 25 |
| `[6, 6]` | 2 | 30 |
| `[4, 4, 4]` | 3 | 35 |
| `[3, 3, 3, 3]` | 4 | 40 |

This codebase combines the [original DUIDD repository](https://gitlab.ethz.ch/rwiesmayr/nrx_duidd) [1] with the 5G NR PUSCH model and training framework from [`NVlabs/neural_rx`](https://github.com/NVlabs/neural_rx). It further extends this combined codebase with site-specific finetuning on real-world measurements and site-specific parameter adaptation through covariance estimation, using the estimation algorithm presented in [2].

## Setup

The code was tested with Python 3.10.12. Install the Python dependencies listed in [`requirements.txt`](requirements.txt).

Real-data finetuning, covariance estimation, and evaluation additionally require:

- NVIDIA pyAerial 2025.2.dev1
- NVIDIA Aerial Data Lake release 25-2
- Access to a database containing the `fapi` and `fh` tables
- CUDA toolkit 12.9

The real-data scripts connect to ClickHouse at `localhost` by default.

## Contributions


This fork adds the following components to the original workflow:

- [`train_neural_rx.py`](scripts/train_neural_rx.py) supports DUIDD finetuning with real-world data when `channel_type = 'Datalake'`.
- [`eval_duidd_from_datalake.py`](scripts/eval_duidd_from_datalake.py) evaluates pretrained or finetuned DUIDD, untuned classical IDD, and the MMSE Reference Rx on real-world measurements. The MMSE Reference Rx corresponds to the pyAerial PUSCH Rx presented in [3]. Timestamp files enable deterministic replay.
- The tools in [`cov_est/`](cov_est/) estimate spatial, frequency, and temporal covariance matrices from direct LS estimates. They remove the estimated channel-estimation error contribution using the noise-only fourteenth OFDM symbol.
- Configurations, weights, covariance estimates, and evaluation timestamps are provided for the DUIDD, IDD, and LMMSE channel-estimation experiments described below.



## Workflow

Experiments are configured through files in [`config/`](config/). The schedule `[N_1, ..., N_I]` gives the number of flooding min-sum LDPC message-passing iterations performed after each of the `I` MMSE-PIC detection stages. The configurations used in the paper keep the total number of decoding iterations fixed at 12.

A tutorial for the complete workflows is provided in the [DUIDD site-specific training tutorial notebook](notebooks/duidd_site_specific_training_tutorial.ipynb). Refer to the [Technical Configurations](#db) section and [`datasets.md`](../datasets.md) for the configuration, database, TFRecord, and timestamp mappings.

Site-specific finetuning and site-specific parameter adaptation are separate workflows. Finetuning updates the DUIDD parameters with labeled data. Parameter adaptation estimates covariance matrices for a classical LMMSE channel estimator and does not train the DUIDD parameters.

### Site-Specific DUIDD Finetuning

0. **Pretrain on synthetic channels.** Train DUIDD using [`train_neural_rx.py`](scripts/train_neural_rx.py) and a configuration that matches the desired schedule. For example, use [`duidd_aerial_6_6.cfg`](config/duidd_aerial_6_6.cfg) for the `[6, 6]` schedule. Pretraining parameters can be further adjusted from the configuration file.
1. **Get real-world measurements.** Ensure that the real-world measurement database is stored in NVIDIA Aerial Data Lake.
2. **Prepare the finetuning data.** Place the matching pre-extracted TFRecord in [`/finetuning_datasets/`](../finetuning_datasets/). Due to the decoder being a part of the training loop, DUIDD finetuning uses samples restricted to one MCS index and one scrambling configuration.
3. **Finetune DUIDD.** Copy the pretrained weights to the label used by the finetuning configuration, such as [`duidd_pixel9pro_j61_2ULL_slot14_mcs10.cfg`](config/duidd_pixel9pro_j61_2ULL_slot14_mcs10.cfg), then run [`train_neural_rx.py`](scripts/train_neural_rx.py). The finetuning schedule can be adjusted in the configuration.
4. **Evaluate DUIDD.** Run [`eval_duidd_from_datalake.py`](scripts/eval_duidd_from_datalake.py) with the matching finetuning configuration, database, and timestamp file. The evaluator loads the weights associated with the configuration label.

### Covariance-Based Site-Specific Parameter Adaptation

1. **Get real-world measurements.** Ensure that the real-world measurement database is stored in NVIDIA Aerial Data Lake.
2. **Estimate real-world covariance matrices.** Estimate the site-specific spatial, frequency, and temporal covariance matrices from the measured data using [`estimate_pusch_spatial_covariance_from_datalake.py`](cov_est/estimate_pusch_spatial_covariance_from_datalake.py), [`estimate_pusch_frequency_covariance_from_datalake.py`](cov_est/estimate_pusch_frequency_covariance_from_datalake.py), and [`estimate_pusch_time_covariance_from_datalake.py`](cov_est/estimate_pusch_time_covariance_from_datalake.py), respectively.
3. **Select the covariance configuration.** Choose the LMMSE channel-estimation configuration that points to either the synthetic or site-specific covariance matrices. Covariance estimation is independent of the DUIDD and IDD parameters; it only produces the matrices used by the LMMSE channel estimator.
4. **Evaluate with classical IDD.** Run [`eval_duidd_from_datalake.py`](scripts/eval_duidd_from_datalake.py) with the selected LMMSE configuration, database, and timestamp file, and pass `--idd-only`. This mode uses the untuned IDD paper-default parameters and does not load DUIDD weights.

## <a id="db"></a>Technical Configurations

The measurement campaigns, Data Lake database selectors, finetuning TFRecords, covariance estimates, and evaluation timestamps are documented in [`datasets.md`](../datasets.md).

### Pretraining

DUIDD training and finetuning support any schedule specified by `duidd_schedule`. The documented experiments use `[6, 6]` for the DUIDD finetuning experiment, while the additional configurations below support training the other architectures.

| Schedule | MMSE-PIC Stages (`I`) | Configuration |
| --- | ---: | --- |
| `[12]` | 1 | [`duidd_aerial_12.cfg`](/nrx_duidd/config/duidd_aerial_12.cfg) |
| `[6, 6]` | 2 | [`duidd_aerial_6_6.cfg`](/nrx_duidd/config/duidd_aerial_6_6.cfg) |
| `[4, 4, 4]` | 3 | [`duidd_aerial_4_4_4.cfg`](/nrx_duidd/config/duidd_aerial_4_4_4.cfg) |
| `[3, 3, 3, 3]` | 4 | [`duidd_aerial_3_3_3_3.cfg`](/nrx_duidd/config/duidd_aerial_3_3_3_3.cfg) |

The channel model and training schedule can be changed in the selected configuration before training or finetuning.


### Finetuning and Evaluation


| Dataset | Configuration | Schedule | Finetuning TFRecord | Evaluation Timestamps |
| --- | --- | --- | --- | --- |
| Small Laboratory Jun.&nbsp;2026&nbsp;(2&nbsp;ULL) | [`duidd_pixel9pro_j61_2ULL_slot14_mcs10.cfg`](/nrx_duidd/config/duidd_pixel9pro_j61_2ULL_slot14_mcs10.cfg) | `[6, 6]` | [`Pixel9Pro_2ULLs_12dB_j61_2026_06_04_no_pyaerial_label.tfrecord`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/tfrecord/`Pixel9Pro_2ULLs_12dB_j61_2026_06_04_no_pyaerial_label.tfrecord) | [`timestamps_junj61_2ULL_fix_100.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/timestamps/timestamps_junj61_2ULL_fix_100.pkl)<br>[`timestamps_junj61_2ULL_fix.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/timestamps/timestamps_junj61_2ULL_fix.pkl)|

The finetuning and evaluation data are restricted to slot 14, MCS index 10, and one scrambling configuration.

<!-- #### Classical IDD Evaluation

The following configurations evaluate untuned classical IDD with `--idd-only`. They do not load or train DUIDD weights.

| Schedule | MMSE-PIC Stages (`I`) | Configuration |
| --- | ---: | --- |
| `[12]` | 1 | [`duidd_aerial_12_idd.cfg`](config/duidd_aerial_12_idd.cfg) |
| `[6, 6]` | 2 | [`duidd_aerial_6_6_idd.cfg`](config/duidd_aerial_6_6_idd.cfg) |
| `[4, 4, 4]` | 3 | [`duidd_aerial_4_4_4_idd.cfg`](config/duidd_aerial_4_4_4_idd.cfg) |
| `[3, 3, 3, 3]` | 4 | [`duidd_aerial_3_3_3_3_idd.cfg`](config/duidd_aerial_3_3_3_3_idd.cfg) |

For `I=1`, MMSE-PIC reduces to non-iterative LMMSE detection followed by 12 LDPC message-passing iterations. -->

### LMMSE Channel-Estimation Evaluation

These configurations evaluate classical IDD with LMMSE channel estimates. The `umi` configurations use covariance matrices estimated from synthetic 3GPP UMi channels, while the `hls` configurations use site-specific covariance estimates. They are evaluation-only configurations used with `--idd-only`.

| Dataset | Schedule&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Synthetic UMi Covariance | Site-Specific Covariance |
| --- | --- | --- | --- |
| Small Laboratory Jun.&nbsp;2026 (1 ULL) | `[12]` | [`duidd_12_lmmse_idd_umi_1ull.cfg`](config/duidd_12_lmmse_idd_umi_1ull.cfg) | [`duidd_12_lmmse_idd_hls_1ull.cfg`](/nrx_duidd/config/duidd_12_lmmse_idd_hls_1ull.cfg) |
| Small Laboratory Jun.&nbsp;2026 (2 ULL) | `[12]` | [`duidd_12_lmmse_idd_umi.cfg`](/nrx_duidd/config/duidd_12_lmmse_idd_umi.cfg) | [`duidd_12_lmmse_idd_hls.cfg`](config/duidd_12_lmmse_idd_hls.cfg) |
| Small Laboratory Jun.&nbsp;2026 (2 ULL) | `[6, 6]` | [`duidd_6_6_lmmse_umi.cfg`](/nrx_duidd/config/duidd_6_6_lmmse_umi.cfg) | [`duidd_6_6_lmmse_hls.cfg`](config/duidd_6_6_lmmse_hls.cfg) |
| Small Laboratory Jun.&nbsp;2026 (2 ULL) | `[4, 4, 4]` | [`duidd_4_4_4_lmmse_umi.cfg`](/nrx_duidd/config/duidd_4_4_4_lmmse_umi.cfg) | [`duidd_4_4_4_lmmse_hls.cfg`](config/duidd_4_4_4_lmmse_hls.cfg) |
| Small&nbsp;Laboratory Jun.&nbsp;2026&nbsp;(2&nbsp;ULL) | `[3, 3, 3, 3]` | [`duidd_3_3_3_3_lmmse_umi.cfg`](/nrx_duidd/config/duidd_3_3_3_3_lmmse_umi.cfg) | [`duidd_3_3_3_3_lmmse_hls.cfg`](/nrx_duidd/config/duidd_3_3_3_3_lmmse_hls.cfg) |

Each LMMSE configuration loads label-matched `<label>_{space,freq,time}_cov_mat.npy` files from [`weights/`](weights/). The 1 ULL and 2 ULL evaluations use [`idd_1ULL.pkl`](eval_timestamps/idd_1ULL.pkl) and [`idd.pkl`](eval_timestamps/idd.pkl) as evaluation timestamps, respectively.

## Repository Map

- [`config/`](config/): configurations used for DUIDD training, finetuning, classical IDD, and LMMSE channel-estimation evaluation.
- [`scripts/`](scripts/): scripts for training, synthetic evaluation, real-world evaluation, and synthetic covariance generation.
- [`cov_est/`](cov_est/): tools for site-specific spatial, frequency, and temporal covariance estimation.
- [`utils/`](utils/): DUIDD, IDD, channel, parameter, and training implementations.
- [`notebooks/`](notebooks/): tutorial and analysis notebooks, including the [site-specific training tutorial](notebooks/duidd_site_specific_training_tutorial.ipynb).
- [`weights/`](weights/): pretrained and finetuned DUIDD weights and covariance matrices.
- [`eval_timestamps/`](eval_timestamps/): timestamp files used for deterministic evaluation and covariance estimation.
- [`logs/`](logs/) and [`results/`](results/): training logs and evaluation outputs.

## References

<a id="ref-1"></a>[1] R. Wiesmayr, C. Dick, J. Hoydis, and C. Studer, “DUIDD: Deep-Unfolded Interleaved Detection and Decoding for MIMO Wireless Systems,” in *Proc. Asilomar Conference on Signals, Systems, and Computers*, 2022. Available: https://arxiv.org/abs/2212.07816

<a id="ref-2"></a>[2] R. Wiesmayr, N. B. Baytekin, C. Dick, and C. Studer, “On the Impact of Site-Specific Training for a Real-World 5G NR System,” in *Proc. Asilomar Conference on Signals, Systems, and Computers*, 2026. Available: https://arxiv.org/abs/2609.04004

<a id="ref-3"></a>[3] NVIDIA Corporation, “Aerial CUDA-Accelerated RAN,” release 25-2. Available: https://docs.nvidia.com/aerial/cuda-accelerated-ran/25-2/index.html
