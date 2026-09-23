# Datasets and Experiment Mapping

This document describes the real-world measurement datasets used for the NRX, MDX, and DUIDD experiments and maps them to the corresponding Data Lake databases, finetuning TFRecord files, and evaluation timestamp lists.

The datasets consist of 5G NR physical uplink shared channel (PUSCH) slot samples, each comprising the OFDM-domain receive signals of one complete uplink transmission slot. They cover three deployment scenarios: an indoor small laboratory, an indoor large office floor with both line-of-sight and non-line-of-sight propagation conditions, and an outdoor high-mobility scenario with a UE mounted on an unmanned aerial vehicle (UAV).

The two indoor scenarios were measured during campaigns in November 2025 and June 2026. The outdoor UAV measurements belong to the November 2025 campaign. The November 2025 datasets contain single-layer transmissions, while the June 2026 datasets contain separate single-layer and dual-layer measurements. An uplink layer (ULL) denotes one spatial stream.

All data required to reproduce the experiments presented in [1, 2] is organized as follows:

- The Data Lake database contains the captured raw measurements. The fronthaul `fh` table contains the received IQ samples in the OFDM-domain, while the `fapi` table contains the corresponding transmission configuration, decoding result, and payload information.
- The `--db` argument selects the correct database for TFRecord extraction in [`create_tfrecord_from_data_multicell.py`](neural_rx/scripts/create_tfrecord_from_data_multicell.py) and for receiver evaluation in [`eval_neural_receiver_from_datalake.py`](neural_rx/scripts/eval_neural_receiver_from_datalake.py), [`eval_mdx_from_datalake_TF.py`](mdx/scripts/eval_mdx_from_datalake_TF.py), and [`eval_duidd_from_datalake.py`](nrx_duidd/scripts/eval_duidd_from_datalake.py).
- The TFRecord files contain the slot samples extracted for finetuning, together with the corresponding ground-truth bit labels for supervised training, and with the LS channel estimates and noise variance estimates to be fed as features to the receiver (if applicable).
- A hardcoded finetuning / testing split specifies the partition used for finetuning and testing. The November small-laboratory and large-office-floor datasets use measurements from one UE for finetuning and measurements from the other UE for testing. The November outdoor dataset and the June datasets instead use temporally disjoint partitions: slots received before the specified timestamp are used for finetuning, while slots received after it are used for testing. The splits were selected by inspection and are hardcoded. In all cases, the samples for the evaluation timestamps are selected randomly within their partition.

## Data Lake Tables

Each imported measurement database provides two ClickHouse tables: the FAPI table `export_fapi_<database>` contains the PUSCH scheduling and decoding metadata, and the FH table `export_fh_<database>` contains the corresponding received fronthaul samples. The `<database>` suffix follows the database name listed below. Further information with regards to the tables and their content is provided below.

### FAPI Table

Each row in `export_fapi_<database>` describes one scheduled PUSCH transmission and its decoding result. The extraction and evaluation scripts use the following fields:

| Field | Description |
| --- | --- |
| `TsTaiNs` | Timestamp stored as `DateTime64`. It is used to match the FAPI row to the corresponding fronthaul samples. |
| `CellId` | Identifier of the serving cell. |
| `SFN`, `Slot` | System frame number and slot index used to reconstruct the transmission timing. |
| `rnti` | Radio Network Temporary Identifier of the scheduled UE. It affects the scrambling. |
| `rbStart`, `rbSize` | Start and size of the PUSCH allocation in physical resource blocks. The documented experiments use full-band allocations with 273 resource blocks. |
| `StartSymbolIndex`, `NrOfSymbols` | First OFDM symbol and number of OFDM symbols occupied by the PUSCH allocation. The selected measurements use 13 PUSCH symbols. |
| `CyclicPrefix` | Cyclic-prefix configuration used to reconstruct the carrier configuration. |
| `nrOfLayers` | Number of spatial uplink layers in the transmission: one or two for the datasets documented here. |
| `mcsTable`, `mcsIndex` | Modulation-and-coding-scheme table and index. The general experiments retain 16-QAM transmissions with MCS indices 6–10; the DUIDD finetuning experiment is restricted to MCS index 10. |
| `qamModOrder`, `targetCodeRate`, `TBSize` | Modulation order, target code rate, and scheduled transport-block size used to reconstruct the PUSCH receiver configuration. |
| `ulDmrsSymbPos` | Bitmap specifying the OFDM symbols containing demodulation reference signals (DMRS). |
| `dmrsConfigType`, `dmrsPorts`, `numDmrsCdmGrpsNoData` | DMRS configuration type, active-port bitmap, and number of code-division-multiplexing groups without data. These fields determine the pilot locations for each layer. |
| `ulDmrsScramblingId`, `SCID` | DMRS scrambling identifier and sequence-initialization selector. |
| `dataScramblingId` | Scrambling identifier used for the coded PUSCH data. |
| `rvIndex`, `harqProcessID` | Redundancy version and HARQ process identifier. The scripts use them to locate a later successful retransmission of a failed transport block. HARQ retransmissions reuse the same process ID with a new redundancy version. |
| `tbCrcFail` | Transport-block CRC result: `0` for a successful decode and `1` for a failed decode. |
| `pduData` | Decoded transport-block payload bytes. For a successful transmission, these bytes provide the information-bit label. For a failed transmission the field is left empty. |

### Fronthaul Table

Each row in `export_fh_<database>` contains the received samples for one cell and slot:

| Field | Description |
| --- | --- |
| `TsTaiNs` | Slot timestamp used to match the samples to the FAPI scheduling record. |
| `CellId` | Cell identifier used to disambiguate simultaneous records at the same timestamp. |
| `fhData` | Packed complex OFDM-domain IQ samples captured from the serving radio unit. The scripts interpret the packed real and imaginary components as complex values and reshape them into four receive antennas, 14 OFDM symbols, and 3276 subcarriers. |

The FAPI table supplies the transmission configuration and transport-block label, whereas `fhData` supplies the received signal processed by the receiver. The extraction and evaluation scripts match the received raw IQ sample with the transmission configuration through the `TsTaiNs` and the `CellId` parameters..



## User Equipments (UEs) Used
| Device | Index |
| --- | --- |
Samsung Galaxy S23 | UE 0 |
iPhone 14 Pro | UE 1 |
iPhone 16e | UE 2 |
Google Pixel 9 Pro | UE 3 |

## <a id="db"></a>Data Lake Databases


The table contains the download links for all of the databases used for the experiment. The databases are presented as a compressed archive as a  `<database_name>.tar.zst` file. The `<database_name>` name creates a common naming convention for the extracted files that is relevant to each experiment. The downloaded archives contain two `.parquet` files corresponding to the `fh` and the `fapi` table, with filenames `export_{fh, fapi}_<database_name>.parquet`.

| Measurement Scenario | Measurement Campaign | Nr. of ULLs | `db` Index | Measurement UEs | Finetuning / Testing Split | Database |
| --- | --- | --- | --- | --- | --- | --- |
| Indoor Small Laboratory | Nov. 2025 | 1 | `0` |  0 and 1 | Finetuning: UE 0; testing: UE 1 |[`nrx_measurement_2025_11_03.tar.zst`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/nov_25/database/nrx_measurement_2025_11_03.tar.zst) |
| Indoor Small Laboratory | Jun. 2026 | 1 | `9` |  0 | Finetuning: before `2026-06-04 12:59:00.372`; testing: after | [`sgs23_1ULLs_5dB_j61_2026_06_04.tar.zst`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/database/finetuning_measurements_1_2_ULLs_j61_2026_06_04.tar.zst) |
| Indoor Small Laboratory | Jun. 2026 | 2 | `8` |  3 | Finetuning: before `2026-06-04 13:27:17.372`; testing: after |[`Pixel9Pro_2ULLs_12dB_j61_2026_06_04.tar.zst`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/database/finetuning_measurements_1_2_ULLs_j61_2026_06_04.tar.zst) |
| Indoor Large Office Floor | Nov. 2025 | 1 | `1` | 0 and 1 | Finetuning: UE 0; testing: UE 1 | [`nrx_jfloor_1st_2025_12_02.tar.zst`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_floor/nov_25/database/nrx_jfloor_1st_2025_12_02.tar.zst) |
| Indoor Large Office Floor | Jun. 2026 | 1 | `11` |  0 | Finetuning: before `2026-06-10 15:04:00`; testing: after |[`sgs23_1ULLs_5dB_jFloorRealLoop_2026_06_10.tar.zst`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_floor/jun_26/database/finetuning_measurements_1_2_ULLs_jFloorLoop_2026_06_10.tar.zst) |
| Indoor Large Office Floor | Jun. 2026 | 2 | `10` |  3 | Finetuning: before `2026-06-10 14:47:00`; testing: after | [`Pixel9Pro_2ULLs_12dB_jFloorRealLoop_2026_06_10.tar.zst`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_floor/jun_26/database/finetuning_measurements_1_2_ULLs_jFloorLoop_2026_06_10.tar.zst) |
| Outdoor UAV | Nov. 2025 | 1 | `2` | 2 | Finetuning: before`2025-12-03 12:21:05.202`; testing: after |[`nrx_drone_2025_12_03.tar.zst`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/outdoor_uav/nov_25/database/nrx_drone_2025_12_03.tar.zst) | 


## Finetuning TFRecords

The following table lists and provides a direct download link for all the TFRecord files that were extracted from the corresponding databases using the [`create_tfrecord_from_data_multicell.py`](neural_rx/scripts/create_tfrecord_from_data_multicell.py) script. They contain 7500 samples each, and have all the relevant data for finetuning.

The filenames correspond to the database names from the [Data Lake Databases](#db) section, named as `<database_name>.tfrecord`. Copy or link the selected TFRecord into [`finetuning_datasets/`](finetuning_datasets/).

| Measurement | TFRecord File |
| --- | --- |
| Small Laboratory Nov. 2025 (1 ULL) | [`nrx_measurement_2025_11_03.tfrecord`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/nov_25/tfrecord/nrx_measurement_2025_11_03.tfrecord) |
| Small Laboratory Jun. 2026 (1 ULL) | [`sgs23_1ULLs_5dB_j61_2026_06_04.tfrecord`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/tfrecord/sgs23_1ULLs_5dB_j61_2026_06_04.tfrecord) |
| Small Laboratory Jun. 2026  (2 ULLs) | [`Pixel9Pro_2ULLs_12dB_j61_2026_06_04.tfrecord`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/tfrecord/Pixel9Pro_2ULLs_12dB_j61_2026_06_04.tfrecord) |
| Large Office Floor Nov. 2025 (1 ULL) | [`nrx_jfloor_1st_2025_12_02.tfrecord`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_floor/nov_25/tfrecord/nrx_jfloor_1st_2025_12_02.tfrecord) |
| Large Office Floor Jun. 2026  (1 ULL) | [`sgs23_1ULLs_5dB_jFloorRealLoop_2026_06_10.tfrecord`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_floor/jun_26/tfrecord/sgs23_1ULLs_5dB_jFloorRealLoop_2026_06_10.tfrecord) |
| Large Office Floor Jun. 2026  (2 ULLs) | [`Pixel9Pro_2ULLs_12dB_jFloorRealLoop_2026_06_10.tfrecord`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_floor/jun_26/tfrecord/Pixel9Pro_2ULLs_12dB_jFloorRealLoop_2026_06_10.tfrecord) |
| Outdoor UAV Nov. 2025 (1 ULL) | [`nrx_drone_2025_12_03.tfrecord`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/outdoor_uav/nov_25/tfrecord/nrx_drone_2025_12_03.tfrecord) |


## Evaluation Timestamps

The following timestamp pickle files are used select deterministic measurement slots used for evaluation by the experiments:

| Measurement | Use Case | Number of Slots |Timestamp Files | 
| --- | --- | --- | --- |
| Small Laboratory Nov. 2025 (1 ULL) | NRX evaluation | 2,000 | [`timestamps_novj61_ue1.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/nov_25/timestamps/timestamps_novj61_ue1.pkl)  |
| Small Laboratory Jun. 2026 (1 ULL) | NRX / MDX evaluation | 1,000 | [`timestamps_junj61_1ULL.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/timestamps/timestamps_junj61_1ULL.pkl) |
| Small Laboratory Jun. 2026 (1 ULL) | DUIDD (LMMSE) evaluation | 100 | [`timestamps_junj61_1ULL_100.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/timestamps/timestamps_junj61_1ULL_100.pkl) |
| Small Laboratory Jun. 2026 (2 ULLs) | NRX evaluation | 1,000 | [`timestamps_junj61_2ULL.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/timestamps/timestamps_junj61_2ULL.pkl) |
| Small Laboratory Jun. 2026 (2 ULLs) | DUIDD (LMMSE) evaluation | 100 | [`timestamps_junj61_2ULL_100.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/timestamps/timestamps_junj61_2ULL_100.pkl) |
| Small Laboratory Jun. 2026 (2 ULLs) | DUIDD evaluation (fixed scrambling) | 100 | [`timestamps_junj61_2ULL_fix_100.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/timestamps/timestamps_junj61_2ULL_fix_100.pkl) |
| Small Laboratory Jun. 2026 (2 ULLs) | DUIDD evaluation (fixed scrambling) | 1,000 | [`timestamps_junj61_2ULL_fix.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_lab/jun_26/timestamps/timestamps_junj61_2ULL_fix.pkl) |
| Large Office Floor Nov. 2025 (1 ULL) | NRX evaluation | 1,000 | [`timestamps_novjfloor_ue1.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_floor/nov_25/timestamps/timestamps_novjfloor_ue1.pkl)
| Large Office Floor Jun. 2026 (1 ULL) | MDX evaluation | 1,000 | [`timestamps_junjfloor_1ULL.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/indoor_floor/jun_26/timestamps/timestamps_junjfloor_1ULL.pkl) |
| Outdoor Nov. 2025 (1 ULL) | NRX evaluation | 1,000 | [`timestamps_drone3.pkl`](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/finetuning_datasets/outdoor_uav/nov_25/timestamps/timestamps_drone3.pkl) |


## Estimated Site-Specific Covariance Matrices

The following covariance estimates are used by the site-specific parameter adaptation experiments:

| Source of Channel Data | Use Case | Numpy File |
| --- | --- | --- |
| 3GPP UMi | LMMSE channel estimation with covariance matrix estimates from synthetic data | [duidd_lmmse_umi_space_cov_mat.npy](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/covariance_estimates/duidd_lmmse_umi_space_cov_mat.npy) [duidd_lmmse_umi_freq_cov_mat.npy](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/covariance_estimates/duidd_lmmse_umi_freq_cov_mat.npy) [duidd_lmmse_umi_time_cov_mat.npy](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/covariance_estimates/duidd_lmmse_umi_time_cov_mat.npy) |
| Small Laboratory Jun. 2026 (1+2 ULLs) | LMMSE channel estimation with site-specific covariance matrices | [duidd_lmmse_junj61_space_cov_mat.npy](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/covariance_estimates/duidd_lmmse_junj61_space_cov_mat.npy) [duidd_lmmse_junj61_freq_cov_mat.npy](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/covariance_estimates/duidd_lmmse_junj61_freq_cov_mat.npy) [duidd_lmmse_junj61_time_cov_mat.npy](https://iis-people.ee.ethz.ch/~iisdatasets/iip/site_specific_training/covariance_estimates/duidd_lmmse_junj61_time_cov_mat.npy) |

## References

<a id="ref-1"></a>[1] R. Wiesmayr, N. B. Baytekin, C. Dick, and C. Studer, “On the Impact of Site-Specific Training for a Real-World 5G NR System,” in *Proc. Asilomar Conference on Signals, Systems, and Computers*, 2026. Available: https://arxiv.org/abs/2609.04004

<a id="ref-2"></a>[2] N. B. Baytekin, R. Wiesmayr, S. Cammerer, C. Dick, and C. Studer, “Site-Specific Finetuning of Neural Receivers with Real-World 5G NR Measurements,” arXiv:2603.09644, Mar. 2026. Available: https://arxiv.org/abs/2603.09644