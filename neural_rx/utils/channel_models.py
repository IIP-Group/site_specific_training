# Copy of the original file from NVabs/neural_rx with an 
# additional DataLakeChannel class to allow for real-world training
# by loading and framing pre-computed datasets from a TFRecord file.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Implements different channel models for performance evaluation

from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
import sionna
from sionna.channel import GenerateOFDMChannel, ApplyOFDMChannel, ChannelModel
from sionna.channel.tr38901 import TDL
import pickle
import clickhouse_connect

def gnb_correlation_matrix(num_ant, alpha):
    assert num_ant in [1,2,4,8]
    if num_ant==1:
        exponents = np.array([0])
    elif num_ant==2:
        exponents =  np.array([0, 1])
    elif num_ant==4:
        exponents = np.array([0, 1/9, 4/9, 1])
    elif num_ant==8:
        exponents = np.array([0, 1/49, 4/49, 9/49, 16/49, 25/49, 36/49, 1])
    row = alpha**exponents
    col = np.conj(row)
    r = tf.linalg.LinearOperatorToeplitz(col, row)
    return tf.cast(r.to_dense(), tf.complex64)

def ue_correlation_matrix(num_ant, beta):
    assert num_ant in [1,2,4]
    return gnb_correlation_matrix(num_ant, beta)

class DoubleTDLChannel(tf.keras.layers.Layer):
    """
    Channel model that stacks a 3GPP TDL-B100-400 and TDL-C-300-100 channel
    model. This allows to benchmark a two user system in a 3GPP compliant
    scenario.

    Parameters
    ---------
    carrier_frequency: float
        Carrier frequency of the simulation.

    resource_grid: ResourceGrid
        Resource grid used for the simulation.

    num_rx_ant: int
        Number of receiver antennas.

    num_tx_ant: int
        Number of transmit antennas for each user.

    norm_channel: bool
        If True, the channel is normalized.

    correlation: "low" | "medium" | "high"
        Antenna correlation according to 38.901.

    Input
    -----

    (x, no) or x:
        Tuple or Tensor:

    x :  [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size],
         tf.complex
        Channel inputs

    no : Scalar or Tensor, tf.float
        Scalar or tensor whose shape can be broadcast to the shape of the
        channel outputs

    Output
    -------
    y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        Channel outputs
    h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant,
              num_ofdm_symbols, fft_size], tf.complex
        Channel frequency responses.
    """
    def __init__(self,
                 carrier_frequency,
                 resource_grid,
                 num_rx_ant=4,
                 num_tx_ant=2,
                 norm_channel=False,
                 correlation="low"):
        super().__init__()

        assert correlation in ["low", "medium", "high"]

        print(f"Loading DoubleTDL with {correlation} correlation.")

        if correlation=="low":
            alpha = beta = 0
        elif correlation=="medium":
            alpha = 0.9
            beta = 0.3
        else:
            alpha = 0.9
            beta = 0.9

        tx_corr_mat = ue_correlation_matrix(num_tx_ant, beta)
        rx_corr_mat = gnb_correlation_matrix(num_rx_ant, alpha)

        # TDL B100 model
        delay_spread_1 = 100e-9
        doppler_spread_1 = 400
        speed_1 = doppler_spread_1 * 299792458 / carrier_frequency
        tdl1 = TDL("B100",
           delay_spread_1,
           carrier_frequency,
           max_speed=speed_1,
           num_tx_ant=num_tx_ant,
           num_rx_ant=num_rx_ant,
           rx_corr_mat=rx_corr_mat,
           tx_corr_mat=tx_corr_mat)

        # TDL C300 model
        delay_spread_2 = 300e-9
        doppler_spread_2 = 100
        speed_2 = doppler_spread_2 * 299792458 / carrier_frequency
        tdl2 = TDL("C300",
           delay_spread_2,
           carrier_frequency,
           max_speed=speed_2,
           num_tx_ant=num_tx_ant,
           num_rx_ant=num_rx_ant,
           rx_corr_mat=rx_corr_mat,
           tx_corr_mat=tx_corr_mat)

        self._gen_channel_1 = GenerateOFDMChannel(
                                        tdl1,
                                        resource_grid,
                                        normalize_channel=norm_channel)

        self._gen_channel_2 = GenerateOFDMChannel(
                                        tdl2,
                                        resource_grid,
                                        normalize_channel=norm_channel)

        self._apply_channel = ApplyOFDMChannel()

    # def call(self, x, no=None):
    def call(self, inputs):
        x, no = inputs
        batch_size = tf.shape(x)[0]
        h1 = self._gen_channel_1(batch_size)
        h2 = self._gen_channel_2(batch_size)

        # stack the two models
        h = tf.concat([h1, h2], axis=3)

        y = self._apply_channel([x, h, no])
        return y, h

class DatasetChannel(ChannelModel):
    """Channel model from a TFRecords Dataset File
       The entire dataset is read in memory.

       This version supports XLA acceleration.


    Parameter
    ---------
    tfrecord_filename: str
        Filename of the pre-computed dataset.

    max_num_examples: int
        Max number of samples loaded from dataset. If equals to "-1"
        the entire dataset will be loaded. Defines memory occupation.

    Input
    -----
    batchsize: int
        How many samples shall be returned.

    Output
    ------
    a: [batch_size,...]
        batch_size samples from ``a``. Exact shape depends on dataset.

    tau: [batch_size,...]
        batch_size samples from ``tau``. Exact shape depends on dataset.

    """
    def __init__(self, tfrecord_filename, max_num_examples=-1, training=True,
                 num_tx=1, random_subsampling=True):

        self._training = training
        self._num_tx = num_tx
        self._random_subsampling = random_subsampling

        # Read raw dataset
        dataset = tf.data.TFRecordDataset([tfrecord_filename]) \
                  .map(self._parse_function,
                       num_parallel_calls=tf.data.AUTOTUNE) \
                  .take(max_num_examples) \
                  .batch(1024)

        # Load entire dataset into memory as large tensor
        a = None
        tau = None
        for example in dataset:
            # aggregate all channels in batch direction to multiple users.
            # i.e., move batch direction to num_tx direction.
            #
            # Evaluation data set already has two active users for each batch
            # sample.
            # Thus, every other sample after the aggregation belong to the same
            # user.
            a_ex, tau_ex = example
            a_ex = tf.split(a_ex, a_ex.shape[0], axis=0)
            a_ex = tf.concat(a_ex, axis=3)
            tau_ex = tf.split(tau_ex, tau_ex.shape[0], axis=0)
            tau_ex = tf.concat(tau_ex, axis=2)
            if a is None:
                a = a_ex
                tau = tau_ex
            else:
                a = tf.concat([a, a_ex], axis=3)
                tau = tf.concat([tau, tau_ex], axis=2)

        if training:
            # User positions are randomly sampled. In order to avoid sampling
            # the same positions multiple times within one batch sample, we
            # split the dataset into equal parts for each user to sample from
            # during simulations.
            num_examples = int(a.shape[3]/self._num_tx)
            self._num_examples = num_examples
            self._a = []
            self._tau = []
            for i in range(self._num_tx):
                self._a.append(a[:,:,:,i*num_examples:(i+1)*num_examples])
                self._tau.append(tau[:,:,i*num_examples:(i+1)*num_examples])
        else:
            self._num_examples = a.shape[3]
            self._a = [a,]
            self._tau = [tau,]

    @staticmethod
    def _parse_function(proto):
        description = {
                'a': tf.io.FixedLenFeature([], tf.string),
                'tau': tf.io.FixedLenFeature([], tf.string),
            }
        features = tf.io.parse_single_example(proto, description)
        a = tf.io.parse_tensor(features['a'], out_type=tf.complex64)
        tau = tf.io.parse_tensor(features['tau'], out_type=tf.float32)
        # tf.print(tf.shape(a))
        return a, tau


    def __call__(self, batch_size=None,
                       num_time_steps=None,
                       sampling_frequency=None):
        # default values are used for compatibility with other TF functions.

        # Remark: this is random subsampling
        # random sampling is also done in eval mode; keep in mind that even
        # though UE is on trajectory, we need many slot realizations for good
        # BLER curves (in any case we sample new AWGN noise)

        a = None
        tau = None

        if self._training:
            if not self._random_subsampling:
                ind = tf.random.uniform([batch_size],
                                     maxval=self._num_examples, dtype=tf.int32)
            # randomly subsample from different subsets
            for ue_idx in range(self._num_tx):
                if self._random_subsampling:
                    ind = tf.random.uniform(
                                        [batch_size],
                                        maxval=self._num_examples,
                                        dtype=tf.int32)

                # Gather reshape and combine
                a_ = tf.gather(self._a[ue_idx], ind, axis=3)
                a_ = tf.transpose(a_, perm=[3, 1, 2, 0, 4, 5, 6])
                tau_ = tf.gather(self._tau[ue_idx], ind, axis=2)
                tau_ = tf.transpose(tau_, perm=[2, 1, 0, 3])
                if a is not None:
                    a = tf.concat([a, a_], axis=3)
                    tau = tf.concat([tau, tau_], axis=2)
                else:
                    a = a_
                    tau = tau_
        else:
            # samples in self._a alternating between both trajectories
            if not self._random_subsampling:
                # no random sub-sampling: take subsequent two samples
                ind = tf.random.uniform([batch_size],
                                     maxval=self._num_examples//self._num_tx,
                                     dtype=tf.int32)
                ind = tf.repeat(tf.expand_dims(ind, axis=-1),
                                repeats=self._num_tx, axis=-1)
            else:
                ind = tf.random.uniform([batch_size, self._num_tx],
                                     maxval=self._num_examples//self._num_tx,
                                     dtype=tf.int32)
            # sample subsequent points from all ues
            ind = self._num_tx * ind + tf.expand_dims(
                                        tf.range(self._num_tx, dtype=tf.int32),
                                        axis=0)

            a = tf.transpose(
                    tf.squeeze(tf.gather(self._a[0], ind, axis=3), axis=0),
                    perm=[2,0,1,3,4,5,6])
            tau = tf.transpose(
                    tf.squeeze(tf.gather(self._tau[0], ind, axis=2), axis=0),
                    perm=[1,0,2,3])

        return a, tau


class DataLakeChannel:
    """DataLake channel for loading pre-computed .tfrecord data instead of simulating channels.

    Parameters
    ----------
    max_num_tx : int
        Maximum number of transmit antennas/users.
        
    resource_grid_shape : tuple
        Shape of the resource grid [num_subcarriers, num_ofdm_symbols].
        
    num_rx_ant : int
        Number of receiver antennas.
        
    training : bool
        If True, enables training mode with data iteration.
        
    Methods
    -------
    get_batch(batch_size)
        Returns pre-computed received signal, channel, ground truth bits, and LS channel estimate
        Uses random sampling.

    get_total_samples()
        Returns total number of available data samples.
    """

    def __init__(self, tf_fn, n_size_bwp=4, max_num_tx=None, min_num_tx=None,
                 resource_grid_shape=None, num_rx_ant=None, training=True):
        self.tf_fn = tf_fn
        self.n_size_bwp = n_size_bwp
        self.max_num_tx = max_num_tx
        self.min_num_tx = min_num_tx
        self.resource_grid_shape = resource_grid_shape
        self.num_rx_ant = num_rx_ant
        self.training = training

        self._num_examples = None

        # Mixed-layer mode: when min_num_tx < max_num_tx

        self._mixed = (min_num_tx is not None
                       and max_num_tx is not None
                       and min_num_tx < max_num_tx)

        print(f"Max num TX: {max_num_tx}, Min num TX: {min_num_tx}, "
              f"Resource grid: {resource_grid_shape}, RX antennas: {num_rx_ant}")
        print(f"Training mode: {training}, Mixed-layer mode: {self._mixed}")

        if not self._mixed:
            fn = tf_fn[0] if isinstance(tf_fn, (list, tuple)) else tf_fn
            tfrecord_filename = f'../../finetuning_datasets/{fn}'
            max_num_examples = 10000  # DON'T Load entire dataset
            self._y, self._bits, self._h = self._load_and_frame(tfrecord_filename, max_num_examples)
            self._num_examples = self._y.shape[0]

        else:
            # ---- Mixed-layer mode -------------------------------------------
            # Expect two datasets: tf_fn = [fn_1tx, fn_2tx]
            assert isinstance(tf_fn, (list, tuple)) and len(tf_fn) == 2, \
                "For mixed-layer training tf_fn must be a list/tuple " \
                "[fn_1tx, fn_2tx]."
            max_num_examples = 5000
            fn_1, fn_2 = tf_fn
            y1, bits1, h1 = self._load_and_frame(
                f'../../finetuning_datasets/{fn_1}', max_num_examples)
            y2, bits2, h2 = self._load_and_frame(
                f'../../finetuning_datasets/{fn_2}', max_num_examples)

            # xla things
            bits1 = self._pad_to_ref(bits1, bits2, pad_value=-1)
            h1 = self._pad_to_ref(h1, h2, pad_value=0.)
            assert y1.shape[1:] == y2.shape[1:], "1-tx and 2-tx received grids must have the same shape."

            self._y_1, self._bits_1, self._h_1 = y1, bits1, h1
            self._y_2, self._bits_2, self._h_2 = y2, bits2, h2
            self._num_examples_1 = int(y1.shape[0])
            self._num_examples_2 = int(y2.shape[0])
            self._num_examples = self._num_examples_1 + self._num_examples_2

    def _load_and_frame(self, tfrecord_filename, max_num_examples):
        """Load a tfrecord and apply the PRB-windowing/framing.
        """
        dataset = (
            tf.data.TFRecordDataset([tfrecord_filename])
            .map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
            .take(max_num_examples)
            .batch(batch_size=2, drop_remainder=True)  # Fixed batch size for XLA
        )

        y = None
        bits = None
        h = None

        for example in dataset:
            if len(example) == 4:
                y_ex, bits_ex, h_ex, no_ex = example  # don't need no for NRX
            else:
                y_ex, bits_ex, h_ex = example

            window = 12*self.n_size_bwp
            stride = window  # no overlap

            y_ex = tf.squeeze(y_ex, axis=1)  # remove batch dim
            y_framed = tf.signal.frame(y_ex, frame_length=window, frame_step=stride, axis=-1)
            y_ex = tf.reshape(tf.transpose(y_framed, [0, 4, 1, 2, 3, 5]), [-1, *y_ex.shape[1:-1], window])  # combine the batch and frame dim

            bits_ex = tf.squeeze(bits_ex, axis=1)  # remove batch dim
            bits_framed = tf.signal.frame(bits_ex, frame_length=window, frame_step=stride, axis=-2)
            bits_ex = tf.transpose(bits_framed, [0, 3, 1, 2, 4, 5])
            bits_ex = tf.reshape(bits_ex, [-1, *bits_ex.shape[2:]])

            h_ex = tf.squeeze(h_ex, axis=1)  # remove batch dim
            h_framed = tf.signal.frame(h_ex, frame_length=window, frame_step=stride, axis=-3)
            h_ex = tf.transpose(h_framed, [0, 2, 1, 3, 4, 5])
            h_ex = tf.reshape(h_ex, [-1, *h_ex.shape[2:]])

            if y is None:
                y = y_ex
                bits = bits_ex
                h = h_ex
            else:
                y = tf.concat([y, y_ex], axis=0)
                bits = tf.concat([bits, bits_ex], axis=0)
                h = tf.concat([h, h_ex], axis=0)

        return y, bits, h

    @staticmethod
    def _pad_to_ref(x, ref, pad_value):
        """Pad ``x`` to match ``ref``'s per-sample shape.

        Pads along the single (tx/user) axis that differs between ``x`` and
        ``ref``, appending ``pad_value`` entries. Axis 0 (the examples axis) is
        never padded, as the two datasets may hold different sample counts.
        """
        xs = x.shape.as_list()
        rs = ref.shape.as_list()
        assert len(xs) == len(rs), "1-tx and 2-tx tensors must have the same rank."
        paddings = [[0, 0]]  # never pad the examples axis
        diff_axes = 0

        for ax in range(1, len(xs)):
            sx, sr = xs[ax], rs[ax]
            assert sx is not None and sr is not None, \
                f"static shape required for padding (axis {ax})."
            if sx == sr:
                paddings.append([0, 0])
            else:
                assert sr > sx, \
                    f"1-tx tensor is larger than 2-tx tensor along axis {ax}."
                paddings.append([0, sr - sx])
                diff_axes += 1

        assert diff_axes == 1, f"expected exactly one differing (tx) axis, found {diff_axes}."
        return tf.pad(x, paddings, mode="CONSTANT", constant_values=pad_value)

    def get_batch(self, batch_size, num_tx=None):

        if not self._mixed:
            ind = tf.random.uniform([batch_size], maxval=self._num_examples, dtype=tf.int32)
            rx_tensor = tf.gather(self._y, ind, axis=0)
            bits = [tf.gather(self._bits, ind, axis=0)]
            h = tf.gather(self._h, ind, axis=0)
            active_dmrs = tf.ones([batch_size, self.max_num_tx], tf.float32)
            return rx_tensor, bits, h, active_dmrs

        num_tx = tf.cast(num_tx, tf.int32)

        def _sample(y_pool, bits_pool, h_pool, n):
            ind = tf.random.uniform([batch_size], maxval=n, dtype=tf.int32)
            return (tf.gather(y_pool, ind, axis=0),
                    tf.gather(bits_pool, ind, axis=0),
                    tf.gather(h_pool, ind, axis=0))

        # xla things
        rx_tensor, bits_t, h = tf.cond(
            tf.equal(num_tx, 2),
            lambda: _sample(self._y_2, self._bits_2, self._h_2, self._num_examples_2),
            lambda: _sample(self._y_1, self._bits_1, self._h_1, self._num_examples_1))

        r = tf.range(self.max_num_tx, dtype=tf.int32)
        active = tf.cast(r < num_tx, tf.float32)
        active_dmrs = tf.tile(active[tf.newaxis], [batch_size, 1])

        return rx_tensor, [bits_t], h, active_dmrs

    def get_total_samples(self):
        return self._num_examples
    
    @staticmethod
    def _parse_function(proto):
        description = {
                'y': tf.io.FixedLenFeature([], tf.string),
                'h': tf.io.FixedLenFeature([], tf.string),
                'b': tf.io.FixedLenFeature([], tf.string),
            }
        features = tf.io.parse_single_example(proto, description)
        rx_tensor = tf.io.parse_tensor(features['y'], out_type=tf.complex64)
        bits = tf.io.parse_tensor(features['b'], out_type=tf.int8)
        h = tf.io.parse_tensor(features['h'], out_type=tf.float32)


        # Reshape tensors to known fixed shapes for XLA compatibility
        # rx_tensor: (1, 1, 4, 14, 48) - [batch, rx, rx_ant, ofdm_symbols, subcarriers]
        # rx_tensor = tf.reshape(rx_tensor, [1, 3276, 14, 4])
        
        # tx_tensor: (1, 1, 2, 14, 48) - [batch, tx, tx_ant, ofdm_symbols, subcarriers]
        # tx_tensor = tf.reshape(tx_tensor, [1, 2, 2, 14, 48])
        # h: (1, 1, 4, 2, 2, 14, 48) - [batch, rx, rx_ant, tx, tx_ant, ofdm_symbols, subcarriers]
        # h = tf.reshape(h, [1, 1, 1, 4, 1, 1, 14, 3276])
        
        # bits: (1, 1, 14, 48, 4) - [batch, tx, ofdm_symbols, subcarriers, bits_per_symbol]
        # bits = tf.reshape(bits, [1, 1, 1, 14, 3276, 8])

        # timestamp_ns = 0

        return rx_tensor, bits, h

