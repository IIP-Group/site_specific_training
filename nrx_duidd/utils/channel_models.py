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
        speed_1 = doppler_spread_1 * sionna.SPEED_OF_LIGHT / carrier_frequency
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
        speed_2 = doppler_spread_2 * sionna.SPEED_OF_LIGHT / carrier_frequency
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


_DEFAULT_DATALAKE_NO = np.float32(0.1)


class DataLakeChannel:
    """In-memory loader for datalake TFRecords (``y``, ``b``, optional ``h``/``no``/``b_info``).

    Full-BWP captures are split into non-overlapping ``n_size_bwp``-PRB windows
    (default 4 PRB). Set ``n_size_bwp`` to the full BWP width (e.g. 273) to
    disable framing: ``window == stride == full grid width`` yields exactly one
    window per example, i.e. the untouched full-BWP slot. Full-BWP is required
    when supervising the TB decoder, since a PRB window of a captured multi-CB
    codeword is not itself a valid codeword.

    Supports optional mixed-layer training via two TFRecords.

    ``h`` is optional: DUIDD (``lslin``/``lsnn``/``lmmse``) estimates the channel
    online from ``y``. Legacy neural_rx TFRecords that store LS ``h`` still load.
    """

    def __init__(self, tf_fn, n_size_bwp=4, max_num_tx=None, min_num_tx=None,
                 resource_grid_shape=None, num_rx_ant=None, training=True,
                 datalake_no=None):
        self.tf_fn = tf_fn
        self.n_size_bwp = n_size_bwp
        self.max_num_tx = max_num_tx
        self.min_num_tx = min_num_tx
        self.resource_grid_shape = resource_grid_shape
        self.num_rx_ant = num_rx_ant
        self.training = training
        self._default_no = (
            float(datalake_no) if datalake_no is not None else _DEFAULT_DATALAKE_NO)
        self._has_h = None

        self._num_examples = None
        self._mixed = (min_num_tx is not None
                       and max_num_tx is not None
                       and min_num_tx < max_num_tx)

        print(f"DataLakeChannel: max_num_tx={max_num_tx}, "
              f"min_num_tx={min_num_tx}, mixed={self._mixed}, "
              f"default_no={self._default_no}")

        if not self._mixed:
            fn = tf_fn[0] if isinstance(tf_fn, (list, tuple)) else tf_fn
            tfrecord_filename = f'../../finetuning_datasets/{fn}'
            self._y, self._bits, self._h, self._no, self._bits_info = self._load_and_frame(
                tfrecord_filename, max_num_examples=10000)
            self._num_examples = int(self._y.shape[0])
        else:
            assert isinstance(tf_fn, (list, tuple)) and len(tf_fn) == 2
            max_num_examples = 5000
            fn_1, fn_2 = tf_fn
            y1, bits1, h1, no1, bits_info1 = self._load_and_frame(
                f'../../finetuning_datasets/{fn_1}', max_num_examples)
            y2, bits2, h2, no2, bits_info2 = self._load_and_frame(
                f'../../finetuning_datasets/{fn_2}', max_num_examples)
            bits1 = self._pad_to_ref(bits1, bits2, pad_value=-1)
            if bits_info1 is not None and bits_info2 is not None:
                bits_info1 = self._pad_to_ref(bits_info1, bits_info2, pad_value=0)
            else:
                bits_info1 = bits_info2 = None
            if h1 is not None and h2 is not None:
                h1 = self._pad_to_ref(h1, h2, pad_value=0.)
            else:
                h1 = h2 = None
            self._y_1, self._bits_1, self._h_1, self._no_1, self._bits_info_1 = y1, bits1, h1, no1, bits_info1
            self._y_2, self._bits_2, self._h_2, self._no_2, self._bits_info_2 = y2, bits2, h2, no2, bits_info2
            self._num_examples_1 = int(y1.shape[0])
            self._num_examples_2 = int(y2.shape[0])
            self._num_examples = self._num_examples_1 + self._num_examples_2

    def _apply_prb_framing(self, y_ex, bits_ex, no_ex, h_ex=None, b_info_ex=None):
        """Slice full-BWP tensors into non-overlapping ``n_size_bwp`` PRB windows."""
        window = 12 * self.n_size_bwp
        stride = window

        y_ex = tf.squeeze(y_ex, axis=1)
        y_framed = tf.signal.frame(
            y_ex, frame_length=window, frame_step=stride, axis=-1)
        y_ex = tf.reshape(
            tf.transpose(y_framed, [0, 4, 1, 2, 3, 5]),
            [-1, *y_ex.shape[1:-1], window])

        bits_ex = tf.squeeze(bits_ex, axis=1)
        if bits_ex.shape.rank == 4:
            bits_ex = tf.expand_dims(bits_ex, axis=1)
        bits_framed = tf.signal.frame(
            bits_ex, frame_length=window, frame_step=stride, axis=-2)
        bits_ex = tf.transpose(bits_framed, [0, 3, 1, 2, 4, 5])
        bits_ex = tf.reshape(bits_ex, [-1, *bits_ex.shape[2:]])

        framed_h = None
        if h_ex is not None:
            h_ex = tf.squeeze(h_ex, axis=1)
            h_framed = tf.signal.frame(
                h_ex, frame_length=window, frame_step=stride, axis=-3)
            framed_h = tf.reshape(
                tf.transpose(h_framed, [0, 2, 1, 3, 4, 5]),
                [-1, *h_ex.shape[2:]])

        no_ex = tf.reshape(no_ex, [-1])
        repeat_factor = tf.shape(y_ex)[0] // tf.shape(no_ex)[0]
        no_ex = tf.repeat(no_ex, repeats=repeat_factor, axis=0)

        if b_info_ex is not None:
            b_info_ex = tf.squeeze(b_info_ex, axis=1)
            b_info_ex = tf.repeat(b_info_ex, repeats=repeat_factor, axis=0)

        return y_ex, bits_ex, framed_h, no_ex, b_info_ex

    def _load_and_frame(self, tfrecord_filename, max_num_examples):
        dataset = (
            tf.data.TFRecordDataset([tfrecord_filename])
            .map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
            .take(max_num_examples)
            .batch(batch_size=2, drop_remainder=True)
        )

        y = bits = h = no = bits_info = None
        has_h = has_b_info = None
        for y_ex, bits_ex, h_raw, no_ex, b_info_raw in dataset:
            if has_h is None:
                has_h = bool(
                    tf.reduce_min(tf.strings.length(h_raw)).numpy() > 0)
                self._has_h = has_h
                if not has_h:
                    print("DataLakeChannel: no 'h' in TFRecord "
                          "(channel will be estimated online).")

            if has_b_info is None:
                has_b_info = bool(
                    tf.reduce_min(tf.strings.length(b_info_raw)).numpy() > 0)
                self._has_b_info = has_b_info
                if not has_b_info:
                    print("DataLakeChannel: no 'b_info' in TFRecord.")

            if has_h:
                h_ex = tf.stack([
                    tf.io.parse_tensor(h_raw[i], out_type=tf.float32)
                    for i in range(int(h_raw.shape[0]))
                ])
            else:
                h_ex = None

            if has_b_info:
                b_info_ex = tf.stack([
                    tf.io.parse_tensor(b_info_raw[i], out_type=tf.int8)
                    for i in range(int(b_info_raw.shape[0]))
                ])
            else:
                b_info_ex = None

            y_ex, bits_ex, h_ex, no_ex, b_info_ex = self._apply_prb_framing(
                y_ex, bits_ex, no_ex, h_ex=h_ex, b_info_ex=b_info_ex)

            if y is None:
                y, bits, h, no, bits_info = y_ex, bits_ex, h_ex, no_ex, b_info_ex
            else:
                y = tf.concat([y, y_ex], axis=0)
                bits = tf.concat([bits, bits_ex], axis=0)
                if h is not None and h_ex is not None:
                    h = tf.concat([h, h_ex], axis=0)
                no = tf.concat([no, no_ex], axis=0)
                if bits_info is not None and b_info_ex is not None:
                    bits_info = tf.concat([bits_info, b_info_ex], axis=0)

        return y, bits, h, no, bits_info

    @staticmethod
    def _pad_to_ref(x, ref, pad_value):
        xs = x.shape.as_list()
        rs = ref.shape.as_list()
        paddings = [[0, 0]]
        diff_axes = 0
        for ax in range(1, len(xs)):
            sx, sr = xs[ax], rs[ax]
            if sx == sr:
                paddings.append([0, 0])
            else:
                paddings.append([0, sr - sx])
                diff_axes += 1
        assert diff_axes == 1
        return tf.pad(x, paddings, mode="CONSTANT", constant_values=pad_value)

    def get_batch(self, batch_size, num_tx=None, return_b_info=False):
        if not self._mixed:
            ind = tf.random.uniform(
                [batch_size], maxval=self._num_examples, dtype=tf.int32)
            rx_tensor = tf.gather(self._y, ind, axis=0)
            bits = [tf.gather(self._bits, ind, axis=0)]
            no = tf.gather(self._no, ind, axis=0)
            active_dmrs = tf.ones([batch_size, self.max_num_tx], tf.float32)
            if return_b_info:
                if self._bits_info is None:
                    raise ValueError("TFRecord has no 'b_info' feature.")
                return (rx_tensor, bits, no, active_dmrs,
                        tf.gather(self._bits_info, ind, axis=0))
            return rx_tensor, bits, no, active_dmrs

        assert num_tx is not None
        num_tx = tf.cast(num_tx, tf.int32)

        def _sample(y_pool, bits_pool, no_pool, bits_info_pool, n):
            ind = tf.random.uniform([batch_size], maxval=n, dtype=tf.int32)
            out = (tf.gather(y_pool, ind, axis=0),
                   tf.gather(bits_pool, ind, axis=0),
                   tf.gather(no_pool, ind, axis=0))
            if return_b_info:
                if bits_info_pool is None:
                    raise ValueError("TFRecord has no 'b_info' feature.")
                out = out + (tf.gather(bits_info_pool, ind, axis=0),)
            return out

        if return_b_info:
            sampled = tf.cond(
                tf.equal(num_tx, 2),
                lambda: _sample(self._y_2, self._bits_2, self._no_2,
                                self._bits_info_2, self._num_examples_2),
                lambda: _sample(self._y_1, self._bits_1, self._no_1,
                                self._bits_info_1, self._num_examples_1))
            rx_tensor, bits_t, no, bits_info = sampled
        else:
            rx_tensor, bits_t, no = tf.cond(
                tf.equal(num_tx, 2),
                lambda: _sample(self._y_2, self._bits_2, self._no_2, None,
                                self._num_examples_2)[:3],
                lambda: _sample(self._y_1, self._bits_1, self._no_1, None,
                                self._num_examples_1)[:3])

        r = tf.range(self.max_num_tx, dtype=tf.int32)
        active = tf.cast(r < num_tx, tf.float32)
        active_dmrs = tf.tile(active[tf.newaxis], [batch_size, 1])
        if return_b_info:
            return rx_tensor, [bits_t], no, active_dmrs, bits_info
        return rx_tensor, [bits_t], no, active_dmrs

    def get_total_samples(self):
        return self._num_examples

    def _parse_function(self, proto):
        description = {
            'y': tf.io.FixedLenFeature([], tf.string),
            'b': tf.io.FixedLenFeature([], tf.string),
            'h': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'no': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'b_info': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
        features = tf.io.parse_single_example(proto, description)
        rx_tensor = tf.io.parse_tensor(features['y'], out_type=tf.complex64)
        bits = tf.io.parse_tensor(features['b'], out_type=tf.int8)
        no_bytes = features['no']
        no = tf.cond(
            tf.equal(tf.strings.length(no_bytes), 0),
            lambda: tf.constant(self._default_no, dtype=tf.float32),
            lambda: tf.reshape(
                tf.io.parse_tensor(no_bytes, out_type=tf.float32), []))
        # Keep raw bytes; decode later only if present (avoids tf.cond shape issues)
        return rx_tensor, bits, features['h'], no, features['b_info']
