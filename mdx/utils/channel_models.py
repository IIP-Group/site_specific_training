# Copy of the original file from MAbdollahpour/MDX with modifications to support MDX and extra features.
# Additional DataLakeChannel class to allow for real-world training
# by loading and framing pre-computed datasets from a TFRecord file.

# Modified to support MDX & Extra features
# NTDL added in graph mode with random delay-spread
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

# Implements different channel models for performance evaluation

from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
import sionna
from sionna.channel import GenerateOFDMChannel, ApplyOFDMChannel, ChannelModel
from sionna.channel.tr38901 import CDL #,TDL
from .tdl import TDL
from sionna.channel.utils import exp_corr_mat
import random

# Serialized default for backwards compatible TFRecords (no per-example noise).
# This must be a Python `bytes` object for `tf.io.FixedLenFeature(default_value=...)`.
_DATALAKE_DEFAULT_NO_BYTES: bytes = tf.io.serialize_tensor(
    tf.constant(0.1, dtype=tf.float32)
).numpy()

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







class NTDLChannel(tf.keras.layers.Layer):

    def __init__(self,
                 carrier_frequency,
                 resource_grid,
                 num_rx_ant=16,
                 num_tx_ant=2,
                 max_num_tx=4,
                 norm_channel=False,
                 correlation="low",
                 tdl_models=["A"], # A, B, C, D, E
                 delay_spread_min=10,   # in nano seconds
                 delay_spread_max=300,  # in nano seconds
                 doppler_shift_max=325  # Hz
                 ):
        super().__init__()




        assert correlation in ["low", "medium", "high"]

        print(f"Loading NTDL {tdl_models} with {correlation} correlation.\n")


        if correlation=="low":
            alpha = beta = 0
        elif correlation=="medium":
            alpha = 0.9
            beta = 0.3
        else:
            alpha = 0.9
            beta = 0.9


        self._tx_corr_mat = exp_corr_mat(beta, num_tx_ant)
        self._rx_corr_mat = exp_corr_mat(alpha, num_rx_ant)

        self._max_num_tx = max_num_tx
        self._num_rx_ant = num_rx_ant
        self._num_tx_ant = num_tx_ant
        self._delay_spread_min = delay_spread_min
        self._delay_spread_max = delay_spread_max
        self._doppler_shift_max = doppler_shift_max
        self._carrier_frequency = carrier_frequency
        self._tdl_models = tdl_models
        self._resource_grid = resource_grid
        self._norm_channel = norm_channel
        self._apply_channel = ApplyOFDMChannel()
        self._speed = self._doppler_shift_max * sionna.SPEED_OF_LIGHT / self._carrier_frequency
        self._num_tdls = len(tdl_models)


        if len(tdl_models) < max_num_tx:
            # Calculate how many times to repeat the list so that its length becomes greater than max_num_tx
            repeat_times = (max_num_tx // len(tdl_models)) + 1
            tdl_models = tdl_models * repeat_times


        self._gens = []
        self._tdls = []
        for model in tdl_models:

            # delay_spread = tf.random.uniform(
            #     shape=(), 
            #     minval=self._delay_spread_min, 
            #     maxval=self._delay_spread_max, 
            #     dtype=tf.float32
            # )
            # delay_spread = delay_spread*1e-9 # seconds
            delay_spread = delay_spread_max*1e-9 # seconds

            # Randomly select a model if more than one provided
            # selected_model = self._tdl_models[0]
            # if len(self._tdl_models)>1:
            #     selected_model = random.choice(self._tdl_models)


            tdl = TDL(model,
                      delay_spread,
                      self._carrier_frequency,
                      max_speed=self._speed,
                      num_tx_ant=self._num_tx_ant,
                      num_rx_ant=self._num_rx_ant,
                      rx_corr_mat=self._rx_corr_mat,
                      tx_corr_mat=self._tx_corr_mat)

            gen_channel = GenerateOFDMChannel_(
                tdl,
                self._resource_grid,
                normalize_channel=self._norm_channel
            )
            # self._tdls.append(tdl)
            self._gens.append(gen_channel)


    def call(self, inputs):

        x, no = inputs
        size = tf.shape(x)
        batch_size = size[0]
        num_tx = size[1]

        k = len(self._gens) 

        # Step 1: Compute all channels with random delay spread
        outputs = [
            gen(
                batch_size,
                delay_spread=tf.random.uniform(
                    shape=(),
                    minval=self._delay_spread_min,
                    maxval=self._delay_spread_max,
                    dtype=tf.float32
                ) * 1e-9  # nano Seconds
            )
            for gen in self._gens
        ]

        # Step 2: Stack outputs into a tensor
        all_outputs = tf.stack(outputs, axis=0)

        # Step 3: Shuffle indices
        indices = tf.range(k)
        shuffled_indices = tf.random.shuffle(indices)

        # Step 4: Select first n indices
        selected_indices = shuffled_indices[:num_tx]

        # Step 5: Gather selected outputs
        selected_outputs = tf.gather(all_outputs, selected_indices, axis=0)

        # Step 6: Unstack into a list
        h_list = tf.unstack(selected_outputs, axis=0)

        h = tf.concat(h_list, axis=3)
        y = self._apply_channel([x, h, no])
        # print(f"[tdl call]h:{tf.shape(h)}, y:{tf.shape(y)}, x:{x.shape}")
        return y, h









#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class for generating channel frequency responses"""


from sionna.channel.utils import subcarrier_frequencies, cir_to_ofdm_channel
class GenerateOFDMChannel_:
    # pylint: disable=line-too-long
    r"""GenerateOFDMChannel(channel_model, resource_grid, normalize_channel=False)

    Generate channel frequency responses.
    The channel impulse response is constant over the duration of an OFDM symbol.

    Given a channel impulse response
    :math:`(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1`, generated by the ``channel_model``,
    the channel frequency response for the :math:`s^{th}` OFDM symbol and
    :math:`n^{th}` subcarrier is computed as follows:

    .. math::
        \widehat{h}_{s, n} = \sum_{m=0}^{M-1} a_{m}(s) e^{-j2\pi n \Delta_f \tau_{m}}

    where :math:`\Delta_f` is the subcarrier spacing, and :math:`s` is used as time
    step to indicate that the channel impulse response can change from one OFDM symbol to the
    next in the event of mobility, even if it is assumed static over the duration
    of an OFDM symbol.

    Parameters
    ----------
    channel_model : :class:`~sionna.channel.ChannelModel` object
        An instance of a :class:`~sionna.channel.ChannelModel` object, such as
        :class:`~sionna.channel.RayleighBlockFading` or
        :class:`~sionna.channel.tr38901.UMi`.

    resource_grid : :class:`~sionna.ofdm.ResourceGrid`
        Resource grid

    normalize_channel : bool
        If set to `True`, the channel is normalized over the resource grid
        to ensure unit average energy per resource element. Defaults to `False`.

    dtype : tf.DType
        Complex datatype to use for internal processing and output.
        Defaults to `tf.complex64`.

    Input
    -----

    batch_size : int
        Batch size. Defaults to `None` for channel models that do not require this paranmeter.

    Output
    -------
    h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex
        Channel frequency responses
    """

    def __init__(self, channel_model, resource_grid, normalize_channel=False,
                 dtype=tf.complex64):

        # Callable used to sample channel input responses
        self._cir_sampler = channel_model

        # We need those in call()
        self._num_ofdm_symbols = resource_grid.num_ofdm_symbols
        self._subcarrier_spacing = resource_grid.subcarrier_spacing
        self._num_subcarriers = resource_grid.fft_size
        self._normalize_channel = normalize_channel
        self._sampling_frequency = 1./resource_grid.ofdm_symbol_duration

        # Frequencies of the subcarriers
        self._frequencies = subcarrier_frequencies(self._num_subcarriers,
                                                   self._subcarrier_spacing,
                                                   dtype)

    def __call__(self, batch_size=None, delay_spread=None):

        # Sample channel impulse responses
        h, tau = self._cir_sampler( batch_size,
                                    self._num_ofdm_symbols,
                                    self._sampling_frequency,
                                    delay_spread)

        h_freq = cir_to_ofdm_channel(self._frequencies, h, tau,
                                     self._normalize_channel)

        return h_freq




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

        # tx_corr_mat = ue_correlation_matrix(num_tx_ant, beta)
        # rx_corr_mat = gnb_correlation_matrix(num_rx_ant, alpha)

        tx_corr_mat = exp_corr_mat(beta, num_tx_ant)
        rx_corr_mat = exp_corr_mat(alpha, num_rx_ant)

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
                                        tdl2,#2
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

    def __init__(self, tf_fn, n_size_bwp=4, max_num_tx=None, resource_grid_shape=None, num_rx_ant=None, training=True):
        self.tf_fn = tf_fn
        self.n_size_bwp = n_size_bwp
        self.max_num_tx = max_num_tx
        self.resource_grid_shape = resource_grid_shape
        self.num_rx_ant = num_rx_ant
        self.training = training

        self._num_examples = None
        
        print(f"Max num TX: {max_num_tx}, Resource grid: {resource_grid_shape}, RX antennas: {num_rx_ant}")
        print(f"Training mode: {training}")

      
        tfrecord_filename = f'../data/{tf_fn}'
        max_num_examples = 10000  # DON'T Load entire dataset

        dataset = (
            tf.data.TFRecordDataset([tfrecord_filename]) \
            .map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE) \
            # .shuffle(buffer_size=1024) \
            # .repeat() \
            .take(max_num_examples) \
            .batch(batch_size=2, drop_remainder=True)  # Fixed batch size for XLA
            # .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        # y = []
        # bits = []
        # h = []

        y = None
        bits = None
        h = None
        h_ls = None
        no = None

        for example in dataset:
            y_ex, bits_ex, h_ex, no_ex = example

            window = 12*self.n_size_bwp
            stride = window # no overlap
            # y_ex = tf.expand_dims(tf.transpose(y_ex, perm=[0, 3, 2, 1]), axis=1)  # [batch, num_rx_ant, ofdm_symbols, subcarriers]
            
            y_ex = tf.squeeze(y_ex, axis=1) # remove batch dim
            y_framed = tf.signal.frame(y_ex, frame_length=window, frame_step=stride, axis=-1)
            y_ex =  tf.reshape(tf.transpose(y_framed, [0, 4, 1, 2, 3, 5]),[-1, *y_ex.shape[1:-1], window]) # combine the batch and frame dim

            bits_ex = tf.squeeze(bits_ex, axis=1) # remove batch dim
            bits_framed = tf.signal.frame(bits_ex, frame_length=window, frame_step=stride, axis=-2)
            bits_ex = tf.transpose(bits_framed, [0, 3, 1, 2, 4, 5])
            bits_ex = tf.reshape(bits_ex,[-1, *bits_ex.shape[2:]])

            h_ex = tf.squeeze(h_ex, axis=1) # remove batch dim
            h_framed = tf.signal.frame(h_ex, frame_length=window, frame_step=stride, axis=-3)
            # h_ex = tf.reshape(tf.transpose(h_framed, [0, 2, 1, 3, 4, 5]),[-1, 1, window, 13, 8])
            h_ex = tf.transpose(h_framed, [0, 2, 1, 3, 4, 5])
            h_ex = tf.reshape(h_ex, [-1, *h_ex.shape[2:]])

            no_ex = tf.reshape(no_ex, [-1])  # [batch]
            no_ex = tf.repeat(
                no_ex,
                repeats=tf.shape(y_ex)[0] // tf.shape(no_ex)[0],
                axis=0,
            )  # [batch*num_frames] == [samples]

            if y is None:
                y = y_ex
                bits = bits_ex
                h = h_ex
                no = no_ex

            else:
                y = tf.concat([y, y_ex], axis=0)
                bits = tf.concat([bits, bits_ex], axis=0)
                h = tf.concat([h, h_ex], axis=0)
                no = tf.concat([no, no_ex], axis=0)

        self._y = y
        self._bits = bits
        self._h = h
        self._no = no

        self._num_examples = self._y.shape[0]
    
    def get_batch(self, batch_size):
        
        ind = tf.random.uniform([batch_size],
                             maxval=self._num_examples, dtype=tf.int32)
        rx_tensor = tf.gather(self._y, ind, axis=0)
        bits = [tf.gather(self._bits, ind, axis=0)]
        h = tf.gather(self._h, ind, axis=0)
        no = tf.gather(self._no, ind, axis=0)
        # h_ls = tf.gather(self._h_ls, ind, axis=0)
        h_ground = tf.zeros((batch_size, 1, 4, 1, 1, 14, 3276))  # Placeholder for compatibility

        active_dmrs = tf.ones([batch_size, self.max_num_tx], tf.float32)

        return rx_tensor, h_ground, bits, h, no, active_dmrs

    def get_total_samples(self):
        return self._num_examples
    
    @staticmethod
    def _parse_function(proto):
        description = {
                'y': tf.io.FixedLenFeature([], tf.string),
                'h': tf.io.FixedLenFeature([], tf.string),
                'b': tf.io.FixedLenFeature([], tf.string),
                'no': tf.io.FixedLenFeature([], tf.string, default_value=_DATALAKE_DEFAULT_NO_BYTES),
            }
        features = tf.io.parse_single_example(proto, description)
        rx_tensor = tf.io.parse_tensor(features['y'], out_type=tf.complex64)
        bits = tf.io.parse_tensor(features['b'], out_type=tf.int8)
        h = tf.io.parse_tensor(features['h'], out_type=tf.float32)
        no = tf.io.parse_tensor(features['no'], out_type=tf.float32)


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

        return rx_tensor, bits, h, no