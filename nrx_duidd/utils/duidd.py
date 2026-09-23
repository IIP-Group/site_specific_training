# Modified from the original implementation with the addition of supoprt for 
# real-world training and testing on a fixed data scrambling configuration.
# Modifications: Nuri Berke Baytekin

import numpy as np
import tensorflow as tf
# from sionna.fec.ldpc import LDPC5GDecoder
from sionna.nr import TBDecoder, PUSCHReceiver
from sionna.channel import time_to_ofdm_channel
from sionna.ofdm import MMSEPICDetector, LinearDetector, OFDMDetectorWithPrior
from sionna.mimo import StreamManagement

from sionna.mimo import MMSEPICDetector as MMSEPICDetector_

from utils.damped_decoding import dampedLDPC5GDecoder

# @TODO implement low complexity MMSE PIC --> XLA!

class MMSEPICDetectorLowComplexity_(MMSEPICDetector_):

    def LLRs2SymbolLogitsLowComplexity(self, llr_a):
        # dummy function to simply push through LLRs
        return llr_a

    def SymbolLogits2MomentsLowComplexity(self, llr_a):
        # input is LLR!!

        p0 = 0.5 * (1 - tf.math.tanh(0.5 * llr_a))
        p1 = 1 - p0
        if self._constellation.num_bits_per_symbol == 1:
            # BPSK
            s_real = (1 - 2 * tf.gather(p1, indices=0, axis=-1))
            s_imag = 0

            c = 1
            d = 0
        elif self._constellation.num_bits_per_symbol == 2:
            # QPSK
            s_real = (1 - 2 * tf.gather(p1, indices=0, axis=-1))
            s_imag = (1 - 2 * tf.gather(p1, indices=1, axis=-1))

            c = 2
            d = 0
        elif self._constellation.num_bits_per_symbol == 4:
            # 16-QAM
            s_real = (1 - 2 * tf.gather(p1, indices=0, axis=-1)) * (1 + 2 * tf.gather(p1, indices=2, axis=-1))
            s_imag = (1 - 2 * tf.gather(p1, indices=1, axis=-1)) * (1 + 2 * tf.gather(p1, indices=3, axis=-1))

            c = 1 + 8 * tf.gather(p1, indices=2, axis=-1)
            d = 1 + 8 * tf.gather(p1, indices=3, axis=-1)
        
        s_hat = self._qam_normalization_factor * tf.complex(s_real, s_imag)  # normalization can be included in previous scaling factor...
        error_var = self._qam_normalization_factor ** 2 * ((c + d) - tf.square(s_real) - tf.square(s_imag)) 

        # s_hat = tf.squeeze(s_hat, axis=-1)
        # error_var = tf.squeeze(error_var, axis=-1)

        return s_hat, error_var

    def DemapperWithPriorLowComplexity(self, inputs):
        x_hat, llr_a, no_eff = inputs

        z_i = x_hat / self._qam_normalization_factor
        z_i = tf.expand_dims(z_i, axis=-1)
        no_eff = tf.expand_dims(no_eff, axis=-1)

        if self._constellation.num_bits_per_symbol == 1: 
            # BPSK
            lambda_b_1 = 4 * tf.math.real(z_i)
            lambda_b = lambda_b_1
        elif self._constellation.num_bits_per_symbol == 2:
            # QPSK
            lambda_b_1 = 4 * tf.math.real(z_i)
            lambda_b_2 = 4 * tf.math.imag(z_i)
            lambda_b = tf.concat([lambda_b_1, lambda_b_2], axis=-1)
        elif self._constellation.num_bits_per_symbol == 4:
            # 16-QAM
            z_i_real = tf.math.real(z_i)
            z_i_imag = tf.math.imag(z_i)
            lambda_b_1 = tf.where(tf.math.less_equal(tf.abs(z_i_real), 2), 4 * z_i_real,
                                    8 * z_i_real - 8 * tf.sign(z_i_real))
            lambda_b_2 = 8 - 4 * tf.abs(z_i_real)
            lambda_b_3 = tf.where(tf.math.less_equal(tf.abs(z_i_imag), 2), 4 * z_i_imag,
                                    8 * z_i_imag - 8 * tf.sign(z_i_imag))
            lambda_b_4 = 8 - 4 * tf.abs(z_i_imag)
            lambda_b = tf.concat([lambda_b_1, lambda_b_3, lambda_b_2, lambda_b_4], axis=-1)

        lambda_b = self._qam_normalization_factor ** 2 * lambda_b
        llr_e = - lambda_b / no_eff  # minus because of inverse LLR definition

        # out_shape = tf.concat([tf.shape(llr_e)[:-1],
        #                        [llr_e.shape[-1] * \
        #                         self._constellation.num_bits_per_symbol]], 0)
        # llr_e = tf.reshape(llr_e, out_shape)

        llr_d = llr_e + llr_a

        return llr_d


    def __init__(self,
                 output,
                 demapping_method="maxlog_low_complexity",
                 num_iter=1,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=tf.complex64,
                 **kwargs):
        
        if demapping_method == "maxlog_low_complexity":
            demapping_method = "maxlog"
            low_complexity = True
        else:
            low_complexity = False

        super().__init__(output=output, demapping_method=demapping_method, 
                         num_iter=num_iter, constellation_type=constellation_type, 
                         num_bits_per_symbol=num_bits_per_symbol, 
                         constellation=constellation, hard_out=hard_out, 
                         dtype=dtype, **kwargs)
        
        if low_complexity and self._constellation._constellation_type == "qam" and self._constellation.num_bits_per_symbol <= 4 and output=="bit":
            self._llr_2_symbol_logits = self.LLRs2SymbolLogitsLowComplexity
            self._symbol_logits_2_moments = self.SymbolLogits2MomentsLowComplexity
            self._bit_demapper = self.DemapperWithPriorLowComplexity

            if self._constellation.normalize:
                n = int(num_bits_per_symbol / 2)
                qam_var = 1 / (2 ** (n - 2)) * np.sum(np.linspace(1, 2 ** n - 1, 2 ** (n - 1)) ** 2)
                self._qam_normalization_factor = 1 / np.sqrt(qam_var)

            else:
                self._qam_normalization_factor = 1


class MMSEPICDetectorLowComplexity(OFDMDetectorWithPrior):
    def __init__(self,
                 output,
                 resource_grid,
                 stream_management,
                 demapping_method="maxlog",
                 num_iter=1,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=tf.complex64,
                 **kwargs):
        # Instantiate the EP detector
        detector = MMSEPICDetectorLowComplexity_(output=output,
                                    demapping_method=demapping_method,
                                    num_iter=num_iter,
                                    constellation_type=constellation_type,
                                    num_bits_per_symbol=num_bits_per_symbol,
                                    constellation=constellation,
                                    hard_out=hard_out,
                                    dtype=dtype,
                                    **kwargs)

        super().__init__(detector=detector,
                         output=output,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         constellation_type=constellation_type,
                         num_bits_per_symbol=num_bits_per_symbol,
                         constellation=constellation,
                         dtype=dtype,
                         **kwargs)


class LinearDetectorIDDWrapper(tf.keras.layers.Layer):
    """IDD-loop detector shim: same 5-tuple API as MMSE-PIC, runs plain LMMSE.

    Decoder bit priors are ignored so we can A/B test whether MMSE-PIC (vs the
    first-pass linear detector) causes datalake IDD failures at wide bandwidth.
    """

    def __init__(self, linear_detector, **kwargs):
        super().__init__(**kwargs)
        self._linear_detector = linear_detector

    def call(self, inputs):
        y, h_hat, _prior, err_var, no = inputs
        return self._linear_detector([y, h_hat, err_var, no])


class SisoTBDecoder(TBDecoder):
    def __init__(self,
                 encoder,
                 mu, xi,
                 num_bp_iter=20,
                 cn_type="boxplus-phi",
                 output_dtype=tf.float32,
                 weighted_bp=False,
                 last_decoder=True,
                 training=False,
                 **kwargs):

        super().__init__(encoder=encoder,
                         num_bp_iter=num_bp_iter,
                         cn_type=cn_type,
                         output_dtype=output_dtype,
                         **kwargs)
        
        # self._training = training

        self._decoder = dampedLDPC5GDecoder(encoder=encoder.ldpc_encoder,
                                            mu=mu, xi=xi,
                                      num_iter=num_bp_iter,
                                      cn_type=cn_type,
                                      hard_out=last_decoder and not training, # IDD: get LLR outputs if not last decoder
                                      return_infobits=last_decoder,    #### return info bits if last decoder, this is important for IDD
                                      stateful=True, ### this is also important for IDD
                                      trainDamping=training, ### we train damping with DUIDD
                                      weighted_bp=weighted_bp,
                                      output_dtype=tf.float32)
        
        self._last_decoder = last_decoder
        
    def build(self, input_shapes):
        """Test input shapes for consistency."""

        assert input_shapes[0][-1]==self.n, \
            f"Invalid input shape. Expected input length is {self.n}."
    
    def call(self, inputs):
        """Apply transport block decoding."""

        inputs, msg_vn, idd_it = inputs

        # store shapes
        input_shape = inputs.shape.as_list()
        llr_ch = tf.cast(inputs, tf.float32)

        llr_ch = tf.reshape(llr_ch,
                            (-1, self._tb_encoder.num_tx, self._tb_encoder.n))

        # undo scrambling (only if scrambler was used)
        if self._descrambler is not None:
            llr_scr = self._descrambler(llr_ch)
        else:
            llr_scr = llr_ch

        # undo CB interleaving and puncturing
        num_fillers = self._tb_encoder.ldpc_encoder.n * self._tb_encoder.num_cbs - np.sum(self._tb_encoder.cw_lengths)
        llr_int = tf.concat([llr_scr,
                            tf.zeros([tf.shape(llr_scr)[0], self._tb_encoder.num_tx, num_fillers])], axis=-1)
        llr_int = tf.gather(llr_int, self._tb_encoder.output_perm_inv, axis=-1)

        cb_shape = llr_int.shape.as_list()
        cb_shape[0] = -1

        # undo CB concatenation
        llr_cb = tf.reshape(llr_int,
                        (-1, self._tb_encoder.num_tx, self._num_cbs, self._tb_encoder.ldpc_encoder.n))

        # LDPC decoding
        u_hat_cb, msg_vn = self._decoder((llr_cb, msg_vn, idd_it))

        if not self._last_decoder:
            # prepare LLRs for another IDD iteration
            #redo CB concatenation
            llr_dec = tf.reshape(u_hat_cb, cb_shape)

            # redo interleaving and puncturing
            llr_dec = tf.gather(llr_dec, self._tb_encoder._output_perm, axis=-1)
            if num_fillers > 0:
                llr_dec = llr_dec[..., 0:-num_fillers]

            # redo scrambling (with same descrambler)
            if self._descrambler is not None:
                llr_dec = self._descrambler(llr_dec)
            else:
                llr_dec = llr_dec

            # restore input shape
            output_shape = input_shape
            output_shape[0] = -1

            llr_dec = tf.reshape(llr_dec, output_shape)

            return (llr_dec, msg_vn)
        
        else:
            # CB CRC removal (if relevant)
            if self._cb_crc_decoder is not None:
                # we are ignoring the CB CRC status for the moment
                # Could be combined with the TB CRC for even better estimates
                u_hat_cb_crc, _ = self._cb_crc_decoder(u_hat_cb)
            else:
                u_hat_cb_crc = u_hat_cb

            # undo CB segmentation
            u_hat_tb = tf.reshape(u_hat_cb_crc,
                    (-1, self._tb_encoder.num_tx, self.tb_size+self._tb_encoder.tb_crc_encoder.crc_length))

            # TB CRC removal
            u_hat, tb_crc_status = self._tb_crc_decoder(u_hat_tb)

            # restore input shape
            output_shape = input_shape
            output_shape[0] = -1
            output_shape[-1] = self.tb_size
            u_hat = tf.reshape(u_hat, output_shape)
            # also apply to tb_crc_status
            output_shape[-1] = 1 # but last dim is 1
            tb_crc_status = tf.reshape(tb_crc_status, output_shape)

            # remove if zero-padding was applied
            if self._tb_encoder.k_padding>0:
                u_hat = u_hat[...,:-self._tb_encoder.k_padding]

            # cast to output dtype
            u_hat = tf.cast(u_hat, self.dtype)
            tb_crc_status = tf.squeeze(tf.cast(tb_crc_status, tf.bool), axis=-1)

            return u_hat, tb_crc_status

class DUIDDPUSCHReceiver(PUSCHReceiver):
    def __init__(self,
                 pusch_transmitter,
                 duidd_schedule=[10,10],
                 channel_estimator=None,
                 mimo_detector=None,
                 siso_mimo_detector=None,
                 demepping_type="maxlog",
                 tb_decoder=None,
                 return_tb_crc_status=False,
                 stream_management=None,
                 input_domain="freq",
                 l_min=None,
                 low_complexity=False,
                 training=False,
                 weighted_bp=False,
                 sys_parameters=None,
                 dtype=tf.complex64,
                 **kwargs):
        
        # Use or create default StreamManagement
        if stream_management is None:
            # Default StreamManagement
            rx_tx_association = np.ones([1, pusch_transmitter._num_tx], bool)
            self._stream_management = StreamManagement(
                                        rx_tx_association,
                                        pusch_transmitter._num_layers)
        else:
            # User-provided StramManagement
            self._stream_management = stream_management

        if mimo_detector is None:
            # Default MIMO detector
            mimo_detector = LinearDetector("lmmse", "bit", demepping_type,
                                        pusch_transmitter.resource_grid,
                                        self._stream_management,
                                        "qam",
                                        pusch_transmitter._num_bits_per_symbol,
                                        dtype=dtype)

        super().__init__(pusch_transmitter=pusch_transmitter,
                         channel_estimator=channel_estimator,
                         mimo_detector=mimo_detector,
                         tb_decoder=tb_decoder,
                         return_tb_crc_status=return_tb_crc_status,
                         stream_management=self._stream_management ,
                         input_domain=input_domain,
                         l_min=l_min,
                         dtype=dtype, **kwargs)
        
        if siso_mimo_detector is None:
            # Default MIMO detector
            if low_complexity:
                self._siso_mimo_detector = MMSEPICDetectorLowComplexity(output="bit", 
                                                       resource_grid=pusch_transmitter.resource_grid, 
                                                       stream_management=self._stream_management, 
                                                       constellation_type="qam", 
                                                       num_bits_per_symbol=pusch_transmitter._num_bits_per_symbol, 
                                                       demapping_method=demepping_type if (not low_complexity) or (demepping_type != "maxlog") else demepping_type + "_low_complexity",
                                                       dtype=dtype)
            else:
                self._siso_mimo_detector = MMSEPICDetector(output="bit", 
                                                        resource_grid=pusch_transmitter.resource_grid, 
                                                        stream_management=self._stream_management, 
                                                        constellation_type="qam", 
                                                        num_bits_per_symbol=pusch_transmitter._num_bits_per_symbol, 
                                                        demapping_method=demepping_type,
                                                        dtype=dtype)

        else:
            self._siso_mimo_detector = siso_mimo_detector
        
        self._duidd_schedule = duidd_schedule
        self._I = len(self._duidd_schedule)

        self._layer_mapper = pusch_transmitter._layer_mapper
        self._num_bits_per_symbol = pusch_transmitter._num_bits_per_symbol
        self._pusch_transmitter = pusch_transmitter
        self._resource_grid_mapper = pusch_transmitter._resource_grid_mapper

        self._training = training

        # Nachmani-style edge weights on VN→CN messages (Sionna LDPC5GDecoder).
        # Prefer explicit ctor arg; otherwise read config [duidd] weighted_bp.
        if sys_parameters is not None:
            weighted_bp = bool(
                getattr(sys_parameters, "weighted_bp", weighted_bp))
        self._weighted_bp = bool(weighted_bp)

        ## define trainable variables
        # extrinsic vs intrinsic
        self._alpha = tf.Variable(tf.ones(self._I), dtype=tf.float32, trainable=training, name="alpha")
        self._beta = tf.Variable(tf.zeros(self._I), dtype=tf.float32, trainable=training, name="beta")
        self._delta = tf.Variable(tf.ones(self._I), dtype=tf.float32, trainable=training, name="delta")
        self._epsilon = tf.Variable(tf.zeros(self._I), dtype=tf.float32, trainable=training, name="epsilon")

        # cest var scaling
        self._eta = tf.Variable(1.0, trainable=training, dtype=tf.float32, name="eta")

        # state forwarding
        self._gamma = tf.Variable(np.ones(self._I), trainable=training, dtype=tf.float32, name="gamma")

        # # message damping (SISO decoder weights supervised during finetuning)
        self._mu = tf.Variable(tf.zeros([self._I, np.max(self._duidd_schedule)]), dtype=tf.float32, trainable=training,
                                  name="mu_damping", constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0))
        self._xi = tf.Variable(tf.zeros([self._I, np.max(self._duidd_schedule)]), dtype=tf.float32, trainable=training,
                                 name="xi_damping", constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0))
        
        self._siso_decoder = SisoTBDecoder(pusch_transmitter._tb_encoder,
                                           mu=self._mu, xi=self._xi,
                                           output_dtype=dtype.real_dtype,
                                           cn_type=self._tb_decoder._decoder._cn_type,
                                           weighted_bp=self._weighted_bp,
                                           last_decoder=False,
                                           training=training)
        
        self._output_decoder = SisoTBDecoder(pusch_transmitter._tb_encoder,
                                             mu=self._mu, xi=self._xi,
                                           output_dtype=dtype.real_dtype,
                                           cn_type=self._tb_decoder._decoder._cn_type,
                                           weighted_bp=self._weighted_bp,
                                           last_decoder=True,
                                           training=training)

        self._datalake_training = (
            training
            and sys_parameters is not None
            and getattr(sys_parameters, "channel_type", None) == "Datalake"
        )

    def _layer_demapper_inverse(self, llr_coded):
        """Inverse of ``LayerDemapper``: coded-bit order -> per-stream detector layout.

        ``LayerDemapper`` maps detector output ``[..., num_layers, bits_per_layer]``
        to serial coded bits ``[..., n]`` by (1) grouping ``Qm`` bits per symbol,
        (2) swapping layer/symbol axes, (3) flattening. This method undoes that.

        ``LayerMapper`` is **not** the inverse: it splits a *symbol* sequence across
        layers and must not be applied to flat coded-bit LLRs when ``Qm > 1``.
        """
        qm = self._num_bits_per_symbol
        num_layers = self._layer_mapper.num_layers
        prefix = tf.shape(llr_coded)[:-1]
        n = tf.shape(llr_coded)[-1]
        sym_per_layer = n // (num_layers * qm)

        # Undo flatten + swap from LayerDemapper.call()
        x = tf.reshape(
            llr_coded,
            tf.concat([prefix, [sym_per_layer, num_layers, qm]], axis=0))
        nd = tf.rank(x)
        perm = tf.concat([tf.range(nd - 3), [nd - 2, nd - 3, nd - 1]], axis=0)
        x = tf.transpose(x, perm)
        return tf.reshape(
            x, tf.concat([prefix, [num_layers, sym_per_layer * qm]], axis=0))

    def map_llrs_to_bit_grid(self, llr_coded, transmitter=None):
        """Map coded soft LLRs onto the bit-grid via the PUSCH transmitter mappers.

        Same structure as ``map_bits_to_resource_grid`` used for TFRecord labels:
        reshape to symbols, ``layer_mapper``, ``resource_grid_mapper``, stack Qm,
        mask DMRS REs with -1.
        """
        if transmitter is None:
            transmitter = self._pusch_transmitter
        layer_mapper = transmitter._layer_mapper
        resource_grid_mapper = transmitter._resource_grid_mapper
        mod_order = transmitter._num_bits_per_symbol

        n_bits = tf.shape(llr_coded)[-1]
        n_symbols = n_bits // mod_order
        new_shape = tf.concat(
            [tf.shape(llr_coded)[:-1], [n_symbols, mod_order]], axis=0)
        symbol_llrs = tf.reshape(llr_coded, new_shape)

        # Place each bit-plane through the same TX mappers used for hard bits
        planes = tf.unstack(symbol_llrs, axis=-1)
        mapped = []
        for plane in planes:
            x = layer_mapper(tf.cast(plane, tf.complex64))
            x = resource_grid_mapper(x)
            mapped.append(tf.math.real(x))
        llr_grid = tf.stack(mapped, axis=-1)
        llr_grid = tf.squeeze(llr_grid, axis=1)

        pilot_position = transmitter._resource_grid.build_type_grid()[:, 0] == 1
        pilot_position = tf.cast(pilot_position, tf.float32)
        pilot_position = tf.expand_dims(pilot_position, axis=-1)
        pilot_position = tf.broadcast_to(pilot_position, tf.shape(llr_grid))
        llr_grid = tf.where(
            pilot_position == 1, tf.constant(-1., llr_grid.dtype), llr_grid)
        return llr_grid

    def call(self, inputs, b_info=None, active_dmrs=None, bit_grid=None):
        supervise_datalake = (
            b_info is not None and self._datalake_training)

        if self._perfect_csi:
            y, h, no = inputs
        else:
            y, no = inputs

        # (Optional) OFDM Demodulation
        if self._input_domain=="time":
            y = self._ofdm_demodulator(y)

        # Channel estimation
        if self._perfect_csi:

            # Transform time-domain to frequency-domain channel
            if self._input_domain=="time":
                h = time_to_ofdm_channel(h, self.resource_grid, self._l_min)


            if self._w is not None:
                # Reshape h to put channel matrix dimensions last
                # [batch size, num_rx, num_tx, num_ofdm_symbols,...
                #  ...fft_size, num_rx_ant, num_tx_ant]
                h = tf.transpose(h, perm=[0,1,3,5,6,2,4])

                # Multiply by precoding matrices to compute effective channels
                # [batch size, num_rx, num_tx, num_ofdm_symbols,...
                #  ...fft_size, num_rx_ant, num_streams]
                h = tf.matmul(h, self._w)

                # Reshape
                # [batch size, num_rx, num_rx_ant, num_tx, num_streams,...
                #  ...num_ofdm_symbols, fft_size]
                h = tf.transpose(h, perm=[0,1,5,2,6,3,4])
            h_hat = h
            err_var = tf.cast(0, dtype=h_hat.dtype.real_dtype)
        else:
            # no = no * 0.1 # tfrecord has pessimistic no
            # no = 0.06
            h_hat,err_var = self._channel_estimator([y, no])

        # DUIDD: err-var scaling
        err_var = self._eta * err_var

        # first detection
        llr_ch = self._mimo_detector([y, h_hat, err_var, no])
        llr_ch = self._layer_demapper(llr_ch)

        llr_a_dec = llr_ch

        # iterative detection and decoding
        num_idd_iter = self._I
        msg_vn = None

        if num_idd_iter >= 2:
            # unrolled loop, requires model recompilation if DUIDD parameters change
            for i in range(num_idd_iter-1):
                self._siso_decoder._decoder._num_iter = self._duidd_schedule[i]
                # DUIDD: scale decoder state forwarding
                [llr_d_dec, msg_vn] = self._siso_decoder([llr_a_dec, self._gamma[i] * msg_vn if msg_vn is not None else msg_vn, i])

                # DUIDD: extrinsic vs intrinsic state forwarding from decoder to detector
                llr_a_det = self._alpha[i + 1] * llr_d_dec - self._beta[i + 1] * llr_a_dec 

                llr_a_det_ = self._layer_demapper_inverse(llr_a_det)
                llr_ch = self._siso_mimo_detector([y, h_hat, llr_a_det_, err_var, no])
                llr_ch = self._layer_demapper(llr_ch)

                # DUIDD: extrinsic vs intrinsic state forwarding from detector to decoder
                llr_a_dec = self._delta[i + 1] * llr_ch - self._epsilon[i + 1] * llr_a_det


        # Last TB decoding (same path as eval BLER/BER)
        self._output_decoder._decoder.num_iter = self._duidd_schedule[-1]
        msg_vn_out = (self._gamma[-1] * msg_vn if msg_vn is not None else msg_vn)
        b_hat, tb_crc_status = self._output_decoder(
            [llr_a_dec, msg_vn_out, num_idd_iter - 1])

        if supervise_datalake:
            labels = tf.cast(b_info, tf.float32)
            return labels, b_hat, active_dmrs

        if self._return_tb_crc_status:
            return b_hat, tb_crc_status
        return b_hat
    
        