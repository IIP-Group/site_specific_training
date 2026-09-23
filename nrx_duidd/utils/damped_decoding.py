import tensorflow as tf
from sionna.fec.ldpc import LDPC5GDecoder
from sionna.fec.utils import llr2mi


class dampedLDPC5GDecoder(LDPC5GDecoder):
    # pylint: disable=line-too-long
    r"""
    LDPC 5G decoder from Sionna extended by OMS and Damping
    """

    def __init__(self,
                 encoder,
                 mu, xi,
                 weighted_bp=False,
                 cn_type='boxplus-phi',
                 hard_out=True,
                 track_exit=False,
                 return_infobits=True,
                 prune_pcm=True,
                 num_iter=20,
                 stateful=False,
                 output_dtype=tf.float32,
                 trainOms=False, trainDamping=False,
                 llr_max=20.0,
                 grad_pass_through=False,
                 **kwargs):

        super().__init__(encoder,
                 trainable=weighted_bp,
                 cn_type=cn_type,
                 hard_out=hard_out,
                 track_exit=track_exit,
                 return_infobits=return_infobits,
                 prune_pcm=prune_pcm,
                 num_iter=num_iter,
                 stateful=stateful,
                 output_dtype=output_dtype,
                 **kwargs)

        self._cn_type = cn_type
        # pre-load the CN function for performance reasons
        if self._cn_type == 'boxplus':
            # check node update using the tanh function
            self._cn_update = self._cn_update_tanh
        elif self._cn_type == 'boxplus-phi':
            # check node update using the "_phi" function
            self._cn_update = self._cn_update_phi
        elif self._cn_type == 'minsum':
            # check node update using the min-sum approximation
            self._cn_update = self._cn_update_minsum
        # elif self._cn_type == 'offset-minsum':
        #     self._cn_update = self._cn_update_offset_minsum
        else:
            raise ValueError('Unknown node type.')

        self._num_iter = tf.constant(num_iter, dtype=tf.int32)
        self._trainable = trainOms or trainDamping

        self._mu = mu
        self._xi = xi
        

        # self._train_clipping = train_clipping
        # llr_max = tf.Variable(llr_max, trainable=train_clipping, dtype=tf.float32, name="msg_clipping_value",
        #                       constraint=lambda x: tf.maximum(x, 5.0))

        self._llr_max = llr_max  # internal max value for LLR initialization

        self._grad_pass_through = grad_pass_through

    #########################################
    # Public methods and properties
    #########################################

    def trainable_clipping_pass_through(self, msg):
        if self._grad_pass_through and self._train_clipping:
            # custom gradient: identity for clipped msg, if clipped: gradient of clipping parameter is +-1
            @tf.custom_gradient
            def my_passthrough_clipping(msg, llr_max):
                def grad(d_y):
                    return (d_y * 1,
                            tf.reduce_sum(
                                d_y * tf.where(tf.greater_equal(msg, llr_max),
                                               llr_max * tf.ones_like(msg),
                                               tf.where(tf.less(msg, -llr_max),
                                                        -llr_max * tf.ones_like(msg),
                                                        tf.zeros_like(msg))),
                                axis=None  # across all dimensions
                            )
                            )

                return tf.clip_by_value(msg, clip_value_min=-llr_max, clip_value_max=llr_max), grad

            msg = my_passthrough_clipping(msg, self._llr_max)
        elif self._grad_pass_through and not self._train_clipping:
            msg = tf.grad_pass_through(lambda x: tf.clip_by_value(x, clip_value_min=-self._llr_max,
                                                                  clip_value_max=self._llr_max))(msg)
        else:
            # clipped messages don't propagate gradient through clipping, but clipping parameter will have gradient
            msg = tf.clip_by_value(msg,
                                   clip_value_min=-self._llr_max,
                                   clip_value_max=self._llr_max)
        return msg

    # def _cn_update_offset_minsum(self, msg, it=0):
    #     """ Check node update function implementing the min-sum approximation with trainable offset parameter.

    #     This function approximates the (extrinsic) check node update
    #     function based on the offset min-sum approximation (cf. [XYZ Who invented OMS?]_).
    #     It calculates the "extrinsic" min function (including some offset parameter, default 0.3) over all incoming messages
    #     ``msg`` excluding the intrinsic (=outgoing) message itself.

    #     The input is expected to be a ragged Tensor of shape
    #     `[num_vns, None, batch_size]`.
    #     """
    #     # a constant used overwrite the first min
    #     LARGE_VAL = 100000. # pylint: disable=invalid-name

    #     # clip values for numerical stability
    #     # msg = tf.clip_by_value(msg,
    #     #                        clip_value_min=-self._llr_max_minsum,
    #     #                        clip_value_max=self._llr_max_minsum)
    #     msg = tf.ragged.map_flat_values(self.trainable_clipping_pass_through, msg)

    #     # calculate sign of outgoing msg
    #     sign_val = tf.ragged.map_flat_values(self._sign_val_minsum, msg)

    #     sign_node = tf.reduce_prod(sign_val, axis=1)

    #     # TF2.9 does not support XLA for the multiplication of ragged tensors
    #     # the following code provides a workaround that supports XLA

    #     # sign_val = self._stop_ragged_gradient(sign_val) \
    #     #             * tf.expand_dims(sign_node, axis=1)
    #     sign_val = tf.ragged.map_flat_values(
    #                                     lambda x, y, row_ind:
    #                                     tf.multiply(x, tf.gather(y, row_ind)),
    #                                     self._stop_ragged_gradient(sign_val),
    #                                     sign_node,
    #                                     sign_val.value_rowids())

    #     msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign

    #     # Calculate the extrinsic minimum per CN, i.e., for each message of
    #     # index i, find the smallest and the second smallest value.
    #     # However, in some cases the second smallest value may equal the
    #     # smallest value (multiplicity of mins).
    #     # Please note that this needs to be applied to raggedTensors, e.g.,
    #     # tf.top_k() is currently not supported and the ops must support graph
    #     # # mode.

    #     # find min_value per node
    #     min_val = tf.reduce_min(msg, axis=1, keepdims=True)

    #     # TF2.9 does not support XLA for the subtraction of ragged tensors
    #     # the following code provides a workaround that supports XLA

    #     # and subtract min; the new array contains zero at the min positions
    #     # benefits from broadcasting; all other values are positive
    #     # msg_min1 = msg - min_val
    #     msg_min1 = tf.ragged.map_flat_values(lambda x, y, row_ind:
    #                                          x- tf.gather(y, row_ind),
    #                                          msg,
    #                                          tf.squeeze(min_val, axis=1),
    #                                          msg.value_rowids())

    #     # replace 0 (=min positions) with large value to ignore it for further
    #     # min calculations
    #     msg = tf.ragged.map_flat_values(lambda x:
    #                                     tf.where(tf.equal(x, 0), LARGE_VAL, x),
    #                                     msg_min1)

    #     # find the second smallest element (we add min_val as this has been
    #     # subtracted before)
    #     min_val2 = tf.reduce_min(msg, axis=1, keepdims=True) + min_val

    #     # Detect duplicated minima (i.e., min_val occurs at two incoming
    #     # messages). As the LLRs per node are <LLR_MAX and we have
    #     # replace at least 1 position (position with message "min_val") by
    #     # LARGE_VAL, it holds for the sum < LARGE_VAL + node_degree*LLR_MAX.
    #     # if the sum > 2*LARGE_VAL, the multiplicity of the min is at least 2.
    #     node_sum = tf.reduce_sum(msg, axis=1, keepdims=True) - (2*LARGE_VAL-1.)
    #     # indicator that duplicated min was detected (per node)
    #     double_min = 0.5*(1-tf.sign(node_sum))      # 0 if double_min ocured

    #     # if a duplicate min occurred, both edges must have min_val, otherwise
    #     # the second smallest value is taken
    #     min_val_e = (1-double_min) * min_val + (double_min) * min_val2

    #     # replace all values with min_val except the position where the min
    #     # occurred (=extrinsic min).
    #     msg_e = tf.where(msg==LARGE_VAL, min_val_e, min_val)

    #     # subtract offset
    #     msg_e = tf.maximum(msg_e - self._beta[it], 0)

    #     # it seems like tf.where does not set the shape of tf.ragged properly
    #     # we need to ensure the shape manually
    #     msg_e = tf.ragged.map_flat_values(
    #                                 lambda x:
    #                                 tf.ensure_shape(x, msg.flat_values.shape),
    #                                 msg_e)

    #     # TF2.9 does not support XLA for the multiplication of ragged tensors
    #     # the following code provides a workaround that supports XLA

    #     # and apply sign
    #     #msg = sign_val * msg_e
    #     msg = tf.ragged.map_flat_values(tf.multiply,
    #                                     sign_val,
    #                                     msg_e)

    #     return msg

    def build(self, input_shape):
        """Build model."""
        if self._stateful:
            assert(len(input_shape)==3), \
                "For stateful decoding, a tuple of two inputs is expected."
            input_shape = input_shape[0]
        # check input dimensions for consistency
        assert (input_shape[-1]==self.encoder.n), \
                                'Last dimension must be of length n.'
        assert (len(input_shape)>=2), 'The inputs must have at least rank 2.'

        self._old_shape_5g = input_shape

    def super_call(self, inputs):
        """Iterative BP decoding function.

        This function performs ``num_iter`` belief propagation decoding
        iterations and returns the estimated codeword.

        Args:
            inputs (tf.float32): Tensor of shape `[...,n]` containing the
                channel logits/llr values.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]` containing
            bit-wise soft-estimates (or hard-decided bit-values) of all
            codeword bits.

        Raises:
            ValueError: If ``inputs`` is not of shape `[batch_size, n]`.

            InvalidArgumentError: When rank(``inputs``)<2.
        """

        # Extract inputs
        if self._stateful:
            llr_ch, msg_vn, idd_it = inputs
        else:
            llr_ch, idd_it = inputs

        tf.debugging.assert_type(llr_ch, self.dtype, 'Invalid input dtype.')

        # internal calculations still in tf.float32
        # llr_ch = tf.cast(llr_ch, tf.float32)
        llr_ch = self.trainable_clipping_pass_through(llr_ch)

        # # clip llrs for numerical stability
        # llr_ch = tf.clip_by_value(llr_ch,
        #                           clip_value_min=-self._llr_max,
        #                           clip_value_max=self._llr_max)

        # last dim must be of length n
        tf.debugging.assert_equal(tf.shape(llr_ch)[-1],
                                  self._num_vns,
                                  'Last dimension must be of length n.')

        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, self._num_vns]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)

        # must be done during call, as XLA fails otherwise due to ragged
        # indices placed on the CPU device.
        # create permutation index from cn perspective
        self._cn_mask_tf = tf.ragged.constant(self._gen_node_mask(self._cn_con),
                                              row_splits_dtype=tf.int32)

        # batch dimension is last dimension due to ragged tensor representation
        llr_ch = tf.transpose(llr_ch_reshaped, (1,0))

        llr_ch = -1. * llr_ch # logits are converted into "true" llrs

        # init internal decoder state if not explicitly
        # provided (e.g., required to restore decoder state for iterative
        # detection and decoding)
        # load internal state from previous iteration
        # required for iterative det./dec.
        if not self._stateful or msg_vn is None:
            msg_shape = tf.stack([tf.constant(self._num_edges),
                                  tf.shape(llr_ch)[1]],
                                 axis=0)
            msg_vn = tf.zeros(msg_shape, dtype=tf.float32)
        else:
            msg_vn = msg_vn.flat_values

        # track exit decoding trajectory; requires all-zero cw?
        if self._track_exit:
            self._ie_c = tf.zeros(self._num_iter + 1)
            self._ie_v = tf.zeros(self._num_iter + 1)

        # perform one decoding iteration
        # Remark: msg_vn cannot be ragged as input for tf.while_loop as
        # otherwise XLA will not be supported (with TF 2.5)
        def dec_iter(llr_ch, msg_vn, it):
            it += 1
            # msg_vn_old are the cn2vn messages from the previous iteration
            msg_vn_old = tf.RaggedTensor.from_row_splits(
                        values=msg_vn,
                        row_splits=tf.constant(self._vn_row_splits, tf.int32))
            # variable node update
            # msg_vn are now the vn2cn messages from the vn perspective
            msg_vn = self._vn_update(msg_vn_old, llr_ch)

            # track exit decoding trajectory; requires all-zero cw
            if self._track_exit:
                # neg values as different llr def is expected
                mi = llr2mi(-1. * msg_vn.flat_values)
                self._ie_v = tf.tensor_scatter_nd_add(self._ie_v,
                                                     tf.reshape(it, (1, 1)),
                                                     tf.reshape(mi, (1)))

            # scale outgoing vn messages (weighted BP); only if activated
            if self._has_weights:
                msg_vn = tf.ragged.map_flat_values(self._mult_weights,
                                                   msg_vn)
            # permute edges into CN perspective
            msg_cn = tf.gather(msg_vn.flat_values, self._cn_mask_tf, axis=None)

            # check node update using the pre-defined function
            if self._cn_type == 'offset-minsum':
                msg_cn = self._cn_update(msg_cn, it-1)
            else:
                msg_cn = self._cn_update(msg_cn)

            # track exit decoding trajectory; requires all-zero cw?
            if self._track_exit:
                # neg values as different llr def is expected
                mi = llr2mi(-1.*msg_cn.flat_values)
                # update pos i+1 such that first iter is stored as 0
                self._ie_c = tf.tensor_scatter_nd_add(self._ie_c,
                                                     tf.reshape(it, (1, 1)),
                                                     tf.reshape(mi, (1)))

            # re-permute edges to variable node perspective + daming via vn2cn messages + damping via old and new state (cn2vn messages)
            # msg_vn = (1-self._mu[it-1]-self._xi[it-1])*tf.gather(msg_cn.flat_values, self._ind_cn_inv, axis=None) + \
            #          self._xi[it-1]*msg_vn.flat_values + \
            #          self._mu[it-1]*msg_vn_old.flat_values
            msg_vn = (1-self._mu[idd_it, it-1]-self._xi[idd_it, it-1]) * tf.gather(msg_cn.flat_values, self._ind_cn_inv, axis=None) + \
                     self._xi[idd_it, it - 1] * msg_vn.flat_values + \
                     self._mu[idd_it, it - 1] * msg_vn_old.flat_values
            return llr_ch, msg_vn, it

        # stopping condition (required for tf.while_loop)
        def dec_stop(llr_ch, msg_vn, it): # pylint: disable=W0613
            return tf.less(it, self._num_iter)

        # start decoding iterations
        it = tf.constant(0)
        # maximum_iterations required for XLA
        _, msg_vn, _ = tf.while_loop(dec_stop,
                                     dec_iter,
                                     (llr_ch, msg_vn, it),
                                     parallel_iterations=1,
                                     maximum_iterations=self._num_iter)


        # raggedTensor for final marginalization
        msg_vn = tf.RaggedTensor.from_row_splits(
                        values=msg_vn,
                        row_splits=tf.constant(self._vn_row_splits, tf.int32))

        # marginalize and remove ragged Tensor
        x_hat = tf.add(llr_ch, tf.reduce_sum(msg_vn, axis=1))

        # restore batch dimension to first dimension
        x_hat = tf.transpose(x_hat, (1,0))

        x_hat = -1. * x_hat # convert llrs back into logits

        if self._hard_out: # hard decide decoder output if required
            x_hat = tf.cast(tf.less(0.0, x_hat), self._output_dtype)

        # Reshape c_short so that it matches the original input dimensions
        output_shape = llr_ch_shape
        output_shape[0] = -1 # overwrite batch dim (can be None in Keras)

        x_reshaped = tf.reshape(x_hat, output_shape)

        # cast output to output_dtype
        x_out = tf.cast(x_reshaped, self._output_dtype)

        if not self._stateful:
            return x_out
        else:
            return x_out, msg_vn

    def call(self, inputs):
        """Iterative BP decoding function.

        This function performs ``num_iter`` belief propagation decoding
        iterations and returns the estimated codeword.

        Args:
            inputs (tf.float32): Tensor of shape `[...,n]` containing the
                channel logits/llr values.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]` or `[...,k]`
            (``return_infobits`` is True) containing bit-wise soft-estimates
            (or hard-decided bit-values) of all codeword bits (or info
            bits, respectively).

        Raises:
            ValueError: If ``inputs`` is not of shape `[batch_size, n]`.

            ValueError: If ``num_iter`` is not an integer greater (or equal)
                `0`.

            InvalidArgumentError: When rank(``inputs``)<2.
        """
        # Modified from sionna code: super().call ==> implements other signature for vn_update function (also takes in iterations count)

        # Extract inputs
        if self._stateful:
            llr_ch, msg_vn, idd_it = inputs
        else:
            llr_ch, idd_it = inputs

        tf.debugging.assert_type(llr_ch, self.dtype, 'Invalid input dtype.')

        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, llr_ch_shape[-1]]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)
        batch_size = tf.shape(llr_ch_reshaped)[0]

        # invert if rate-matching output interleaver was applied as defined in
        # Sec. 5.4.2.2 in 38.212
        if self._encoder.num_bits_per_symbol is not None:
            llr_ch_reshaped = tf.gather(llr_ch_reshaped,
                                        self._encoder.out_int_inv,
                                        axis=-1)

        # undo puncturing of the first 2*Z bit positions
        llr_5g = tf.concat(
            [tf.zeros([batch_size, 2 * self.encoder.z], self._output_dtype),
             llr_ch_reshaped],
            1)

        # undo puncturing of the last positions
        # total length must be n_ldpc, while llr_ch has length n
        # first 2*z positions are already added
        # -> add n_ldpc - n - 2Z punctured positions
        k_filler = self.encoder.k_ldpc - self.encoder.k  # number of filler bits
        nb_punc_bits = ((self.encoder.n_ldpc - k_filler)
                        - self.encoder.n - 2 * self.encoder.z)

        llr_5g = tf.concat([llr_5g,
                            tf.zeros([batch_size, nb_punc_bits - self._nb_pruned_nodes],
                                     self._output_dtype)],
                           1)

        # undo shortening (= add 0 positions after k bits, i.e. LLR=LLR_max)
        # the first k positions are the systematic bits
        x1 = tf.slice(llr_5g, [0, 0], [batch_size, self.encoder.k])

        # parity part
        nb_par_bits = (self.encoder.n_ldpc - k_filler
                       - self.encoder.k - self._nb_pruned_nodes)
        x2 = tf.slice(llr_5g,
                      [0, self.encoder.k],
                      [batch_size, nb_par_bits])

        # negative sign due to logit definition
        z = -self._llr_max * tf.ones([batch_size, k_filler], self._output_dtype)

        llr_5g = tf.concat([x1, z, x2], 1)

        # and execute the decoder (modified super-call because of damping)
        if not self._stateful:
            x_hat = self.super_call([llr_5g, idd_it])
        else:
            x_hat, msg_vn = self.super_call([llr_5g, msg_vn, idd_it])

        if self._return_infobits:  # return only info bits
            # reconstruct u_hat # code is systematic
            u_hat = tf.slice(x_hat, [0, 0], [batch_size, self.encoder.k])
            # Reshape u_hat so that it matches the original input dimensions
            output_shape = llr_ch_shape[0:-1] + [self.encoder.k]
            # overwrite first dimension as this could be None (Keras)
            output_shape[0] = -1
            u_reshaped = tf.reshape(u_hat, output_shape)

            # enable other output datatypes than tf.float32
            u_out = tf.cast(u_reshaped, self._output_dtype)

            if not self._stateful:
                return u_out
            else:
                return u_out, msg_vn

        else:  # return all codeword bits
            # the transmitted CW bits are not the same as used during decoding
            # cf. last parts of 5G encoding function

            # remove last dim
            x = tf.reshape(x_hat, [batch_size, self._n_pruned])

            # remove filler bits at pos (k, k_ldpc)
            x_no_filler1 = tf.slice(x, [0, 0], [batch_size, self.encoder.k])

            x_no_filler2 = tf.slice(x,
                                    [0, self.encoder.k_ldpc],
                                    [batch_size,
                                     self._n_pruned - self.encoder.k_ldpc])

            x_no_filler = tf.concat([x_no_filler1, x_no_filler2], 1)

            # shorten the first 2*Z positions and end after n bits
            x_short = tf.slice(x_no_filler,
                               [0, 2 * self.encoder.z],
                               [batch_size, self.encoder.n])

            # if used, apply rate-matching output interleaver again as
            # Sec. 5.4.2.2 in 38.212
            if self._encoder.num_bits_per_symbol is not None:
                x_short = tf.gather(x_short, self._encoder.out_int, axis=-1)

            # Reshape x_short so that it matches the original input dimensions
            # overwrite first dimension as this could be None (Keras)
            llr_ch_shape[0] = -1
            x_short = tf.reshape(x_short, llr_ch_shape)

            # enable other output datatypes than tf.float32
            x_out = tf.cast(x_short, self._output_dtype)

            if not self._stateful:
                return x_out
            else:
                return x_out, msg_vn