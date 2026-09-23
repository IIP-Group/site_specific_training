# Copy of the original file from MAbdollahpour/MDX with an added loss calculation 
# for the Datalake path to allow for real world data training.
# The loss calculation is modified to keep it independent from scrambling parameters and
# is directly calculated over the mapped OFDM grid.
# Modifications: Nuri Berke Baytekin

# Mahdi Abdollahpour (mahdi.abdollahpour@unibo.it)
# 2025

# Some functions (CGNN, CGNNOFDM, MDNeuralPUSCHReceiver) from 'neural_rx.py' modified and used here



import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, SeparableConv2D, Layer, LayerNormalization, BatchNormalization
from tensorflow.nn import relu
from sionna.utils import flatten_dims, split_dim, flatten_last_dims, insert_dims, expand_to_rank, matrix_inv
from sionna.ofdm import ResourceGridDemapper
from sionna.nr import TBDecoder, LayerDemapper, PUSCHLSChannelEstimator
from sionna.mapping import Demapper, SymbolDemapper, Constellation

from sionna.mimo import lmmse_equalizer


from sionna.ofdm import LinearDetector

from sionna.mimo.utils import whiten_channel
import sionna as sn



class BDemapper(Layer):
    def __init__(   self,
                    num_bits_per_symbol,
                    demapping_type,
                    dtype=tf.float32,
                    **kwargs):

        super().__init__(dtype=dtype,**kwargs)


        self._dtype = dtype
        self._cdtype = tf.complex64
        if dtype==tf.float64:
            self._cdtype = tf.complex128


        constellation_ = Constellation.create_or_check_constellation(
                                    "qam",
                                    num_bits_per_symbol,
                                    constellation=None,
                                    dtype=self._cdtype)
        self._demapper = Demapper(demapping_type,
                constellation=constellation_,
                hard_out=False,
                dtype=self._cdtype)



    def call(self,inputs):
        x, no = inputs
        # x  [batch_size, num_tx, num_subcarriers, num_ofdm_symbols]
        #       received signal

        # no [batch_size, num_tx, num_subcarriers, num_ofdm_symbols]


        # ---------------------------------
        x_shape = tf.shape(x)
        # [batch_size, num_tx, num_subcarriers* num_ofdm_symbols], complex
        x = flatten_last_dims(x,2)
        no = flatten_last_dims(no,2)

        # [batch_size, num_tx, num_subcarriers* num_ofdm_symbols*num_bits_per_symbol]
        llrs = self._demapper([x,no])
                
        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_bits_per_symbol]
        llrs = tf.reshape(llrs, tf.concat([x_shape, [-1]], axis=0) )




        tf.debugging.assert_all_finite(llrs, "llrs contains non-finite values")


        return llrs



class LMMSE(Layer):

    def __init__(   self,
                    num_units_agg=[[2],[1],[1],[0],[1],[1],[1],[1],[1]],
                    num_mcs=1,
                    name_suffix="",
                    dtype=tf.float32,
                    **kwargs):
        name = f"LMMSE_{name_suffix}"


        self._name_suffix = name_suffix
        self._dtype = dtype
        self._cdtype = tf.complex64
        if dtype==tf.float64:
            self._cdtype = tf.complex128
        super().__init__(name=name, dtype=dtype,**kwargs)
        # super().__init__(name=name, dtype=self._cdtype,**kwargs)
        self._num_mcs = num_mcs
        self._num_units_agg = num_units_agg
        self._gamma = None
        self._theta = None
        self._zeta  = None
        self._mcs_mul = None

    def build(self, input_shape):
        num_ofdm_symbols = input_shape[0][2]
        # input noise mul-------------------
        shape = None
        if self._num_units_agg[1][0]==2: # mat in mul
            shape=(12,num_ofdm_symbols)
            name = f"in_noise_multiplier_mat_{self._name_suffix}"
        if self._num_units_agg[1][0]==1: # scaler in mul
            shape=()
            name = f"noise_multiplier_{self._name_suffix}"
        if shape is not None:
            
            self._gamma = self.add_weight(
                name=name,
                shape=shape,
                initializer=tf.keras.initializers.Constant(1.),
                trainable=True,
                dtype = self._dtype
            )

        # output noise mul-----------------
        shape = None
        if self._num_units_agg[2][0]==2: # mat in mul
            shape=(12,num_ofdm_symbols)
            name = f"out_noise_multiplier_mat_{self._name_suffix}"
        if self._num_units_agg[2][0]==1: # scaler in mul
            shape=()
            name = f"noise_multiplier_out_{self._name_suffix}"
        if shape is not None:
            
            self._theta = self.add_weight(
                name=name,
                shape=shape,
                initializer=tf.keras.initializers.Constant(1.),
                trainable=True,
                dtype = self._dtype
            )

        # output noise mul ----------
        name = f"x_multiplier_out_{self._name_suffix}"
        self._zeta = self.add_weight(
            name=name,
            shape=(),
            initializer=tf.keras.initializers.Constant(1.),
            trainable=True,
            dtype = self._dtype
        )

        # ------------- MCS mul
        if self._num_units_agg[5][0] == 0:
            self._mcs_mul = []
            if self._num_mcs>1:
                for i in range(self._num_mcs):
                    name = f"x_mcs_mul_{i}"
                    mcs_mul_ = self.add_weight(
                        name=name,
                        shape=(),
                        initializer=tf.keras.initializers.Constant(1.),
                        trainable=True,
                        dtype = self._dtype
                    )
                    self._mcs_mul.append(mcs_mul_)

    def cholesky_inverse(self, matrix):
        # Ensure the matrix is symmetric and positive definite
        matrix = (matrix + tf.linalg.adjoint(matrix)) / tf.constant(2, self._cdtype)
        # tf.debugging.assert_positive(tf.linalg.det(matrix), "Matrix must be positive definite.")

        # Compute Cholesky decomposition
        L = tf.linalg.cholesky(matrix)


        identity = tf.eye(tf.shape(L)[-1], dtype=L.dtype)
        matrix_inv = tf.linalg.cholesky_solve(L, identity)
        tf.debugging.assert_all_finite(tf.math.real(matrix_inv), f"[cholesky_inverse] [{self._name_suffix}] inverted matrix, (real part) contains non-finite values")
        tf.debugging.assert_all_finite(tf.math.imag(matrix_inv), f"[cholesky_inverse] [{self._name_suffix}] inverted matrix, (imag part) contains non-finite values")


        return matrix_inv

    def matrix_inv(self, tensor): # Sionna 

        if tensor.dtype in [tf.complex64, tf.complex128] \
                        and sn.config.xla_compat \
                        and not tf.executing_eagerly():
            s, u = tf.linalg.eigh(tensor)

            # Compute inverse of eigenvalues
            s = tf.abs(s)
            # s = tf.cast(s, self._dtype)
            # tf.debugging.assert_positive(s, "Input must be positive definite.")
            one = tf.constant(1.,dtype=s.dtype, shape=())
            s = one/s
            s = tf.cast(s, u.dtype)

            # Matrix multiplication
            s = tf.expand_dims(s, -2)
            return tf.matmul(u*s, u, adjoint_b=True)
        else:
            return tf.linalg.inv(tensor)
    def lmmse(self, y, h, s, shape, whiten_interference=False):

        # y [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        shape = tf.shape(y)
        num_prbs = shape[1]//12
        num_subcarriers = shape[1] 
        num_ofdm_symbols = shape[2]
        num_rx_ant = shape[3]
        # s [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        if self._num_units_agg[1][0]==1:
            s = tf.cast(self._gamma, self._cdtype) * s                                     
        if self._num_units_agg[1][0]==2:
            gamma = tf.expand_dims(self._gamma,axis=0)
            gamma = tf.expand_dims(gamma,axis=0)
            gamma = tf.expand_dims(gamma,axis=-1)
            gamma = tf.expand_dims(gamma,axis=-1)
            # [batch_size, num_prbs, 12, num_ofdm_symbols, num_rx_ant, num_rx_ant]
            gamma = tf.broadcast_to(gamma, [1, num_prbs, 12, num_ofdm_symbols, num_rx_ant, num_rx_ant])
            # [batch_size, num_prbs*12, num_ofdm_symbols, num_rx_ant, num_rx_ant]
            gamma = tf.reshape(gamma, [1, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_rx_ant])
            s = tf.cast(gamma, self._cdtype) * s        

        # Ensure positive noise variance
        e = tf.constant(0.00001,self._dtype)
        s =  tf.nn.relu( tf.math.real(s) ) + e
        s = tf.cast(s,self._cdtype)


        # Compute G
        g = tf.matmul(h, h, adjoint_b=True) + s           # [Nr,Nr]

        g_inv = self.cholesky_inverse(g)

        g = tf.matmul(h, g_inv, adjoint_a=True)     # [Nt,Nr]
        tf.debugging.assert_all_finite(tf.math.real(g_inv), "g_inv real contains non-finite values")
        tf.debugging.assert_all_finite(tf.math.imag(g_inv), "g_inv imag contains non-finite values")


        # Compute Gy
        y = tf.expand_dims(y, -1)
        gy = tf.squeeze(tf.matmul(g, y), axis=-1)   # [Nt,Nr][Nr,1]: [Nt,1]

        # Compute GH
        gh = tf.matmul(g, h)                        # [Nt,Nr][Nr,Nt]: [Nt,Nt]

        # Compute diag(GH)
        d = tf.linalg.diag_part(gh)                 # Nt

        # Compute x_hat
        x_hat = tf.math.divide_no_nan(gy,d)                                     
        x_hat = tf.cast(self._zeta,self._cdtype) * x_hat                        

        # --------------- Compute residual error variance
        d = tf.math.real(d)
        one = tf.constant(1.,dtype=self._dtype, shape=())
        d = tf.cast(d, dtype=self._dtype)
        no_eff = tf.math.divide_no_nan(one,d) - one


        num_tx = tf.shape(x_hat)[-1]
        if self._num_units_agg[2][0]==1:
            no_eff = tf.cast(self._theta, self._dtype) * no_eff
            # no_eff = tf.cast(self._theta, self._dtype) * tf.math.divide_no_nan(one,d) - one + 0.0001

        if self._num_units_agg[2][0]==2:
            theta = tf.expand_dims(self._theta,axis=0)
            theta = tf.expand_dims(theta,axis=0)
            theta = tf.expand_dims(theta,axis=-1)
            # [batch_size, num_prbs, 12, num_ofdm_symbols, num_tx]
            theta = tf.broadcast_to(theta, [1, num_prbs, 12, num_ofdm_symbols, num_tx])
            # [batch_size, num_prbs*12, num_ofdm_symbols, num_tx]
            theta = tf.reshape(theta, [1, num_subcarriers, num_ofdm_symbols, num_tx])
            no_eff = tf.cast(theta, self._dtype) * no_eff

        # Ensure positive error variance
        no_eff =  tf.nn.relu( no_eff ) + e


        if self._num_units_agg[8][0]==0:
            no_eff = d
                                     
        tf.debugging.assert_all_finite(no_eff, "no_eff contains non-finite values")
        tf.debugging.assert_all_finite(tf.math.real(x_hat), "x_real contains non-finite values")
        tf.debugging.assert_all_finite(tf.math.imag(x_hat), "x_imag contains non-finite values")

        return x_hat, no_eff

    def check_is_finit(self, x_real,x_imag,no_eff):

        tf.debugging.assert_all_finite(x_real, "x_real contains non-finite values")
        tf.debugging.assert_all_finite(x_imag, "x_imag contains non-finite values")
        tf.debugging.assert_all_finite(no_eff, "no_eff contains non-finite values")


    def call(self, inputs, mcs_ue_mask):
        y, h_hat, active_tx_x, s = inputs
        # y : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant],
        #      tf.complex
        #   The received OFDM resource grid after cyclic prefix removal and FFT.

        # h_hat : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
        #          2*num_rx_ant], tf.float
        #     Channel estimate.

        # active_tx_x: [batch_size, num_tx, 1, 1] 
        #     tf.complex

        # no  [batch_size]
        # noise variance

        # err_var [batch_size, num_rx, num_subcarriers, num_ofdm_symbols, num_rx_ant]



        # Complex h_hat, h_tilde
        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant], tf.complex
        real_part, imag_part = tf.split(h_hat, num_or_size_splits=2, axis=-1)
        h_hat = tf.complex(real_part, imag_part)

        # Reshape H
        # [batch_size,num_subcarriers, num_ofdm_symbols,num_rx_ant, num_tx], tf.complex
        h_hat = tf.transpose(h_hat, perm=[0,2,3,4,1])        



        shape = tf.shape(h_hat)
        # [batch_size, num_subcarriers, num_ofdm_symbols, num_tx]
        # x_hat, no_eff = lmmse_equalizer(y, h_hat, s, whiten_interference=False)
        x_hat, no_eff = self.lmmse(y, h_hat, s, shape, whiten_interference=False)


        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols]
        x_hat = tf.transpose(x_hat, perm=[0,3,1,2])
        no_eff = tf.transpose(no_eff, perm=[0,3,1,2])



        # -------  MCS error multiplier
        # mcs_ue_mask  [batch_size, max_num_tx, depth]

        if self._num_mcs>1 and self._num_units_agg[5][0]==0:
            no_eff_ = tf.zeros_like(no_eff)
            for i in range(self._num_mcs):
                # [batch_size, max_num_tx, 1]
                mask_i = tf.expand_dims(mcs_ue_mask[:, :, i], axis=-1)
                # [batch_size, max_num_tx, 1, 1]
                mask_i = tf.expand_dims(mask_i, axis=-1)
                no_eff_ = no_eff_ + self._mcs_mul[i]* mask_i * no_eff
            no_eff = no_eff_



        x_hat = tf.multiply(active_tx_x,x_hat)                            


        self.check_is_finit(tf.math.real(x_hat),tf.math.imag(x_hat),no_eff)

        return x_hat, no_eff



class LS(Layer):
    def __init__(   self,
                    num_tx,
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._dtype = dtype
        self._num_tx = num_tx
    def ls_res(self,y,x,h,batch_size):

        # h [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant*2]
        #    tf.float

        # y : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant],
        #      tf.complex
        #   The received OFDM resource grid after cyclic prefix removal and FFT.

        # x : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols],
        #     tf.complex
        #     symbol estimate.



        
        # num_tx = tf.shape(x)[1]
        num_tx = self._num_tx
        
        # Complex H
        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        real_part, imag_part = tf.split(h, num_or_size_splits=2, axis=-1)
        h = tf.complex(real_part, imag_part)
        
        # Reshape H
        # [batch_size,num_subcarriers, num_ofdm_symbols, num_rx_ant, num_tx], tf.complex
        h = tf.transpose(h, perm=[0,2,3,4,1])

        # Reshape x_hat
        # [batch_size, num_subcarriers, num_ofdm_symbols, num_tx], tf.complex
        x = tf.transpose(x,perm=[0,2,3,1])

    # ---------------------------- Eager & Graph ------------------------

        # # list of [batch_size,num_subcarriers, num_ofdm_symbols, num_rx_ant, 1]
        # h_hats = tf.split(h_hat, num_or_size_splits=num_tx, axis=-1)
        batch_size_, num_subcarriers, num_ofdm_symbols, num_rx_ant = h.shape[:-1]
        Hx = [
            tf.zeros((batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant), dtype=h.dtype)
            for _ in range(num_tx)
        ]
        for i in range(num_tx):

            # [batch_size, num_subcarriers, num_ofdm_symbols,1]
            xi = tf.expand_dims(x[...,i],axis=-1)

            # [batch_size,num_subcarriers, num_ofdm_symbols, num_rx_ant]
            hi = h[...,i]
            # hi = tf.expand_dims(hi,axis=-1)

            # [batch_size,num_subcarriers, num_ofdm_symbols, num_rx_ant]
            Hx_i = tf.multiply(hi, xi)  

            # list [batch_size,num_subcarriers, num_ofdm_symbols, num_rx_ant]
            Hx[i]=Hx_i
            

        # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        Hx_sum = tf.add_n(Hx) # =Hx = matmul(h,x)

        # [batch_size,num_subcarriers, num_ofdm_symbols, num_rx_ant,num_tx]
        Hxs = tf.stack(Hx,axis=-1)
    # -----------------------------------------------------------



        # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant,1]
        Hx_sum = tf.expand_dims(Hx_sum,axis=-1)


        # res[batch_size,num_subcarriers, num_ofdm_symbols, num_rx_ant,num_tx]
        res_tx = Hx_sum - Hxs

        # Reshape y
        # from [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant],
        # ro   [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, 1],
        y = tf.expand_dims(y,axis=-1)

        # hx for every tx
        # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_tx]
        hx_tx = y - res_tx

        # Reshape x
        # x_hat [batch_size, num_subcarriers, num_ofdm_symbols, 1, num_tx]
        x = tf.expand_dims(x,axis=3)

        # H_LS: Data aided h_ls for every tx
        # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_tx]
        # h_ls_d = tf.math.divide_no_nan(hx_tx , x)
        h_ls_d = tf.multiply(hx_tx,tf.math.conj(x))

        # Output Reshape
        # res_tx [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant*2]
        h_ls_d = tf.transpose(h_ls_d,perm=[0,4,1,2,3])
        h_ls_d = tf.concat([tf.math.real(h_ls_d), tf.math.imag(h_ls_d)], axis=-1)


        return h_ls_d


    def check_is_finit(self, ls):


        tf.debugging.assert_all_finite(ls, "x_real contains non-finite values")



    def call(self, inputs, res=False):

        if res:
            y, x, pilot_mask, h, batch_size = inputs
            ls = self.ls_res(y,x,h, batch_size)
            ls = tf.multiply(pilot_mask,ls)
            self.check_is_finit(ls)
            return ls
        else:
            y, x, pilot_mask, h, batch_size = inputs

        # y : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant],
        #      tf.complex
        #   The received OFDM resource grid after cyclic prefix removal and FFT.

        # x : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols],
        #     tf.complex
        #     symbol estimate.

        # pilot_mask [1,num_tx,num_effective_subcarriers, num_ofdm_symbols,1], tf.float

        # [batch_size, 1, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        y = tf.expand_dims(y,axis=-1)
        y = tf.transpose(y, perm=[0,4,1,2,3])

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 1]
        x = tf.expand_dims(x,axis=-1)

        # ls[batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        # ls = tf.math.divide_no_nan(y, x)
        ls = tf.multiply(y, tf.math.conj(x))

        # ls[batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        ls = tf.concat([tf.math.real(ls), tf.math.imag(ls)], axis=-1)

        # Zero pilots
        ls = tf.multiply(pilot_mask,ls)
        self.check_is_finit(ls)

        return ls




class ResBlock(Layer):

    def __init__(   self,
                    num_units, # num filters
                    num_rx_ant,
                    name_suffix="",
                    h_k0=6, # input channels
                    pe_k0=2, # input dimensionality of Positional Encoding
                    pe_type=0,
                    input_relu=0,
                    input_norm=0,
                    num_skip_connections=1,
                    dtype=tf.float32,
                    **kwargs):
        self._name_suffix = name_suffix
        name = f"ResBlock_{name_suffix}"
        super().__init__(name=name,dtype=dtype,**kwargs)

        self._pe_type = pe_type
        self._num_rx_ant = num_rx_ant
        self._num_units = num_units
        self._input_relu = input_relu
        self._input_norm = input_norm
        self._name_suffix = name_suffix
        self._num_skip_connections = num_skip_connections
        # ----------- Preprocess ---------------
        if input_norm:
            # input: h [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, h_k0]
            self._BN = BatchNormalization(axis=-1, name=f"input_norm")
            self._BN1 = BatchNormalization(axis=-1, name=f"input_norm1")

        # ------------- Input Layer of V&H Passes -----------------
        act_v = None
        act_h = None
        self._input_vert_h=[]
        self._input_horz_h=[]
        self._input_vert_pe=[]
        self._input_horz_pe=[]

        in_num_ant = 1
        k_last = num_units
        str = ""
        # input_shape= [num_subcarriers, num_ofdm_symbols, 1, h_k0*in_num_ant]
        # output_shape=[num_subcarriers, num_ofdm_symbols, 1, h_k0*in_num_ant]
        input_horz_h = Conv3D(filters=h_k0*in_num_ant, kernel_size=(3,3,1), activation=act_h,
                        data_format='channels_last', padding='same', strides=(1,1,1), dtype=dtype, name=f"input_horzH{str}", groups=h_k0*in_num_ant )

        # input_shape=[num_subcarriers, num_ofdm_symbols, 1,h_k0*in_num_ant]
        # output_shape=[num_subcarriers, num_ofdm_symbols, 1, k_last]
        input_vert_h = Conv3D(filters=k_last, kernel_size=(1,1,1), activation=act_v,
                        data_format='channels_last', padding='valid', strides=(1,1,1), dtype=dtype, name=f"input_vertH{str}")
        self._input_vert_h.append(input_vert_h)
        self._input_horz_h.append(input_horz_h)

        # ---- pe
        # input_shape= [num_subcarriers, num_ofdm_symbols, 1, pe_k0]
        # output_shape=[num_subcarriers, num_ofdm_symbols, 1, pe_k0]
        input_horz_pe = Conv3D(filters=pe_k0, kernel_size=(3,3,1), activation=act_h,
                        data_format='channels_last', padding='same', strides=(1,1,1), dtype=dtype, name=f"input_horzP", groups=pe_k0)
        
        # input_shape= [num_subcarriers, num_ofdm_symbols, 1, pe_k0]
        # output_shape=[num_subcarriers, num_ofdm_symbols, 1, k_last]
        input_vert_pe = Conv3D(filters=k_last, kernel_size=(1,1,1), activation=act_v,
                        data_format='channels_last', padding='valid', strides=(1,1,1), dtype=dtype, name=f"input_vertP")

        self._input_vert_pe.append(input_vert_pe)
        self._input_horz_pe.append(input_horz_pe)


        # ------------------- Output Layers -----------------

        self._outpt_vert_h=[]
        self._outpt_horz_h=[]
        # k0 = num_units_state[-1]
        k0 = k_last
        k1 = 2


        # input_shape= [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, 1, k0]
        # output_shape=[num_subcarriers, num_ofdm_symbols, 1, k0]
        horz_h = Conv3D(filters=k0, kernel_size=(3,3,1), activation=act_h,
                        data_format='channels_last', padding='same', strides=(1,1,1), dtype=dtype, name=f"outpt_horzH", groups=k0)

        # input_shape= [num_subcarriers, num_ofdm_symbols, 1, k0]
        # output_shape=[num_subcarriers, num_ofdm_symbols, 1, k1]
        vert_h = Conv3D(filters=k1, kernel_size=(1,1,1), activation=act_v,
                        data_format='channels_last', padding='same', strides=(1,1,1), dtype=dtype, name=f"outpt_vertH" )
           
        self._outpt_vert_h.append(vert_h)
        self._outpt_horz_h.append(horz_h)

        if num_skip_connections==2:
            # input_shape= [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, 1, k0]
            # output_shape=[num_subcarriers, num_ofdm_symbols, 1, k0]
            horz_h = Conv3D(filters=k0, kernel_size=(3,3,1), activation=act_h,
                            data_format='channels_last', padding='same', strides=(1,1,1), dtype=dtype, name=f"outpt_horzH1", groups=k0)

            # input_shape= [num_subcarriers, num_ofdm_symbols, 1, k0]
            # output_shape=[num_subcarriers, num_ofdm_symbols, 1, k1]
            vert_h = Conv3D(filters=k1, kernel_size=(1,1,1), activation=act_v,
                            data_format='channels_last', padding='same', strides=(1,1,1), dtype=dtype, name=f"outpt_vertH1" )
            
            self._outpt_vert_h.append(vert_h)
            self._outpt_horz_h.append(horz_h)


    def call(self,h1, h2, pe):
        # h1  [batch_size * num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
        # h2  [batch_size * num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
        # pe [batch_size * num_tx, num_subcarriers, num_ofdm_symbols, 2]
        # pe [12,14,pe_k0(4)]

        # output h
        # [batch_size * num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]

        shape = tf.shape(h1)
        num_prbs = shape[1]//12
        num_subcarriers = shape[1] 
        num_ofdm_symbols = shape[2]
        
        # ----------------- preprocess ----------------
        if self._input_norm:
            h1 = self._BN(h1)
            h2 = self._BN1(h2)
        if self._input_relu:
            h1 =  tf.nn.relu(h1)
            h2 =  tf.nn.relu(h2)

        # h  [batch_size * num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, h_k0]
        h = tf.concat([h2, h1], axis=-1)
        # -------------------------------------  Input ----------------------------------------
        # ------------------ PE ----------------
        # pe [batch_size * num_tx, num_subcarriers, num_ofdm_symbols, pe_k0]
        # pe [1, 12,14,pe_k0]
        if self._pe_type==1 or self._pe_type==2:
            # [1,12,14,pe_k0]
            pe = tf.expand_dims(pe,axis=0)

        # [batch_size/1, num_subcarriers, num_ofdm_symbols, 1, pe_k0]
        pe = tf.expand_dims(pe,axis=-1)
        pe = tf.transpose(pe,perm=[0,1,2,4,3])

        # input_shape= [num_subcarriers, num_ofdm_symbols, 1, pe_k0]
        # output_shape=[num_subcarriers, num_ofdm_symbols, 1, pe_k0]
        pe = self._input_horz_pe[0]( pe )
        

        # # [num_subcarriers, num_ofdm_symbols, pe_k0,1]
        # pe = tf.transpose(pe,perm=[0,1,2,4,3])

        # input_shape= [batch, num_subcarriers, num_ofdm_symbols, 1, pe_k0]
        # output_shape=[batch, num_subcarriers, num_ofdm_symbols, 1, k_last]
        # output_shape=[1, 12, 14, 1, k_last]
        pe = self._input_vert_pe[0](pe)
        

        if self._pe_type==1 or self._pe_type==2:
            # [1, 1, 12, 14, 1, k_last]
            pe = tf.expand_dims(pe,axis=0)

        # -------------------- H -----------------
        h_=[]
        h_1=[]
        for i in range(self._num_rx_ant):
            # h_i [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, 1, h_k0]
            h_i = h  [:,:,:,i,:]
            h_i = tf.expand_dims(h_i,axis=3)


   
            # -------------------- input ---------------
            # input_shape= (num_subcarriers, num_ofdm_symbols, 1,h_k0*1)
            # output_shape=(num_subcarriers, num_ofdm_symbols, 1,h_k0*1)
            h_i = self._input_horz_h[0](  h_i  )

            # input_shape=[num_subcarriers, num_ofdm_symbols, 1, h_k0*1]
            # output_shape=[num_subcarriers, num_ofdm_symbols, 1, k_last]
            h_i = self._input_vert_h[0](  h_i  )

            # -------------- Add Positional Encoding ------------
            if self._pe_type==1 or self._pe_type==2:
                # [num_prbs, 12, num_ofdm_symbols, 1, k_last*1]
                h_i = tf.reshape(h_i, [-1, num_prbs, 12, num_ofdm_symbols, 1, self._num_units])
                
                # pe[1, 1, 12, 14, 1, h_k0]

                h_i = h_i + pe

                # Back to origional shape
                # [num_subcarriers, num_ofdm_symbols, 1, k_last*1]
                h_i = tf.reshape(h_i, [-1, num_subcarriers, num_ofdm_symbols, 1, self._num_units])
            else:
                # pe [batch, num_subcarriers, num_ofdm_symbols, 1, k_last]
                h_i = h_i + pe

            # ---------------- Activation ---------------------
            h_i = tf.nn.relu(h_i)

            # -------------------- output ---------------
            if self._num_skip_connections==2:
                # input_shape= [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, 1, k_last]
                # output_shape=[num_subcarriers, num_ofdm_symbols, 1, k_last]
                h_i1 = self._outpt_horz_h[1](h_i)

                # input_shape= [num_subcarriers, num_ofdm_symbols, 1, k_last]
                # output_shape=[num_subcarriers, num_ofdm_symbols, 1, 2]
                h_i1 = self._outpt_vert_h[1](h_i1)
                # ---------------- store ---------------------
                h_1.append(h_i1)

            # input_shape= [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, 1, k_last]
            # output_shape=[num_subcarriers, num_ofdm_symbols, 1, k_last]
            h_i = self._outpt_horz_h[0](h_i)

            # input_shape= [num_subcarriers, num_ofdm_symbols, 1, k_last]
            # output_shape=[num_subcarriers, num_ofdm_symbols, 1, 2]
            h_i = self._outpt_vert_h[0](h_i)
            # ---------------- store ---------------------
            h_.append(h_i)
        


        # [num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
        h_ = tf.concat(h_,axis=3)
        if self._num_skip_connections==2:
            h_1 = tf.concat(h_1,axis=3)

        return h_, h_1


class CHNN(Layer):

    def __init__(   self,
                    num_units_state,
                    arch="2D",
                    layer_type="sepconv",
                    num_rx_ant=4, #required for 2D arch
                    norm="batch_norm", # "batch_norm", "layer_norm", None
                    name_suffix="",
                    h_k0=6, # input dimensionality per antenna
                    pe_k0=2, # input dimensionality of Positional Encoding
                    pe_type=0,
                    d_s=1,
                    num_units_init=[5],
                    num_units_agg=[2],
                    num_mcs=1, # num supported modulation orders
                    dtype=tf.float32,
                    **kwargs):
        self._name_suffix = name_suffix
        name = f"CHNN_{name_suffix}"
        super().__init__(name=name,dtype=dtype,**kwargs)

        self._num_mcs = num_mcs
        self._mcs_mul = None
        self._gamma_real = None
        self._gamma_imag = None
        self._gamma_real_sclr = None
        self._h_k0 = h_k0
        self._arch = arch
        self._norm = norm
        self._num_rx_ant = num_rx_ant
        self._pe_k0 = pe_k0
        self._input_type = d_s
        self._pe_type = pe_type
        self._num_units = num_units_state
        self._num_units_init = num_units_init
        self._num_units_agg = num_units_agg
        if self._input_type == 1:
            self._h_k0 = 2
        if self._input_type == 2:
            self._h_k0 = 3
        if self._input_type == 3:
            self._h_k0 = 4
        if self._input_type == 4:
            self._h_k0 = 6
        if self._input_type == 5:
            self._h_k0 = 10
        if self._input_type == 6:
            self._h_k0 = 2
        if self._input_type == 7:
            self._h_k0 = 3
        if self._input_type == 8:
            self._h_k0 = 9
        h_k0 = self._h_k0
        if norm is not None:
            if norm=="batch_norm":
                norm_layer = BatchNormalization 
            if norm=="layer_norm":
                norm_layer = LayerNormalization


    # -------------------- Arch:  -------------------
        self._num_input_layers = 1

        if "res_blocks" in arch:
            in_num_ant = 1
            self._num_input_layers = 1

        self._in_num_ant = in_num_ant
        self._arch = arch

        # input_shape=(num_subcarriers, num_ofdm_symbols, 6*num_rx_ant,1)
        # output_shape=(num_subcarriers, num_ofdm_symbols, num_rx_ant, k*2)


        # ------------- Preprocess -----------------
        self._norm_layer = []
        if norm is not None:
            # self._norm_layer.append(norm_layer(axis=[-4,-3], name=f"layer_norm_0"))# on channels axis
            axis = -1
            self._norm_layer.append(norm_layer(axis=axis, name=f"input_norm1"))# on channels axis
            self._norm_layer.append(norm_layer(axis=axis, name=f"input_norm2"))# on channels axis
            self._norm_layer.append(norm_layer(axis=axis, name=f"input_norm3"))# on channels axis

        # ------------- NN -----------------
        act_v = None
        act_h = None
        self._input_vert_h=[]
        self._input_horz_h=[]
        self._input_vert_pe=[]
        self._input_horz_pe=[]

       
        if "res_blocks" in arch:
            self._num_res_blocks = len(num_units_init)
            if "2" in arch:
                self._num_skip_connections=2
            else:
                self._num_skip_connections=1
            self._res_blocks = []
            for i in range(len(num_units_init)):
                if i==0:
                    input_relu=False
                    input_norm=False
                else:
                    input_relu=True
                    input_norm=True

                if i<len(num_units_init)-1:
                    num_skip_connections = self._num_skip_connections
                else:
                    num_skip_connections = 1

                name_suffix = f"_{i}"
                res_block = ResBlock(num_units_init[i], num_rx_ant, name_suffix=name_suffix, h_k0=h_k0, pe_k0=pe_k0,
                                     pe_type=pe_type, input_relu=input_relu, input_norm=input_norm, 
                                     num_skip_connections=num_skip_connections)
                self._res_blocks.append(res_block)

    def build(self, input_shape):
        num_ofdm_symbols = input_shape[0][3]
        if self._num_units_agg[0][0] == 0: # scaler mul
            name = f"skip_multiplier_real_scaler"
            self._gamma_real_sclr = self.add_weight(
                name=name,
                shape=(),
                initializer=tf.keras.initializers.Constant(0.0001),
                trainable=True
            )
            name = f"skip_multiplier_imag_scaler"
            self._gamma_imag_sclr = self.add_weight(
                name=name,
                shape=(),
                initializer=tf.keras.initializers.Constant(0.0001),
                trainable=True
            )
        if self._num_units_agg[0][0] == 1: # one skip mul matrix for both real, imag parts
            name = f"skip_multiplier_real"
            self._gamma_real = self.add_weight(
                name=name,
                shape=(12,num_ofdm_symbols),
                initializer=tf.keras.initializers.Constant(0.0001),
                trainable=True
            )
        if "res_blocks2" in self._arch and self._num_units_agg[3][0] > 0: # remove
            name = f"skip_multiplier_tilde"
            self._gamma_tilde = self.add_weight(
                name=name,
                shape=(12,num_ofdm_symbols),
                initializer=tf.keras.initializers.Constant(0.0001),
                trainable=True
            )


        if self._num_units_agg[0][0] == 2: # separated real & imag skip multiplier 
            name = f"skip_multiplier_real"
            self._gamma_real = self.add_weight(
                name=name,
                shape=(12,num_ofdm_symbols),
                initializer=tf.keras.initializers.Constant(0.0001),
                trainable=True
            )
            name = f"skip_multiplier_imag"
            self._gamma_imag = self.add_weight(
                name=name,
                shape=(12,num_ofdm_symbols),
                initializer=tf.keras.initializers.Constant(0.0001),
                trainable=True
            )
        
        if self._num_units_agg[0][0] == 3: # one skip mul mat per resBlock
            self._gamma_real = []
            for i in range(self._num_res_blocks):
                name = f"skip_multiplier_{i}"
                gamma_ = self.add_weight(
                    name=name,
                    shape=(12,num_ofdm_symbols),
                    initializer=tf.keras.initializers.Constant(0.0001),
                    trainable=True
                )
                self._gamma_real.append(gamma_)

        if self._num_units_agg[9][0] == 0: # modulation specific skip mul
            self._mcs_mul = []
            if self._num_mcs>1:
                for i in range(self._num_mcs):
                    name = f"mcs_mul_{i}"
                    mcs_mul_ = self.add_weight(
                        name=name,
                        shape=(),
                        initializer=tf.keras.initializers.Constant(1.),
                        trainable=True,
                        dtype = self._dtype
                    )
                    self._mcs_mul.append(mcs_mul_)


    def resNN(self, h_hat, hdres, pe, batch_size=None, num_tx=None, mcs_ue_mask=None):
        # h  [batch_size * num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
        # pe [batch_size * num_tx, num_subcarriers, num_ofdm_symbols, 2]
        # pe [12,14,pe_k0(4)]

        # h_hat [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]

        # output h
        # [num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]


        shape = tf.shape(h_hat)
        num_prbs = shape[1]//12
        num_subcarriers = shape[1] 
        num_ofdm_symbols = shape[2]


        # ---------------- initial h -------------
        # # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols,  2*num_rx_ant]
        # h_hat = flatten_dims(h_hat, num_dims=2, axis=0)
        # # h [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
        # real, imag = tf.split(h_hat, num_or_size_splits=2, axis=-1)
        # h_hat = tf.stack([real, imag], axis=-1)

        # # ----------------- h_tilde, h_hat -------------
        # hdres = h[...,0:2]
        # h_hat = h[...,2:4]

        for i, block in enumerate(self._res_blocks):
            # [num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
            h_hat_new, hdres_new = block(h_hat, hdres,pe)

            # ----------------------- Prepare Skip Mul --------------------
            # if i < len(self._res_blocks)-1:   #

            # skip_multiplier [1, 1, 12, 14, 1, 1]
            if self._num_units_agg[0][0]==0:
                gamma_real = self._gamma_real_sclr
                gamma_imag = self._gamma_real_sclr
            if self._num_units_agg[0][0]==1:
                gamma_real = insert_dims(self._gamma_real, num_dims=2, axis=0)
                gamma_real = insert_dims(gamma_real, num_dims=2, axis=-1)
                gamma_imag = gamma_real
            if self._num_units_agg[0][0]==2:
                gamma_real = insert_dims(self._gamma_real, num_dims=2, axis=0)
                gamma_real = insert_dims(gamma_real, num_dims=2, axis=-1)
                gamma_imag = insert_dims(self._gamma_imag, num_dims=2, axis=0)
                gamma_imag = insert_dims(gamma_imag, num_dims=2, axis=-1)
            if self._num_units_agg[0][0]==3:
                gamma_real = insert_dims(self._gamma_real[i], num_dims=2, axis=0)
                gamma_real = insert_dims(gamma_real, num_dims=2, axis=-1)
                gamma_imag = gamma_real



            # ----------------------- Skip Mul h_hat --------------------
            # if i==len(self._res_blocks)-1: # skip mul after last block
            # [batch_size*num_tx, [num_prbs, 12], num_ofdm_symbols, num_rx_ant, 2]
            h_hat_new = split_dim(h_hat_new, [num_prbs, 12], 1)
            # 2 *[batch_size*num_tx, [num_prbs, 12], num_ofdm_symbols, num_rx_ant, 1]
            real, imag = tf.expand_dims(h_hat_new[...,0],axis=-1), tf.expand_dims(h_hat_new[...,1],axis=-1)

            real = tf.multiply(real, gamma_real)        
            imag = tf.multiply(imag, gamma_imag)

            # [batch_size*num_tx, [num_prbs, 12], num_ofdm_symbols, num_rx_ant, 2]
            h_hat_new = tf.concat([real, imag], axis=-1)

            # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
            h_hat_new = tf.reshape(h_hat_new, [-1, num_subcarriers, num_ofdm_symbols, self._num_rx_ant, 2])


            # mcs specific skip mul
            if self._num_units_agg[9][0]==0 and self._num_mcs>1: # modulation specific skip mul
                # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
                h_hat_new = split_dim(h_hat_new, [batch_size, num_tx], 0)

                h__ = tf.zeros_like(h_hat_new)
                for i_ in range(self._num_mcs):
                    # [batch_size, max_num_tx, 1]
                    mask_i = tf.expand_dims(mcs_ue_mask[:, :, i_], axis=-1)
                    # [batch_size, max_num_tx, 1,1,1,1]
                    mask_i = tf.expand_dims(mask_i, axis=-1)
                    mask_i = tf.expand_dims(mask_i, axis=-1)
                    mask_i = tf.expand_dims(mask_i, axis=-1)
                    # \nh_hat_new[batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]:{tf.shape(h_hat_new)}")
                    h__ = h__ + self._mcs_mul[i_] * mask_i * h_hat_new
                h_hat_new = h__
                
                # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
                h_hat_new = flatten_dims(h_hat_new, num_dims=2, axis=0)
            # --------------------------- skip Add h_hat -------------------------
            h_hat = h_hat + h_hat_new

            # ---------------------------- skip h_dres --------------------
            if self._num_skip_connections==2:
                # ------------------ skip Mul h_dres--------------
                if i==len(self._res_blocks)-1 and self._num_units_agg[3][0] > 0: # skip mul after last block
                    gamma_tilde = insert_dims(self._gamma_tilde, num_dims=2, axis=0)
                    gamma_tilde = insert_dims(gamma_tilde, num_dims=2, axis=-1)

                    hdres_new = split_dim(hdres_new, [num_prbs, 12], 1)
                    # 2 *[batch_size*num_tx, [num_prbs, 12], num_ofdm_symbols, num_rx_ant, 1]
                    real1, imag1 = tf.expand_dims(hdres_new[...,0],axis=-1), tf.expand_dims(hdres_new[...,1],axis=-1)

                    real1 = tf.multiply(real1, gamma_tilde)
                    imag1 = tf.multiply(imag1, gamma_tilde)

                    # [batch_size*num_tx, [num_prbs, 12], num_ofdm_symbols, num_rx_ant, 2]
                    hdres_new = tf.concat([real1, imag1], axis=-1)
                    
                    # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
                    hdres_new = tf.reshape(hdres_new, [-1, num_subcarriers, num_ofdm_symbols, self._num_rx_ant, 2])
                # -------------------- skip Add h_dres ---------------
                if i < len(self._res_blocks)-1:
                    hdres = hdres + hdres_new

            # # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 4] # works only with self._input_type == 3
            # h = tf.concat([hdres, h_hat], axis=-1)

        if self._num_skip_connections==1:
            hdres = h_hat

        return h_hat, hdres


    def preprocess(self, h1, h2, h3):
        # h [batch_size*num_tx, num_subcarriers, num_ofdm_symbols,  2*num_rx_ant]
        # output h [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, h_k0]


        # h1: h_hat
        # h2: h_d_res
        # h3: h_d

        shape = tf.shape(h1)
        num_subcarriers = shape[1] 
        num_ofdm_symbols = shape[2]
        num_rx_ant = shape[3]//2

        # ---------- Make Data Contiguous per Ant ---------------
        # h [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
        real, imag = tf.split(h1, num_or_size_splits=2, axis=-1)
        h1 = tf.stack([real, imag], axis=-1)

        real, imag = tf.split(h2, num_or_size_splits=2, axis=-1)
        h2 = tf.stack([real, imag], axis=-1)

        real, imag = tf.split(h3, num_or_size_splits=2, axis=-1)
        h3 = tf.stack([real, imag], axis=-1)

        # ---------------------- Normalize --------------------
        if self._norm is not None:
            h1 = self._norm_layer[0](h1)
            h2 = self._norm_layer[1](h2)
            h3 = self._norm_layer[2](h3)






        #----------------------------[h_dres, h_hat], h_k0=4    ******

        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 4]
        h = tf.concat([h2, h1], axis=-1)
        # # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant* 4]
        # h = flatten_last_dims(h, num_dims=2)

        

        return h

    def check_is_finit(self,h1,h2,h1_old,h2_old,pe):


        tf.debugging.assert_all_finite(h1, "x_real contains non-finite values")
        tf.debugging.assert_all_finite(h2, "x_imag contains non-finite values")
        tf.debugging.assert_all_finite(h1_old, "no_eff contains non-finite values")
        tf.debugging.assert_all_finite(h2_old, "no_eff contains non-finite values")


    def call(self, inputs, mcs_ue_mask=None):

        h_hat, h2, h3, pe, active_tx_h = inputs


        # h1; h_hat [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        #      tf.float

        # h2: h_d_res [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        #      tf.float
        
        # h3: h_d [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]

        # pe : [num_tx, num_subcarriers, num_ofdm_symbols, 2],
        # pe : [12,14,4]
        #      tf.float
        #  Map showing the position of the nearest pilot for every user in time
        #     and frequency.
        #     This can be seen as a form of positional encoding.


        h1 = tf.identity(h_hat)

        shape = tf.shape(h1)
        num_prbs = shape[2]//12
        num_subcarriers = shape[2] 
        num_ofdm_symbols = shape[3]
        batch_size = shape[0]
        num_tx = shape[1]


        self.check_is_finit(h1,h2,h1,h2,pe)


        h1 = tf.multiply(h1,active_tx_h)
        h2 = tf.multiply(h2,active_tx_h)
        h3 = tf.multiply(h3,active_tx_h)



        # move num_tx to batch dimension
        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols,  2*num_rx_ant]
        h1 = flatten_dims(h1, num_dims=2, axis=0)
        h2 = flatten_dims(h2, num_dims=2, axis=0)
        h3 = flatten_dims(h3, num_dims=2, axis=0)
        #       \n {tf.shape(h1)},\n{tf.shape(h2)}")
        
        # if tf.rank(pe)>3:
        if self._pe_type==0 or self._pe_type==3:
            # [batch_size * num_tx, num_subcarriers, num_ofdm_symbols, 2]
            pe = tf.tile(tf.expand_dims(pe, axis=0), [batch_size, 1, 1, 1, 1])
            pe = flatten_dims(pe, 2, 0)
        

        # ---------------------- PreProcess -----------------
        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, h_k0]
        hp = self.preprocess(h1,h2,h3)

       

        # --------------------------- res_blocks archs ---------------------------
        h_tilde = h1
        h_hat = h1

        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
        h_hat, h_tilde = self.resNN(hp[...,2:4],hp[...,0:2], pe, batch_size, num_tx, mcs_ue_mask) # works with input type 3

        real, imag = tf.split(h_hat, num_or_size_splits=2, axis=-1)
        h_hat = tf.concat([tf.squeeze(real,axis=[-1]), tf.squeeze(imag,axis=[-1])], axis=-1)
        h_hat = split_dim(h_hat, [batch_size, num_tx], 0)

        real, imag = tf.split(h_tilde, num_or_size_splits=2, axis=-1)
        h_tilde = tf.concat([tf.squeeze(real,axis=[-1]), tf.squeeze(imag,axis=[-1])], axis=-1)
        h_tilde = split_dim(h_tilde, [batch_size, num_tx], 0)


        return h_hat, h_tilde



class CGNN(Model):
   
    def __init__(   self,
                    num_bits_per_symbol,
                    num_rx_ant,
                    num_it,
                    arch,
                    d_s,
                    num_units_init,
                    num_units_agg,
                    num_units_state ,
                    num_units_readout,
                    layer_type_dense,
                    layer_type_conv,
                    layer_type_readout,
                    pilot_mask, # [1,num_tx,num_effective_subcarriers, num_ofdm_symbols,1]
                    constellation,#####
                    demapping_type,
                    max_num_tx, ####
                    training=False,
                    apply_multiloss=False,
                    var_mcs_masking=False,
                    pe_d=2,
                    pe_type=False,
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(dtype=dtype,**kwargs)

        self._training = training

        self._apply_multiloss = apply_multiloss
        self._var_mcs_masking = var_mcs_masking
        self._pilot_mask = pilot_mask
        self._dtype = dtype
        self._cdtype = tf.complex64
        self._num_units_agg = num_units_agg
        if dtype==tf.float64:
            self._cdtype = tf.complex128
        self._pe_type = pe_type
        # Iterations blocks
        # self._oamps = []
        self._chnn = []
        self._num_mcs = len(num_bits_per_symbol)
        self._lmmse0 = LMMSE(name_suffix=f"dummy",dtype=dtype)

        if num_units_agg[4][0] > 0:
            self._lmmse_init = LMMSE(num_mcs=self._num_mcs,name_suffix=f"init",dtype=dtype)
        if num_units_agg[4][0] == 0:
            self._lmmse_init = LMMSE(num_mcs=self._num_mcs,num_units_agg=num_units_agg,name_suffix=f"init",dtype=dtype)


        self._lmmse = []

        for i in range(num_it):

            lmmse = LMMSE(num_mcs=self._num_mcs,num_units_agg=num_units_agg, name_suffix=f"{i}",dtype=dtype)
            self._lmmse.append(lmmse)
            chnn = CHNN( num_units_state[i],
                        arch=arch,
                        num_rx_ant=num_rx_ant,
                        name_suffix=f"{i}",
                        pe_k0=pe_d,
                        d_s = d_s,
                        num_units_init=num_units_init,
                        num_units_agg=num_units_agg,
                        pe_type=pe_type,
                        num_mcs=self._num_mcs,
                        dtype=dtype)
            self._chnn.append(chnn)

        self._bdemapper = []
        for num_bits_per_symbol_ in num_bits_per_symbol:
            bd = BDemapper(num_bits_per_symbol_, demapping_type)
            self._bdemapper.append(bd)



        self._ls = LS(num_tx=max_num_tx,dtype=dtype)
        # self._lmmse = LMMSE(dtype=dtype)
        self._num_it = num_it

        self._num_mcss_supported = len(num_bits_per_symbol)
        self._num_bits_per_symbol = num_bits_per_symbol

        self._inc_num_it = 1
        self._step = 0


    @property
    def apply_multiloss(self):
        """Average loss over all iterations or eval just the last iteration."""
        return self._apply_multiloss

    @apply_multiloss.setter
    def apply_multiloss(self, val):
        assert isinstance(val, bool), "apply_multiloss must be bool."
        self._apply_multiloss = val

    @property
    def num_it(self):
        """Number of receiver iterations."""
        return self._num_it

    @num_it.setter
    def num_it(self, val):
        assert (val >= 1) and (val <= len(self._chnn)),\
            "Invalid number of iterations"
        self._num_it = val

   
    def call(self, inputs):

        y, pe, h_hat, active_tx, mcs_ue_mask, no, x, h, err_var, batch_size = inputs

        if x is not None:
            x = tf.complex(x[...,0],x[...,1]) # ground truth for debug porposes


        # y : [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant],
        #     tf.float
        #   The received OFDM resource grid after cyclic prefix removal and FFT.

        # pe : [num_tx, num_subcarriers, num_ofdm_symbols, 2],
        #      tf.float
        #  Map showing the position of the nearest pilot for every user in time
        #     and frequency.
        #     This can be seen as a form of positional encoding.

        # h_hat : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant], tf.float
        #     Channel estimate.

        # active_tx: [batch_size, num_tx], tf.float
        #      Active user mask.

        # mcs_ue_mask: [batch_size, max_num_tx, depth], depth being num supported MCSs (one hot)

        # constellation_points (ragged) [batch_size, max_num_tx, (num_points)]

        # err_var
        #  [batch_size, num_rx, num_effective_subcarriers, num_ofdm_symbols, num_rx_ant]

        # no [batch_size]


        batch_size = tf.shape(y)[0]


        # Initialize X
        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols], complex
        x_hat = tf.zeros(tf.shape(h_hat)[0:-1],dtype=self._cdtype)

        # Mask H
        # Multply by Zero non-active users
        active_tx_h = expand_to_rank(active_tx, tf.rank(h_hat), axis=-1)
        # h_hat = tf.multiply(h_hat, active_tx_h)
        h_ls = h_hat
        h_tilde = h_hat

        # Mask Pilots
        # [1,num_tx,num_effective_subcarriers, num_ofdm_symbols,1], tf.float
        pilot_mask = self._pilot_mask

        # Mask for X
        active_tx_x = expand_to_rank(active_tx, 4, axis=-1)
        active_tx_x = tf.complex(active_tx_x, tf.zeros_like(active_tx_x) )
        active_tx_x = tf.cast(active_tx_x, self._cdtype)

        active_tx_llr = expand_to_rank(active_tx, 5, axis=-1)


        # Reshape & Complex y 
        # from [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        # To   [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant], tf.complex
        real_part, imag_part = tf.split(y, num_or_size_splits=2, axis=-1)
        y = tf.complex(real_part, imag_part)


        ########################################
        # Run Joint CHannel Estimation & Detection (JCHED) iterations
        ########################################
        # Remark: each iteration uses a different NN with different weights
        # weight sharing could possibly be used, but degrades the performance
        # supports only var_mcs_masking

        llrs = []
        h_hats = []
        h_tildes = []
        x_hats = []

        batch_size = tf.shape(y)[0]
        num_rx_ant = tf.shape(y)[-1]

        # [batch_size, num_rx, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        err_var = tf.squeeze(err_var,axis=1)
        err_var_ = tf.linalg.diag(err_var)
        err_var_ = tf.cast(err_var_,self._cdtype)

        # no [batch_size]
        # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        no_ = expand_to_rank(no,2,-1)
        no_ = tf.broadcast_to(no_,[batch_size, num_rx_ant])
        no_ = insert_dims(no_,num_dims=2,axis=1)
        no_ = tf.linalg.diag(no_)
        no_ = tf.cast(no_,self._cdtype)

        s = no_ + err_var_


        # -------------- debug: PerfChe/ls solution at it -1
        h_debug = h_ls # h, or h_ls
        x_hat_debug, no_eff = self._lmmse0([y, h_debug, active_tx_x, s], mcs_ue_mask)

        if self._num_units_agg[8][0]==1:
            llrs_ = []
            for idx in range(self._num_mcss_supported):
                # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_bits_per_symbol]
                llrs__ = self._bdemapper[idx]([x_hat_debug, no_eff])
                # llrs__ = tf.multiply(active_tx_llr, llrs__)
                llrs_.append(llrs__)

            llrs.append(llrs_)
            h_hats.append(h_debug)
            h_tildes.append(h_debug)
            x_hats.append(x_hat_debug)


        # ------------ LMMSE Init
        # x_hat, no_eff = self._lmmse[0]([y, h_hat, active_tx_x, s])
        x_hat, no_eff = self._lmmse_init([y, h_hat, active_tx_x, s],mcs_ue_mask)
        if self._num_units_agg[8][0]==1:
            llrs_ = []
            for idx in range(self._num_mcss_supported):
                # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_bits_per_symbol]
                llrs__ = self._bdemapper[idx]([x_hat, no_eff])
                # llrs__ = tf.multiply(active_tx_llr, llrs__)
                llrs_.append(llrs__)
            llrs.append(llrs_)


        # ----------------------------- 
        for i in range(self._num_it):

            # H_D --------------
            # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
            h_d_res = self._ls([y,x_hat,pilot_mask,h_hat, batch_size], res=True)
            h_d = self._ls([y,x_hat,pilot_mask,h_hat, batch_size], res=False)



            # Estimate Channel --------------
            # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant*2] 
            h_hat, h_tilde = self._chnn[i]([h_hat, h_d_res, h_d, pe, active_tx_h], mcs_ue_mask=mcs_ue_mask)


            x_hat, no_eff = self._lmmse[i]([y, h_hat, active_tx_x, s],mcs_ue_mask)



            # only during training every intermediate iteration is tracked
            if self._training or i==self._num_it-1:

                h_hats.append(h_hat)                
                h_tildes.append(h_tilde)   

                x_hats.append(x_hat)
                llrs_ = []
                # iterate over all MCS schemes individually
                for idx in range(self._num_mcss_supported):

                    # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_bits_per_symbol]
                    llrs__ = self._bdemapper[idx]([x_hat, no_eff])
                    # llrs__ = tf.multiply(active_tx_llr, llrs__)
                    llrs_.append(llrs__)
                llrs.append(llrs_)


        return llrs, h_hats, x_hats, no_eff, h_tildes

class CGNNOFDM(Model):
   
    def __init__(self,
                 sys_parameters,
                 max_num_tx,
                 training,
                 num_it=5,
                 d_s=1,
                 num_units_init=[64],
                 num_units_agg=[[64]],
                 num_units_state=[[64]],
                 num_units_readout=[64],
                 layer_demappers=None,
                 layer_type_dense="dense",
                 layer_type_conv="sepconv",
                 layer_type_readout="dense",
                 nrx_dtype=tf.float32,
                 **kwargs):
        super().__init__(**kwargs)
        
        self._num_units_agg = num_units_agg
        self._training = training
        self._max_num_tx = max_num_tx
        self._layer_demappers = layer_demappers
        self._sys_parameters = sys_parameters
        self._nrx_dtype = nrx_dtype
        self._nrx_cdtype = tf.complex64
        if self._nrx_dtype==tf.float64:
            self._nrx_cdtype=tf.complex128
        self._num_it = num_it

        self._num_mcss_supported = len(sys_parameters.mcs_index)

        self._rg = sys_parameters.transmitters[0]._resource_grid


        # all UEs in the same pusch config must use the same MCS
        self._num_bits_per_symbol = []
        self._constellation = []
        self._demapper = []
        _sys_demapper = []
        for mcs_list_idx in range(self._num_mcss_supported):
            num_bits_per_symbol_ = sys_parameters.pusch_configs[mcs_list_idx][0].tb.num_bits_per_symbol
            constellation_ = Constellation.create_or_check_constellation(
                                    "qam",
                                    num_bits_per_symbol_,
                                    constellation=None,
                                    dtype=self._nrx_cdtype)
            demapper_ = Demapper(self._sys_parameters.demapping_type,
                    constellation=constellation_,
                    hard_out=False,
                    dtype=self._nrx_cdtype)
            sym_demapper = SymbolDemapper(constellation=constellation_,
                                            hard_out=False,
                                            dtype=self._nrx_cdtype)
            self._num_bits_per_symbol.append(num_bits_per_symbol_)
            self._constellation.append( constellation_ )
            self._demapper.append( demapper_ )
            _sys_demapper.append( sym_demapper )

        ###################################################
        # Resource grid demapper to extract the
        # data-carrying resource elements from the
        # resource grid
        ###################################################
        self._rg_demapper = ResourceGridDemapper(self._rg,
                                                 sys_parameters.sm)
        # # Precompute indices to extract data symbols
        # mask = resource_grid.pilot_pattern.mask
        self._num_data_symbols = self._rg.pilot_pattern.num_data_symbols
        # data_ind = tf.argsort(flatten_last_dims(mask), direction="ASCENDING")
        # self._data_ind = data_ind[...,:num_data_symbols]
        #################################################
        # Instantiate the loss function if training
        #################################################
        if training:
            # Loss function
            self._bce = tf.keras.losses.BinaryCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.NONE)
            # Loss function
            self._mse = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE)
        ###############################################
        # Pre-compute positional encoding.
        # Positional encoding consists in the distance
        # to the nearest pilot in time and frequency.
        # It is therefore a 2D positional encoding.
        ##############################################

        # Indices of the pilot-carrying resource elements and pilot symbols
        rg_type = self._rg.build_type_grid()[:,0] # One stream only
        pilot_ind = tf.where(rg_type==1)
        pilots = flatten_last_dims(self._rg.pilot_pattern.pilots, 3)
        # Resource grid carrying only the pilots
        # [max_num_tx, num_effective_subcarriers, num_ofdm_symbols]
        pilots_only = tf.scatter_nd(pilot_ind, pilots,
                                    rg_type.shape)
        # Indices of pilots carrying RE (transmitter, freq, time)
        pilot_ind = tf.where(tf.abs(pilots_only) > 1e-3)
        pilot_ind = np.array(pilot_ind)

        # Sort the pilots according to which to which TX they are allocated
        pilot_ind_sorted = [ [] for _ in range(max_num_tx) ]

        for p_ind in pilot_ind:
            tx_ind = p_ind[0]
            re_ind = p_ind[1:]
            pilot_ind_sorted[tx_ind].append(re_ind)
        pilot_ind_sorted = np.array(pilot_ind_sorted)

        # Distance to the nearest pilot in time
        # Initialized with zeros and then filled.
        pilots_dist_time = np.zeros([   max_num_tx,
                                        self._rg.num_ofdm_symbols,
                                        self._rg.fft_size,
                                        pilot_ind_sorted.shape[1]])
        # Distance to the nearest pilot in frequency
        # Initialized with zeros and then filled
        pilots_dist_freq = np.zeros([   max_num_tx,
                                        self._rg.num_ofdm_symbols,
                                        self._rg.fft_size,
                                        pilot_ind_sorted.shape[1]])

        t_ind = np.arange(self._rg.num_ofdm_symbols)
        f_ind = np.arange(self._rg.fft_size)

        for tx_ind in range(max_num_tx):
            for i, p_ind in enumerate(pilot_ind_sorted[tx_ind]):

                pt = np.expand_dims(np.abs(p_ind[0] - t_ind), axis=1)
                pilots_dist_time[tx_ind, :, :, i] = pt

                pf = np.expand_dims(np.abs(p_ind[1] - f_ind), axis=0)
                pilots_dist_freq[tx_ind, :, :, i] = pf

        # Normalizing the tensors of distance to force zero-mean and
        # unit variance.
        nearest_pilot_dist_time = np.min(pilots_dist_time, axis=-1)
        nearest_pilot_dist_freq = np.min(pilots_dist_freq, axis=-1)
        nearest_pilot_dist_time -= np.mean(nearest_pilot_dist_time,
                                            axis=1, keepdims=True)
        std_ = np.std(nearest_pilot_dist_time, axis=1, keepdims=True)
        nearest_pilot_dist_time = np.where(std_ > 0.,
                                           nearest_pilot_dist_time / std_,
                                           nearest_pilot_dist_time)
        nearest_pilot_dist_freq -= np.mean(nearest_pilot_dist_freq,
                                            axis=2, keepdims=True)
        std_ = np.std(nearest_pilot_dist_freq, axis=2, keepdims=True)
        nearest_pilot_dist_freq = np.where(std_ > 0.,
                                           nearest_pilot_dist_freq / std_,
                                           nearest_pilot_dist_freq)

        # Stacking the time and frequency distances and casting to TF types.
        nearest_pilot_dist = np.stack([ nearest_pilot_dist_time,
                                        nearest_pilot_dist_freq],
                                        axis=-1)
        nearest_pilot_dist = tf.constant(nearest_pilot_dist, tf.float32)
        # Reshaping to match the expected shape.
        # [max_num_tx, num_subcarriers, num_ofdm_symbols, 2]
        self._nearest_pilot_dist = tf.transpose(nearest_pilot_dist,
                                                [0, 2, 1, 3])



        ####################################################
        # Core neural receiver
        ####################################################
        # Number of receive antennas
        num_rx_ant = sys_parameters.num_rx_antennas
        arch = sys_parameters.arch


        pilot_pattern = self._rg.pilot_pattern
        num_pilot_symbols = pilot_pattern.num_pilot_symbols
        # [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], int32/bool
        # 0's and 1's; 1's for pilot locations
        pilot_mask = pilot_pattern.mask


        # [num_tx,num_effective_subcarriers, num_ofdm_symbols]
        # 0s for pilot locations, 
        pilot_mask = 1 - pilot_mask[:,0]
        pilot_mask = tf.transpose(pilot_mask, perm=[0,2,1])
        # [1,num_tx,num_effective_subcarriers, num_ofdm_symbols,1]
        pilot_mask = insert_dims(pilot_mask,num_dims=1,axis=0)
        pilot_mask = insert_dims(pilot_mask,num_dims=1,axis=-1)
        pilot_mask = tf.cast(pilot_mask,self._dtype)
        
        pe_d = self._sys_parameters.pe_d # pos. encoding dim.

        if num_units_agg[8][0]==-1:
            layer_type_readout="sepconv"

        self._cgnn = CGNN(self._num_bits_per_symbol,  # is a list
                          num_rx_ant,
                          num_it,
                          arch,
                          d_s,
                          num_units_init,
                          num_units_agg,
                          num_units_state,
                          num_units_readout,
                          pilot_mask=pilot_mask, 
                          constellation=self._constellation,
                          demapping_type=self._sys_parameters.demapping_type,
                          max_num_tx = self._sys_parameters.max_num_tx,
                          training=training,
                          layer_type_dense=layer_type_dense,
                          layer_type_conv=layer_type_conv,
                          layer_type_readout=layer_type_readout,
                          var_mcs_masking=None,
                          pe_d=self._sys_parameters.pe_d,
                          pe_type=self._sys_parameters.pe_type,
                          dtype=nrx_dtype)
        # [1, num_tx, num_subcarriers, num_ofdm_symbols]
        self._pilot_mask = pilot_mask
        self._pilot_mask = tf.squeeze(self._pilot_mask,axis=-1)


                       
    @property
    def num_it(self):
        """Number of receiver iterations. No weight sharing is used."""
        return self._cgnn.num_it

    @num_it.setter
    def num_it(self, val):
        self._cgnn.num_it = val

    def call(self, inputs, mcs_arr_eval, mcs_ue_mask_eval=None, no=None):

        # training requires to feed the inputs
        # mcs_ue_mask: [batch_size, num_tx, num_mcss], tf.float
        # x [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,2]
        # h     [batch_size, num_tx, num_effective_subcarriers,num_ofdm_symbols, 2*num_rx_ant]
        # h_hat [batch_size, num_tx, num_effective_subcarriers,num_ofdm_symbols, 2*num_rx_ant]
        # y [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_subcarriers]
        if self._training:
            y, h_hat_init, active_tx, bits, h, mcs_ue_mask, x, no,err_var, batch_size = inputs
        else:
            y, h_hat_init, active_tx, no, x, h,err_var, batch_size = inputs
            if mcs_ue_mask_eval is None:
                mcs_ue_mask = tf.one_hot(mcs_arr_eval[0],
                                         depth=self._num_mcss_supported)
            else:
                mcs_ue_mask = mcs_ue_mask_eval
            mcs_ue_mask = expand_to_rank(mcs_ue_mask, 3, axis=0)


        # total number of possible streams; not all of them might be active.
        num_tx = tf.shape(active_tx)[1]

        num_prbs = tf.shape(y)[4]//12


        # mask pilots for pilotless communications
        if self._sys_parameters.mask_pilots:
            rg_type = self._rg.build_type_grid()
            # add batch dim
            rg_type = tf.expand_dims(rg_type, axis=0)
            rg_type = tf.broadcast_to(rg_type, tf.shape(y))
            y = tf.where(rg_type==1, tf.constant(0., y.dtype), y)

        ##############################################
        # Core Neural Receiver
        ##############################################

        # Reshaping to the expected shape
        # [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        y = y[:,0]
        y = tf.transpose(y, [0, 3, 2, 1])
        y = tf.concat([tf.math.real(y), tf.math.imag(y)], axis=-1)



        
        #  0:NRX default | 1:sin coding | 2:PRB coding | 3:NRX default + PRB coding
        if self._sys_parameters.pe_type==0:
            # [num_tx, num_subcarriers, num_ofdm_symbols, 2]
            pe = self._nearest_pilot_dist[:num_tx]
        if self._sys_parameters.pe_type==1 or self._sys_parameters.pe_type==2:
            # pe [12, 14, 2]
            pe = self._sys_parameters.PE
        if self._sys_parameters.pe_type==3:
            # pe [1, 1, 12, self._rg.num_ofdm_symbols, 2]
            pe = tf.expand_dims(self._sys_parameters.PE,axis=0)
            pe = tf.expand_dims(pe,axis=0)

            # pe [num_tx, num_prbs, 12, num_ofdm_symbols, 2]
            pe = tf.broadcast_to(pe,[num_tx, num_prbs, 12,self._rg.num_ofdm_symbols,2])

            # [num_tx, num_subcarriers, num_ofdm_symbols, 2]
            pe = tf.reshape(pe, [num_tx, num_prbs*12, self._rg.num_ofdm_symbols, 2])

            # [num_tx, num_subcarriers, num_ofdm_symbols, 2]
            pe = tf.concat([pe, self._nearest_pilot_dist[:num_tx]], axis=-1)

        # Calling the detector to compute LLRs.
        # List of size num_it of tensors of LLRs with shape:
        # llrs : [batch_size, num_tx, num_effective_subcarriers,
        #       num_ofdm_symbols, num_bits_per_symbol]
        # h_hats : [batch_size, num_tx, num_effective_subcarriers,
        #       num_ofdm_symbols, 2*num_rx_ant]

        # cast to desired dtypes
        y = tf.cast(y, self._nrx_dtype)
        pe = tf.cast(pe, self._nrx_dtype)

        if h_hat_init is not None:
            h_hat_init = tf.cast(h_hat_init, self._nrx_dtype)
        active_tx = tf.cast(active_tx, self._nrx_dtype)

        # and run the neural receiver
        h_temp=0
        llrs_, h_hats_, x_hats_, no_, h_tildes_ = self._cgnn([y, pe, h_hat_init, active_tx, mcs_ue_mask, no, x, h_temp,err_var, batch_size])

        indices = mcs_arr_eval

        # list of lists; outer list separates iterations, inner list the MCSs
        llrs = []

        h_hats = []
        h_tildes = []
        x_hats = []
        # process each list entry (=iteration) individually
        # for llrs_, h_hat_, x_hat_, h_tilde_ in zip(llrs_, h_hats_, x_hats_, h_tildes_):
        for llrs_ in llrs_:

            # local llrs list to only process LLRs of MCS indices specified in
            # mcs_arr_eval
            _llrs_ = []
            # loop over all evaluated mcs indices
            for idx in indices:


                
                
                # --------------------------- llrs at CGNN
                # cast back to tf.float32 (if NRX uses quantization)
                llrs_[idx] = tf.cast(llrs_[idx], tf.float32)
                
                
                # llr[batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_bits_per_symbol]

                ## [batch_size, 1, num_tx, num_ofdm_symbols,fft_size, num_bits_per_symbol]
                llrs_[idx] = tf.transpose(llrs_[idx], [0, 1, 3, 2, 4])

                if self._training and self._sys_parameters.channel_type == "Datalake":
                    (B, R) = llrs_[idx].shape[0:2]
                    llrs_[idx] = tf.reshape(llrs_[idx], (B, R, -1))
                    
                else:  #if not self._training:
                    llrs_[idx] = tf.expand_dims(llrs_[idx], axis=1)
                    
                    # [batch_size, num_tx, 1, num_data_symbols, num_bit_per_symbols]
                    llrs_[idx] = self._rg_demapper(llrs_[idx])
                    llrs_[idx] = llrs_[idx][:,:num_tx]

                    # batch_size_, num_tx, one, num_data_symbols, num_bit_per_symbols = llrs_[idx].shape

                    # [batch_size, num_tx, 1, num_data_symbols*num_bit_per_symbols]
                    # llrs_[idx] = flatten_last_dims(llrs_[idx], 2)
                    llrs_[idx] = tf.reshape(llrs_[idx], [batch_size, self._max_num_tx, 1, self._num_data_symbols*self._num_bits_per_symbol[idx]])



                    # -----------------------

                    # Remove stream dimension, NOTE: does not support
                    # multiple-streams
                    # per user; conceptually the neural receiver does, but would
                    # require modified reshapes
                    if self._layer_demappers is None:
                        llrs_[idx] = tf.squeeze(llrs_[idx], axis=-2)
                    else:
                        llrs_[idx] = self._layer_demappers[idx](llrs_[idx])
                
                
                _llrs_.append(llrs_[idx])



            # llr is of shape
            # [batch_size, num_tx, num_data_symbols*num_bit_per_symbols]

            # h_hat is of shape
            # [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols,
            #   2*num_rx_ant]

            # x_hat 
            # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2], float32

            llrs.append(_llrs_)
        
        # Datalake: build the DMRS/no-data loss mask ONCE per evaluated MCS,
        # before the labels are sanitized. Using a per-MCS list (instead of the
        # single tensor built from the leftover `idx`/`B`/`R` loop variables)
        # keeps multi-MCS correct and removes the reliance on loop leakage.
        dmrs_loss_mask = None
        if self._training and self._sys_parameters.channel_type == "Datalake":
            dmrs_loss_mask = []
            for idx in range(len(indices)):
                B = tf.shape(bits[idx])[0]
                R = tf.shape(bits[idx])[1]
                bits_flat = tf.reshape(bits[idx], (B, R, -1))
                # 1 for data-carrying REs, 0 for DMRS/no-data REs (sentinel -1)
                dmrs_loss_mask.append(tf.cast(bits_flat != -1, tf.float32))
                # replace -1 with 0 for a numerically stable BCE label
                bits[idx] = tf.where(bits_flat == -1, 0.,
                                     tf.cast(bits_flat, tf.float32))


        for h_hat_, x_hat_, h_tilde_ in zip(h_hats_, x_hats_, h_tildes_):
            x_hat_ = tf.stack([tf.math.real(x_hat_), tf.math.imag(x_hat_)], axis=-1)
            x_hat_ = tf.cast(x_hat_, tf.float32)
            h_hat_ = tf.cast(h_hat_, tf.float32)
            h_tilde_ = tf.cast(h_tilde_, tf.float32)

            h_hats.append(h_hat_)
            h_tildes.append(h_tilde_)
            x_hats.append(x_hat_)


        if self._training:
            
            mcs_mul = tf.constant([10, 1, 1/5], dtype=tf.float32) 

            exclude_n = 1 # exclude the dummy lmmse (used for debug)
            if self._num_units_agg[8][0]<1: # if not maxlog there is no lmmse_init and dummy losses
                exclude_n = 0 # i starts from 1
            # ------------------ Cross Entropy Loss on data
            
            loss_data = tf.constant(0.0, dtype=tf.float32)
            # l0=[]
            # l0_rel=[]
            l0 = [tf.constant(0.0, dtype=tf.float32) for _ in range(self._sys_parameters.num_nrx_iter+2)] # only one macs
            l0_rel = [tf.constant(0.0, dtype=tf.float32) for _ in range(self._sys_parameters.num_nrx_iter+2)]
            i=0
            for llrs_ in llrs:
                i=i+1
                for idx in range(len(indices)):

                    # [batch_size, max_num_tx]
                    loss_data_ = self._bce(bits[idx], llrs_[idx])

                    if self._sys_parameters.channel_type == "Datalake":
                        loss_data_ = tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=bits[idx],
                                    logits=llrs_[idx]) # important to preserve the bce shape for the dmrs masking
                        loss_data_ = tf.multiply(loss_data_, dmrs_loss_mask[idx])
                        # divide_no_nan guards fully-masked users (e.g. a padded
                        # inactive layer in mixed-layer training, whose bits are
                        # all -1): 0/0 -> 0 instead of NaN.
                        loss_data_ = tf.math.divide_no_nan(
                            tf.reduce_sum(loss_data_, axis=-1),
                            tf.reduce_sum(dmrs_loss_mask[idx], axis=-1))

                    mcs_ue_mask_ = expand_to_rank(
                        tf.gather(mcs_ue_mask, indices=indices[idx], axis=2),
                        tf.rank(loss_data_), axis=-1)

                    # select data loss only for associated MCSs
                    loss_data_ = tf.multiply(loss_data_, mcs_ue_mask_)

                    # only focus on active users
                    active_tx_data = expand_to_rank(active_tx,
                                                    tf.rank(loss_data_),
                                                    axis=-1)
                                                    
                    loss_data_ = tf.multiply(loss_data_, active_tx_data)

                    l0_ = loss_data_
                    # weight loss with snr
                    if self._num_units_agg[6][0]==1:
                        snr_mul = tf.math.log(1 + 1/no) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
                        snr_mul = expand_to_rank(snr_mul,2,-1)
                        snr_mul = tf.broadcast_to(snr_mul,[batch_size, num_tx])
                        snr_mul = tf.math.log(1 + snr_mul) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
                        loss_data_ = tf.multiply(loss_data_, snr_mul)

                    # weight loss with mcs index mul
                    if  self._num_units_agg[7][0]==1 and len(indices)>1:
                        # mcs_mul = tf.constant([4, 1, 1/4], dtype=tf.float32) 
                        loss_data_ = mcs_mul[i-1]*loss_data_



                    l0_ = tf.reduce_mean(l0_)
                    loss_data_ = tf.reduce_mean(loss_data_)
                    



                    if i==1:
                        ref = l0_
                    if i>exclude_n:
                        # Average over batch, transmitters, and resource grid
                        # loss_data += l0_
                        loss_data += loss_data_
                    # for prints
                    l0[i-1] = l0_
                    l0_rel[i-1] = l0_-ref

            # ------------------- MSE Loss on channel estimation
            # h [batch size, num_tx, num_subcarriers, num_ofdm_symbols,2*num_rx_ant]
            loss_chest = tf.constant(0.0, dtype=tf.float32)
            l1=[tf.constant(0.0, dtype=tf.float32) for _ in range(self._sys_parameters.num_nrx_iter+1)]

            loss_tilde = tf.constant(0.0, dtype=tf.float32)
            l1t=[tf.constant(0.0, dtype=tf.float32) for _ in range(self._sys_parameters.num_nrx_iter+1)]

            # weight loss with snr
            if self._num_units_agg[6][0]==1:
                snr_mul = tf.math.log(1 + 1/no) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
                snr_mul = expand_to_rank(snr_mul,4,-1)
                if h is not None:
                    snr_mul = tf.broadcast_to(snr_mul,[batch_size, tf.shape(h)[1], tf.shape(h)[2], tf.shape(h)[3]])
                elif h_hat_init is not None:
                    snr_mul = tf.broadcast_to(snr_mul,[batch_size, tf.shape(h_hat_init)[1], tf.shape(h_hat_init)[2], tf.shape(h_hat_init)[3]])
                snr_mul = tf.math.log(1 + snr_mul) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
                
            


            i = 0
            if h_hats is not None: # h_hat might not be available
                for h_hat_, h_tilde_ in zip(h_hats, h_tildes):
                    i = i + 1
                    if h is not None:
                        if i>exclude_n:
                            # [batch size, num_tx, num_subcarriers, num_ofdm_symbols]
                            loss_ = self._mse(h, h_hat_)

                            # weight with snr
                            if self._num_units_agg[6][0]==1:
                                loss_ = tf.multiply(loss_, snr_mul)
                            # weight loss with mcs index mul
                            if  self._num_units_agg[7][0]==1 and len(indices)>1:
                               
                                # [batch_size, max_num_tx, 1, 1]
                                mask_i = tf.expand_dims(mcs_ue_mask[:, :, 0], axis=-1)
                                mask_i = tf.expand_dims(mask_i,axis=-1)
                                loss__ = mcs_mul[0]*mask_i*loss_

                                mask_i = tf.expand_dims(mcs_ue_mask[:, :, 1], axis=-1)
                                mask_i = tf.expand_dims(mask_i,axis=-1)
                                loss__ = loss__ + mcs_mul[1]*mask_i*loss_

                                if len(indices) > 2:
                                    mask_i = tf.expand_dims(mcs_ue_mask[:, :, 2], axis=-1)
                                    mask_i = tf.expand_dims(mask_i,axis=-1)
                                    loss__ = loss__ + mcs_mul[2]*mask_i*loss_
                                loss_ = loss__

                            # Accumulate loss
                            loss_chest += loss_

                            loss_t = self._mse(h, h_tilde_)
                            # weight with snr
                            if self._num_units_agg[6][0]==1:
                                loss_t = tf.multiply(loss_t, snr_mul)
                            # Accumulate loss
                            loss_tilde += loss_t


                        l1[i-1] = self._mse(h, h_hat_)
                        l1t[i-1] = self._mse(h, h_tilde_)

            # only focus on active users
            active_tx_chest = expand_to_rank(active_tx,
                                             tf.rank(loss_chest), axis=-1)
            loss_chest = tf.multiply(loss_chest, active_tx_chest)
            # Average over batch, transmitters, and resource grid
            loss_chest = tf.reduce_mean(loss_chest)

            loss_tilde = tf.multiply(loss_tilde, active_tx_chest)
            # Average over batch, transmitters, and resource grid
            loss_tilde = tf.reduce_mean(loss_tilde)




            # -------------------- Prints for debug & log
            for i in range(len(l1)):
                l1_ = tf.multiply(l1[i], active_tx_chest)
                l1_ = tf.reduce_mean(l1_)
                l1[i] = l1_

                l1t_ = tf.multiply(l1t[i], active_tx_chest)
                l1t_ = tf.reduce_mean(l1t_)
                l1t[i] = l1t_

                # l2_ = tf.multiply(l2[i], self._pilot_mask)
                # l2_ = tf.multiply(l2_, active_tx_chest)
                # l2_ = tf.reduce_mean(l2_)


            #     # -- prints (disable for graph)
            #     print(f"i{i:02d}, loss_bce:{l0_rel[i].numpy():.5f}:{l0[i].numpy():.5f} loss_chest:{l1_.numpy():.8f}", end="")
            #     if i==0:
            #         print(" --> LSlin+LMMSE")
            #     else:
            #         print("")
            # print(f"out, loss_bce:{l0_rel[-1].numpy():.5f}:{l0[-1].numpy():.5f}")
            # print("---------------\n")
            


            return loss_data, loss_chest, l0_rel, l1, loss_tilde, l1t
        else:
            # Only return the last iteration during inference
            return llrs[-1][0], h_hats[-1]

class MDNeuralPUSCHReceiver(Layer):


    def __init__(self,
                sys_parameters,
                training=False,
                **kwargs):


        super().__init__(**kwargs)

        self._sys_parameters = sys_parameters

        self._training = training

        # init transport block enc/decoder
        self._tb_encoders = []   # @TODO encoderS and decoderS
        self._tb_decoders= []

        self._num_mcss_supported = len(sys_parameters.mcs_index)
        for mcs_list_idx in range(self._num_mcss_supported):
                self._tb_encoders.append(
                    self._sys_parameters.transmitters[mcs_list_idx]._tb_encoder)

                self._tb_decoders.append(
                    TBDecoder(self._tb_encoders[mcs_list_idx],
                              num_bp_iter=sys_parameters.num_bp_iter,
                              cn_type=sys_parameters.cn_type))

        # Precoding matrix to post-process the ground-truth channel when
        # training
        #  [num_tx, num_tx_ant, num_layers = 1]
        if hasattr(sys_parameters.transmitters[0], "_precoder"):
            self._precoding_mat = sys_parameters.transmitters[0]._precoder._w
        else:
            self._precoding_mat = tf.ones([sys_parameters.max_num_tx,
                                           sys_parameters.num_antenna_ports, 1], tf.complex64)

        # LS channel estimator
        # rg independent of MCS index
        rg = sys_parameters.transmitters[0]._resource_grid
        # get pc from first MCS and first Tx
        pc =  sys_parameters.pusch_configs[0][0]
        self._ls_est = PUSCHLSChannelEstimator(
                resource_grid=rg,
                dmrs_length=pc.dmrs.length,
                dmrs_additional_position=pc.dmrs.additional_position,
                num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
                interpolation_type="nn")

        # rg_type[num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        # rg_type[:,0][num_tx, num_ofdm_symbols, fft_size]
        rg_type = rg.build_type_grid()[:,0] # One stream only
        # [Num all pilots, [tx ind, ofdm_symbol ind, fft ind] ] it has 2 dims
        pilot_ind = tf.where(rg_type==1)
        self._pilot_ind = np.array(pilot_ind)

        # required to remove layers
        self._layer_demappers = []
        for mcs_list_idx in range(self._num_mcss_supported):
                self._layer_demappers.append(
                    LayerDemapper(
                            self._sys_parameters.transmitters[mcs_list_idx]._layer_mapper,
                            sys_parameters.transmitters[mcs_list_idx]._num_bits_per_symbol))

        self._neural_rx = CGNNOFDM(
                    sys_parameters,
                    max_num_tx=sys_parameters.max_num_tx,
                    training=training,
                    num_it=sys_parameters.num_nrx_iter,
                    d_s=sys_parameters.d_s,
                    num_units_init=sys_parameters.num_units_init,
                    num_units_agg=sys_parameters.num_units_agg,
                    num_units_state=sys_parameters.num_units_state,
                    num_units_readout=sys_parameters.num_units_readout,
                    layer_demappers=self._layer_demappers,
                    layer_type_dense=sys_parameters.layer_type_dense,
                    layer_type_conv=sys_parameters.layer_type_conv,
                    layer_type_readout=sys_parameters.layer_type_readout,
                    dtype=sys_parameters.nrx_dtype)

    def estimate_channel(self, y, num_tx,no):

        # y has shape
        #[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_subcarriers]


        if self._sys_parameters.initial_chest == 'ls':
            if self._sys_parameters.mask_pilots:
                raise ValueError("Cannot use initial channel estimator if " \
                                "pilots are masked.")
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #    num_ofdm_symbols, num_effective_subcarriers]
            # Dummy value for N0 as it is not used anyway.
            h_hat, err_var = self._ls_est([y, no])

            # [batch_size, num_rx, num_ofdm_symbols,...
            #  ..., num_effective_subcarriers, num_rx_ant, num_tx*num_streams]
            err_var_dt = tf.broadcast_to(err_var, tf.shape(h_hat))
            err_var_dt = tf.transpose(err_var_dt, [0, 1, 5, 6, 2, 3, 4])
            err_var_dt = flatten_last_dims(err_var_dt, 2)

            # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant]
            err_var = tf.reduce_sum(err_var_dt, -1)


            # Reshaping to the expected shape
            # [batch_size, num_tx, num_effective_subcarriers,
            #       num_ofdm_symbols, 2*num_rx_ant]
            h_hat = h_hat[:,0,:,:num_tx,0]
            h_hat = tf.transpose(h_hat, [0, 2, 4, 3, 1])
            h_hat = tf.concat([tf.math.real(h_hat), tf.math.imag(h_hat)],
                              axis=-1)

            # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant]
            # [batch_size, num_rx, num_effective_subcarriers, num_ofdm_symbols, num_rx_ant]
            err_var = tf.transpose(err_var,perm=[0,1,3,2,4])


        elif self._sys_parameters.initial_chest == None:
            h_hat = None
        return h_hat, err_var

    def preprocess_channel_ground_truth(self, h):
        # h : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_effective_subcarriers]

        # Assume only one rx
        # [batch_size, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        h = tf.squeeze(h, axis=1)

        # Reshape h
        # [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols, num_rx_ant, num_tx_ant]
        h = tf.transpose(h, perm=[0,2,5,4,1,3])



        # Multiply by precoding matrices to compute effective channels
        # [1, num_tx, 1, 1, num_tx_ant, 1]
        w = insert_dims(tf.expand_dims(self._precoding_mat, axis=0), 2, 2)
        # [batch size, num_tx, num_effective_subcarriers, num_ofdm_symbols, num_rx_ant]
        h = tf.squeeze(tf.matmul(h, w), axis=-1)



        # Complex-to-real
        # [batch size, num_tx, num_effective_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        h = tf.concat([tf.math.real(h), tf.math.imag(h)], axis=-1)
        return h

    def preprocess_x_ground_truth(self, x):
        # x: [batch_size, max_num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        # transmitters is a list of PUSCHTransmitters, one for each MCS
        # self._precoding_mat [num_tx, num_tx_ant, 1]


        # reshape x
        # [batch_size, max_num_tx, num_ofdm_symbols, fft_size, num_tx_ant, 1]
        x = tf.transpose(x,perm=[0,1,3,4,2])
        x = tf.expand_dims(x,axis=-1)

        w = self._precoding_mat

        # Invert Precoding Matrix
        w = 1/ (2 * self._precoding_mat)

        # reshape precoding matrix
        # [num_tx, num_tx_ant, 1]
        # to [1, num_tx,1,1, num_tx_ant, 1]
        w = insert_dims(tf.expand_dims(w, axis=0), 2, 2)

        # [batch_size, max_num_tx, num_ofdm_symbols, fft_size, 1, 1]
        x = tf.matmul(w,x,transpose_a=True)
        
        # [batch_size, max_num_tx, num_ofdm_symbols, fft_size]
        x = tf.squeeze(x, axis=[4,5])

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,2]     
        x = tf.transpose(x,perm=[0,1,3,2])
        x = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)

        return x








    def call(self, inputs, mcs_arr_eval=[0], mcs_ue_mask_eval=None, h_ls=None):


        # assume u is provided as input in training mode
        if self._training:
            y, active_tx, b, h, mcs_ue_mask, x, no, batch_size  = inputs
            # re-encode bits in training mode to generate labels
            # avoids the need for post-FEC bits as labels
            if len(mcs_arr_eval)==1 and not isinstance(b, list):
                b = [b] # generate new list if b is not provided as list
            if self._sys_parameters.channel_type != "Datalake":
                bits = []
                for idx in range(len(mcs_arr_eval)):
                    bits.append(
                        self._sys_parameters.transmitters[mcs_arr_eval[idx]]._tb_encoder(b[idx]))
            else:
                bits = b # assume b are the encoded bits already

            num_tx = tf.shape(active_tx)[1]
            h_hat, err_var = self.estimate_channel(y, num_tx, no)

            # overwrite h_hat by DataLake LS channel estimate
            if h_ls is not None:
                h_hat = h_ls # use provided LS estimate

            # Reshaping `h` to the expected shape and apply precoding matrices
            # [batch size, num_tx, num_effective_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
            if h is not None:
                h = self.preprocess_channel_ground_truth(h)

            # Reshape x from [batch_size, max_num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
            # to [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,2] real 
            if x is not None:
                x = self.preprocess_x_ground_truth(x)


            # Apply neural receiver and return loss
            losses = self._neural_rx((y, h_hat, active_tx,
                                      bits, h, mcs_ue_mask, x, no,err_var, batch_size),
                                      mcs_arr_eval)
            return losses

        else:
            y, active_tx, no, mcs_ue_mask, x, h, batch_size = inputs

            # Initial channel estimation
            num_tx = tf.shape(active_tx)[1]
            h_hat, err_var = self.estimate_channel(y, num_tx, no)

            # overwrite h_hat by DataLake LS channel estimate
            if h_ls is not None:
                h_hat = h_ls # use provided LS estimate

            # Reshape x from [batch_size, max_num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
            # to [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,2] real 
            if x is not None:
                x = self.preprocess_x_ground_truth(x)

            # Reshaping `h` to the expected shape and apply precoding matrices
            # [batch size, num_tx, num_effective_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
            if h is not None:
                h = self.preprocess_channel_ground_truth(h)

            llr, h_hat_refined = self._neural_rx(
                                            (y, h_hat, active_tx, no,x,h, err_var, batch_size),
                                            [mcs_arr_eval[0]],
                                            mcs_ue_mask_eval=mcs_ue_mask_eval, no=no)

            # apply TBDecoding
            b_hat, tb_crc_status = self._tb_decoders[mcs_arr_eval[0]](llr)

            return b_hat, h_hat_refined, h_hat, tb_crc_status
        



################################
## ONNX Layers / Wrapper
################################
# The following layers provide an adapter to the Aerial PUSCH pipeline
# the code is only relevant for for ONNX/TensorRT exports but can be ignored
# for Sionna-based simulations.

class NRPreprocessing(Layer):
    # pylint: disable=line-too-long
    r"""
    Pre-preprocessing layer for the neural receiver applying initial channel
    estimation. In particular, the layer takes the channel estimates at pilot
    positions and performs nearest neighbor interpolation on a "per PRB" basis.
    As such it scales to arbitrary resource grid sizes.

    The input/output shapes are Aerial compatible and not directly compatible
    with Sionna. The returned "resourcegrid of LLRs" can be further processed
    using PyAerial.

    Note that all operations are real-valued.

    Parameters
    -----------
    num_tx : int
        Number of transmitters (i.e., independent streams)

    Input
    ------
    (y, h_hat, dmrs_ofdm_pos, dmrs_subcarrier_pos)
    Tuple:

    y : [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant], tf.float32
        The received OFDM resource grid after cyclic prefix removal and FFT.
        Real and imaginary part are stacked in the rx_antenna direction.

    h_hat : [bs, num_pilots, num_streams, 2*num_rx_ant], tf.float32
        Channel estimates at pilot positions. Real and imaginary part are stacked in the rx_antenna direction.

    dmrs_ofdm_pos: [num_tx, num_dmrs_symbols], tf.int
        DMRS symbol positions within slot.

    dmrs_subcarrier_pos: [num_tx, num_pilots_per_PRB], tf.int
        Pilot position per PRB.

    Output
    -------
    [h_hat, pe]: Tuple

    h_hat : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
             2*num_rx_ant], tf.float
        Channel estimate after nearest neighbor interpolation.

    pe : [num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
        Map showing the position of the nearest pilot for every user in time
        and frequency. This can be seen as a form of positional encoding.
    """

    def __init__(self,
                 num_tx,
                 **kwargs):

        super().__init__(**kwargs)

        self._num_tx = num_tx
        self._num_res_per_prb = 12 # fixed in 5G

    def _focc_removal(self, h_hat):
        """
        Apply FOCC removal to h_hat.

        Parameters
        ----------
        h_hat: [bs, 2*num_rx_ants, num_layers, num_pilots], tf.complex64
            Channel estimates at pilot positions.

        Outputs
        -------
        h_hat: [bs, 2*num_rx_ants, num_layers, num_pilots], tf.complex64
            Channel estimates at pilot positions after FOCC removal.
        """

        shape = [-1, 2]
        s = tf.shape(h_hat)
        new_shape = tf.concat([s[:3], shape], 0)
        # [bs, num_rx_ants, num_tx, num_pilots//2, 2]
        h_hat = tf.reshape(h_hat, new_shape)

        # [bs, num_rx_ants, num_tx, num_pilots//2, 1]
        h_hat = tf.reduce_sum(h_hat, axis=-1, keepdims=True) \
                                    / tf.cast(2., dtype=h_hat.dtype)
        # [bs, num_rx_ants, num_tx, num_pilots//2, 2]

        # we split into 2 as 4 was only required if we feed the zeroed/masked
        # pilots as well. This is not the case in the Aerial format
        # (only non-zero pilots are returned)
        h_hat = tf.repeat(h_hat, 2, axis=-1)

        # [bs, num_rx_ants, num_tx, num_pilots]
        shape = [-1]
        s = tf.shape(h_hat)
        new_shape = tf.concat([s[:3], shape], 0)
        h_ls = tf.reshape(h_hat, new_shape)

        return h_ls

    def _calculate_nn_indices(self, dmrs_ofdm_pos, dmrs_subcarrier_pos,
                              num_ofdm_symbols, num_prbs):
        """
        Calculates nearest neighbor interpolation indices for a single PRB.

        Parameters
        ----------
        dmrs_ofdm_pos: [num_tx, num_dmrs_symbols], tf.int
            DMRS symbol position within slot.

        dmrs_subcarrier_pos: [num_tx, num_pilots_per_PRB], tf.int
            Pilot position per PRB.

        num_ofdm_symbol: tf.int
            Number of OFDM symbols per slot.

        num_prbs: tf.int
            Number of allocated PRBs.

        Output
        ------
        nn_idx : [num_tx, num_streams_per_tx, num_ofdm_symbols,
                  num_effective_subcarriers=12], tf.int32
            Indices of nearest pilot RG grid. Can be used for nearest neighbor
            interpolation.

        pe : [num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
            Map showing the position of the nearest pilot for every user in time
            and frequency. This can be seen as a form of positional encoding.
        """

        re_pos = tf.meshgrid(tf.range(self._num_res_per_prb),
                             tf.range(num_ofdm_symbols))
        re_pos = tf.stack(re_pos, axis=-1)
        # enable broadcasting for distance calculation
        re_pos = tf.reshape(re_pos, (-1,1,2))

        pes = []
        nn_idxs = []
        # calculate distance per tx
        for tx_idx in range(self._num_tx):
            # Combining the coordinates into a single matrix
            p_idx= tf.meshgrid(dmrs_subcarrier_pos[tx_idx],
                               dmrs_ofdm_pos[tx_idx])
            pilot_pos = tf.stack(p_idx, axis=-1)
            # Reshaping to get a list of coordinate pairs
            pilot_pos = tf.reshape(pilot_pos, (-1, 2))

            pilot_pos = tf.reshape(pilot_pos, (1,-1,2))
            diff = tf.abs(re_pos - pilot_pos)
            # Manhattan distance
            dist = tf.reduce_sum(diff, axis=-1)

            # find indices of closes pilot
            nn_idx = tf.argmin(dist, axis=1)

            # and bring into shape [num_tx, num_streams_per_tx,...
            # ... num_ofdm_symbols,num_effective_subcarriers]
            nn_idx = tf.reshape(nn_idx,
                                (1, 1, num_ofdm_symbols, self._num_res_per_prb))

            # [num_tx, num_subcarriers, num_ofdm_symbols, 2],
            pe = tf.reduce_min(diff, axis=1)
            pe = tf.reshape(pe,
                            (1, num_ofdm_symbols, self._num_res_per_prb, 2))
            pe = tf.transpose(pe, (0,2,1,3))

            # normalize per axis(t and f)
            # remark leads to effect that in f dim, we have +/-1
            p = []
            pe = tf.cast(pe, tf.float32)

            # ofdm symbol axis
            pe_ = pe[...,1:2]
            pe_ -= tf.reduce_mean(pe_)#,axis=2, keepdims=True)
            std_ = tf.math.reduce_std(pe_)#,axis=2, keepdims=True)
            pe_ = tf.where(std_>0., pe_/std_, pe_)
            p.append(pe_)

            # subcarrier axis
            pe_ = pe[...,0:1]
            pe_ -= tf.reduce_mean(pe_)#,axis=1, keepdims=True)
            std_ = tf.math.reduce_std(pe_)#,axis=1, keepdims=True)
            pe_ = tf.where(std_>0., pe_/std_, pe_)
            p.append(pe_)

            pe = tf.concat(p ,axis=-1)

            pes.append(pe)
            nn_idxs.append(nn_idx)

        pe = tf.concat(pes, axis=0)
        pe = tf.tile(pe, (1, num_prbs, 1, 1)) # broadcasting over all PRBs
        nn_idx = tf.concat(nn_idxs, axis=0)
        nn_idx = tf.concat(nn_idxs, axis=0)
        return nn_idx, pe

    def _nn_interpolation(self, h_hat, num_ofdm_symbols,dmrs_ofdm_pos,
                          dmrs_subcarrier_pos):
        """
        Applies nearest neighbor interpolation of pilots to all data
        symbols in the resource grid.

        Remark: NN interpolation is done per PRB.

        Parameters
        ----------
        h_hat: [bs, 2*num_rx_ants, num_layers, num_pilots], tf.complex
            Channel estimates at pilot locations

        num_ofdm_symbols: tf.int
            Total number of OFDM symbols per slot.

        dmrs_ofdm_pos: [num_tx, num_dmrs_symbols], tf.int
            DMRS symbol position within slot.

        dmrs_subcarrier_pos: [num_tx, num_pilots_per_PRB], tf.int
            Pilot position per PRB.

        Output
        ------
        h_hat: [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols,
                num_effective_subcarriers], tf.complex
            Interpolated channel estimates of the entire resource grid.

        pe : [num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
            Map showing the position of the nearest pilot for every user in time
            and frequency. This can be seen as a form of positional encoding.
        """
        # derive shapes from h_hat (and not y) as TRT has issues otherwise with
        # dynamic input shapes
        num_pilots_per_dmrs = tf.shape(dmrs_subcarrier_pos)[1] # pilot symbols
        num_prbs = tf.cast(tf.shape(h_hat)[-1]
                        / (num_pilots_per_dmrs * tf.shape(dmrs_ofdm_pos)[-1]),
                           tf.int32)

        # permute input estimates such that the first half of pilots of the
        # first PRB follows its second half (or third if multiple dmrs symbols)
        s = tf.shape(h_hat)
        h_hat = split_dim(h_hat, shape=(-1, num_pilots_per_dmrs), axis=3)
        h_hat = split_dim(h_hat, shape=(-1, num_prbs), axis=3)
        h_hat = tf.transpose(h_hat, (0,1,2,4,3,5))
        # flatten dims does not work with ONNX (partially unknown shapes)
        # h_hat = flatten_dims(h_hat, 3, axis=3)
        h_hat = tf.reshape(h_hat, s)

        # bring into Sionna compatible shape
        h_hat = tf.expand_dims(h_hat, axis=1) # num_rx
        h_hat = tf.expand_dims(h_hat, axis=4) # num_streams_per_tx
        perm = tf.roll(tf.range(tf.rank(h_hat)), -3, 0)
        h_hat = tf.transpose(h_hat, perm)

        # compute indices on the fly
        ls_nn_ind, pe = self._calculate_nn_indices(dmrs_ofdm_pos,
                                                   dmrs_subcarrier_pos,
                                                   num_ofdm_symbols,
                                                   num_prbs)

        # reshape such that interpolation is applied to each PRB individually
        s = tf.shape(h_hat)
        h_hat_prb = split_dim(h_hat, shape=(num_prbs, -1), axis=2)
        h_hat_prb = tf.transpose(h_hat_prb, (0,1,3,2,4,5,6))
        # apply nn interpolation
        outputs = tf.gather(h_hat_prb, ls_nn_ind, 2, batch_dims=2)
        outputs = tf.transpose(outputs, (0,1,2,4,3,5,6,7))

        # and remove artificial "per-prb" dimension
        s = tf.shape(outputs)
        s = tf.concat((tf.constant((-1,), tf.int32), s[1:3],
                       tf.expand_dims(num_prbs*self._num_res_per_prb, axis=0),
                       s[5:]), axis=0)
        # and combine all PRBs
        outputs = tf.reshape(outputs, s)

        # Transpose outputs to bring batch_dims first again. New shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols,
        #  num_effective_subcarriers]
        perm = tf.roll(tf.range(tf.rank(outputs)), 3, 0)
        h_hat = tf.transpose(outputs, perm)
        return h_hat, pe

    def call(self, inputs):

        y, h_hat_ls, dmrs_ofdm_pos, dmrs_subcarrier_pos = inputs

        num_ofdm_symbols = tf.shape(y)[2]

        # shape y [bs, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        # shape h_hat_ls [bs, num_pilots, num_layers, 2*num_rx_ants]
        # desired shape of h_hat_ls [bs, 2*num_rx_ants, num_layers, num_pilots]
        h_hat_ls = tf.transpose(h_hat_ls, (0,3,2,1))

        ######################
        ### FOCC removal
        ######################

        h_hat_ls = self._focc_removal(h_hat_ls)

        ######################
        ### NN interpolator
        ######################
        # Interpolate channel estimates over the RG
        h_hat, pe = self._nn_interpolation(h_hat_ls,
                                           num_ofdm_symbols,
                                           dmrs_ofdm_pos,
                                           dmrs_subcarrier_pos)

        # Reshaping to the expected shape
        # [batch_size, num_tx, num_effective_subcarriers,
        #       num_ofdm_symbols, 2*num_rx_ant]
        h_hat = h_hat[:,0,:,:self._num_tx,0]
        h_hat = tf.transpose(h_hat, [0, 2, 4, 3, 1])
        return [h_hat, pe]

class MDNeuralReceiverONNX(Model):
    # pylint: disable=line-too-long
    r"""
    Wraps the 5G NR neural receiver in an ONNX compatible format.

    Note that the shapes are optimized for Aerial and not directly compatible
    with Sionna.

    Parameters
    -----------
    num_it : int
        Number of iterations.

    d_s : int
        Size of the state vector.

    num_units_init : list of int
        Number of hidden units for the init network.

    num_units_agg : list of int
        Number of kernel for the hidden dense layers of the aggregation network

    num_units_state : list of int
        Number of hidden units for the state-update network.

    num_units_readout : list of int
        Number of hidden units for the read-out network.

    num_bits_per_symbol: int
        Number of bits per symbol.

    num_tx: int
        Max. number of layers/DMRS ports. Note that DMRS ports can be
        dynamically deactivated via the dmrs_port_mask.

    Input
    ------
    (rx_slot, h_hat, dmrs_port_mask)
    Tuple:

    rx_slot_real : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant],
                    tf.float32
        Real part of the received OFDM resource grid after cyclic prefix
        removal and FFT.

    rx_slot_imag : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant],
                    tf.float32
        Imaginary part of the received OFDM resource grid after cyclic prefix
        removal and FFT.

    h_hat_real : [batch_size, num_pilots, num_streams, num_rx_ant], tf.float32
        Real part of the LS channel estimates at pilot positions.

    h_hat_imag : [batch_size, num_pilots, num_streams, num_rx_ant], tf.float32
        Imaginary part of the LS channel estimates at pilot positions.

    dmrs_port_mask: [bs, num_tx], tf.float32
        Mask of 0s and 1s to indicate that DMRS ports are active or not.

    dmrs_ofdm_pos: [num_tx, num_dmrs_symbols], tf.int
        DMRS symbol positions within slot.

    dmrs_subcarrier_pos: [num_tx, num_pilots_per_prb], tf.int
        Pilot positions per PRB.

    no : scalar, tf.float32, optional
        Noise variance per complex sample. If omitted, ``datalake_no`` from
        construction is used.

    Output
    -------
    (llr, h_hat)
    Tuple:

    llr : [batch_size, num_bits_per_symbol, num_tx, num_effective_subcarriers,
          num_ofdm_symbols], tf.float
        LLRs on bits.

    h_hat : [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols,
             2*num_rx_ant], tf.float
        Refined channel estimates.
    """

    def __init__(self,
                 num_it,
                 d_s,
                 num_units_init,
                 num_units_agg,
                 num_units_state ,
                 num_units_readout,
                 num_bits_per_symbol,
                 layer_type_dense,
                 layer_type_conv,
                 layer_type_readout,
                 nrx_dtype,
                 num_tx,
                 num_rx_ant,
                 pilot_pattern,
                 arch,
                 pe_type,
                 pe_d,
                 demapping_type,
                 datalake_no,
                 **kwargs):

        super().__init__(**kwargs)
        # assert len(num_units_agg) == num_it and len(num_units_state) == num_it

        # hard-coded for simplicity
        self._num_tx = num_tx # we assume 1 stream per user

        self._datalake_no = datalake_no

        ####################################################
        # Detector
        ####################################################
        # arch = self._sys_parameters.arch
        # pilot_pattern = self._rg.pilot_pattern
        num_pilot_symbols = pilot_pattern.num_pilot_symbols
        # [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], int32/bool
        # 0's and 1's; 1's for pilot locations
        pilot_mask = pilot_pattern.mask


        # [num_tx,num_effective_subcarriers, num_ofdm_symbols]
        # 0s for pilot locations, 
        pilot_mask = 1 - pilot_mask[:,0]
        pilot_mask = tf.transpose(pilot_mask, perm=[0,2,1])
        # [1,num_tx,num_effective_subcarriers, num_ofdm_symbols,1]
        pilot_mask = insert_dims(pilot_mask,num_dims=1,axis=0)
        pilot_mask = insert_dims(pilot_mask,num_dims=1,axis=-1)
        pilot_mask = tf.cast(pilot_mask,self._dtype)

        assert pe_type==0, "PE only with NRX default"

        self._cgnn = CGNN([num_bits_per_symbol], # no support for mixed MCS
                          num_rx_ant,
                          num_it,
                          arch,
                          d_s,
                          num_units_init,
                          num_units_agg,
                          num_units_state,
                          num_units_readout,
                          pilot_mask=pilot_mask, 
                          constellation=None,
                          demapping_type=demapping_type,
                          max_num_tx=num_tx,
                          layer_type_dense=layer_type_dense,
                          layer_type_conv=layer_type_conv,
                          layer_type_readout=layer_type_readout,
                          pe_d=pe_d,
                          pe_type=pe_type,
                          dtype=nrx_dtype)

        self._preprocessing = NRPreprocessing(self._num_tx)

    @property
    def num_it(self):
        return self._num_it

    @num_it.setter
    def num_it(self, val):
        assert (val >= 1) and (val <= len(self._iterations)),\
            "Invalid number of iterations"
        self._num_it = val

    def call(self, inputs):

        if len(inputs) == 8:
            y_real, y_imag, h_hat_real, h_hat_imag, \
                dmrs_port_mask, dmrs_ofdm_pos, dmrs_subcarrier_pos, no = inputs
        else:
            y_real, y_imag, h_hat_real, h_hat_imag, \
                dmrs_port_mask, dmrs_ofdm_pos, dmrs_subcarrier_pos = inputs
            no = self._datalake_no
        no = tf.cast(no, tf.float32)

        y = tf.concat((y_real, y_imag), axis=-1)
        h_hat_p = tf.concat((h_hat_real, h_hat_imag), axis=-1)

        # nearest neighbor interpolation of channel estimates
        h_hat, pe = self._preprocessing((y,
                                         h_hat_p,
                                         dmrs_ofdm_pos,
                                         dmrs_subcarrier_pos))

        # dummy MCS mask (no support for mixed MCS)
        mcs_ue_mask = tf.ones((1,self._num_tx,1), tf.float32)

        # add MDX specific dummy inputs
        x = None
        # effective scaling factor for selected pilot pattern

        # must be [batch_size, num_rx, num_effective_subcarriers, num_ofdm_symbols, num_rx_ant]
        batch_size = tf.shape(y)[0]
        num_effective_subcarriers = tf.shape(y)[1]
        num_ofdm_symbols = tf.shape(y)[2]
        num_rx_ant = tf.shape(y_real)[3]
        h_err_var = 0.25*no * tf.ones(shape=[batch_size, 1, num_effective_subcarriers, num_ofdm_symbols, num_rx_ant])
        h = None

        # and run NRX
        llr, h_hat, _, _, _ = self._cgnn([y, pe, h_hat, dmrs_port_mask, mcs_ue_mask, no, x, h, h_err_var, batch_size])

        # cgnn returns list of results for each iteration
        # (not needed for inference)
        llr = llr[-1][0] # take LLRs of first MCS (no support for mixed MCS)
        h_hat = h_hat[-1]

        # cast back to tf.float32 (if NRX uses quantization)
        llr = tf.cast(llr, tf.float32)
        h_hat = tf.cast(h_hat, tf.float32)

        # reshape llrs in Aerial format
        llr = tf.transpose(llr, (0,4,1,2,3))
        # Sionna defines LLRs with different sign
        llr = -1. * llr

        return llr, h_hat
