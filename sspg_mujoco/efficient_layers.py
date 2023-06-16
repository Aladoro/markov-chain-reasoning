import tensorflow as tf
import numpy as np


def run_layers(layers, inputs, **kwargs):
    out = inputs
    for layer in layers:
        out = layer(out, **kwargs)
    return out


tfl = tf.keras.layers
tfm = tf.keras.models
LN4 = np.log(4)


class BaseCritics(tfl.Layer):
    def __init__(self):
        super(BaseCritics, self).__init__()

    def call(self, inputs):
        raise NotImplementedError

    def get_qs(self, observation_batch, action_batch, permutation=None):
        raise NotImplementedError

    def preprocess_obs(self, observation_batch):
        if self._norm_mean is not None and self._norm_stddev is not None:
            return (observation_batch - self.norm_mean) / (self.norm_stddev + 1e-7)
        return observation_batch


class Critics(BaseCritics):
    def __init__(self, layers_lists, norm_mean=None, norm_stddev=None):
        super(Critics, self).__init__()
        self._cri_layers_lists = layers_lists
        self._norm_mean = norm_mean
        self._norm_stddev = norm_stddev
        self._num_shared_critics = layers_lists[0][-1].units
        self._num_critics = len(layers_lists) * self._num_shared_critics
        if self._num_shared_critics > 1:
            self.requires_permute = True
        else:
            self.requires_permute = False

    def call(self, inputs):
        out_list = []
        for cri_layers in self._cri_layers_lists:
            out = tf.identity(inputs)
            for layer in cri_layers:
                out = layer(out)
            out_list.append(out)
        return tf.concat(out_list, axis=-1)

    def build(self, input_shape):
        self._input_dims = input_shape[-1]
        self._permutation_reshape_dims = (self._input_dims // self._num_shared_critics) * (self._num_shared_critics - 1)
        super(Critics, self).build(input_shape)

    def get_qs(self, observation_batch, action_batch, permutation=None):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        input_batch = tf.concat([pre_obs_batch, action_batch], axis=-1)
        if self._num_shared_critics > 1:
            if permutation is not None:
                input_batch_permutations = tf.gather(input_batch, indices=permutation, axis=0)
                input_batch_permutations = tf.reshape(input_batch_permutations, [-1, self._permutation_reshape_dims])
                full_input_batch = tf.concat([input_batch, input_batch_permutations], axis=-1)
            else:
                full_input_batch = tf.tile(input_batch, multiples=[1, self._num_shared_critics])
            qs = self.__call__(full_input_batch)
        else:
            qs = self.__call__(input_batch)
        return qs


class ParallelLayer(tfl.Layer):
    def __init__(self, n_parallel):
        super(ParallelLayer, self).__init__()
        self.n_parallel = n_parallel

    def build(self, input_shape):
        raise NotImplementedError

    @tf.function
    def call(self, inputs):
        raise NotImplementedError


class DenseParallel(ParallelLayer):
    def __init__(self, units, n_parallel, activation=None, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None):
        super(DenseParallel, self).__init__(n_parallel=n_parallel)
        self.units = units

        if isinstance(activation, str):
            self.activation = tfl.Activation(activation)
        else:
            self.activation = activation

        self._kernel_init = kernel_initializer
        self._kernel_reg = kernel_regularizer

        self._bias_init = bias_initializer
        self._bias_reg = bias_regularizer

    def build(self, input_shape):
        self.kernels = self.add_weight(shape=[self.n_parallel, input_shape[-1], self.units],
                                       initializer=self._kernel_init, regularizer=self._kernel_reg,
                                       name='kernels')
        self.biases = self.add_weight(shape=[self.n_parallel, 1, self.units], initializer=self._bias_init,
                                      regularizer=self._bias_reg, name='biases')

    @tf.function
    def call(self, inputs):
        o = tf.linalg.matmul(inputs, self.kernels) + self.biases
        if self.activation is not None:
            o = self.activation(o)
        return o


class ParallelLayerNormalization(tfl.LayerNormalization):
    def __init__(self,
                 axis=-1,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(ParallelLayerNormalization, self).__init__(
            axis=axis,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            **kwargs)
    def build(self, input_shape):
        super(ParallelLayerNormalization, self).build(input_shape)
        assert len(input_shape) == 3
        assert self.axis == [2]
        # Override
        param_shape = [input_shape[0], 1, input_shape[2]]
        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False,)
        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)


class EfficientParallelCritics(BaseCritics):
    def __init__(self, layers_list, norm_mean=None, norm_stddev=None):
        super(EfficientParallelCritics, self).__init__()
        self._cri_layers_list = layers_list
        self._norm_mean = norm_mean
        self._norm_stddev = norm_stddev
        self._num_shared_critics = 1
        self._num_critics = self._cri_layers_list[0].n_parallel
        self.requires_permute = False

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        batch_dims = inputs_shape[:-1] # save batch dimensions of input
        inputs_reshaped = tf.reshape(inputs, [tf.reduce_prod(batch_dims), inputs_shape[-1]])
        out = run_layers(layers=self._cri_layers_list, inputs=inputs_reshaped)
        out = tf.transpose(out, [1, 0, 2])
        return tf.reshape(out, tf.concat([batch_dims, [self._num_critics]], axis=0)) # reshape output

    def build(self, input_shape):
        self._input_dims = input_shape[-1]
        super(EfficientParallelCritics, self).build(input_shape)

    def get_qs(self, observation_batch, action_batch, permutation=None):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        input_batch = tf.concat([pre_obs_batch, action_batch], axis=-1)
        return self.__call__(input_batch)


class EfficientQuantileParallelCritics(BaseCritics):
    def __init__(self, layers_list, norm_mean=None, norm_stddev=None):
        super(EfficientQuantileParallelCritics, self).__init__()
        self._cri_layers_list = layers_list
        self._norm_mean = norm_mean
        self._norm_stddev = norm_stddev
        self._num_shared_critics = 1
        self._num_critics = self._cri_layers_list[0].n_parallel
        self._num_quantiles = self._cri_layers_list[-1].units
        self.requires_permute = False

    def call(self, inputs):
        out = run_layers(layers=self._cri_layers_list, inputs=inputs)
        out = tf.transpose(out, [1, 0, 2])
        return tf.reshape(out, [-1, self._num_critics, self._num_quantiles])

    def build(self, input_shape):
        self._input_dims = input_shape[-1]
        super(EfficientQuantileParallelCritics, self).build(input_shape)

    def get_qs(self, observation_batch, action_batch, permutation=None):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        input_batch = tf.concat([pre_obs_batch, action_batch], axis=-1)
        return self.__call__(input_batch)
