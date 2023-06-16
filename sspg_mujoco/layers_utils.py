import numpy as np
import tensorflow as tf

from efficient_layers import DenseParallel, ParallelLayer, ParallelLayerNormalization

tfl = tf.keras.layers
tfi = tf.keras.initializers


class ParallelOrthogonal(tfi.Initializer):
    def __init__(self, gain=1.0, seed=None):
        self.orthogonal_init = tfi.Orthogonal(gain=gain, seed=seed)

    def __call__(self, shape, dtype=None):
        weight = np.zeros(shape)
        for i in range(shape[0]):
            weight[i] = self.orthogonal_init(shape=shape[1:], dtype=dtype)
        return tf.cast(weight, dtype=dtype)


def run_layers(layers, inputs, **kwargs):
    out = inputs
    for layer in layers:
        out = layer(out, **kwargs)
    return out


class LayersWrapper(tfl.Layer):
    def __init__(self, layers):
        super(LayersWrapper, self).__init__()
        self._wrapper_layers = layers

    def call(self, inputs):
        return run_layers(self._wrapper_layers, inputs)


class SpectralNormalization(tfl.Wrapper):

    def __init__(self, layer, power_iterations=1, eps=1e-12):
        assert isinstance(layer, tf.keras.layers.Layer)
        if isinstance(layer, tfl.Dense):
            self.units = layer.units

        if isinstance(layer, ParallelLayer):
            self.n_parallel = layer.n_parallel
            if isinstance(layer, DenseParallel):
                self.units = layer.units
            self.pi_fn = self.parallel_power_iteration
        else:
            self.pi_fn = self.power_iteration
        self.power_iterations = power_iterations
        self._eps = eps
        super(SpectralNormalization, self).__init__(layer)

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        if isinstance(self.layer, ParallelLayer):
            self.kernels_shape = tf.shape(self.layer.kernels)
            self.u = self.add_weight(shape=[self.n_parallel, 1, self.kernels_shape[-1]],
                                     initializer=tf.keras.initializers.RandomNormal(),
                                     trainable=False, name='u_vec')
        else:
            self.kernel_shape = tf.shape(self.layer.kernel)
            self.u = self.add_weight(shape=[1, self.kernel_shape[-1]],
                                     initializer=tf.keras.initializers.RandomNormal(),
                                     trainable=False, name='u_vec')
        self.built = True

    def call(self, inputs):
        self.pi_fn(self.power_iterations)
        return self.layer(inputs)

    def power_iteration(self, iterations):
        reshaped_kernel = tf.reshape(self.layer.kernel, [-1, self.kernel_shape[-1]])  # k_dims x o_dims
        u = tf.identity(self.u)  # 1 x o_dims
        for _ in range(iterations):
            v = tf.matmul(u, tf.transpose(reshaped_kernel))  # 1 x k_dims
            v = tf.nn.l2_normalize(v, epsilon=self._eps)  # 1 x k_dims
            u = tf.matmul(v, reshaped_kernel)  # 1 x o_dims
            u = tf.nn.l2_normalize(u, epsilon=self._eps)  # 1 x o_dims
        u, v = tf.stop_gradient(u), tf.stop_gradient(v)
        self.u.assign(u)
        norm_value = tf.matmul(tf.matmul(v, reshaped_kernel), tf.transpose(u))  # 1 x 1
        self.layer.kernel.assign(self.layer.kernel / norm_value)

    def parallel_power_iteration(self, iterations):
        reshaped_kernel = tf.reshape(self.layer.kernels,
                                     [self.n_parallel, -1, self.kernels_shape[-1]])  # n_par x k_dims x o_dims
        u = tf.identity(self.u)  # n_par x 1 x o_dims
        for _ in range(iterations):
            v = tf.matmul(u, tf.transpose(reshaped_kernel, [0, 2, 1]))  # n_par x 1 x k_dims
            v = tf.nn.l2_normalize(v, epsilon=self._eps, axis=-1)
            u = tf.matmul(v, reshaped_kernel)  # n_par x 1 x o_dims
            u = tf.nn.l2_normalize(u, epsilon=self._eps, axis=-1)  # n_par x 1 x o_dims
        u, v = tf.stop_gradient(u), tf.stop_gradient(v)
        self.u.assign(u)
        norm_value = tf.matmul(tf.matmul(v, reshaped_kernel), tf.transpose(u, [0, 2, 1]))  # n_par x 1 x 1
        self.layer.kernels.assign(self.layer.kernels / norm_value)


class ModernResidualBlock(tfl.Layer):
    def __init__(self, out_dims, bottleneck_dims, kernel_initializer=None,
                 layer_normalization=True, spectral_normalization=True,
                 parallel_models=None):
        super(ModernResidualBlock, self).__init__()
        self._od = out_dims
        self._bd = bottleneck_dims
        self._ki = kernel_initializer
        self._ln = layer_normalization
        self._sn = spectral_normalization
        self._pm = parallel_models

    def build(self, input_shape):
        super(ModernResidualBlock, self).build(input_shape)

        def make_norm():
            if self._pm is not None:
                return ParallelLayerNormalization()
            else:
                return tfl.LayerNormalization()

        def make_dense(out_dims):
            if self._pm is not None:
                l = DenseParallel(units=out_dims, n_parallel=self._pm,
                                  kernel_initializer=self._ki)
            else:
                l = tfl.Dense(out_dims, kernel_initializer=self._ki)
            if self._sn:
                return SpectralNormalization(l)
            else:
                return l

        input_dims = input_shape[-1]
        self._short_layers = []
        if self._ln:
            self._res_layers = [make_norm()]
        else:
            self._res_layers = []
        self._res_layers += [make_dense(self._bd), tfl.ReLU(),
                             make_dense(self._od)]
        if input_dims != self._od:
            self._short_layers.append(make_dense(self._od))

    def call(self, inputs):
        residual = run_layers(self._res_layers, inputs)
        shortcut = run_layers(self._short_layers, inputs)
        return residual + shortcut


def make_simple_dense_layers(num_hidden, dims, nonlinearity, init, final_init, out_dim, sn_hidden_layers,
                             sn_first_layer=False, sn_last_layer=False, regularizer=None, parallel_models=None):
    def make_dense(out_dims, activation, initializer):
        if parallel_models is not None:
            l = DenseParallel(units=out_dims, activation=activation,
                              n_parallel=parallel_models,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer)
        else:
            l = tfl.Dense(out_dims, activation=activation,
                          kernel_initializer=initializer, kernel_regularizer=regularizer)
        return l

    if sn_first_layer:
        layers = [SpectralNormalization(make_dense(out_dims=dims, activation=nonlinearity, initializer=init))]
    else:
        layers = [make_dense(out_dims=dims, activation=nonlinearity, initializer=init)]
    hidden_layers = [make_dense(out_dims=dims, activation=nonlinearity, initializer=init) for _ in range(num_hidden)]
    if sn_hidden_layers:
        hidden_layers = [SpectralNormalization(l) for l in hidden_layers]
    layers += hidden_layers
    if sn_last_layer:
        layers += [SpectralNormalization(make_dense(out_dims=out_dim, activation=None, initializer=final_init))]
    else:
        layers += [make_dense(out_dims=out_dim, activation=None, initializer=final_init)]
    return layers


def make_modern_arch(num_modern_blocks, dims, bottleneck_dims, init, final_init, out_dim, layer_norm=True,
                     parallel_models=None):
    def make_dense(out_dims, activation, initializer):
        if parallel_models is not None:
            l = DenseParallel(units=out_dims, activation=activation,
                              n_parallel=parallel_models,
                              kernel_initializer=initializer)
        else:
            l = tfl.Dense(out_dims, activation=activation, kernel_initializer=initializer)
        return l

    layers = [make_dense(out_dims=dims, activation='relu', initializer=init)]
    layers += [ModernResidualBlock(out_dims=dims, bottleneck_dims=bottleneck_dims,
                                   kernel_initializer=init, layer_normalization=layer_norm, spectral_normalization=True,
                                   parallel_models=parallel_models)
               for _ in range(num_modern_blocks)]
    layers += [make_dense(out_dims=out_dim, activation=None, initializer=final_init)]

    return layers


def make_modern_arch_mod(num_modern_blocks, dims, bottleneck_dims, init, small_final_gain, out_dim, layer_norm=True,
                     parallel_models=None):
    gain = np.sqrt(2)
    if init == 'orthogonal':
        init = tfi.Orthogonal(gain=gain)
        if small_final_gain:
            final_init = tfi.Orthogonal(gain=0.01 * gain)
        else:
            final_init = init
    else:
        raise NotImplementedError
    def make_dense(out_dims, activation, initializer):
        if parallel_models is not None:
            l = DenseParallel(units=out_dims, activation=activation,
                              n_parallel=parallel_models,
                              kernel_initializer=initializer)
        else:
            l = tfl.Dense(out_dims, activation=activation, kernel_initializer=initializer)
        return l

    layers = [make_dense(out_dims=dims, activation='relu', initializer=init)]
    layers += [ModernResidualBlock(out_dims=dims, bottleneck_dims=bottleneck_dims,
                                   kernel_initializer=init, layer_normalization=layer_norm, spectral_normalization=True,
                                   parallel_models=parallel_models)
               for _ in range(num_modern_blocks)]
    layers += [make_dense(out_dims=out_dim, activation=None, initializer=final_init)]

    return layers

def make_actor_layers(action_size, output_extra_dims, modern, algorithm_specific_extra_dims=0, **kwargs):
    if modern:
        layers = make_modern_arch_mod(out_dim=action_size+output_extra_dims+algorithm_specific_extra_dims, **kwargs)
    else:
        layers = make_simple_dense_layers(out_dim=action_size+output_extra_dims+algorithm_specific_extra_dims, **kwargs)
    return layers