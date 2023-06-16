import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from fsres_sac_models import ResSAC

from modular_sac_models import StochasticActor

tfl = tf.keras.layers
tfi = tf.keras.initializers
tfo = tf.keras.optimizers
tfd = tfp.distributions
tfb = tfp.bijectors
LN4 = np.log(4)


class StepsUpdater:
    def __init__(self, init, decay=0.99):
        assert (decay < 1.0) and (decay > 0.0)
        self.moving_average = tf.Variable(init, dtype=tf.float32, trainable=False)
        self.moving_deviation = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.decay = decay
        self.res = 1 - self.decay

    def update(self, sample):
        sample = tf.cast(sample, tf.float32)
        self.moving_average.assign(self.moving_average*self.decay + self.res*sample)
        deviation = tf.abs(self.moving_average - sample)
        self.moving_deviation.assign(self.moving_deviation*self.decay + self.res*deviation)

    def get(self,):
        return {'updater_average': self.moving_average,
                'updater_deviation': self.moving_deviation}

    def get_steps_estimate(self, steps_estimate='mean_ceil'):
        if steps_estimate == 'mean_ceil':
            return tf.cast(tf.math.ceil(self.moving_average), tf.int32)
        if steps_estimate == 'mean_floor':
            return tf.cast(self.moving_average, tf.int32)
        else:
            raise NotImplementedError

