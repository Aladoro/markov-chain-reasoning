import math
import tensorflow as tf
import numpy as np

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)
MIN_RANGE_FLOAT = np.finfo(np.float32).eps
MAX_RANGE_FLOAT = 1 - MIN_RANGE_FLOAT


def standard_cdf(x):
    return 0.5 * (1 + tf.math.erf(x * CONST_INV_SQRT_2))


def standard_icdf(x):
    return CONST_SQRT_2 * tf.math.erfinv(2 * x - 1)


@tf.function
def stable_sample_tn(loc, scale, min=-1.0, max=1.0, batch_shape=[]):
    '''
    Stably sampling from a truncated normal distribution,
    based on 'https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py'
    '''
    transformed_min = (min - loc) / scale
    transformed_max = (max - loc) / scale
    cdf_min = standard_cdf(transformed_min)
    cdf_max = standard_cdf(transformed_max)
    Z = tf.maximum((cdf_max - cdf_min), MIN_RANGE_FLOAT)
    random_samples = tf.random.uniform(
        shape=tf.concat([batch_shape, tf.shape(loc)], axis=0),
        minval=MIN_RANGE_FLOAT, maxval=MAX_RANGE_FLOAT)
    unscaled_samples = standard_icdf(cdf_min + random_samples * Z)
    return tf.stop_gradient(unscaled_samples) * scale + loc
