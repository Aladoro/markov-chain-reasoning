import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from fsres_sac_models import ReasoningStochasticActor
from truncated_normal_utils import stable_sample_tn

tfl = tf.keras.layers
tfi = tf.keras.initializers
tfo = tf.keras.optimizers
tfd = tfp.distributions
tfb = tfp.bijectors
LN4 = np.log(4)


class TruncReasoningStochasticActor(ReasoningStochasticActor):
    def __init__(self, *args, **kwargs):
        super(TruncReasoningStochasticActor, self).__init__(*args, **kwargs)
        if self.residual:
            assert (self._act_layers[-1].units % 2) == 1 # odd number of units since we want a final interpolation
            self._act_dims = self._act_layers[-1].units // 2

    def call(self, inputs):
        out = inputs
        for layer in self._act_layers:
            out = layer(out)
        out = tf.math.tanh(out)  # scales to -1/1 for both mean/log_stddev together, since we are using a truncated normal
        if self.residual:
            mean, log_stddev, mixing = tf.split(out, [self._act_dims, self._act_dims, 1], axis=-1)
        else:
            mean, log_stddev = tf.split(out, 2, -1)
        scaled_log_stddev = self._min_log_stddev + (log_stddev + 1) / 2 * self._range_log_stddev
        stddev = tf.exp(scaled_log_stddev)
        if self.residual:
            mixing = (mixing + 1)/2
            return mean, stddev, mixing
        return mean, stddev

    def call_verbose(self, inputs, idx):
        out = inputs
        tf.print('i')
        tf.print(tf.gather(out, idx, axis=0))
        for i, layer in enumerate(self._act_layers):
            out = layer(out)
            tf.print(i)
            tf.print(tf.gather(out, idx, axis=0))
        out = tf.math.tanh(out) # scales to -1/1 for both mean/log_stddev together, since we are using a truncated normal
        tf.print('tanh')
        tf.print(tf.gather(out, idx, axis=0), summarize=-1)
        mean, log_stddev = tf.split(out, 2, -1)
        scaled_log_stddev = self._min_log_stddev + (log_stddev + 1) / 2 * self._range_log_stddev
        tf.print('scal_std')
        tf.print(tf.gather(scaled_log_stddev, idx, axis=0))
        stddev = tf.exp(scaled_log_stddev)
        return mean, stddev

    def get_mean_std(self, observation_batch, action_batch, preprocess=True):
        if preprocess:
            pre_obs_batch = self.preprocess_obs(observation_batch)
        else:
            pre_obs_batch = observation_batch
        if self.residual:
            unscaled_action = action_batch / self._action_scale
            input_batch = tf.concat([pre_obs_batch, unscaled_action], axis=-1)
        else:
            input_batch = tf.concat([pre_obs_batch, action_batch], axis=-1)
        if self.residual:
            mean, stddev, mixing = self.__call__(input_batch)
            mean = mixing*mean + (1-mixing)*unscaled_action
        else:
            mean, stddev = self.__call__(input_batch)
        return mean, stddev

    def get_mean_std_verbose(self, observation_batch, action_batch, idx, preprocess=True):
        if preprocess:
            pre_obs_batch = self.preprocess_obs(observation_batch)
        else:
            pre_obs_batch = observation_batch
        if self.residual:
            unscaled_action = action_batch / self._action_scale
            input_batch = tf.concat([pre_obs_batch, unscaled_action], axis=-1)
        else:
            input_batch = tf.concat([pre_obs_batch, action_batch], axis=-1)
        if self.residual:
            mean, stddev, mixing = self.call_verbose(input_batch)
            mean = mixing * mean + (1 - mixing) * unscaled_action
        else:
            mean, stddev = self.call_verbose(input_batch)
        return mean, stddev

    def get_action(self, observation_batch, action_batch, noise_stddev, *args, **kwargs):
        mean, stddev = self.get_mean_std(observation_batch, action_batch)
        if noise_stddev == 0.0:
            return mean * self._action_scale
        action = stable_sample_tn(loc=mean, scale=stddev, min=-1.0, max=1.0)
        return action * self._action_scale

    def get_action_probability(self, observation_batch, action_batch):
        mean, stddev = self.get_mean_std(observation_batch, action_batch)
        actions = stable_sample_tn(loc=mean, scale=stddev, min=-1.0, max=1.0)
        dist = tfp.distributions.TruncatedNormal(mean, stddev, low=-1.0, high=1.0)
        log_probs = tf.reduce_sum(dist.log_prob(actions), axis=-1, keepdims=True)
        if self._correct_logprob_for_action_scale:
            log_probs -= self._log_action_scale_shift
        return actions * self._action_scale, log_probs

    def get_action_chain_probability(self, observation_batch, action_batch, chain_length, unbiased, chain_backprop,
                                     perturb_type, perturb_prob, perturb_coeff, only_last=False,
                                     conditioning_probabilities=False, first_conditioning_distribution=None,
                                     reverse_transition_probabilities=False,
                                     get_raw_statistics=False):
        raw_statistics = dict()
        pre_obs_batch = self.preprocess_obs(observation_batch)
        all_mean = []
        all_stddev = []
        all_raw_actions = [] # raw_actions unscaled actions
        all_actions = []
        current_action_batch = action_batch
        if conditioning_probabilities or reverse_transition_probabilities:
            all_raw_actions.append(tf.clip_by_value(action_batch/self.action_scale,
                                                    clip_value_min=-0.9999, clip_value_max=0.9999))
            all_actions.append(action_batch)
        for _ in range(chain_length):
            if not chain_backprop:
                input_actions = tf.stop_gradient(current_action_batch)
            else:
                input_actions = current_action_batch
            if perturb_prob and perturb_type:
                input_actions = self.perturb_actions(action_batch=input_actions, noise_type=perturb_type,
                                                     prob=perturb_prob, coeff=perturb_coeff)
            mean, stddev = self.get_mean_std(pre_obs_batch, input_actions, preprocess=False)
            raw_actions = stable_sample_tn(loc=mean, scale=stddev, min=-1.0, max=1.0)
            current_action_batch = raw_actions * self._action_scale
            all_mean.append(mean)
            all_stddev.append(stddev)
            all_raw_actions.append(raw_actions)
            all_actions.append(current_action_batch)
        if reverse_transition_probabilities:
            assert not perturb_type
            if not chain_backprop:
                input_actions = tf.stop_gradient(current_action_batch)
            else:
                input_actions = current_action_batch
            mean, stddev = self.get_mean_std(pre_obs_batch, input_actions, preprocess=False)
            all_mean.append(mean)
            all_stddev.append(stddev)


        all_mean = tf.stack(all_mean, axis=-2)  # batch_dims x chain_length (+1? if reversed) x action_size
        all_stddev = tf.stack(all_stddev, axis=-2)  # batch_dims x chain_length (+1? if reversed) x action_size
        all_raw_actions = tf.stack(all_raw_actions, axis=-2)  # batch_dims x chain_length x action_size
        all_actions = tf.stack(all_actions, axis=-2)  # batch_dims x chain_length x action_size
        joint_distribution = tfd.TruncatedNormal(loc=tf.expand_dims(all_mean, axis=-3), scale=tf.expand_dims(all_stddev, axis=-3), low=-1.0, high=1.0)
        all_log_probs = joint_distribution.log_prob(tf.expand_dims(all_raw_actions, axis=-2))  # batch_dims x chain_length x chain_length (+1? if reversed) x action_size - for each action - chain_length log prob predictions (of each action dim)
        all_log_probs = tf.reduce_sum(all_log_probs, axis=-1) # batch_dims x chain_length x chain_length - accumulate for all action dims
        if reverse_transition_probabilities:
            forward_transition_probabilities = tf.expand_dims(tf.linalg.diag_part(all_log_probs[..., 1:, :-1]), axis=-1)
            backward_transition_probabilities = tf.expand_dims(tf.linalg.diag_part(all_log_probs[..., :-1, 1:]), axis=-1)
            all_log_probs = all_log_probs[..., :-1]
        if conditioning_probabilities:
            all_conditioning_log_probs = tf.stop_gradient(all_log_probs[..., :-1, :])
            if first_conditioning_distribution is None:
                uniform_logprobs = tf.tile(self.uniform_log_probs, [tf.shape(all_conditioning_log_probs)[0], chain_length, 1])
                all_conditioning_log_probs = tf.concat([uniform_logprobs, all_conditioning_log_probs[..., :-1]], axis=-1)
            else:
                raise NotImplementedError
            conditioning_log_probs = self.reduce_log_probs(all_log_probs=all_conditioning_log_probs, chain_length=chain_length, unbiased=False)

        if conditioning_probabilities or reverse_transition_probabilities:
            all_log_probs = all_log_probs[..., 1:, :]
            all_actions = all_actions[..., 1:, :]
            all_mean = all_mean[..., 1:, :]
            all_stddev = all_stddev[..., 1:, :]
            all_raw_actions = all_raw_actions[..., 1:, :]

        log_probs = self.reduce_log_probs(all_log_probs=all_log_probs, chain_length=chain_length, unbiased=unbiased)
        if self._correct_logprob_for_action_scale:
            log_probs -= self._log_action_scale_shift
            if conditioning_probabilities:
                conditioning_log_probs -= self._log_action_scale_shift
            if reverse_transition_probabilities:
                forward_transition_probabilities -= self._log_action_scale_shift
                backward_transition_probabilities -= self._log_action_scale_shift

        if get_raw_statistics:
            raw_statistics['raw_actions_L1'] = tf.reduce_mean(tf.abs(all_raw_actions))
            raw_statistics['pre_tanh_stddev'] = tf.reduce_mean(all_stddev)

        auxiliary_returns = dict()
        auxiliary_returns['all_mean'] = all_mean
        auxiliary_returns['all_stddev'] = all_stddev
        auxiliary_returns['all_raw_actions'] = all_raw_actions
        if conditioning_probabilities:
            auxiliary_returns['conditioning_log_probs'] = conditioning_log_probs
        if reverse_transition_probabilities:
            auxiliary_returns['transition_probabilities'] = (forward_transition_probabilities, backward_transition_probabilities)
        return all_actions, log_probs, auxiliary_returns, raw_statistics  # batch_dims x chain_length x (action_dims, 1)

    def get_action_distribution(self, observation_batch, action_batch):
        mean, stddev = self.get_mean_std(observation_batch, action_batch)
        dist = tfp.distributions.TruncatedNormal(mean, stddev, low=-1, high=1)
        return dist

