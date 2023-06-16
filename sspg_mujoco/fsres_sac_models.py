import copy

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from modular_generalized_sac_models import GPSAC
from modular_sac_models import StochasticActor

from layers_utils import ModernResidualBlock


tfl = tf.keras.layers
tfi = tf.keras.initializers
tfo = tf.keras.optimizers
tfd = tfp.distributions
tfb = tfp.bijectors
LN4 = np.log(4)


class ReasoningStochasticActor(StochasticActor):
    def __init__(self, *args, residual=False, residual_eps=1e-6, auto_steps_max=None, **kwargs):
        super(ReasoningStochasticActor, self).__init__(*args, **kwargs)
        self.uniform_log_probs = tf.math.log(1/2)*tf.ones([1, 1, 1])
        self.residual = residual
        self.residual_eps = residual_eps
        self._copied_act_layers = [copy.deepcopy(l) for l in self._act_layers]
        for layer in self._copied_act_layers:
            layer.trainable = False

    def build(self, input_shape):
        super(ReasoningStochasticActor, self).build(input_shape)
        random_inp = tf.random.normal(shape=input_shape)
        self.run_copied(random_inp)
        for layer in self._copied_act_layers:
            for i, w in enumerate(layer.weights):
                new_name = w.name + '_cp'
                layer.weights[i]._handle_name = new_name

    def call(self, inputs):
        out = inputs
        for layer in self._act_layers:
            out = layer(out)
        mean, log_stddev = tf.split(out, 2, -1)
        scaled_log_stddev = self._min_log_stddev + (tf.tanh(log_stddev) + 1) / 2 * self._range_log_stddev
        stddev = tf.exp(scaled_log_stddev)
        return mean, stddev

    def run_copied(self, inputs):
        out = inputs
        for layer in self._copied_act_layers:
            out = layer(out)
        mean, log_stddev = tf.split(out, 2, -1)
        scaled_log_stddev = self._min_log_stddev + (tf.tanh(log_stddev) + 1) / 2 * self._range_log_stddev
        stddev = tf.exp(scaled_log_stddev)
        return mean, stddev

    def copy_weights(self, ):
        for l, cl in zip(self._act_layers, self._copied_act_layers):
            l_ws = l.weights
            cl_ws = cl.weights
            if isinstance(l, ModernResidualBlock):
                for rl, cprl in zip(l._res_layers, cl._res_layers):
                    rl_ws = rl.weights
                    cprl_ws = cprl.weights
                    for i, (l_w, cl_w) in enumerate(zip(rl_ws, cprl_ws)):
                        cl_w.assign(l_w)
                        cl_w.assign(l_w)
                for rl, cprl in zip(l._short_layers, cl._short_layers):
                    rl_ws = rl.weights
                    cprl_ws = cprl.weights
                    for i, (l_w, cl_w) in enumerate(zip(rl_ws, cprl_ws)):
                        cl_w.assign(l_w)
                        cl_w.assign(l_w)
            else:
                for i, (l_w, cl_w) in enumerate(zip(l_ws, cl_ws)):
                    cl_w.assign(l_w)

    def get_mean_std(self, observation_batch, action_batch):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        if self.residual:
            unscaled_action = action_batch / self._action_scale
            pre_tanh_action = tf.atanh(tf.clip_by_value(unscaled_action,
                clip_value_min=-1.0+self.residual_eps, clip_value_max=1.0-self.residual_eps))
            input_batch = tf.concat([pre_obs_batch, pre_tanh_action], axis=-1)
        else:
            input_batch = tf.concat([pre_obs_batch, action_batch], axis=-1)
        mean, stddev = self.__call__(input_batch)
        if self.residual:
            mean = mean + pre_tanh_action
        return mean, stddev

    def get_mean_std_copied(self, observation_batch, action_batch):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        if self.residual:
            unscaled_action = action_batch / self._action_scale
            pre_tanh_action = tf.atanh(tf.clip_by_value(unscaled_action,
                clip_value_min=-1.0+self.residual_eps, clip_value_max=1.0-self.residual_eps))
            input_batch = tf.concat([pre_obs_batch, pre_tanh_action], axis=-1)
        else:
            input_batch = tf.concat([pre_obs_batch, action_batch], axis=-1)
        mean, stddev = self.run_copied(input_batch)
        if self.residual:
            mean = mean + pre_tanh_action
        return mean, stddev

    def get_action(self, observation_batch, action_batch, noise_stddev, *args, **kwargs):
        mean, stddev = self.get_mean_std(observation_batch, action_batch)
        if noise_stddev == 0.0:
            return tf.tanh(mean) * self._action_scale
        return tf.tanh(mean + tf.random.normal(tf.shape(mean)) * stddev) * self._action_scale

    def get_action_probability(self, observation_batch, action_batch):  # Biased
        mean, stddev = self.get_mean_std(observation_batch, action_batch)
        random_component = tf.random.normal(tf.shape(mean))
        raw_actions = mean + random_component * stddev
        actions = tf.tanh(raw_actions)
        log_probs = (-1 / 2 * tf.reduce_sum(tf.square(random_component), axis=-1) -
                     tf.reduce_sum(tf.math.log(stddev), axis=-1) - self._log_prob_offset)
        squash_features = -2 * raw_actions
        squash_correction = tf.reduce_sum(LN4 + squash_features - 2 * tf.math.softplus(squash_features), axis=1)
        log_probs -= squash_correction
        log_probs = tf.reshape(log_probs, [-1, 1])
        if self._correct_logprob_for_action_scale:
            log_probs -= self._log_action_scale_shift
        return actions * self._action_scale, log_probs

    def perturb_actions(self, action_batch, noise_type, prob, coeff):
        if noise_type:
            random_samples = tf.random.uniform(shape=[tf.shape(action_batch)[0], 1])
            resets = random_samples < prob
            if noise_type == 'uniform':
                assert coeff <= 1.0 and coeff > 0.0
                random_actions = tf.random.uniform(shape=tf.shape(action_batch), minval=-self._action_scale,
                                                   maxval=self._action_scale)
                random_actions = action_batch * (1 - coeff) + random_actions * coeff
            elif noise_type == 'gaussian':
                assert coeff > 0.0
                random_actions = action_batch + tf.random.normal(shape=tf.shape(action_batch), stddev=coeff)
                random_actions = tf.clip_by_value(random_actions, clip_value_min=-self._action_scale,
                                                  clip_value_max=self._action_scale)
            else:
                raise NotImplementedError
            input_actions = tf.where(resets, x=random_actions, y=action_batch)
            return input_actions
        else:
            return action_batch

    def get_action_chain_probability(self, observation_batch, action_batch, chain_length, unbiased, chain_backprop,
                                     perturb_type, perturb_prob, perturb_coeff, only_last=False,
                                     conditioning_probabilities=False, first_conditioning_distribution=None,
                                     reverse_transition_probabilities=False,
                                     get_raw_statistics=False):  # Unbiased
        raw_statistics = dict()
        if type(chain_backprop) is int:
            self.copy_weights()
        pre_obs_batch = self.preprocess_obs(observation_batch)
        all_mean = []
        all_stddev = []
        all_raw_actions = []
        all_actions = []
        current_action_batch = action_batch
        if conditioning_probabilities or reverse_transition_probabilities:
            all_raw_actions.append(tf.atanh(tf.clip_by_value(action_batch, clip_value_min=-0.99999, clip_value_max=0.99999)))
            all_actions.append(action_batch)
        for curr_iter in range(chain_length):
            if not chain_backprop:
                input_actions = tf.stop_gradient(current_action_batch)
            else:
                input_actions = current_action_batch
            if perturb_prob and perturb_type:
                input_actions = self.perturb_actions(action_batch=input_actions, noise_type=perturb_type,
                                                     prob=perturb_prob, coeff=perturb_coeff)
            input_batch = tf.concat([pre_obs_batch, input_actions], axis=-1)
            if type(chain_backprop) is int:
                if curr_iter >= int(chain_backprop):
                    # mean, stddev = self.copied_self(input_batch)
                    mean, stddev = self.run_copied(input_batch)
                else:
                    mean, stddev = self.__call__(input_batch)
            else:
                mean, stddev = self.__call__(input_batch)
            random_component = tf.random.normal(tf.shape(mean))
            raw_actions = mean + random_component * stddev
            current_action_batch = tf.tanh(raw_actions) * self._action_scale
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
            input_batch = tf.concat([pre_obs_batch, input_actions], axis=-1)
            mean, stddev = self.__call__(input_batch)
            all_mean.append(mean)
            all_stddev.append(stddev)


        all_mean = tf.stack(all_mean, axis=-2)  # batch_dims x chain_length (+1? if reversed) x action_size
        all_stddev = tf.stack(all_stddev, axis=-2)  # batch_dims x chain_length (+1? if reversed) x action_size
        all_raw_actions = tf.stack(all_raw_actions, axis=-2)  # batch_dims x chain_length x action_size
        all_actions = tf.stack(all_actions, axis=-2)  # batch_dims x chain_length x action_size
        joint_distribution = tfd.MultivariateNormalDiag(loc=tf.expand_dims(all_mean, axis=-3), scale_diag=tf.expand_dims(all_stddev, axis=-3))  # batch_dims x 1 x chain_length (+1? if reversed)x action_size
        all_log_probs = joint_distribution.log_prob(tf.expand_dims(all_raw_actions, axis=-2))  # batch_dims x chain_length x chain_length (+1? if reversed) - for each action - chain_length log prob predictions
        squash_features = -2 * all_raw_actions  # batch_dims x chain_length x action_size
        squash_correction = tf.reduce_sum(LN4 + squash_features - 2 * tf.math.softplus(squash_features), axis=-1, keepdims=True)  # batch_dims x chain_length x 1
        all_log_probs -= squash_correction  # batch_dims x chain_length x chain_length
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

    def reduce_log_probs(self, all_log_probs, chain_length, unbiased):
        if unbiased:
            all_log_probs_mask = tf.eye(
                num_rows=chain_length) * 1e9
            all_log_probs_masked = all_log_probs - all_log_probs_mask  # batch_dims x chain_length x chain_length -1e-9 whenever the prev. action generated the current action
            log_probs = tf.reduce_logsumexp(all_log_probs_masked, axis=-1, keepdims=True) # batch_dims x chain_length x 1 - sum unbiased log prob computed in log space
            log_probs = log_probs - tf.math.log(tf.cast(chain_length - 1, tf.float32))  # batch_dims x chain_length x 1 - average unbiased log prob computed in log space
        else:
            log_probs = tf.reduce_logsumexp(all_log_probs, axis=-1, keepdims=True) - tf.math.log(tf.cast(chain_length, tf.float32))  # batch_dims x chain_length x 1 - average unbiased log prob computed in log space
        return log_probs

    def get_action_distribution(self, observation_batch, action_batch):
        observation_shape = tf.shape(observation_batch)
        pre_obs_batch = self.preprocess_obs(observation_batch)
        input_batch = tf.concat([pre_obs_batch, action_batch], axis=-1)
        mean, stddev = self.__call__(input_batch)
        if observation_shape[0] == 1:
            mean = tf.squeeze(mean, axis=0)
            stddev = tf.squeeze(stddev, axis=0)
        multivariate_gaussian = tfd.MultivariateNormalDiag(loc=mean, scale_diag=stddev)
        return tfd.TransformedDistribution(distribution=multivariate_gaussian, bijector=self.distribution_bijector)

    def preprocess_obs(self, observation_batch):
        if self._norm_mean is not None and self._norm_stddev is not None:
            return (observation_batch - self.norm_mean) / (self.norm_stddev + 1e-7)
        return observation_batch


class ReasoningStochasticActorCopy(ReasoningStochasticActor):
    def __init__(self, *args, **kwargs):
        super(ReasoningStochasticActor, self).__init__(*args, **kwargs)
        self.uniform_log_probs = tf.math.log(1 / 2) * tf.ones([1, 1, 1])
        self._act_layers = [copy.deepcopy(l) for l in self._act_layers]
        for layer in self._act_layers:
            layer.trainable = False

    def build(self, input_shape):
        super(ReasoningStochasticActor, self).build(input_shape)

    def copy_weights(self, actor):
        actor_weights = actor.trainable_weights
        self_weights = self.weights
        for a_w, s_w in zip(actor_weights, self_weights):
            s_w.assign(a_w)

class ResSAC(GPSAC):
    def __init__(self, num_chains,
                 training_chain_init,
                 target_chain_init,
                 training_chain_mode,
                 target_chain_mode,
                 exploration_chain_mode,
                 training_steps,
                 target_steps,
                 exploration_steps,
                 test_steps,
                 training_unbiased_logprobs,
                 target_unbiased_logprobs,
                 training_chain_backpropagation,
                 training_chain_perturb,
                 training_chain_perturb_prob,
                 training_chain_perturb_coeff,
                 training_policy_with_next_state,
                 deterministic_test_sampling,
                 *args,

                 random_chain_init=False,

                 training_is=False,
                 target_is=False,
                 exploration_is=False,
                 test_is=False,

                 unbiased_is=False,

                 training_burnin=0,

                 raw_mean_decay=None,

                 detailed_balance_optim=None,
                 detailed_balance_enforce=False,

                 save_detailed_fs_statistics=False,
                 save_mh_statistics=False,

                 **kwargs):
        GPSAC.__init__(self, *args, **kwargs)
        assert isinstance(self._act, ReasoningStochasticActor)
        assert training_chain_init in ['rb_action', 'target_chain_action']
        assert target_chain_init in ['rb_action']
        assert training_chain_mode in ['all', 'first']
        assert target_chain_mode in ['last', 'random']
        assert exploration_chain_mode in ['last', 'random']
        if training_chain_perturb:
            assert training_chain_perturb in ['uniform', 'gaussian']
        if detailed_balance_optim:
            raise NotImplementedError
            assert detailed_balance_optim in []
        self.training_unbiased_logprobs = training_unbiased_logprobs
        self.target_unbiased_logprobs = target_unbiased_logprobs
        self.training_chain_backpropagation = training_chain_backpropagation
        self.training_chain_backpropagation = training_chain_backpropagation
        self.training_chain_init = training_chain_init
        self.target_chain_init = target_chain_init
        self.training_chain_mode = training_chain_mode
        self.target_chain_mode = target_chain_mode
        self.exploration_chain_mode = exploration_chain_mode
        self.num_chains = num_chains
        self.training_steps = training_steps
        self.target_steps = target_steps
        self.exploration_steps = exploration_steps
        self.test_steps = test_steps

        self.training_burnin = training_burnin

        self.training_chain_reset_prob = float(training_chain_perturb_prob)
        self.training_chain_perturb = training_chain_perturb
        self.training_chain_perturb_coeff = float(training_chain_perturb_coeff)
        self.training_policy_with_next_state = training_policy_with_next_state

        self.deterministic_test_sampling = deterministic_test_sampling


        self.random_chain_init = random_chain_init

        self.training_is = training_is
        self.target_is = target_is
        self.exploration_is = exploration_is
        self.test_is = test_is

        self.unbiased_is = unbiased_is

        self.raw_mean_decay = raw_mean_decay

        self.detailed_balance_optim = detailed_balance_optim
        self.detailed_balance_enforce = detailed_balance_enforce

        self.exploration_chains = tf.Variable(
            initial_value=tf.random.uniform([num_chains, self._act._action_dim],
                                            minval=-1 * self._act._action_scale,
                                            maxval=self._act._action_scale),
            trainable=False,
            name='expl_chains')

        self.test_chains = tf.Variable(
            initial_value=tf.random.uniform([num_chains, self._act._action_dim],
                                            minval=-1 * self._act._action_scale,
                                            maxval=self._act._action_scale),
            trainable=False,
            name='test_chains')

        if self.training_chain_init == 'target_chain_action':
            self.latest_target_chain_action = None

        self._save_detailed_fs_statistics = save_detailed_fs_statistics
        self._save_mh_statistics = save_mh_statistics

        self.define_training_statistics()
        self.initialize_training_statistics()

    def define_training_statistics(self,):
        if self._save_training_statistics:
            self._scalar_statistics = dict()
            self._scalar_statistics['raw_actions_L1'] = tf.keras.metrics.Mean('raL1')
            self._scalar_statistics['pre_tanh_stddev'] = tf.keras.metrics.Mean('std')
            self._scalar_statistics['auxiliary_loss'] = tf.keras.metrics.Mean('auxL')
            if self.raw_mean_decay:
                self._scalar_statistics['raw_mean_l2'] = tf.keras.metrics.Mean('rmL2')
            self._training_steps_statistics = dict()
            if self._save_detailed_fs_statistics:
                self._training_steps_statistics['tr_q'] = tf.keras.metrics.MeanTensor('trq')
                self._training_steps_statistics['tr_p'] = tf.keras.metrics.MeanTensor('trp')
            if self.training_is:
                self._training_steps_statistics['training_is'] = tf.keras.metrics.MeanTensor('tis')
                if self._save_detailed_fs_statistics:
                    self._training_steps_statistics['cond_q'] = tf.keras.metrics.MeanTensor('coq')
                    self._training_steps_statistics['cond_p'] = tf.keras.metrics.MeanTensor('cop')
            if self._save_mh_statistics or self.detailed_balance_enforce:
                self._training_steps_statistics['mh_for_p'] = tf.keras.metrics.MeanTensor('mfp')
                self._training_steps_statistics['mh_back_p'] = tf.keras.metrics.MeanTensor('mbp')
                self._training_steps_statistics['mh_acceptance_p'] = tf.keras.metrics.MeanTensor('mhp')

    def initialize_training_statistics(self,):
        if self._save_training_statistics:
            for tr_stat in self._training_steps_statistics.values():
                tr_stat(tf.zeros([self.training_steps]))
                tr_stat.reset_states()

    def sample_chain(self, steps, observation_batch, initial_action_batch, noise_stddev):
        action_chains = []
        current_action_batch = initial_action_batch
        if len(tf.shape(observation_batch)) == 1:
            observation_batch = tf.expand_dims(observation_batch, axis=0)
        if tf.shape(observation_batch)[0] == 1:
            observation_batch = tf.repeat(observation_batch,
                                          repeats=[tf.shape(initial_action_batch)[0]],
                                          axis=0)
        for _ in range(steps):
            current_action_batch = self._act.get_action(observation_batch, current_action_batch, noise_stddev, )
            action_chains.append(current_action_batch)
        action_chains = tf.stack(action_chains, axis=1)  # bs x steps x act_dim
        return action_chains, current_action_batch

    def sample_logprob_chain(self, steps, observation_batch, initial_action_batch):
        action_chains = []
        logprob_chains = []
        current_action_batch = initial_action_batch
        for _ in range(steps):
            current_action_batch, current_log_probs = self._act.get_action_probability(observation_batch,
                                                                                       current_action_batch)
            action_chains.append(current_action_batch)
            logprob_chains.append(current_action_batch)
        action_chains = tf.stack(action_chains, axis=1)  # bs x steps x act_dim
        logprob_chains = tf.stack(logprob_chains, axis=1)  # bs x steps x act_dim
        return action_chains, logprob_chains, current_action_batch

    def get_action(self, observation_batch, noise_stddev, max_noise=0.5):
        return self.get_max_action(observation_batch, noise_stddev, max_noise)

    def get_random_chain(self,):
        return tf.random.uniform([self.num_chains, self._act._action_dim], minval=-1 * self._act._action_scale, maxval=self._act._action_scale)

    def get_input_chain(self, noise_stddev):
        if self.random_chain_init:
            return self.get_random_chain()
        elif noise_stddev == 0.0:
            return self.test_chains
        else:
            return self.exploration_chains

    @tf.function
    def get_max_action(self, observation_batch, noise_stddev, max_noise=0.5):
        input_chains = self.get_input_chain(noise_stddev)
        if noise_stddev == 0.0:
            # action_chains: # chains x action_dim
            # action_batch: # chains x test_steps x action_dim
            action_chains, action_batch = self.sample_chain(steps=self.test_steps, observation_batch=observation_batch,
                                                            initial_action_batch=input_chains,
                                                            noise_stddev=0.2)
            self.test_chains.assign(action_batch)
            if self.deterministic_test_sampling:
                _, final_det_action_batch = self.sample_chain(steps=1,
                                                                observation_batch=observation_batch,
                                                                initial_action_batch=input_chains,
                                                                noise_stddev=0.0)
                flat_action_chains = tf.reshape(action_chains,
                                                [self.test_steps * self.num_chains, self._act._action_dim])
                q_pred = self.evaluate_actions(observation_batch, flat_action_chains)
                best_action_idx = tf.argmax(q_pred[:, 0])
                action = tf.gather(flat_action_chains, indices=[best_action_idx], axis=0)
                if self._save_training_statistics:
                    pass
            else:
                if self.exploration_chain_mode == 'last':
                    action_idx = tf.random.uniform(shape=[1], maxval=self.num_chains, dtype=tf.int32)
                    action = tf.gather(action_batch, indices=action_idx, axis=0)
                elif self.exploration_chain_mode == 'random':
                    action_idx = tf.random.uniform(shape=[1], maxval=self.num_chains, dtype=tf.int32)
                    step_idx = tf.random.uniform(shape=[1], maxval=self.test_steps, dtype=tf.int32)
                    indexed_chain_step = tf.gather(action_chains, indices=step_idx, axis=1)[:, 0]
                    action = tf.gather(indexed_chain_step, indices=action_idx, axis=0)
                else:
                    raise NotImplementedError
                if self._save_training_statistics:
                    pass
        else:
            action_chains, action_batch = self.sample_chain(steps=self.exploration_steps,
                                                            observation_batch=observation_batch,
                                                            initial_action_batch=input_chains,
                                                            noise_stddev=noise_stddev)
            self.exploration_chains.assign(action_batch)
            if self.exploration_chain_mode == 'last':
                action_idx = tf.random.uniform(shape=[1], maxval=self.num_chains, dtype=tf.int32)
                action = tf.gather(action_batch, indices=action_idx, axis=0)
            elif self.exploration_chain_mode == 'random':
                action_idx = tf.random.uniform(shape=[1], maxval=self.num_chains, dtype=tf.int32)
                step_idx = tf.random.uniform(shape=[1], maxval=self.exploration_steps, dtype=tf.int32)
                indexed_chain_step = tf.gather(action_chains, indices=step_idx, axis=1)[:, 0]
                action = tf.gather(indexed_chain_step, indices=action_idx, axis=0)
            else:
                raise NotImplementedError
        return action

    @tf.function
    def evaluate_actions(self, observation, action_batch):
        action_batch_dims = tf.shape(action_batch)[:-1]
        num_actions = tf.reduce_prod(action_batch_dims)

        tiled_observations = tf.repeat(observation, repeats=[num_actions], axis=0)
        flattened_actions = tf.reshape(action_batch, [num_actions, self._act._action_dim])
        next_q_pred, next_q_uncertainty = self.get_qs_prediction_uncertainty(tiled_observations, flattened_actions)
        penalized_q_pred = next_q_pred - self._uncertainty_coeff * next_q_uncertainty
        penalized_q_pred = tf.expand_dims(tf.reshape(penalized_q_pred, action_batch_dims), axis=-1)
        return penalized_q_pred

    def call(self, inputs):
        out = {}
        out['act'] = self.get_action(inputs, noise_stddev=0.0)
        out['act_out'] = self._act.get_mean_std(inputs, out['act'])
        out['copied_act_out'] = self._act.get_mean_std_copied(inputs, out['act'])
        self._act.copy_weights()
        out['q'] = tf.math.reduce_min(self._cri.get_qs(inputs, out['act']), axis=-1,
                                      keepdims=True)
        out['t_q'] = tf.math.reduce_min(self._targ_cri.get_qs(inputs, out['act']), axis=-1,
                                        keepdims=True)
        return out

    def reset_loss_metrics(self, ):
        super(ResSAC, self).reset_loss_metrics()
        for tr_stat in self._training_steps_statistics.values():
            tr_stat.reset_states()
        for raw_stat in self._scalar_statistics.values():
            raw_stat.reset_states()
        pass

    def make_latest_log_dict(self, ):
        super(ResSAC, self).make_latest_log_dict()
        for tr_stat_name, tr_stat in self._training_steps_statistics.items():
            tr_stat_res = tr_stat.result().numpy()
            for i in range(self.training_steps):
                self._latest_log_dict['{}_pos_{}'.format(tr_stat_name, i)] = tr_stat_res[i]
        for raw_stat_name, raw_stat in self._scalar_statistics.items():
            self._latest_log_dict[raw_stat_name] = raw_stat.result().numpy()
        pass

    @tf.function
    def run_full_training_impl(self, observations, actions, next_observations, rewards, done_mask, tune_uncertainty):
        loss_critic = self.run_delayed_training_impl(observations, actions, next_observations, rewards, done_mask,
                                                     tune_uncertainty)
        if self.training_policy_with_next_state:
            loss_actor, loss_alpha = self._train_act_and_alpha(next_observations, actions)
        else:
            loss_actor, loss_alpha = self._train_act_and_alpha(observations, actions)
        if self._save_training_statistics:
            self._al_metric(loss_actor)
            self._alpha_metric(loss_alpha)
        return loss_critic, loss_actor, loss_alpha

    def make_critics_train_op(self, discount):
        if self._critics_type == 'det':
            def train(observation_batch, action_batch, next_observation_batch,
                      reward_batch, done_mask, tune_uncertainty):
                if self.target_chain_init == 'rb_action':
                    init_action_batch = action_batch
                else:
                    raise NotImplementedError
                next_actions_chain, next_log_probs_chain, auxiliary_returns, raw_statistics = self._act.get_action_chain_probability(
                    next_observation_batch, init_action_batch, self.target_steps,
                    unbiased=self.target_unbiased_logprobs,
                    perturb_type=None, perturb_prob=0, perturb_coeff=1.0,
                    chain_backprop=False, )  # batch_dims x chain_length x (action_dims, 1)
                if self.target_chain_mode == 'last':
                    next_actions = next_actions_chain[..., -1, :]  # batch_dims x action_dims
                    next_log_probs = next_log_probs_chain[..., -1, :]  # batch_dims x 1
                    next_q_pred, next_q_uncertainty = self.get_target_qs_prediction_uncertainty(next_observation_batch,
                                                                                             next_actions)
                    penalized_q_pred = next_q_pred - self._uncertainty_coeff * next_q_uncertainty
                elif self.target_chain_mode == 'random':
                    num_actions = tf.shape(next_actions_chain)[0]
                    step_idx = tf.random.uniform(shape=[num_actions, 1], maxval=self.test_steps, dtype=tf.int32)
                    next_actions = tf.gather(next_actions_chain, indices=step_idx, axis=1, batch_dims=1)[:, 0]
                    next_log_probs = tf.gather(next_log_probs_chain, indices=step_idx, axis=1, batch_dims=1)[:, 0]
                    next_q_pred, next_q_uncertainty = self.get_target_qs_prediction_uncertainty(next_observation_batch,
                                                                                                next_actions)
                    penalized_q_pred = next_q_pred - self._uncertainty_coeff * next_q_uncertainty
                elif self.target_chain_mode == 'all':
                    raise NotImplementedError
                if self.training_chain_init == 'target_chain_action':
                    if self.latest_target_chain_action is None:
                        self.latest_target_chain_action = tf.Variable(next_actions, trainable=False)
                    self.latest_target_chain_action.assign(next_actions)
                targets = tf.reshape(reward_batch, [-1, 1]) + tf.reshape(
                    1 - done_mask, [-1, 1]) * discount * (
                                  penalized_q_pred - tf.exp(self._log_alpha) * next_log_probs)
                with tf.GradientTape() as tape:
                    critic_qs = self._cri.get_qs(observation_batch, action_batch)
                    errors = critic_qs - targets
                    squared_errors = tf.square(errors)
                    q_loss = 0.5 * tf.reduce_mean(
                        tf.reduce_sum(squared_errors, axis=-1))
                    loss = q_loss
                gradients = tape.gradient(loss, self.get_critics_trainable_weights())
                self.critic_opt.apply_gradients(zip(gradients, self.get_critics_trainable_weights()))
                if self._tune_uncertainty_coeff and tune_uncertainty:
                    self.step_uncertainty_coeff(errors, next_q_uncertainty)
                return q_loss, tf.reduce_mean(next_q_uncertainty)
        else:
            raise NotImplementedError

        return tf.function(train)


    def actor_loss_and_logprobs(self, observation_batch, actions_chain, log_probs_chain, auxiliary_returns=None,
                                first_actions=None, ):
        if self.training_chain_mode == 'first':
            assert not self.training_is # first action is cond. action
            actions, log_probs = actions_chain[..., 0, :], log_probs_chain[..., 0, :]
            next_q_pred, next_q_uncertainty = self.get_qs_prediction_uncertainty(observation_batch, actions)
            penalized_q_pred = next_q_pred - self._actor_uncertainty_coeff * next_q_uncertainty
            actor_loss = tf.reduce_mean(tf.exp(self._log_alpha) * log_probs - penalized_q_pred)
        elif self.training_chain_mode == 'all':
            tiled_observations = tf.expand_dims(observation_batch, axis=-2)  # batch_dims x 1 x obs_dims
            tiled_observations = tf.repeat(tiled_observations, repeats=[tf.shape(actions_chain)[-2]],
                                           axis=-2)  # batch_dims x chain_length (+1?) x obs_dims
            # batch_dims x chain_length x (1, 1)
            next_q_pred_chain, next_q_uncertainty_chain = self.get_qs_prediction_uncertainty(tiled_observations,
                                                                                             actions_chain)
            # batch_dims x chain_length x 1
            penalized_q_pred_chain = next_q_pred_chain - self._actor_uncertainty_coeff * next_q_uncertainty_chain
            # Expectation averaged across batch and across chain
            alpha = tf.exp(self._log_alpha)
            # batch_dims x chain_length x 1
            actor_loss = alpha * log_probs_chain - penalized_q_pred_chain
            if self._training_steps_statistics and self._save_detailed_fs_statistics:
                self._training_steps_statistics['tr_q'](tf.reduce_mean(penalized_q_pred_chain[..., 0], axis=0))
                self._training_steps_statistics['tr_p'](tf.reduce_mean(log_probs_chain[..., 0], axis=0))


            if self._save_mh_statistics or self.training_is:
                first_q_pred, first_q_uncertainty = self.get_qs_prediction_uncertainty(observation_batch,
                                                                                       first_actions)
                penalized_q_pred_first = tf.expand_dims(first_q_pred - self._actor_uncertainty_coeff * first_q_uncertainty, axis=-2)  # batch_dims x 1 x 1
                conditioning_q_pred_chain = tf.stop_gradient(
                    tf.concat([penalized_q_pred_first, penalized_q_pred_chain[..., :-1, :]],
                              axis=-2))  # batch_dims x chain_length x 1 - replacing last q_pred
            if self._save_mh_statistics:
                forward_transition_probabilities, backward_transition_probabilities = auxiliary_returns['transition_probabilities']
                log_mh_acceptance = (penalized_q_pred_chain - conditioning_q_pred_chain)/alpha + backward_transition_probabilities - forward_transition_probabilities
                mh_acceptances = tf.minimum(log_mh_acceptance, 0.0) # logprobs
                self._training_steps_statistics['mh_for_p'](tf.reduce_mean(forward_transition_probabilities[..., 0], axis=0))
                self._training_steps_statistics['mh_back_p'](tf.reduce_mean(backward_transition_probabilities[..., 0], axis=0))
                self._training_steps_statistics['mh_acceptance_p'](tf.reduce_mean(mh_acceptances[..., 0], axis=0))
            if self.training_is:
                conditioning_log_probs_chain = auxiliary_returns['conditioning_log_probs']
                shifted_conditioning_q_pred_chain = conditioning_q_pred_chain/alpha - conditioning_log_probs_chain # batch_dims x chain_length x 1
                importance_weights = tf.nn.softmax(shifted_conditioning_q_pred_chain, axis=-2)
                if self._save_training_statistics:
                    self._training_steps_statistics['training_is'](tf.reduce_mean(importance_weights[..., 0], axis=0))
                    if self._save_detailed_fs_statistics:
                        self._training_steps_statistics['cond_q'](tf.reduce_mean(conditioning_q_pred_chain[..., 0], axis=0))
                        self._training_steps_statistics['cond_p'](tf.reduce_mean(conditioning_log_probs_chain[..., 0], axis=0))
                actor_loss = tf.reduce_sum(actor_loss * tf.stop_gradient(importance_weights), axis=-2)
            actor_loss = tf.reduce_mean(actor_loss)
            log_probs = log_probs_chain
        else:
            raise NotImplementedError
        return actor_loss, log_probs


    def make_actor_and_alpha_train_op(self, ):  # TODO
        if self._tune_entropy_coefficient:
            def step_alpha(log_probs):
                with tf.GradientTape() as tape:
                    loss = -tf.reduce_mean(self._log_alpha * tf.stop_gradient(
                        (log_probs + self._target_entropy)))
                gradients = tape.gradient(loss, [self._log_alpha])
                self.entropy_opt.apply_gradients(zip(gradients, [self._log_alpha]))
                return loss
        else:
            def step_alpha(log_probs):
                return 0.0

        def train(observation_batch, action_batch):
            if self.training_chain_init == 'rb_action':
                init_action_batch = action_batch
            elif self.training_chain_init == 'target_chain_action':
                init_action_batch = self.latest_target_chain_action
            else:
                raise NotImplementedError
            if self.training_burnin > 0:
                _, init_action_batch = self.sample_chain(steps=self.training_burnin,
                                                         observation_batch=observation_batch,
                                                         initial_action_batch=init_action_batch, noise_stddev=0.2)
            get_reversed_transition_probabilities = self.detailed_balance_enforce or self._save_mh_statistics
            with tf.GradientTape() as tape:
                actions_chain, log_probs_chain, auxiliary_returns, raw_statistics = self._act.get_action_chain_probability(
                    observation_batch, init_action_batch, self.training_steps,
                    unbiased=self.training_unbiased_logprobs,
                    perturb_type=self.training_chain_perturb,
                    perturb_prob=self.training_chain_reset_prob,
                    perturb_coeff=self.training_chain_perturb_coeff,
                    chain_backprop=self.training_chain_backpropagation,
                    conditioning_probabilities=self.training_is,
                    first_conditioning_distribution=None,
                    reverse_transition_probabilities=get_reversed_transition_probabilities,
                    get_raw_statistics=self._save_training_statistics,
                )
                actor_loss, log_probs = self.actor_loss_and_logprobs(observation_batch=observation_batch,
                                                                     actions_chain=actions_chain,
                                                                     log_probs_chain=log_probs_chain,
                                                                     auxiliary_returns=auxiliary_returns,
                                                                     first_actions=init_action_batch)
                auxiliary_loss = tf.constant(0.0)
                if self.raw_mean_decay:
                    all_mean = auxiliary_returns['all_mean']
                    all_mean_l2 = tf.reduce_mean(tf.square(all_mean))
                    raw_statistics['raw_mean_l2'] = all_mean_l2
                    auxiliary_loss = auxiliary_loss + self.raw_mean_decay*all_mean_l2
                actor_loss = actor_loss + auxiliary_loss

            gradients = tape.gradient(actor_loss, self.get_actor_trainable_weights())
            if self._clip_actor_gradients:
                gradients, _ = tf.clip_by_global_norm(gradients, 40)
            self.actor_opt.apply_gradients(zip(gradients, self.get_actor_trainable_weights()))
            if self.training_chain_mode == 'first':
                alpha_loss = step_alpha(log_probs=log_probs)
            elif self.training_chain_mode == 'all':
                alpha_loss = step_alpha(log_probs=log_probs_chain)
            else:
                raise NotImplementedError
            raw_statistics['auxiliary_loss'] = auxiliary_loss
            if self._save_training_statistics:
                for raw_stat, value in raw_statistics.items():
                    self._scalar_statistics[raw_stat](value)
            return actor_loss, alpha_loss
        return tf.function(train)
