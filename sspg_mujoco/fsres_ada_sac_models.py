import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from fsres_ada_utils import StepsUpdater
from fsres_sac_models import ResSAC

from modular_sac_models import StochasticActor

tfl = tf.keras.layers
tfi = tf.keras.initializers
tfo = tf.keras.optimizers
tfd = tfp.distributions
tfb = tfp.bijectors
LN4 = np.log(4)


class AdaResSAC(ResSAC):
    def __init__(self,
                 auto_steps,
                 auto_steps_max,
                 auto_steps_threshold,
                 auto_steps_min_expl_update,
                 auto_steps_training_update,
                 auto_steps_target_update,
                 *args,
                 **kwargs):
        if auto_steps:
            assert auto_steps in ['gelman_rubin', 'impr_gelman_rubin']
        self.auto_steps = auto_steps
        self.auto_steps_max = auto_steps_max
        self.auto_steps_thresh = auto_steps_threshold
        self.auto_steps_training_update = auto_steps_training_update
        self.auto_steps_target_update = auto_steps_target_update
        self.auto_steps_min_expl_update = auto_steps_min_expl_update
        ResSAC.__init__(self, *args, **kwargs)

        # Variables for dynamic_action_unroll
        self.action_dim = self._act._action_dim
        self.all_actions = tf.Variable(tf.zeros([self.num_chains, self.auto_steps_max, self.action_dim]),
                                       trainable=False)
        self.conv_measures = tf.Variable(tf.zeros([self.auto_steps_max]), trainable=False)
        self.k = tf.Variable(0, trainable=False)
        self.end_loop = tf.Variable(False, trainable=False)

        self.steps_updater = StepsUpdater(init=self.training_steps)
        self.current_min_expl_steps = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.current_training_steps = tf.Variable(self.training_steps, dtype=tf.int32, trainable=False)
        self.current_target_steps = tf.Variable(self.target_steps, dtype=tf.int32, trainable=False)

    def define_training_statistics(self, ):
        super(AdaResSAC, self).define_training_statistics()
        if self._save_training_statistics:
            self._scalar_statistics['auto_exploration_steps'] = tf.keras.metrics.Mean('aexs')
            self._scalar_statistics['auto_exploration_final_conv'] = tf.keras.metrics.Mean('aefc')
            self._scalar_statistics['auto_test_steps'] = tf.keras.metrics.Mean('atss')
            self._scalar_statistics['auto_test_final_conv'] = tf.keras.metrics.Mean('atfc')
            if self.auto_steps_min_expl_update:
                self._scalar_statistics['auto_min_expl_steps'] = tf.keras.metrics.Mean('amexgs')
            if self.auto_steps_target_update:
                self._scalar_statistics['auto_target_steps'] = tf.keras.metrics.Mean('atrgs')
            if self.auto_steps_training_update:
                self._scalar_statistics['auto_training_steps'] = tf.keras.metrics.Mean('atrns')

    def update_training_target_steps(self, ):
        if self.auto_steps_min_expl_update:
            self.current_min_expl_steps.assign(
                self.steps_updater.get_steps_estimate(self.auto_steps_min_expl_update) - 1)
            if self._save_training_statistics:
                self._scalar_statistics['auto_min_expl_steps'](self.current_min_expl_steps)

        if self.auto_steps_target_update:
            self.current_target_steps.assign(self.steps_updater.get_steps_estimate(self.auto_steps_target_update))
            if self._save_training_statistics:
                self._scalar_statistics['auto_target_steps'](self.current_target_steps)

        if self.auto_steps_training_update:
            self.current_training_steps.assign(self.steps_updater.get_steps_estimate(self.auto_steps_training_update))
            if self._save_training_statistics:
                self._scalar_statistics['auto_training_steps'](self.current_training_steps)

    @tf.function
    def calculate_convergence_statistic(self, action_chains, ):
        # input - num_chains x current_length x action_dim
        action_chains_shape = tf.shape(action_chains)
        action_dim = tf.cast(action_chains_shape[2], dtype=tf.float32)
        chain_length = tf.cast(action_chains_shape[1], dtype=tf.float32)
        num_chains = tf.cast(action_chains_shape[0], dtype=tf.float32)
        if self.auto_steps == 'gelman_rubin' or self.auto_steps == 'impr_gelman_rubin':
            chain_means = tf.reduce_mean(action_chains, axis=-2, keepdims=True)  # num_chains x 1 x action_dim
            overall_mean = tf.reduce_mean(action_chains, axis=[-3, -2])  # action_dim
            within_chains_deviations = action_chains - chain_means  # num_chains x current_length x action_dim
            # num_chains x current_length x action_dim x action_dim
            within_chains_sample_cov = (tf.expand_dims(within_chains_deviations, axis=-1)
                                        * tf.expand_dims(within_chains_deviations, axis=-2))
            # num_chains x action_dim x action_dim
            within_chains_sample_cov = tf.reduce_sum(within_chains_sample_cov, axis=-3) / (chain_length - 1)

            mean_within_chains_sample_cov = tf.reduce_mean(within_chains_sample_cov,
                                                           axis=-3)  # action_dims x action_dims

            # num_chains x 1 x action_dim
            between_chains_deviations = chain_means - overall_mean

            # num_chains x action_dim x action_dim
            between_chains_sample_cov = (
                    between_chains_deviations * tf.transpose(between_chains_deviations, perm=[0, 2, 1]))
            # action_dim x action_dim
            between_chains_sample_cov = tf.reduce_sum(between_chains_sample_cov, axis=0) / (num_chains - 1)
            if self.auto_steps == 'gelman_rubin':
                # action_dim x action_dim
                mean_within_sample_cov_inv = tf.linalg.pinv(
                    mean_within_chains_sample_cov)  # todo: safer way to compute inverse - check for correctness
                cov_ratio = tf.matmul(mean_within_sample_cov_inv, between_chains_sample_cov)
                cov_ratio = (cov_ratio + tf.transpose(cov_ratio)) / 2
                eigvals = tf.linalg.eigvalsh(cov_ratio)
                max_eigval = tf.reduce_max(eigvals)
                squared_score = (chain_length - 1) / chain_length + max_eigval
                return tf.math.sqrt(squared_score)
            else:
                det_between_chains_sample_cov = tf.linalg.det(between_chains_sample_cov)
                det_mean_within_chains_sample_cov = tf.linalg.det(mean_within_chains_sample_cov)
                det_ratio = tf.math.divide_no_nan(det_between_chains_sample_cov,
                                                  det_mean_within_chains_sample_cov)  # sets to 0 rather than nan/inf
                scaled_det_ratio = tf.pow(det_ratio, 1 / action_dim)
                squared_score = (chain_length - 1) / chain_length + scaled_det_ratio
                return tf.math.sqrt(squared_score)
        else:
            raise NotImplementedError

    @tf.function
    def dynamic_action_unroll(self, observation_batch, initial_action_batch, min_steps=0, **kwargs):
        """Adaptively compute next actions and check for chains convergence."""
        if tf.shape(observation_batch)[0] == 1:
            observation_batch = tf.repeat(observation_batch,
                                          repeats=[tf.shape(initial_action_batch)[0]],
                                          axis=0)
        current_actions = self._act.get_action(observation_batch, initial_action_batch, **kwargs)
        self.all_actions[:, 0, :].assign(current_actions)
        self.k.assign(1)
        self.end_loop.assign(False)
        conv = tf.constant(0.0)
        while not self.end_loop:
            current_actions = self._act.get_action(observation_batch, current_actions, **kwargs)
            self.all_actions[:, self.k, :].assign(current_actions)
            self.k.assign(self.k + 1)
            if self.k >= min_steps:
                conv = self.calculate_convergence_statistic(action_chains=self.all_actions[:, :self.k])

                self.conv_measures[self.k - 1].assign(conv)
                self.end_loop.assign((conv < self.auto_steps_thresh) or (self.k >= self.auto_steps_max))
            else:
                self.end_loop.assign((self.k >= self.auto_steps_max))
        return current_actions, self.k, conv, self.all_actions, self.conv_measures

    def get_action(self, observation_batch, noise_stddev, max_noise=0.5):
        min_steps = int(self.current_min_expl_steps)
        act = self.get_max_action(observation_batch, noise_stddev, min_steps, max_noise)
        return act

    @tf.function
    def get_max_action(self, observation_batch, noise_stddev, min_steps, min_max_noise=0.5):
        input_chains = self.get_input_chain(noise_stddev)
        if noise_stddev == 0.0:
            action_batch, k, conv, all_actions, conv_measures = self.dynamic_action_unroll(
                observation_batch=observation_batch,
                initial_action_batch=input_chains,
                min_steps=min_steps,
                noise_stddev=0.2)
            self.test_chains.assign(action_batch)
            if self.deterministic_test_sampling:
                flat_action_chains = tf.reshape(all_actions[:, :k],
                                                [-1, self._act._action_dim])
                q_pred = self.evaluate_actions(observation_batch, flat_action_chains)
                best_action_idx = tf.argmax(q_pred[:, 0])
                action = tf.gather(flat_action_chains, indices=[best_action_idx], axis=0)
            else:
                action_idx = tf.random.uniform(shape=[1], maxval=self.num_chains, dtype=tf.int32)
                action = tf.gather(action_batch, indices=action_idx, axis=0)
            if self._save_training_statistics:
                self._scalar_statistics['auto_test_steps'](k)
                self._scalar_statistics['auto_test_final_conv'](conv)
        else:
            action_batch, k, conv, all_actions, conv_measures = self.dynamic_action_unroll(
                observation_batch=observation_batch,
                initial_action_batch=input_chains,
                min_steps=min_steps,
                noise_stddev=noise_stddev)
            self.exploration_chains.assign(action_batch)
            self.steps_updater.update(k)
            self.update_training_target_steps()
            if self.exploration_chain_mode == 'last':
                action_idx = tf.random.uniform(shape=[1], maxval=self.num_chains, dtype=tf.int32)
                action = tf.gather(action_batch, indices=action_idx, axis=0)
            elif self.exploration_chain_mode == 'random':
                action_idx = tf.random.uniform(shape=[1], maxval=self.num_chains, dtype=tf.int32)
                step_idx = tf.random.uniform(shape=[1], maxval=k, dtype=tf.int32)
                indexed_chain_step = tf.gather(all_actions, indices=step_idx, axis=1)[:, 0]
                action = tf.gather(indexed_chain_step, indices=action_idx, axis=0)
            else:
                raise NotImplementedError
            if self._save_training_statistics:
                self._scalar_statistics['auto_exploration_steps'](k)
                self._scalar_statistics['auto_exploration_final_conv'](conv)
        return action

    def make_critics_train_op(self, discount):
        if self._critics_type == 'det':
            def train(observation_batch, action_batch, next_observation_batch,
                      reward_batch, done_mask, tune_uncertainty, target_steps):
                if self.target_chain_init == 'rb_action':
                    init_action_batch = action_batch
                else:
                    raise NotImplementedError
                next_actions_chain, next_log_probs_chain, auxiliary_returns, raw_statistics = self._act.get_action_chain_probability(
                    next_observation_batch, init_action_batch, target_steps,  # self.current_target_steps,
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
                    step_idx = tf.random.uniform(shape=[num_actions, 1], maxval=self.target_steps,
                                                 dtype=tf.int32)
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

    def run_full_training(self, observations, actions, next_observations, rewards, done_mask):
        tune_uncertainty = self._training_steps % self._tune_uncertainty_coefficient_delay == 0
        return self.run_full_training_impl(observations, actions, next_observations, rewards, done_mask,
                                           tune_uncertainty,
                                           training_steps=int(self.current_training_steps),
                                           target_steps=int(self.current_target_steps))

    def run_delayed_training(self, observations, actions, next_observations, rewards, done_mask):
        tune_uncertainty = self._training_steps % self._tune_uncertainty_coefficient_delay == 0
        return self.run_delayed_training_impl(observations, actions, next_observations, rewards, done_mask,
                                              tune_uncertainty,
                                              target_steps=int(self.current_target_steps))

    @tf.function
    def run_full_training_impl(self, observations, actions, next_observations, rewards, done_mask, tune_uncertainty,
                               training_steps, target_steps):
        loss_critic = self.run_delayed_training_impl(observations, actions, next_observations, rewards, done_mask,
                                                     tune_uncertainty, target_steps=target_steps)
        if self.training_policy_with_next_state:
            loss_actor, loss_alpha = self._train_act_and_alpha(next_observations, actions,
                                                               training_steps=training_steps)
        else:
            loss_actor, loss_alpha = self._train_act_and_alpha(observations, actions, training_steps=training_steps)
        if self._save_training_statistics:
            self._al_metric(loss_actor)
            self._alpha_metric(loss_alpha)
        return loss_critic, loss_actor, loss_alpha

    @tf.function
    def run_delayed_training_impl(self, observations, actions, next_observations, rewards, done_mask, tune_uncertainty,
                                  target_steps):
        losses_critic = self._train_cri(observations, actions, next_observations,
                                        rewards, done_mask, tune_uncertainty,
                                        target_steps=target_steps)
        if self._save_training_statistics:
            loss_critic, uncertainty = losses_critic
            self._unc_metric(uncertainty)
            self._ql_metric(loss_critic)
        return losses_critic

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

        def train(observation_batch, action_batch, training_steps):
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
                    observation_batch, init_action_batch, training_steps,  # self.current_training_steps,
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
                    auxiliary_loss = auxiliary_loss + self.raw_mean_decay * all_mean_l2
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
