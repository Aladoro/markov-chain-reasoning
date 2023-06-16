import numpy as np
import tensorflow as tf
from modular_sac_models import SAC

tfl = tf.keras.layers
tfm = tf.keras.models
LN4 = np.log(4)

def huber_quantile_loss(errors, quantiles, kappa):
    abs_diffs = tf.abs(errors)
    quantile_indicator = tf.stop_gradient(tf.cast(errors < 0, tf.float32))
    huber_diffs = tf.where(abs_diffs > kappa, x=kappa*(abs_diffs-0.5*kappa), y=0.5*tf.square(abs_diffs))
    return tf.abs(quantiles - quantile_indicator)*huber_diffs

class GPSAC(SAC):
    def __init__(self,
                 actor,
                 critics,
                 actor_optimizer=tf.keras.optimizers.Adam(1e-3),
                 critic_optimizer=tf.keras.optimizers.Adam(1e-3),
                 entropy_optimizer=tf.keras.optimizers.Adam(1e-4),
                 gamma=0.99,
                 q_polyak=0.995,
                 entropy_coefficient=0.1,
                 tune_entropy_coefficient=False,
                 target_entropy=-6,
                 clip_actor_gradients=True,

                 critics_type='det',
                 uncertainty_coeff=0.5,
                 actor_uncertainty_coeff=0.5,
                 tune_uncertainty_coefficient=False,
                 uncertainty_coeff_optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.5),
                 tie_actor_uncertainty_coeff=False,
                 actor_optimistic_shift_value=0.0,
                 actor_optimistic_shift_value_decay_steps=-1,
                 tune_uncertainty_coefficient_delay=1,
                 num_quantiles=51,
                 kappa=1.0,

                 save_training_statistics=False,
                 debug_invalid_grads=False,
                 **kwargs
                 ):
        initialize_model = True
        if initialize_model:
            self._save_training_statistics = save_training_statistics
            tfm.Model.__init__(self, )
            if self._save_training_statistics:
                self._al_metric = tf.keras.metrics.Mean(name='al')
                self._ql_metric = tf.keras.metrics.Mean(name='ql')
                self._alpha_metric = tf.keras.metrics.Mean(name='alpha')
                self._unc_metric = tf.keras.metrics.Mean(name='unc')

        self._tune_uncertainty_coeff = tune_uncertainty_coefficient
        self._tune_uncertainty_coefficient_delay = tune_uncertainty_coefficient_delay

        self._actor_uncertainty_coeff = tf.Variable(actor_uncertainty_coeff, trainable=False,
                                                    name='var_act_unc_coeff')

        if self._tune_uncertainty_coeff:
            self._uncertainty_coeff = tf.Variable(uncertainty_coeff, trainable=True,
                                                    name='var_unc_coeff')
            self._uncertainty_coeff_opt = uncertainty_coeff_optimizer
            self._tie_actor_uncertainty_coeff = tie_actor_uncertainty_coeff

            if tie_actor_uncertainty_coeff:
                assert actor_uncertainty_coeff == uncertainty_coeff
                self._actor_optimistic_shift_value = tf.Variable(actor_optimistic_shift_value, trainable=False,
                                                                 name='var_opt_shift')
                self._decay_actor_optimistic_shift_value = self._make_actor_uncertainty_coeff_shift_decay(
                    actor_optimistic_shift_value_decay_steps)
        else:
            self._uncertainty_coeff = tf.Variable(uncertainty_coeff, trainable=False,
                                                    name='var_unc_coeff')
        self._critics_type = critics_type

        SAC.__init__(self,
                     actor=actor,
                     critics=critics,
                     actor_optimizer=actor_optimizer,
                     critic_optimizer=critic_optimizer,
                     entropy_optimizer=entropy_optimizer,
                     gamma=gamma,
                     q_polyak=q_polyak,
                     entropy_coefficient=entropy_coefficient,
                     tune_entropy_coefficient=tune_entropy_coefficient,
                     target_entropy=target_entropy,
                     clip_actor_gradients=clip_actor_gradients,
                     save_training_statistics=save_training_statistics,
                     initialize_model=False)

        self._num_critics = self._cri._num_critics
        self._critics_range = tf.range(self._num_critics)
        self._num_uncertainty_diffs = self._num_critics ** 2 - self._num_critics

        if critics_type == 'quant':
            self.num_quantiles = num_quantiles
            self.quantile_locations = tf.reshape(
                (tf.range(self.num_quantiles, dtype=tf.float32) + 0.5) / self.num_quantiles,
                [1, 1, self.num_quantiles, 1])
            self._kappa = kappa

        self._debug_invalid_grads = debug_invalid_grads

    def _make_actor_uncertainty_coeff_shift_decay(self, actor_uncertainty_coeff_shift_decay_steps):
        if actor_uncertainty_coeff_shift_decay_steps > 0:
            decay_step = float(self._actor_optimistic_shift_value / actor_uncertainty_coeff_shift_decay_steps)
            if decay_step < 0.0:
                def decay_actor_uncertainty_coeff_shift():
                    self._actor_optimistic_shift_value.assign(tf.minimum(self._actor_optimistic_shift_value - decay_step, 0.0))
            elif decay_step > 0.0:
                def decay_actor_uncertainty_coeff_shift():
                    self._actor_optimistic_shift_value.assign(tf.maximum(self._actor_optimistic_shift_value - decay_step, 0.0))
            else:
                raise NotImplementedError
        else:
            def decay_actor_uncertainty_coeff_shift():
                pass

        return tf.function(decay_actor_uncertainty_coeff_shift)

    @tf.function
    def step_uncertainty_coeff(self, errors, uncertainty):
        with tf.GradientTape() as tape:
            loss = tf.stop_gradient(tf.reduce_mean(errors)) * self._uncertainty_coeff
        gradients = tape.gradient(loss, [self._uncertainty_coeff])
        self._uncertainty_coeff_opt.apply_gradients(zip(gradients, [self._uncertainty_coeff]))
        if self._tie_actor_uncertainty_coeff:
            self._decay_actor_optimistic_shift_value()
            self._actor_uncertainty_coeff.assign(self._uncertainty_coeff + self._actor_optimistic_shift_value)

    def get_action(self, observation_batch, noise_stddev, max_noise=0.5):
        return self.get_max_action(observation_batch, noise_stddev, max_noise)

    @tf.function
    def get_max_action(self, observation_batch, noise_stddev, max_noise=0.5):
        return self._act.get_action(observation_batch, noise_stddev=noise_stddev, max_noise=max_noise)

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
        out['act'] = self._act.get_action(inputs, noise_stddev=0.0)
        out['q'] = tf.math.reduce_min(self._cri.get_qs(inputs, out['act']), axis=-1,
                                      keepdims=True)
        out['t_q'] = tf.math.reduce_min(self._targ_cri.get_qs(inputs, out['act']), axis=-1,
                                        keepdims=True)
        return out

    def reset_loss_metrics(self, ):
        super(GPSAC, self).reset_loss_metrics()
        self._unc_metric.reset_states()

    def make_latest_log_dict(self, ):
        super(GPSAC, self).make_latest_log_dict()
        self._latest_log_dict['unc'] = self._unc_metric.result().numpy()
        self._latest_log_dict['unc_coeff'] = self._uncertainty_coeff.numpy()

    def run_full_training(self, observations, actions, next_observations, rewards, done_mask):
        tune_uncertainty = self._training_steps % self._tune_uncertainty_coefficient_delay == 0
        return self.run_full_training_impl(observations, actions, next_observations, rewards, done_mask, tune_uncertainty)

    def run_delayed_training(self, observations, actions, next_observations, rewards, done_mask):
        tune_uncertainty = self._training_steps % self._tune_uncertainty_coefficient_delay == 0
        return self.run_delayed_training_impl(observations, actions, next_observations, rewards, done_mask, tune_uncertainty)

    @tf.function
    def run_full_training_impl(self, observations, actions, next_observations, rewards, done_mask, tune_uncertainty):
        loss_critic = self.run_delayed_training_impl(observations, actions, next_observations, rewards, done_mask, tune_uncertainty)
        loss_actor, loss_alpha = self._train_act_and_alpha(observations)
        if self._save_training_statistics:
            self._al_metric(loss_actor)
            self._alpha_metric(loss_alpha)
        return loss_critic, loss_actor, loss_alpha

    @tf.function
    def run_delayed_training_impl(self, observations, actions, next_observations, rewards, done_mask, tune_uncertainty):
        losses_critic = self._train_cri(observations, actions, next_observations,
                                        rewards, done_mask, tune_uncertainty)
        if self._save_training_statistics:
            loss_critic, uncertainty = losses_critic
            self._unc_metric(uncertainty)
            self._ql_metric(loss_critic)
        return losses_critic

    def get_target_qs_prediction_uncertainty(self, observations, actions, use_online=False):
        if self._critics_type == 'det':
            return self.get_target_qs_prediction_uncertainty_det(observations, actions, use_online=use_online)
        elif self._critics_type == 'quant':
            return self.get_target_qs_prediction_uncertainty_quant(observations, actions, use_online=use_online)
        else:
            raise NotImplementedError

    def get_target_qs_prediction_uncertainty_det(self, observations, actions, use_online):
        if use_online:
            qs = self._cri.get_qs(observations, actions)
        else:
            qs = self._targ_cri.get_qs(observations, actions)
        q_pred = tf.reduce_mean(qs, axis=-1, keepdims=True)
        if self._num_critics == 2:
            q_unc = tf.abs(qs[..., 0] - qs[..., 1])
        elif self._num_critics > 2:
            qs_uns = tf.expand_dims(qs, axis=-1)
            qs_uns2 = tf.expand_dims(qs, axis=-2)

            qs_diffs = tf.abs(qs_uns - qs_uns2)
            q_unc = tf.reduce_sum(qs_diffs, axis=[-1, -2]) / self._num_uncertainty_diffs
        else:
            raise NotImplementedError
        q_unc = tf.expand_dims(q_unc, axis=-1)
        return q_pred, q_unc

    def get_target_qs_prediction_uncertainty_quant(self, observations, actions, use_online):
        if use_online:
            qs = self._cri.get_qs(observations, actions)
        else:
            qs = self._targ_cri.get_qs(observations, actions) # n x n_cri x n_quant

        qs_sorted_idxs = tf.argsort(qs, axis=2)
        sorted_qs = tf.gather(qs, indices=qs_sorted_idxs, batch_dims=2, axis=2) # re-ordering inputs first
        q_pred = tf.reduce_mean(sorted_qs, axis=1, keepdims=True) # n x 1 x n_quant
        sorted_qs_uns = tf.expand_dims(sorted_qs, axis=1) # n x 1 x n_cri x n_quant
        sorted_qs_uns2 = tf.expand_dims(sorted_qs, axis=2) # n x n_cri x 1 x n_quant
        qs_diffs = tf.abs(sorted_qs_uns - sorted_qs_uns2) # n x n_cri x n_cri x n_quant
        qs_wass_dists = tf.reduce_mean(qs_diffs, axis=-1, keepdims=True) # n x n_cri x n_cri x 1 - Wass_1
        q_unc = tf.reduce_sum(qs_wass_dists, axis=[1, 2]) / self._num_uncertainty_diffs
        q_unc = tf.reshape(q_unc, [-1, 1, 1]) # n x 1 x 1
        return q_pred, q_unc

    def get_qs_prediction_uncertainty(self, observations, actions):
        if self._critics_type == 'det':
            return self.get_qs_prediction_uncertainty_det(observations, actions)
        elif self._critics_type == 'quant':
            return self.get_qs_prediction_uncertainty_quant(observations, actions)
        else:
            raise NotImplementedError

    def get_qs_prediction_uncertainty_det(self, observations, actions):
        qs = self._cri.get_qs(observations, actions)
        q_pred = tf.reduce_mean(qs, axis=-1, keepdims=True)
        if self._num_critics == 2:
            q_unc = tf.abs(qs[..., 0] - qs[..., 1])
        elif self._num_critics > 2:
            qs_uns = tf.expand_dims(qs, axis=-1)
            qs_uns2 = tf.expand_dims(qs, axis=-2)

            qs_diffs = tf.abs(qs_uns - qs_uns2)
            q_unc = tf.reduce_sum(qs_diffs, axis=[-1, -2]) / self._num_uncertainty_diffs
        else:
            raise NotImplementedError
        q_unc = tf.expand_dims(q_unc, axis=-1)
        return q_pred, q_unc

    def get_qs_prediction_uncertainty_quant(self, observations, actions):
        qs = self._cri.get_qs(observations, actions) # n x n_cri x n_quant
        qs_sorted_idxs = tf.argsort(qs, axis=2) # n x n_cri x n_quant
        sorted_qs = tf.gather(qs, indices=qs_sorted_idxs, batch_dims=2, axis=2) # re-ordering inputs first
        q_pred = tf.reduce_mean(sorted_qs, axis=1, keepdims=True)  # n x 1 x n_quant
        sorted_qs_uns = tf.expand_dims(sorted_qs, axis=1)  # n x 1 x n_cri x n_quant
        sorted_qs_uns2 = tf.expand_dims(sorted_qs, axis=2)  # n x n_cri x 1 x n_quant
        qs_diffs = tf.abs(sorted_qs_uns - sorted_qs_uns2)  # n x n_cri x n_cri x n_quant
        qs_wass_dists = tf.reduce_mean(qs_diffs, axis=-1, keepdims=True)  # n x n_cri x n_cri x 1 - Wass_1
        q_unc = tf.reduce_sum(qs_wass_dists, axis=[1, 2]) / self._num_uncertainty_diffs
        q_unc = tf.reshape(q_unc, [-1, 1, 1])  # n x 1 x 1
        return q_pred, q_unc

    def check_gradients(self, grads, weights, obs, name, inf_ele, sampled_act, **log_kwargs):
        '''Debug function.'''
        none_grads = [tf.reduce_any(tf.math.is_nan(grad)) for grad in grads]
        stacked_none = tf.stack(none_grads, axis=0)
        if tf.reduce_any(stacked_none):
            tf.print(name)
            tf.print('------- KWARGS -------')
            for n, v in log_kwargs.items():
                tf.print('------- {} -------'.format(n))
                tf.print(v)

            tf.print('------- None grads -------')
            for i, ng in enumerate(none_grads):
                if ng:
                    tf.print(i)
                    tf.print(weights[i].name)

            inf_coords = tf.argmin(tf.cast(tf.math.is_finite(inf_ele), tf.float32))
            inf = tf.reduce_min(tf.cast(tf.math.is_finite(inf_ele), tf.float32))
            tf.print('------- INF? -------')
            tf.print(inf)
            tf.print('------- MEAN STD -------')
            mean, stddev = self._act.call_verbose(obs)
            tf.print('------- MEAN -------')
            tf.print(tf.gather(mean, inf_coords, axis=0), summarize=-1)
            tf.print('------- std -------')
            tf.print(tf.gather(stddev, inf_coords, axis=0), summarize=-1)
            tf.print('------- ACT -------')
            tf.print(tf.gather(sampled_act, inf_coords, axis=0), summarize=-1)

    def make_critics_train_op(self, discount):
        if self._critics_type == 'det':
            def train(observation_batch, action_batch, next_observation_batch,
                      reward_batch, done_mask, tune_uncertainty):
                next_actions, next_log_probs = self._act.get_action_probability(next_observation_batch)
                next_q_pred, next_q_uncertainty = self.get_target_qs_prediction_uncertainty(next_observation_batch,
                                                                                            next_actions)
                penalized_q_pred = next_q_pred - self._uncertainty_coeff * next_q_uncertainty

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
                if self._debug_invalid_grads:
                    self.check_gradients(grads=gradients, weights=self.get_critics_trainable_weights(),
                                         obs=next_observation_batch, sampled_act=next_actions,
                                         name='Q-LOSS', inf_ele=next_log_probs,
                                         tmean_QS=tf.reduce_mean(critic_qs),
                                         tar_q_pred=tf.reduce_mean(penalized_q_pred),
                                         mean_targets=tf.reduce_mean(targets), q_loss=q_loss,
                                         mean_log_probs=tf.reduce_mean(next_log_probs), alpha=tf.exp(self._log_alpha))
                self.critic_opt.apply_gradients(zip(gradients, self.get_critics_trainable_weights()))
                if self._tune_uncertainty_coeff and tune_uncertainty:
                    self.step_uncertainty_coeff(errors, next_q_uncertainty)
                return q_loss, tf.reduce_mean(next_q_uncertainty)
        elif self._critics_type == 'quant':
            def train(observation_batch, action_batch, next_observation_batch,
                      reward_batch, done_mask, tune_uncertainty):
                next_actions, next_log_probs = self._act.get_action_probability(next_observation_batch)
                # n x 1 x (n_quant, 1)
                next_q_pred, next_q_uncertainty = self.get_target_qs_prediction_uncertainty(next_observation_batch,
                                                                                            next_actions)
                # n x 1 x n_quant
                penalized_q_pred = next_q_pred - self._uncertainty_coeff * next_q_uncertainty
                # n x 1 x n_quant
                targets = tf.reshape(reward_batch, [-1, 1, 1]) + tf.reshape(
                    1 - done_mask, [-1, 1, 1]) * discount * (
                                  penalized_q_pred - tf.exp(self._log_alpha) *
                                  tf.reshape(next_log_probs, [-1, 1, 1]))
                with tf.GradientTape() as tape:
                    # n x n_cri x n_on_quant
                    critic_qs = self._cri.get_qs(observation_batch, action_batch)
                    # n x n_cri x n_on_quant x n_quant
                    all_errors = tf.expand_dims(critic_qs, axis=-1) - tf.expand_dims(targets, axis=-2)
                    quant_losses = tf.reduce_sum(huber_quantile_loss(errors=-1*all_errors,
                                                                     quantiles=self.quantile_locations,
                                                                     kappa=self._kappa), axis=[1, 2, 3])
                    q_loss = tf.reduce_mean(quant_losses)
                    loss = q_loss
                gradients = tape.gradient(loss, self.get_critics_trainable_weights())
                self.critic_opt.apply_gradients(zip(gradients, self.get_critics_trainable_weights()))
                if self._tune_uncertainty_coeff and tune_uncertainty:
                    self.step_uncertainty_coeff(all_errors, next_q_uncertainty)
                return q_loss, tf.reduce_mean(next_q_uncertainty)
        else:
            raise NotImplementedError

        return tf.function(train)

    def make_actor_and_alpha_train_op(self, ):
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
        def train(observation_batch):
            with tf.GradientTape() as tape:
                actions, log_probs = self._act.get_action_probability(observation_batch)
                next_q_pred, next_q_uncertainty = self.get_qs_prediction_uncertainty(observation_batch, actions)
                penalized_q_pred = next_q_pred - self._actor_uncertainty_coeff * next_q_uncertainty
                actor_loss = tf.reduce_mean(tf.exp(self._log_alpha) * log_probs - penalized_q_pred)
            gradients = tape.gradient(actor_loss, self.get_actor_trainable_weights())

            if self._clip_actor_gradients:
                gradients, _ = tf.clip_by_global_norm(gradients, 40)
            if self._debug_invalid_grads:
                self.check_gradients(grads=gradients, weights=self.get_actor_trainable_weights(),
                                     obs=observation_batch,
                                     name='ACT-LOSS', inf_ele=log_probs[:, 0], sampled_act=actions,
                                     mean_OBJ=actor_loss, mean_log_probs=tf.reduce_mean(log_probs),
                                     alpha=tf.exp(self._log_alpha))
            self.actor_opt.apply_gradients(zip(gradients, self.get_actor_trainable_weights()))
            alpha_loss = step_alpha(log_probs=log_probs)
            return actor_loss, alpha_loss

        return tf.function(train)
