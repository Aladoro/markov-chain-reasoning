import numpy as np
import tensorflow as tf

tfl = tf.keras.layers
LN4 = np.log(4)


class StochasticActor(tf.keras.layers.Layer):
    def __init__(self, layers, norm_mean=None, norm_stddev=None, min_log_stddev=-10,
                 max_log_stddev=2, action_scale=1.0, correct_logprob_for_action_scale=False):
        super(StochasticActor, self).__init__()
        self._act_layers = layers
        self._out_dim = layers[-1].units
        self._action_dim = self._out_dim//2

        self._log_prob_offset = self._action_dim/2*np.log(np.pi*2)

        self._norm_mean = norm_mean
        self._norm_stddev = norm_stddev
        self._min_log_stddev = min_log_stddev
        self._range_log_stddev = max_log_stddev - min_log_stddev
        self._action_scale = action_scale
        self._correct_logprob_for_action_scale = correct_logprob_for_action_scale
        if correct_logprob_for_action_scale:
            self._log_action_scale_shift = np.log(self._action_scale)*self._action_dim

    def call(self, inputs):
        out = inputs
        for layer in self._act_layers:
            out = layer(out)
        mean, log_stddev = tf.split(out, 2, -1)
        scaled_log_stddev = self._min_log_stddev + (tf.tanh(log_stddev) + 1) / 2 * self._range_log_stddev
        stddev = tf.exp(scaled_log_stddev)
        return mean, stddev

    def get_action(self, observation_batch, noise_stddev, *args, **kwargs):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        mean, stddev = self.__call__(pre_obs_batch)
        if noise_stddev == 0.0:
            return tf.tanh(mean)*self._action_scale
        return tf.tanh(mean + tf.random.normal(tf.shape(mean))*stddev)*self._action_scale

    def get_action_probability(self, observation_batch):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        mean, stddev = self.__call__(pre_obs_batch)
        random_component = tf.random.normal(tf.shape(mean))
        raw_actions = mean + random_component*stddev
        actions = tf.tanh(raw_actions)
        log_probs = (-1/2*tf.reduce_sum(tf.square(random_component), axis=-1) -
                     tf.reduce_sum(tf.math.log(stddev), axis=-1) - self._log_prob_offset)
        squash_features = -2 * raw_actions
        squash_correction = tf.reduce_sum(LN4 + squash_features - 2 * tf.math.softplus(squash_features), axis=1)
        log_probs -= squash_correction
        log_probs = tf.reshape(log_probs, [-1, 1])
        if self._correct_logprob_for_action_scale:
            log_probs -= self._log_action_scale_shift
        return actions*self._action_scale, log_probs

    def preprocess_obs(self, observation_batch):
        if self._norm_mean is not None and self._norm_stddev is not None:
            return (observation_batch - self.norm_mean) / (self.norm_stddev + 1e-7)
        return observation_batch


class Critic(tf.keras.layers.Layer):
    def __init__(self, layers, norm_mean=None, norm_stddev=None):
        super(Critic, self).__init__()
        self._cri_layers = layers
        self._norm_mean = norm_mean
        self._norm_stddev = norm_stddev

    def call(self, inputs):
        out = inputs
        for layer in self._cri_layers:
            out = layer(out)
        return out

    def get_q(self, observation_batch, action_batch):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        input_batch = tf.concat([pre_obs_batch, action_batch], axis=-1)
        return self.__call__(input_batch)

    def preprocess_obs(self, observation_batch):
        if self._norm_mean is not None and self._norm_stddev is not None:
            return (observation_batch - self.norm_mean) / (self.norm_stddev + 1e-7)
        return observation_batch


class SAC(tf.keras.Model):
    def __init__(self, make_actor, make_critics,
                 actor_optimizer=tf.keras.optimizers.Adam(1e-3),
                 critic_optimizer=tf.keras.optimizers.Adam(1e-3),
                 entropy_optimizer=tf.keras.optimizers.Adam(1e-4),
                 gamma=0.99,
                 q_polyak=0.995,
                 entropy_coefficient=0.1,
                 tune_entropy_coefficient=False,
                 target_entropy=-6,
                 clip_actor_gradients=True,
                 save_training_statistics=False,
                 **kwargs):
        initialize_model = kwargs.get('initialize_model', True)
        if initialize_model:
            super(SAC, self).__init__()
            self._save_training_statistics = save_training_statistics
            if save_training_statistics:
                self._al_metric = tf.keras.metrics.Mean(name='al')
                self._ql_metric = tf.keras.metrics.Mean(name='ql')
                self._alpha_metric = tf.keras.metrics.Mean(name='alpha')
        self.actor_opt = actor_optimizer
        self.critic_opt = critic_optimizer
        self.entropy_opt = entropy_optimizer
        self._entropy_coefficient = entropy_coefficient
        self._gamma = gamma
        self._tune_entropy_coefficient = tune_entropy_coefficient
        self._target_entropy = float(target_entropy)
        self._act = make_actor()
        self._cri = make_critics()
        self._targ_cri = make_critics()
        self._clip_actor_gradients = clip_actor_gradients

        self._train_cri = self.make_critics_train_op(gamma)
        self._targ_cri_update = self.make_target_update_op(self._cri,
                                                           self._targ_cri,
                                                           q_polyak)

        if self._tune_entropy_coefficient:
            self._log_alpha = tf.Variable(tf.math.log(
                self._entropy_coefficient), trainable=True, name='var_log_alpha')
        else:
            self._log_alpha = tf.Variable(tf.math.log(
                self._entropy_coefficient), trainable=False, name='var_log_alpha')

        self._train_act_and_alpha = self.make_actor_and_alpha_train_op()
        self._training_steps = 0

    def call(self, inputs):
        out = {}
        out['act'] = self._act.get_action(inputs, noise_stddev=0.0)
        out['q'] = self._cri.get_q(inputs, out['act'])
        out['t_q'] = self._targ_cri.get_q(inputs, out['act'])
        return out

    @tf.function
    def get_action(self, observation_batch, noise_stddev, max_noise=0.5):
        return self._act.get_action(observation_batch, noise_stddev=noise_stddev, max_noise=max_noise)

    def reset_loss_metrics(self, ):
        self._ql_metric.reset_states()
        self._al_metric.reset_states()
        self._alpha_metric.reset_states()

    def save_metrics_and_reset(self, ):
        self._latest_log_dict = {'q_loss': self._ql_metric.result().numpy(),
                                 'act_loss': self._al_metric.result().numpy(),
                                 'alpha_loss': self._alpha_metric.result().numpy(),
                                 'alpha': tf.exp(self._log_alpha).numpy()}
        self.reset_loss_metrics()

    def train(self, buffer, batch_size=128, n_updates=1, act_delay=1, tar_delay=1, **kwargs):
        for _ in range(n_updates):
            self._training_steps += 1
            b = buffer.get_random_batch(batch_size)
            (observations, actions, next_observations, rewards, done_mask) = (
                b['obs'], b['act'], b['nobs'], b['rew'], tf.cast(b['don'], tf.float32))
            if self._training_steps % act_delay == 0:
                losses = self.run_full_training(observations, actions, next_observations, rewards, done_mask)
            else:
                losses = self.run_delayed_training(observations, actions, next_observations, rewards, done_mask)
            if self._training_steps % tar_delay == 0:
                self.update_targets()

    @tf.function
    def run_full_training(self, observations, actions, next_observations, rewards, done_mask):
        loss_critic = self._train_cri(observations, actions, next_observations,
                                          rewards, done_mask)
        loss_actor, loss_alpha = self._train_act_and_alpha(observations)
        if self._save_training_statistics:
            self._ql_metric(loss_critic)
            self._al_metric(loss_actor)
            self._alpha_metric(loss_alpha)
        return loss_critic, loss_actor, loss_alpha

    @tf.function
    def run_delayed_training(self, observations, actions, next_observations, rewards, done_mask):
        loss_critic = self._train_cri(observations, actions, next_observations,
                                      rewards, done_mask)
        if self._save_training_statistics:
            self._ql_metric(loss_critic)
        return loss_critic

    @tf.function
    def update_targets(self,):
        self._targ_cri_update()

    def make_critics_train_op(self, discount):
        def q_estimator(observations, actions):
            return tf.reduce_min(self._targ_cri.get_qs(observations, actions), axis=-1, keepdims=True)
        def train(observation_batch, action_batch, next_observation_batch,
                  reward_batch, done_mask):
            next_actions, next_log_probs = self._act.get_action_probability(
                next_observation_batch)
            next_q = q_estimator(next_observation_batch, next_actions)
            targets = tf.reshape(reward_batch, [-1, 1]) + tf.reshape(
                1 - done_mask, [-1, 1]) * discount * (
                              next_q - tf.exp(self._log_alpha) * next_log_probs)
            with tf.GradientTape() as tape:
                loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(
                    self._cri.get_qs(observation_batch, action_batch) - tf.stop_gradient(targets)), axis=-1))
            gradients = tape.gradient(loss, self.get_critics_trainable_weights())
            self.critic_opt.apply_gradients(zip(gradients, self.get_critics_trainable_weights()))
            return loss

        return tf.function(train)

    def make_actor_and_alpha_train_op(self, ):
        def q_estimator(observations, actions):
            return tf.reduce_min(self._cri.get_q(observations, actions), axis=-1, keepdims=True)
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
                q_estimates = q_estimator(observation_batch, actions)
                actor_loss = tf.reduce_mean(tf.exp(self._log_alpha) * log_probs - q_estimates)
            gradients = tape.gradient(actor_loss, self.get_actor_trainable_weights())
            if self._clip_actor_gradients:
                gradients, _ = tf.clip_by_global_norm(gradients, 40)
            self.actor_opt.apply_gradients(zip(gradients, self.get_actor_trainable_weights()))
            alpha_loss = step_alpha(log_probs=log_probs)
            return actor_loss, alpha_loss

        return tf.function(train)

    def make_target_update_op(self, model, target_model, polyak):
        def update_target():
            critic_weights = model.trainable_weights
            target_weights = target_model.trainable_weights
            for c_w, t_w in zip(critic_weights, target_weights):
                t_w.assign((polyak) * t_w + (1 - polyak) * c_w)
        return tf.function(update_target)

    def get_critics_trainable_weights(self,):
        return self._cri.trainable_weights

    def get_actor_trainable_weights(self, ):
        return self._act.trainable_weights
