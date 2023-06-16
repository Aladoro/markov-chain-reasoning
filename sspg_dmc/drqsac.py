
import copy
import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from parallel_layers import ParallelLayerNorm
from drqv2 import RandomShiftsAug, ChangingModule
from truncated_normal import ActualTruncatedNormal

U_ENT = np.log(2)

class StochasticActor(nn.Module):
    def __init__(self, encoder, action_shape, feature_dim, hidden_dim, min_log_std, max_log_std):
        super().__init__()
        self.parallel_encoder = isinstance(encoder, ParallelEncoder)
        if self.parallel_encoder:
            self.trunk = nn.Sequential(nn.Linear(encoder.repr_dim*encoder.num_encoders, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
        else:
            self.trunk = nn.Sequential(nn.Linear(encoder.repr_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())

        self.action_dim = action_shape[0]

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, self.action_dim*2))

        self.apply(utils.weight_init)

        self.min_log_std, self.max_log_std = min_log_std, max_log_std
        self.log_std_range = self.max_log_std - self.min_log_std

    def process_obs(self, obs):
        if self.parallel_encoder:
            # n_enc x bs x -1
            obs = torch.transpose(obs, 0, 1) # bs x n_enc x -1
            obs = torch.flatten(obs, start_dim=-2, end_dim=-1)
        return obs

    def mean(self, obs):
        obs = self.process_obs(obs)
        h = self.trunk(obs)
        out = self.policy(h)
        mean, log_std = torch.tensor_split(out, 2, dim=-1)
        mu = torch.tanh(mean)
        return mu

    def mean_std(self, obs):
        obs = self.process_obs(obs)
        h = self.trunk(obs)

        out = self.policy(h)
        mean, log_std = torch.tensor_split(out, 2, dim=-1)
        log_std = (torch.tanh(log_std) + 1)/2*self.log_std_range + self.min_log_std
        return mean, torch.exp(log_std)

    def forward(self, obs):
        mean, std = self.mean_std(obs)
        dist = utils.SquashedNormal(mean, std)
        return dist


class TruncStochasticActor(StochasticActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mean(self, obs):
        obs = self.process_obs(obs)
        h = self.trunk(obs)
        out = self.policy(h)
        mean, log_std = torch.tensor_split(out, 2, dim=-1)
        mu = torch.tanh(mean)
        return mu

    def mean_std(self, obs):
        obs = self.process_obs(obs)
        h = self.trunk(obs)

        out = self.policy(h)
        out = torch.tanh(out)
        mean, log_std = torch.tensor_split(out, 2, dim=-1)
        log_std = (log_std + 1)/2*self.log_std_range + self.min_log_std
        return mean, torch.exp(log_std)

    def forward(self, obs):
        mean, std = self.mean_std(obs)
        dist = ActualTruncatedNormal(mean, std, a=-1.0, b=1.0)
        return dist


class Critic(nn.Module):
    def __init__(self, encoder, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self._n_parallel = 2

        self.trunk = nn.Sequential(nn.Linear(encoder.repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.QS = nn.Sequential(
            utils.DenseParallel(feature_dim + action_shape[0], hidden_dim, self._n_parallel),
            nn.ReLU(inplace=True),
            utils.DenseParallel(hidden_dim, hidden_dim, self._n_parallel),
            nn.ReLU(inplace=True),
            utils.DenseParallel(hidden_dim, 1, 2))

        self.apply(utils.weight_init)

    @property
    def n_parallel(self):
        return self._n_parallel

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        qs = self.QS(h_action)

        return torch.squeeze(torch.transpose(qs, 0, 1), dim=-1)

    def get_value(self, obs, action):
        QS = self.forward(obs, action)
        return QS.amin(dim=-1, keepdim=True)




class GPCritic(nn.Module):
    def __init__(self, num_critics, encoder, action_shape, feature_dim,
                 hidden_dim, unc_coeff, parallel_trunk=False):
        super().__init__()
        self.parallel_encoder = isinstance(encoder, ParallelEncoder)
        if self.parallel_encoder:
            assert parallel_trunk, 'need a parallel trunk to process input from a parallel encoder'
            assert num_critics == encoder.num_encoders
        self.unc_coeff = unc_coeff
        self._n_parallel = num_critics
        self.parallel_trunk = parallel_trunk
        if parallel_trunk:
            if self.parallel_encoder:
                self.trunk = nn.Sequential(utils.DenseParallel(encoder.repr_dim, feature_dim, encoder.num_encoders),
                                           ParallelLayerNorm(n_parallel=encoder.num_encoders, normalized_shape=[feature_dim]),
                                           nn.Tanh())
            else:
                self.trunk = nn.Sequential(utils.DenseParallel(encoder.repr_dim, feature_dim, num_critics),
                                           ParallelLayerNorm(n_parallel=num_critics, normalized_shape=[feature_dim]),
                                           nn.Tanh())
        else:
            self.trunk = nn.Sequential(nn.Linear(encoder.repr_dim, feature_dim),
                                        nn.LayerNorm(feature_dim), nn.Tanh())

        self.QS = nn.Sequential(
            utils.DenseParallel(feature_dim + action_shape[0], hidden_dim,
                                num_critics),
            nn.ReLU(inplace=True),
            utils.DenseParallel(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(inplace=True),
            utils.DenseParallel(hidden_dim, 1, num_critics))

        self.apply(utils.weight_init)
        self.num_uncertainty_diffs = num_critics ** 2 - num_critics

    @property
    def n_parallel(self):
        return self._n_parallel

    def forward(self, obs, action):

        h = self.trunk(obs)

        if self.parallel_trunk:
            action = action.unsqueeze(0).tile(self._n_parallel, 1, 1)
        h_action = torch.cat([h, action], dim=-1)
        qs = self.QS(h_action)  # num_critics x bs x 1
        return torch.squeeze(torch.transpose(qs, 0, 1), dim=-1)  # bs x num_critics

    def get_pred_target_and_uncertainty(self, obs, action):
        qs = self.__call__(obs, action)
        q_pred = torch.mean(qs, dim=1, keepdim=True)  # bs x 1
        q_diffs = torch.abs(qs.unsqueeze(-1) - qs.unsqueeze(-2))  # bs x n_critic x n_critic
        q_unc = q_diffs.sum(-1).sum(-1, keepdim=True) / self.num_uncertainty_diffs  # bs x 1
        return q_pred, q_unc

    def get_value(self, obs, action):
        q_pred, q_unc = self.get_pred_target_and_uncertainty(obs, action)
        target_V = q_pred - self.unc_coeff * q_unc
        return target_V


    def compute_loss_part(self, obs, action, target_Q):
        QS = self.__call__(obs, action)  # bs x num_critics
        Q_errors = QS - target_Q
        loss = Q_errors.pow(2).sum(1).mean()
        return QS, loss, Q_errors

    def compute_loss(self, obs, action, reward, discount, next_obs,
                     next_action, target_critic, unc_coeff):
        with torch.no_grad():
            target_Q_mean, target_Q_unc = target_critic.get_pred_target_and_uncertainty(next_obs, next_action)
            target_V = target_Q_mean - unc_coeff * target_Q_unc
            target_Q = reward + (discount * target_V)
        QS = self.__call__(obs, action)  # bs x num_critics
        Q_errors = QS - target_Q
        loss = Q_errors.pow(2).sum(1).mean()
        return target_Q, QS, loss, Q_errors



class ParallelEncoder(nn.Module):
    def __init__(self, obs_shape, n_filters, n_encoders):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = n_filters
        self.num_encoders = n_encoders
        self.output_dim = 35
        self.output_logits = False
        # self.feature_dim = feature_dim

        self.convnet = nn.ModuleList([
            nn.Conv2d(obs_shape[0]*self.num_encoders, self.num_filters*self.num_encoders, 3, stride=2, groups=self.num_encoders),
            nn.Conv2d(self.num_filters*self.num_encoders, self.num_filters*self.num_encoders, 3, stride=1, groups=self.num_encoders),
            nn.Conv2d(self.num_filters*self.num_encoders, self.num_filters*self.num_encoders, 3, stride=1, groups=self.num_encoders),
            nn.Conv2d(self.num_filters*self.num_encoders, self.num_filters*self.num_encoders, 3, stride=1, groups=self.num_encoders)
        ])

        self.repr_dim = self.num_filters * 35 * 35

        self.apply(utils.weight_init)
    def forward_conv(self, obs):
        obs = obs / 255. - 0.5

        tiled_obs = torch.tile(obs, [1, self.num_encoders, 1, 1])

        conv = torch.relu(self.convnet[0](tiled_obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convnet[i](conv))

        h = conv.view(conv.size(0), self.num_encoders, -1)
        return torch.transpose(h, 0, 1) # n_enc x bs x -1

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()

        return h

class DrQSACAgent:
    def __init__(self, encoder, actor, critic,
                 obs_shape, action_shape, device, lr,
                 feature_dim, hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps,
                 init_temperature,
                 target_entropy,
                 per_dim_target_entropy,
                 nstep,
                 discount,
                 nstep_entropy_correction,
                 use_tb,
                 aug=RandomShiftsAug(pad=4),
                 srank_batches=8,
                 gradient_steps_per_update=1,
                 **kwargs):

        self.train_log_format = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                                 ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                                 ('episode_reward', 'R', 'float'),
                                 ('buffer_size', 'BS', 'int'), ('fps', 'FPS', 'float'),
                                 ('total_time', 'T', 'time')]

        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.gradient_steps_per_update = gradient_steps_per_update
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.action_dim = action_shape[0]

        self.target_entropy = target_entropy
        self.per_dim_target_entropy = per_dim_target_entropy
        if isinstance(target_entropy, str):
            self.anneal_target_entropy = True
        else:
            if self.per_dim_target_entropy:
                self.target_entropy = target_entropy*self.action_dim
            self.anneal_target_entropy = False

        print(self.target_entropy)

        self.nstep = nstep
        self.nstep_entropy_correction = nstep_entropy_correction
        self.entropy_discount_vec = torch.pow(discount, torch.arange(start=1, end=nstep+1)).to(device=device)
        self.entropy_discount_correction_coeff = self.entropy_discount_vec[:-1].sum()
        print(self.entropy_discount_correction_coeff)
        # models
        if nstep_entropy_correction:
            assert nstep_entropy_correction in ['recompute', 'replicate', 'saved_entropy']
        assert isinstance(actor, StochasticActor)

        self.encoder = encoder.to(device)
        self.actor = actor.to(device)

        self.critic = critic.to(device)
        self.critic_target = copy.deepcopy(critic).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        encoder_lr = kwargs.get('encoder_lr', lr)
        actor_lr = kwargs.get('actor_lr', lr)
        critic_lr = kwargs.get('critic_lr', lr)
        log_alpha_lr = kwargs.get('log_alpha_lr', lr)

        # optimizers
        encoder_optimizer_builder = kwargs.get('encoder_optimizer_builder', None)
        if encoder_optimizer_builder is None:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=encoder_lr)
        else:
            self.encoder_opt = encoder_optimizer_builder(encoder)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(),
                                          lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=critic_lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=log_alpha_lr)

        # data augmentation
        self.aug = aug

        self.train()
        self.critic_target.train()

        if isinstance(self.encoder, ChangingModule):
            self.encoder_change = True
        else:
            self.encoder_change = False
        if isinstance(self.actor, ChangingModule):
            self.actor_change = True
        else:
            self.actor_change = False
        if isinstance(self.critic, ChangingModule):
            self.critic_change = True
        else:
            self.critic_change = False

        self.srank_batches = srank_batches

    def change_modules(self, step):
        metrics = dict()
        if self.encoder_change:
            metrics.update(self.encoder.change(step))
        if self.actor_change:
            metrics.update(self.actor.change(step))
        if self.critic_change:
            metrics.update(self.critic.change(step))
        return metrics

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        # stddev = utils.schedule(self.stddev_schedule, step)
        if eval_mode:
            action = self.actor.mean(obs)
        else:
            dist = self.actor(obs)
            action = dist.sample()
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
                entropy_estimate = U_ENT
            elif self.nstep_entropy_correction == 'saved_entropy':
                log_prob = dist.log_prob(action).sum(-1)
                entropy_estimate = -1*log_prob[0].cpu().item()
            if self.nstep_entropy_correction == 'saved_entropy':
                return (action.cpu().numpy()[0], {'entropy': entropy_estimate})
        return action.cpu().numpy()[0]

    def critic_loss(self, enc_obs, action, reward, discount, enc_next_obs,
                    step, **kwargs):
        with torch.no_grad():
            # stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(enc_next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            if self.nstep_entropy_correction == 'recompute':
                log_prob = log_prob.view(-1, self.nstep, 1)
                enc_next_obs = enc_next_obs.view(-1, self.nstep, self.encoder.repr_dim)[:, -1]
                next_action = next_action.view(-1, self.nstep, self.action_dim)[:, -1]
                correction_value = -1*self.alpha.detach() * (log_prob[:, :-1, 0]*self.entropy_discount_vec[:-1]).sum(dim=-1, keepdims=True)
                log_prob = log_prob[:, -1]
            elif self.nstep_entropy_correction == 'replicate':
                correction_value = -1*self.alpha.detach()*log_prob*self.entropy_discount_correction_coeff
            elif self.nstep_entropy_correction == 'saved_entropy':
                discounted_nstep_entropy_correction = kwargs['discounted_nstep_entropy_correction']
                correction_value = self.alpha.detach()*discounted_nstep_entropy_correction
            else:
                correction_value = 0


            target_V = self.critic_target.get_value(enc_next_obs, next_action)
            target_V = target_V - self.alpha.detach() * log_prob
            target_Q = reward + (discount * target_V) + correction_value

        QS = self.critic(enc_obs, action)
        critic_loss = (QS - target_Q).square().sum(1).mean()
        return critic_loss, QS, target_Q, target_V, next_action, correction_value

    def critic_optim(self, critic_loss):
        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

    def update_critic(self, enc_obs, action, reward, discount, enc_next_obs,
                      step, **kwargs):
        metrics = dict()
        critic_loss, QS, target_Q, target_V, next_action, correction_value = self.critic_loss(
            enc_obs, action, reward, discount, enc_next_obs, step, **kwargs)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            for i in range(self.critic.n_parallel):
                metrics['critic_q{}'.format(i+1)] = QS[..., i].mean().item()
            metrics['critic_loss'] = critic_loss.item()
            if isinstance(correction_value, (int, float)):
                metrics['n_step_entropy_correction'] = correction_value
            else:
                metrics['n_step_entropy_correction'] = correction_value.mean().item()

        self.critic_optim(critic_loss)
        return metrics

    def update_actor_and_log_alpha(self, enc_obs, step):
        metrics = dict()

        dist = self.actor(enc_obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q = self.critic.get_value(enc_obs, action)

        actor_loss = (self.alpha.detach() * log_prob-Q).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = (-log_prob).mean().item()

        if self.anneal_target_entropy:
            target_entropy = utils.schedule(self.target_entropy, step)
            if self.per_dim_target_entropy:
                target_entropy = target_entropy * self.action_dim
        else:
            target_entropy = self.target_entropy
        alpha_loss = (self.alpha * (-log_prob - target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        if self.use_tb:
            metrics['alpha'] = self.alpha.item()
            metrics['alpha_loss'] = alpha_loss.item()
            metrics['target_ent'] = target_entropy.item()
        return metrics

    def update_targets(self,):
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

    def get_data_batch(self, replay_buffer):
        kwargs = dict()
        if self.nstep_entropy_correction == 'recompute':
            batch = replay_buffer.gather_random_all_nstep_batch()
            obs, action, reward, discount, next_obs = utils.to_torch(
                batch, self.device)

        elif self.nstep_entropy_correction == 'saved_entropy':
            batch = next(replay_buffer)
            obs, action, reward, discount, next_obs, discounted_nstep_entropy_correction = utils.to_torch(
                batch, self.device)
            kwargs['discounted_nstep_entropy_correction'] = discounted_nstep_entropy_correction
        else:
            batch = next(replay_buffer)
            obs, action, reward, discount, next_obs = utils.to_torch(
                batch, self.device)
        return obs, action, reward, discount, next_obs, kwargs


    def update(self, replay_buffer, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        metrics.update(self.change_modules(step))
        
        for _ in range(self.gradient_steps_per_update):
            obs, action, reward, discount, next_obs, kwargs = self.get_data_batch(replay_buffer)

            # augment
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
            kwargs['obs'] = obs
            kwargs['next_obs'] = next_obs
            # encode
            enc_obs = self.encoder(obs)
            with torch.no_grad():
                enc_next_obs = self.encoder(next_obs)

            if self.use_tb:
                metrics['batch_reward'] = reward.mean().item()

            # update critic
            metrics.update(
                self.update_critic(enc_obs, action, reward, discount, enc_next_obs, step, **kwargs))

            # update actor
            metrics.update(self.update_actor_and_log_alpha(enc_obs.detach(), step))

            # update critic target
            self.update_targets()

        return metrics

    def calculate_critic_srank(self, replay_buffer, augment=False):
        feat_enc = []
        feat = []
        with torch.no_grad():
            for _ in range(self.srank_batches):
                batch = next(replay_buffer)
                obs, action, _, _, _ = utils.to_torch(batch, self.device)
                if augment:
                    obs = self.aug(obs)
                obs = self.encoder(obs)
                feat_enc.append(obs.cpu().numpy())
                feat.append(utils.get_network_repr_before_final(self.critic, obs, action).cpu().numpy())
            feat_enc = np.concatenate(feat_enc, axis=-2)
            feat = np.concatenate(feat, axis=-2)
            return utils.calculate_feature_srank(feat, delta=0.01), utils.calculate_feature_srank(feat_enc, delta=0.01)
