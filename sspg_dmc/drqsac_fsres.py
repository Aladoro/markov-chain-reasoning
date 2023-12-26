
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd

import utils
from drqsac import StochasticActor, DrQSACAgent, ParallelEncoder

U_ENT = np.log(2)
LN4 = np.log(4)

class ReasoningStochasticActor(StochasticActor):
    def __init__(self, encoder, action_shape, feature_dim, hidden_dim, min_log_std, max_log_std, residual=False):
        super().__init__(encoder, action_shape, feature_dim, hidden_dim, min_log_std, max_log_std)

        self.parallel_encoder = isinstance(encoder, ParallelEncoder)
        if self.parallel_encoder:
            self.trunk = nn.Sequential(nn.Linear(encoder.repr_dim * encoder.num_encoders, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
        else:
            self.trunk = nn.Sequential(nn.Linear(encoder.repr_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())

        self.action_dim = action_shape[0]

        self.residual = residual

        if self.residual:
            out_dims = self.action_dim*2 + 1
        else:
            out_dims = self.action_dim * 2
        self.policy = nn.Sequential(nn.Linear(feature_dim+self.action_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, out_dims))

        self.make_frozen_policy()
        self.apply(utils.weight_init)

        self.min_log_std, self.max_log_std = min_log_std, max_log_std
        self.log_std_range = self.max_log_std - self.min_log_std

    def make_frozen_policy(self,):
        self.frozen_trunk = copy.deepcopy(self.trunk)
        self.frozen_policy = copy.deepcopy(self.policy)
        for param in self.frozen_trunk.parameters():
            param.requires_grad = False
        for param in self.frozen_policy.parameters():
            param.requires_grad = False

    def update_frozen_policy(self,):
        with torch.no_grad():
            for frozen_param, param in zip(self.frozen_policy.parameters(), self.policy.parameters()):
                frozen_param.data.copy_(param.data)
            for frozen_param, param in zip(self.frozen_trunk.parameters(), self.trunk.parameters()):
                frozen_param.data.copy_(param.data)

    def mean(self, obs, act):
        if self.residual:
            raise NotImplementedError
        obs = self.process_obs(obs)
        h = self.trunk(obs)
        h = torch.concat([h, act], dim=-1)
        out = self.policy(h)
        mean, log_std = torch.tensor_split(out, 2, dim=-1)
        mu = torch.tanh(mean)
        return mu

    def mean_std(self, obs, act):
        if self.residual:
            raise NotImplementedError
        obs = self.process_obs(obs)
        h = self.trunk(obs)
        h = torch.concat([h, act], dim=-1)
        out = self.policy(h)
        mean, log_std = torch.tensor_split(out, 2, dim=-1)
        log_std = (torch.tanh(log_std) + 1)/2*self.log_std_range + self.min_log_std
        return mean, torch.exp(log_std)

    def mean_std_frozen(self, obs, act):
        if self.residual:
            raise NotImplementedError
        obs = self.process_obs(obs)
        h = self.frozen_trunk(obs)
        h = torch.concat([h, act], dim=-1)
        out = self.frozen_policy(h)
        mean, log_std = torch.tensor_split(out, 2, dim=-1)
        log_std = (torch.tanh(log_std) + 1)/2*self.log_std_range + self.min_log_std
        return mean, torch.exp(log_std)

    def forward(self, obs, act):
        mean, std = self.mean_std(obs, act)
        dist = utils.SquashedNormal(mean, std)
        return dist

    def perturb_actions(self, act, noise_type, prob, coeff):
        if noise_type:
            random_samples = torch.rand(size=[act.size()[0], 1], device=act.device)
            resets = random_samples < prob
            if noise_type == 'uniform':
                assert coeff <= 1.0 and coeff > 0.0
                random_actions = torch.rand(size=act.size(), device=act.device)
                random_actions = act * (1 - coeff) + random_actions * coeff
            elif noise_type == 'gaussian':
                assert coeff > 0.0
                random_actions = act + torch.randn(size=act.size(), device=act.device)*coeff
                random_actions = torch.clamp_(random_actions, min=-1, max=1)
            else:
                raise NotImplementedError
            input_act = torch.where(resets, random_actions, act)
            return input_act
        else:
            return act

    def get_action_chain_probability(self, obs, act, chain_length, chain_backprop,
                                     perturb_type, perturb_prob, perturb_coeff): # Unbiased
        all_mean = []
        all_stddev = []
        all_raw_actions = []
        all_actions = []
        current_act = act
        if type(chain_backprop) is int:
            self.update_frozen_policy()
        for curr_iter in range(chain_length):
            if not chain_backprop:
                input_act = current_act.clone().detach()
            else:
                input_act = current_act
            if perturb_prob and perturb_type:
                input_act = self.perturb_actions(act=input_act, noise_type=perturb_type,
                                                     prob=perturb_prob, coeff=perturb_coeff)

            if type(chain_backprop) is int:
                if curr_iter >= int(chain_backprop):
                    mean, stddev = self.mean_std_frozen(obs, input_act)
                else:
                    mean, stddev = self.mean_std(obs, input_act)
            else:
                mean, stddev = self.mean_std(obs, input_act)

            random_component = torch.randn(size=mean.size(), device=mean.device)
            raw_act = mean + random_component * stddev
            current_act = torch.tanh(raw_act)
            all_mean.append(mean)
            all_stddev.append(stddev)
            all_raw_actions.append(raw_act)
            all_actions.append(current_act)
        all_mean = torch.stack(all_mean, dim=-2) # batch_dims x chain_length x action_size
        all_stddev = torch.stack(all_stddev, dim=-2) # batch_dims x chain_length x action_size
        all_raw_actions = torch.stack(all_raw_actions, dim=-2) # batch_dims x chain_length x action_size
        all_actions = torch.stack(all_actions, dim=-2)  # batch_dims x chain_length x action_size

        joint_distribution = pyd.Normal(all_mean.unsqueeze(-3), all_stddev.unsqueeze(-3))

        all_log_probs = joint_distribution.log_prob(all_raw_actions.unsqueeze(-2)).sum(-1) # batch_dims x chain_length x chain_length - for each action - chain_length log prob predictions
        squash_features = -2 * all_raw_actions  # batch_dims x chain_length x action_size
        squash_correction = (LN4 + squash_features - 2 * F.softplus(squash_features)).sum(dim=-1, keepdims=True) # batch_dims x chain_length x 1
        all_log_probs -= squash_correction # batch_dims x chain_length x chain_length

        log_probs = torch.logsumexp(all_log_probs, dim=-1, keepdim=True) - torch.log(torch.tensor(chain_length, device=all_log_probs.device, dtype=torch.float32))
        return all_actions, log_probs # batch_dims x chain_length x (action_dims, 1)


class SSPGFixedSteps(DrQSACAgent):
    def __init__(self,
                 num_chains,
                 training_chain_init,
                 target_chain_init,
                 training_chain_mode,
                 target_chain_mode,
                 exploration_chain_mode,
                 training_steps,
                 target_steps,
                 exploration_steps,
                 test_steps,
                 training_chain_perturb,
                 training_chain_perturb_prob,
                 training_chain_perturb_coeff,
                 deterministic_test_sampling,
                 *args,
                 random_chain_init=False,
                 chain_backprop=False,
                 **kwargs):

        super(SSPGFixedSteps, self).__init__(*args, **kwargs)
        assert isinstance(self.actor, ReasoningStochasticActor)
        assert training_chain_init in ['rb_action']
        assert target_chain_init in ['rb_action']
        assert training_chain_mode in ['all', 'first']
        assert target_chain_mode in ['last']
        assert exploration_chain_mode in ['last']
        if training_chain_perturb:
            assert training_chain_perturb in ['uniform', 'gaussian']
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
        self.training_chain_reset_prob = float(training_chain_perturb_prob)
        self.training_chain_perturb = training_chain_perturb
        self.training_chain_perturb_coeff = float(training_chain_perturb_coeff)
        self.deterministic_test_sampling = deterministic_test_sampling
        self.random_chain_init = random_chain_init
        self.chain_backprop = chain_backprop
        self.exploration_chains = torch.rand([num_chains, self.actor.action_dim],
                                             device=self.device)*2-1
        self.test_chains = torch.rand([num_chains, self.actor.action_dim],
                                      device=self.device, )*2-1

    def sample_chain(self, steps, obs, initial_act):
        action_chains = []
        current_act = initial_act
        if obs.size()[-2] == 1:
            obs = torch.repeat_interleave(obs, repeats=current_act.size()[0],
                                          dim=-2)
        for s_idx in range(steps):
            try:
                current_dist = self.actor(obs, current_act)
            except ValueError as err:
                print('STEP IDX')
                print(s_idx)
                raise err
            current_act = current_dist.sample()
            action_chains.append(current_act)
        action_chains = torch.stack(action_chains, dim=1) # bs x steps x act_dim
        return action_chains, current_act

    def evaluate_actions(self, obs, act):
        action_batch_dims = act.size()[0]
        if obs.size()[-2] == 1:
            obs = torch.repeat_interleave(obs, repeats=action_batch_dims, dim=-2)
        return self.critic.get_value(obs, act)

    def get_random_chain(self,):
        return torch.rand([self.num_chains, self.actor.action_dim],
                           device=self.device, )*2-1

    def get_input_chain(self, eval_mode):
        if self.random_chain_init:
            return self.get_random_chain()
        elif eval_mode:
            return self.test_chains
        else:
            return self.exploration_chains

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        input_chain = self.get_input_chain(eval_mode=eval_mode)
        if eval_mode:
            action_chains, final_action = self.sample_chain(
                steps=self.test_steps, obs=obs,
                initial_act=input_chain, )
            self.test_chains = final_action
            if self.deterministic_test_sampling:
                flat_action_chains = torch.flatten(action_chains, start_dim=0, end_dim=-2)
                q_pred = self.evaluate_actions(obs, flat_action_chains)
                best_action_idx = torch.argmax(q_pred[:, 0])
                action = flat_action_chains[best_action_idx]
            elif self.exploration_chain_mode == 'last':
                action_idx = torch.randint(high=self.num_chains, size=[], device=self.device)
                action = final_action[action_idx]
            else:
                raise NotImplementedError
        else:
            if step < self.num_expl_steps:
                action = torch.rand([self.actor.action_dim])*2-1
            else:
                action_chains, final_action = self.sample_chain(
                    steps=self.exploration_steps, obs=obs,
                    initial_act=input_chain,)
                self.exploration_chains = final_action
                if self.exploration_chain_mode == 'last':
                    action_idx = torch.randint(high=self.num_chains, size=[], device=self.device)
                    action = final_action[action_idx]
                else:
                    raise NotImplementedError
        return action.cpu().numpy()

    def critic_loss(self, enc_obs, action, reward, discount, enc_next_obs,
                    step, **kwargs):
        with torch.no_grad():
            if self.target_chain_init == 'rb_action':
                init_act = action
            else:
                raise NotImplementedError
            next_action_chain, next_log_prob_chain = self.actor.get_action_chain_probability(
                enc_next_obs, init_act, self.target_steps, chain_backprop=False,
                perturb_type=None, perturb_prob=0, perturb_coeff=1.0,)  # batch_dims x chain_length x (action_dims, 1)
            if self.target_chain_mode == 'last':
                next_action = next_action_chain[..., -1, :]  # batch_dims x action_dims
                log_prob = next_log_prob_chain[..., -1, :]  # batch_dims x 1
            else:
                raise NotImplementedError

            if self.nstep_entropy_correction:
                raise NotImplementedError
            else:
                correction_value = 0
            target_V = self.critic_target.get_value(enc_next_obs, next_action) #target_QS.amin(dim=1, keepdim=True)
            target_V = target_V - self.alpha.detach() * log_prob
            target_Q = reward + (discount * target_V) + correction_value
        QS = self.critic(enc_obs, action)
        critic_loss = (QS - target_Q).square().sum(1).mean()
        return critic_loss, QS, target_Q, target_V, next_action, correction_value

    def update_actor_and_log_alpha(self, enc_obs, act, step):
        metrics = dict()

        if self.training_chain_init == 'rb_action':
            init_act = act
        else:
            raise NotImplementedError
        # batch_dims x chain_length x (action_dims, 1)
        act_chain, log_prob_chain = self.actor.get_action_chain_probability(
            enc_obs, init_act, self.training_steps, chain_backprop=self.chain_backprop,
            perturb_type=self.training_chain_perturb, perturb_prob=self.training_chain_reset_prob, perturb_coeff=self.training_chain_perturb_coeff,)
        if self.training_chain_mode == 'first':
            action, log_prob = act_chain[..., 0, :], log_prob_chain[..., 0, :]
            Q = self.critic.get_value(enc_obs, action)
            #Q = QS.amin(dim=-1, keepdim=True)
            actor_loss = (self.alpha.detach() * log_prob - Q).mean()
        elif self.training_chain_mode == 'all':
            tiled_enc_obs = enc_obs.unsqueeze(-2)  # batch_dims x 1 x obs_dims
            tiled_enc_obs = torch.repeat_interleave(tiled_enc_obs, repeats=self.training_steps, dim=-2)  # batch_dims x chain_length x obs_dims
            # batch_dims x chain_length x (1, 1)
            # batch_dims x chain_length
            Q = self.critic.get_value(
                torch.flatten(tiled_enc_obs, start_dim=-3, end_dim=-2),
                torch.flatten(act_chain, start_dim=0, end_dim=-2))
            #Q = QS.amin(dim=-1, keepdim=True)
            # Expectation averaged across batch and across chain
            log_prob = torch.flatten(log_prob_chain, start_dim=0, end_dim=-2)
            actor_loss = (self.alpha.detach() * log_prob - Q).mean()
        else:
            raise NotImplementedError
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
            metrics.update(self.update_actor_and_log_alpha(enc_obs.detach(), action, step))

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
