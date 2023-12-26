import numpy as np
import torch

import utils
from drqsac_fsres import SSPGFixedSteps
from fsres_ada_utils import StepsUpdater
U_ENT = np.log(2)
LN4 = np.log(4)


class SSPG(SSPGFixedSteps):
    def __init__(self,
                 auto_steps,
                 auto_steps_max,
                 auto_steps_threshold,
                 auto_steps_min_expl_update,
                 auto_steps_training_update,
                 auto_steps_target_update,
                 *args,
                 **kwargs):

        self.auto_steps = auto_steps
        self.auto_steps_max = auto_steps_max
        self.auto_steps_thresh = auto_steps_threshold
        self.auto_steps_training_update = auto_steps_training_update
        self.auto_steps_target_update = auto_steps_target_update
        self.auto_steps_min_expl_update = auto_steps_min_expl_update
        super(SSPG, self).__init__(*args, **kwargs)

        self.steps_updater = StepsUpdater(init=self.training_steps, decay=0.99)

        self.current_min_expl_steps = torch.tensor(1, dtype=torch.int32, device=self.device)
        self.current_training_steps = torch.tensor(self.training_steps, dtype=torch.int32, device=self.device)
        self.current_target_steps = torch.tensor(self.target_steps, dtype=torch.int32, device=self.device)

        self.last_k = 0
        self.last_final_conv = None

    def update_training_target_steps(self, ):
        metrics = dict()
        if self.auto_steps_min_expl_update:
            self.current_min_expl_steps = self.steps_updater.get_steps_estimate(self.auto_steps_min_expl_update) - 1

        if self.auto_steps_target_update:
            self.current_target_steps = self.steps_updater.get_steps_estimate(self.auto_steps_target_update)

        if self.auto_steps_training_update:
            self.current_training_steps = self.steps_updater.get_steps_estimate(self.auto_steps_training_update)

        if self.use_tb:
            metrics['auto_min_expl_steps'] = self.current_min_expl_steps.item()
            metrics['auto_target_steps'] = self.current_target_steps.item()
            metrics['auto_training_steps'] = self.current_training_steps.item()
            metrics['auto_last_conv_score'] = self.last_final_conv.item()
            metrics['auto_last_k'] = self.last_k
        return metrics

    def calculate_convergence_statistic(self, action_chains,):
        # input - num_chains x current_length x action_dim
        action_chains_shape = action_chains.size()
        action_dim = action_chains_shape[2]
        chain_length = action_chains_shape[1]
        num_chains = action_chains_shape[0]
        if self.auto_steps == 'gelman_rubin' or self.auto_steps == 'impr_gelman_rubin':
            chain_means = action_chains.mean(dim=-2, keepdim=True)  # num_chains x 1 x action_dim
            overall_mean = action_chains.mean(dim=[-3, -2])  # action_dim
            within_chains_deviations = action_chains - chain_means  # num_chains x current_length x action_dim
            # num_chains x current_length x action_dim x action_dim
            within_chains_sample_cov = (within_chains_deviations.unsqueeze(-1) * within_chains_deviations.unsqueeze(-2))
            # num_chains x action_dim x action_dim
            within_chains_sample_cov = within_chains_sample_cov.sum(dim=-3) / (chain_length - 1)

            mean_within_chains_sample_cov = within_chains_sample_cov.mean(dim=-3)  # action_dims x action_dims

            # num_chains x 1 x action_dim
            between_chains_deviations = chain_means - overall_mean

            # num_chains x action_dim x action_dim
            between_chains_sample_cov = between_chains_deviations * between_chains_deviations.squeeze(-2).unsqueeze(-1)
            # action_dim x action_dim
            between_chains_sample_cov = (between_chains_sample_cov.sum(dim=0)) / (num_chains - 1)
            if self.auto_steps == 'gelman_rubin':
                cov_ratio_sol = torch.linalg.lstsq(mean_within_chains_sample_cov,
                                                   between_chains_sample_cov)
                cov_ratio = cov_ratio_sol.solution
                cov_ratio = (cov_ratio + torch.transpose(cov_ratio, 0, 1)) / 2
                try:
                    eigvals = torch.linalg.eigvalsh(cov_ratio)
                    max_eigval = (eigvals).max()
                except Exception as e:
                    print('Exception while computing the eigenvalues at iter {}'.format(chain_length))
                    print(e)
                    max_eigval = torch.tensor(0.0, device=self.device)
                squared_score = (chain_length - 1) / chain_length + max_eigval
                return torch.sqrt(squared_score)
            else:
                det_between_chains_sample_cov = torch.linalg.det(between_chains_sample_cov)
                det_mean_within_chains_sample_cov = torch.linalg.det(mean_within_chains_sample_cov)
                det_ratio = torch.nan_to_num(
                    det_between_chains_sample_cov / det_mean_within_chains_sample_cov)  # sets to 0 rather than nan/inf
                scaled_det_ratio = torch.pow(det_ratio, 1 / action_dim)
                squared_score = (chain_length - 1) / chain_length + scaled_det_ratio
                return torch.sqrt(squared_score)
        else:
            raise NotImplementedError

    def dynamic_action_unroll(self, obs, initial_act, min_steps=0, ):
        current_act = initial_act
        if obs.size()[-2] == 1:
            obs = torch.repeat_interleave(obs, repeats=current_act.size()[0],
                                          dim=-2)
        action_shape = initial_act.size()
        action_chains = torch.empty(action_shape[0], self.auto_steps_max, action_shape[1], device=initial_act.device)
        current_dist = self.actor(obs, current_act)
        current_act = current_dist.sample()
        action_chains[:, 0, :] = current_act
        k = 1
        end_loop = False
        conv = 0.0
        conv_measures = {}
        while not end_loop:
            current_dist = self.actor(obs, current_act)
            current_act = current_dist.sample()
            action_chains[:, k, :] = current_act
            k = k + 1
            if k >= min_steps:
                conv = self.calculate_convergence_statistic(action_chains=action_chains[:, :k])

                conv_measures[k] = conv
                end_loop = ((conv < self.auto_steps_thresh) or (k >= self.auto_steps_max))
            else:
                end_loop = (k >= self.auto_steps_max)
        self.last_final_conv = conv
        self.last_k = k
        return current_act, k, conv, action_chains[:, :k], conv_measures

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        input_chain = self.get_input_chain(eval_mode=eval_mode)
        if eval_mode:
            final_action, k, conv, action_chains, conv_measures = self.dynamic_action_unroll(
                obs=obs, initial_act=input_chain, min_steps=self.current_min_expl_steps)
            self.test_chains = final_action
            if self.deterministic_test_sampling:
                tiled_obs = torch.repeat_interleave(obs, repeats=final_action.size()[0], dim=-2)
                det_action = self.actor.mean(tiled_obs, final_action)
                action_chains = torch.concat([action_chains, det_action.unsqueeze(-2)], dim=-2)
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
                k = 1
            else:
                final_action, k, conv, action_chains, conv_measures = self.dynamic_action_unroll(
                    obs=obs, initial_act=input_chain, min_steps=self.current_min_expl_steps)
                action_chains, final_action = self.sample_chain(
                    steps=self.exploration_steps, obs=obs,
                    initial_act=self.exploration_chains,)
                self.exploration_chains = final_action
                if self.exploration_chain_mode == 'last':
                    action_idx = torch.randint(high=self.num_chains, size=[], device=self.device)
                    action = final_action[action_idx]
                else:
                    raise NotImplementedError
        self.steps_updater.update(k)
        return action.cpu().numpy()

    def critic_loss(self, enc_obs, action, reward, discount, enc_next_obs,
                    step, **kwargs):
        with torch.no_grad():
            if self.target_chain_init == 'rb_action':
                init_act = action
            else:
                raise NotImplementedError
            next_action_chain, next_log_prob_chain = self.actor.get_action_chain_probability(
                enc_next_obs, init_act, self.current_target_steps, chain_backprop=False,
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
            target_V = self.critic_target.get_value(enc_next_obs, next_action)
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
            enc_obs, init_act, self.current_training_steps, chain_backprop=self.chain_backprop,
            perturb_type=self.training_chain_perturb, perturb_prob=self.training_chain_reset_prob, perturb_coeff=self.training_chain_perturb_coeff,)
        if self.training_chain_mode == 'first':
            action, log_prob = act_chain[..., 0, :], log_prob_chain[..., 0, :]
            Q = self.critic.get_value(enc_obs, action)
            #Q = QS.amin(dim=-1, keepdim=True)
            actor_loss = (self.alpha.detach() * log_prob - Q).mean()
        elif self.training_chain_mode == 'all':
            tiled_enc_obs = enc_obs.unsqueeze(-2)  # batch_dims x 1 x obs_dims
            tiled_enc_obs = torch.repeat_interleave(tiled_enc_obs, repeats=act_chain.size()[-2], dim=-2)  # batch_dims x chain_length x obs_dims
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

        metrics.update(self.update_training_target_steps())
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