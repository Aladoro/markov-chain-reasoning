import copy
import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd

import utils
from drqv2 import RandomShiftsAug, ChangingModule
from drqsac import StochasticActor, DrQSACAgent, Critic, ParallelEncoder
from drqsac_fsres import ReasoningStochasticActor
from truncated_normal import ActualTruncatedNormal

U_ENT = np.log(2)
LN4 = np.log(4)


class TruncReasoningStochasticActor(ReasoningStochasticActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mean(self, obs, act):
        obs = self.process_obs(obs)
        h = self.trunk(obs)
        h = torch.concat([h, act], dim=-1)
        out = self.policy(h)
        if self.residual:
            out = torch.tanh(out)
            mean, log_std, mixing = torch.tensor_split(out, (self.action_dim, self.action_dim*2), dim=-1)
            mixing = (1+mixing)/2
            mu = mean*mixing + act * (1-mixing)
        else:
            mean, log_std = torch.tensor_split(out, 2, dim=-1)
            mu = torch.tanh(mean)
        return mu

    def mean_std(self, obs, act):
        obs = self.process_obs(obs)
        h = self.trunk(obs)
        h = torch.concat([h, act], dim=-1)
        out = self.policy(h)
        out = torch.tanh(out)
        if self.residual:
            mean, log_std, mixing = torch.tensor_split(out, (self.action_dim, self.action_dim*2), dim=-1)
            mixing = (1 + mixing) / 2
            mean = mean * mixing + act * (1 - mixing)
        else:
            mean, log_std = torch.tensor_split(out, 2, dim=-1)
        log_std = (log_std + 1)/2*self.log_std_range + self.min_log_std
        return mean, torch.exp(log_std)

    def forward(self, obs, act):
        mean, std = self.mean_std(obs, act)
        dist = ActualTruncatedNormal(mean, std, a=-1.0, b=1.0)
        return dist

    def get_action_chain_probability(self, obs, act, chain_length, chain_backprop,
                                     perturb_type, perturb_prob, perturb_coeff):
        all_mean = []
        all_stddev = []
        all_actions = []
        current_act = act
        for _ in range(chain_length):
            if not chain_backprop:
                input_act = current_act.clone().detach()
            else:
                input_act = current_act
            if perturb_prob and perturb_type:
                input_act = self.perturb_actions(act=input_act, noise_type=perturb_type,
                                                     prob=perturb_prob, coeff=perturb_coeff)
            mean, stddev = self.mean_std(obs, input_act)
            dist = ActualTruncatedNormal(mean, stddev, a=-1.0, b=1.0)
            current_act = dist.rsample()
            all_mean.append(mean)
            all_stddev.append(stddev)
            all_actions.append(current_act)
        all_mean = torch.stack(all_mean, dim=-2) # batch_dims x chain_length x action_size
        all_stddev = torch.stack(all_stddev, dim=-2) # batch_dims x chain_length x action_size
        all_actions = torch.stack(all_actions, dim=-2)  # batch_dims x chain_length x action_size

        joint_distribution = ActualTruncatedNormal(all_mean.unsqueeze(-3), all_stddev.unsqueeze(-3), a=-1.0, b=1.0)

        all_log_probs = joint_distribution.log_prob(all_actions.unsqueeze(-2)).sum(-1) # batch_dims x chain_length x chain_length - for each action - chain_length log prob predictions
        log_probs = torch.logsumexp(all_log_probs, dim=-1, keepdim=True) - torch.log(torch.tensor(chain_length, device=all_log_probs.device, dtype=torch.float32))
        return all_actions, log_probs # batch_dims x chain_length x (action_dims, 1)
