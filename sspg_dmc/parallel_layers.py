import math
import numbers
import os
import random
from collections import deque
from typing import Tuple

import numpy as np
import scipy.linalg as sp_la

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations
from skimage.util.shape import view_as_windows
from torch import distributions as pyd
from parallel_layers_sn import parallel_spectral_norm


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def soft_update_params_verbose(net, target_net, tau):
    for (name, param), (target_name, target_param) in zip(net.named_parameters(), target_net.named_parameters()):
        print('{} is copied into {}'.format(name, target_name))
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

def parallel_orthogonal_(tensor, gain=1):
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() //n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()

        # Compute the qr factorization
        q, r = torch.linalg.qr(flat_tensor)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if rows < cols:
            q.t_()
        qs.append(q)

    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain('relu')
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    else:
        print('Not applying custom init to layer {}'.format(m))


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


class DenseParallel(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_parallel: int,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DenseParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        self.weight = nn.Parameter(torch.empty((n_parallel, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((n_parallel, 1, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return th.matmul(input, self.weight) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, n_parallel={}, bias={}'.format(
            self.in_features, self.out_features, self.n_parallel, self.bias is not None
        )


class ParallelLayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, n_parallel, normalized_shape, eps=1e-5, elementwise_affine=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [normalized_shape, ]
        assert len(normalized_shape) == 1
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
            self.bias = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        norm_input = F.layer_norm(
            input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        else:
            return norm_input

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


def parallel_mlp(input_dim, hidden_dim, output_dim, n_parallel, hidden_depth, output_mod=None, hidden_sn=False):
    if hidden_depth == 0:
        mods = [DenseParallel(input_dim, output_dim, n_parallel)]
    else:
        mods = [DenseParallel(input_dim, hidden_dim, n_parallel), nn.ReLU(inplace=True)]

        def make_hidden():
            l = DenseParallel(hidden_dim, hidden_dim, n_parallel)
            if hidden_sn:
                return parallel_spectral_norm(l)
            else:
                return l

        for i in range(hidden_depth - 1):
            mods += [make_hidden(), nn.ReLU(inplace=True)]
        mods.append(DenseParallel(hidden_dim, output_dim, n_parallel))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class ModernResidualBlock(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim,
                 layer_normalization=True, rescale_ln=True, spectral_normalization=True,
                 n_parallel=None):
        super(ModernResidualBlock, self).__init__()
        self._id = input_dim
        self._od = output_dim
        self._bd = bottleneck_dim
        self._ln = layer_normalization
        self._rln = rescale_ln
        self._sn = spectral_normalization
        self._np = n_parallel

        def make_fc(in_features, out_features):
            if self._np is not None:
                l = DenseParallel(in_features=in_features,
                                  out_features=out_features,
                                  n_parallel=n_parallel)
                if self._sn:
                    l = parallel_spectral_norm(l)
            else:
                l = nn.Linear(in_features=in_features,
                              out_features=out_features)
                if self._sn:
                    l = nn.utils.spectral_norm(l)
            return l

        if self._id != self._od:
            self._short = make_fc(self._id, self._od)
        else:
            self._short = None

        res_layers = []

        if self._ln:
            if n_parallel is not None:
                res_layers.append(ParallelLayerNorm(n_parallel, [self._id], elementwise_affine=self._rln))
            else:
                res_layers.append(nn.LayerNorm([self._id], elementwise_affine=self._rln))

        res_layers += [make_fc(self._id, self._bd), nn.ReLU(inplace=True),
                       make_fc(self._bd, self._od)]
        self._res = nn.Sequential(*res_layers)

    def forward(self, input):
        res_out = self._res(input)
        if self._short is not None:
            id = self._short(input)
        else:
            id = input
        return id + res_out

def modern_mlp(input_dim, hidden_dim, bottleneck_dim, output_dim, n_blocks, output_mod=None, hidden_sn=False,
               rescale_ln=True):
    mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
    mods += [ModernResidualBlock(input_dim=hidden_dim, bottleneck_dim=bottleneck_dim, output_dim=hidden_dim,
                                 layer_normalization=True, rescale_ln=rescale_ln, spectral_normalization=hidden_sn,
                                 n_parallel=None)
             for _ in range(n_blocks)]
    mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def modern_parallel_mlp(input_dim, hidden_dim, bottleneck_dim, output_dim, n_parallel, n_blocks, output_mod=None, hidden_sn=False,
                        rescale_ln=True):
    mods = [DenseParallel(input_dim, hidden_dim, n_parallel), nn.ReLU(inplace=True)]
    mods += [ModernResidualBlock(input_dim=hidden_dim, bottleneck_dim=bottleneck_dim, output_dim=hidden_dim,
                                layer_normalization=True, rescale_ln=rescale_ln, spectral_normalization=hidden_sn,
                                 n_parallel=n_parallel)
             for _ in range(n_blocks)]
    mods.append(DenseParallel(hidden_dim, output_dim, n_parallel))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk