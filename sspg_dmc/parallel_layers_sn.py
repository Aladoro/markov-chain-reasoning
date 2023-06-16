import torch
from torch.nn.functional import normalize
from typing import Any, Optional, TypeVar
from torch.nn import Module


class ParallelSpectralNorm:

    _version: int = 1
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 1, eps: float = 1e-12,
                 n_parallel: int = 5) -> None:
        self.name = name
        self.dim = dim
        assert self.dim != 0, 'first dim is for parallel weights'
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.n_parallel = n_parallel

    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        return weight

    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u') # n_par x 1 x m
        v = getattr(module, self.name + '_v') # n_par x 1 x n
        weight_mat = self.reshape_weight_to_matrix(weight) # n_par x m x n

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v = normalize(torch.matmul(u, weight_mat), dim=-1, eps=self.eps, out=v) # n_par x 1 x n
                    u = normalize(torch.matmul(v, weight_mat.permute(0, 2, 1)), dim=-1, eps=self.eps, out=u) # n_par x 1 x m
                if self.n_power_iterations > 0:
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.matmul(torch.matmul(u, weight_mat), v.permute(0, 2, 1)) # n_par x 1 x 1
        weight = weight / sigma
        return weight

    def remove(self, module: Module) -> None:
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))


    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float, n_parallel: int) -> 'ParallelSpectralNorm':
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, ParallelSpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = ParallelSpectralNorm(name, n_power_iterations, dim, eps, n_parallel)
        weight = module._parameters[name]
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError(
                'The module passed to `SpectralNorm` can\'t have uninitialized parameters. '
                'Make sure to run the dummy forward before applying spectral normalization')

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            n_par, h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(n_par, 1, h).normal_(0, 1), dim=-1, eps=fn.eps)
            v = normalize(weight.new_empty(n_par, 1, w).normal_(0, 1), dim=-1, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(ParallelSpectralNormStateDictHook(fn))
        return fn


class ParallelSpectralNormStateDictHook:
    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError("Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata['spectral_norm'][key] = self.fn._version


T_module = TypeVar('T_module', bound=Module)

def parallel_spectral_norm(module: T_module,
                           name: str = 'weight',
                           n_power_iterations: int = 1,
                           eps: float = 1e-12,
                           dim: Optional[int] = None,
                           n_parallel: int = 5) -> T_module:
    if dim is None:
            dim = 1
    ParallelSpectralNorm.apply(module, name, n_power_iterations, dim, eps, n_parallel)
    return module



def remove_parallel_spectral_norm(module: T_module, name: str = 'weight') -> T_module:
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, ParallelSpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    else:
        raise ValueError("spectral_norm of '{}' not found in {}".format(
            name, module))

    for k, hook in module._state_dict_hooks.items():
        if isinstance(hook, ParallelSpectralNormStateDictHook) and hook.fn.name == name:
            del module._state_dict_hooks[k]
            break

    return module