#import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from drqv2 import Encoder

import utils

class NormEncoder(Encoder):
    def __init__(self, obs_shape, norm_layers=[None, None, None, None], pretrained=False):
        nn.Module.__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        layers = [nn.Conv2d(obs_shape[0], 32, 3, stride=2)]
        assert len(norm_layers) == 4
        for nl in norm_layers[:-1]:
            if nl is not None:
                layers.append(nl)
            layers += [nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1)]
        if norm_layers[-1] is not None:
            layers.append(norm_layers[-1])
        layers.append(nn.ReLU())
        self.convnet = nn.Sequential(*layers)

        self.apply(utils.weight_init)

        if pretrained:
            pretrained_agent = torch.load(pretrained)
            self.load_state_dict(pretrained_agent.encoder.state_dict())
