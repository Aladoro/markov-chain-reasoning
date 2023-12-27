# SSPG for DMC

Minimal implementation of SSPG for pixel-based control on the DeepMind Control suite.

## Instructions

Install [MuJoCo version 2.0](https://www.roboti.us/download.html). For further details about requirements and common issues with this step, see detailed instructions at the  [mujoco-py page](https://github.com/openai/mujoco-py/tree/0711ab58777a28aff847adbf05ba246a337908a0).

Install dependencies via [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html):

```sh
conda env create -f env.yml
conda activate sspg_dmc
```

## Replicating the results

You can run experiments by executing _train.py_ and override the appropriate arguments (see [hydra](https://hydra.cc/docs/intro/) for details), e.g. to run SSPG on _quadruped\_run_:

```sh
python train.py algo=sspg task=quadruped_run
```

## Main dependency licenses

Mujoco and DeepMind Control are licensed under the Apache License, Version 2.0.

_DrQv2_ is licensed under the MIT License.

## Acknowledgements

We would like to thank Denis Yarats for open-sourcing the [DrQv2 codebase](https://github.com/facebookresearch/drqv2). Our implementation builds on top of their repository.