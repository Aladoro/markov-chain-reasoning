# SSPG for Mujoco

Minimal implementation of SSPG for proprioceptive control on OpenAI Gym Mujoco environments.

## Instructions

Install [MuJoCo version 2.0](https://www.roboti.us/download.html). For further details about requirements and common issues with this step, see detailed instructions at the  [mujoco-py page](https://github.com/openai/mujoco-py/tree/0711ab58777a28aff847adbf05ba246a337908a0).

Install dependencies via [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html):

```sh
conda env create -f env.yml
conda activate sspg_mj
```

The code should be compatible but has not been tested with later versions of MuJoCo that do not require mujoco-py.

## Replicating the results

You can run experiments by running _main_hydra.py_ and override the appropriate arguments (see [hydra](https://hydra.cc/docs/intro/) for details), e.g. to run SSPG on _Humanoid-v2_:

```sh
python main_hydra.py agent=sspg task_name=Humanoid-v2
```

## Main dependency licenses

Mujoco is licensed under the Apache License, Version 2.0.