# SSPG for Mujoco

Minimal repository.
A complete, documented version of our implementation will be open-sourced after review.

## Requirements

1) To replicate the experiments in this project you need to install the Mujoco
simulation software with a valid license. You can find instructions [here](https://github.com/openai/mujoco-py).

2) The rest of the requirements can be installed with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html),
by utilizing the provided environment file:

```setup
conda env create -f env.yml
conda activate sspg_mj
```

## Replicating the results

You can run experiments by running _main_hydra.py_ and override the appropriate arguments (see [hydra](https://hydra.cc/docs/intro/) for details), e.g. to run SSPG on _Humanoid-v2_:

```setup
python main_hydra.py agent=sspg task_name=Humanoid-v2
```

## Main dependency licenses

Mujoco is licensed under the Apache License, Version 2.0.