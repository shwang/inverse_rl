# Inverse RL

Implementations for imitation learning / IRL algorithms in RLLAB

Contains:
- GAIL (https://arxiv.org/abs/1606.03476/pdf)
- Guided Cost Learning, GAN formulation (https://arxiv.org/pdf/1611.03852.pdf)
- Tabular MaxCausalEnt IRL (http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf)

Setup Instructions (Linux)
---
### Install Anaconda for Linux
https://www.anaconda.com/distribution/#linux

#### Recommended: Disable automatic base conda environment activation.
Prevents confusion from installing to (base) instead. Also, makes sure
that any non-Conda setups aren't being automatically shadowed by (base)
everytime you load `~/.bashrc`.
```bash
source ~/.bashrc
conda config --set auto_activate_base false
conda deactivate
```

### Install rllab and dependencies into `rllab3` conda environment
```bash
git submodule init
git submodule update
pushd rllab
./chai_setup.sh
popd
```

### Activate rllab3 conda env
```bash
conda activate rllab3
```

### Install `mujoco_py` (Use rllab3 env)
https://github.com/openai/mujoco-py#install-mujoco

You might see a "cffi ImportError". This seems to be fine...

### Install `inverse_rl`  (Use rllab3 env)
```
pip install -e .
```


Examples
---

Running the Pendulum-v0 gym environment (Use rllab3 env):

1) Collect expert data
```
python scripts/pendulum_data_collect.py
```

You should get an "AverageReturn" of around -100 to -150

2) Run imitation learning
```
python scripts/pendulum_gcl.py
```

The "OriginalTaskAverageReturn" should reach around -100 to -150
