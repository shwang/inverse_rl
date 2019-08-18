# Inverse RL

Implementations for imitation learning / IRL algorithms in RLLAB

Contains:
- GAIL (https://arxiv.org/abs/1606.03476/pdf)
- Guided Cost Learning, GAN formulation (https://arxiv.org/pdf/1611.03852.pdf)
- Tabular MaxCausalEnt IRL (http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf)

Setup Instructions (Linux)
---

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
