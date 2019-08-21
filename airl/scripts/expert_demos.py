import datetime
import math
import os.path as osp

import sacred
from sacred.observers import FileStorageObserver
import tensorflow as tf

from airl.algos.trpo import TRPO
from airl.models.tf_util import get_session_config
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from rllab.spaces import Box
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv

from airl.envs.env_utils import CustomGymEnv
from airl.utils.log_utils import rllab_logdir
from airl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial

from airl.scripts.expert_demos_config import expert_demos_ex


@expert_demos_ex.main
def main(log_dir,
         env_name,
         ent_coef,
         n_steps,
         total_timesteps,
         num_vec,
         ):
    tf.reset_default_graph()
    # n_steps is the `batch_size // num_vec` in `imitation`.
    batch_size = n_steps * num_vec
    n_itr = int(math.ceil(total_timesteps / batch_size))

    if env_name.startswith("airl/"):
        env_cls = CustomGymEnv
    else:
        env_cls = GymEnv
    env = TfEnv(env_cls(env_name, record_video=False, record_log=False))

    # NOTE: Haven't yet checked if hidden_sizes=(32, 32) matches the settings in
    # the `imitation` repo. We use the default Stable Baselines MLP policy.
    if isinstance(env.spec.action_space, Box):
        policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    else:
        policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))

    with tf.Session(config=get_session_config()) as sess:
        algo = TRPO(
            env=env,
            policy=policy,
            n_itr=n_itr,

            batch_size=batch_size,
            max_path_length=500,
            discount=0.99,
            store_paths=True,
            entropy_weight=ent_coef,
            baseline=LinearFeatureBaseline(env_spec=env.spec),
            # Maybe it will be the case the not every policy is compatible with
            # the VectorizedSampler. In that case, consider changing to
            # `sampler_cls=None` and adding a dummy `n_envs` kwargs to BatchSampler.
            sampler_cls=VectorizedSampler,
            sampler_args=dict(n_envs=num_vec),
        )
        with rllab_logdir(algo=algo, dirname=log_dir):
            algo.train(sess)


def main_console():
    observer = FileStorageObserver.create(
        osp.join('output', 'sacred', 'expert_demos'))
    expert_demos_ex.observers.append(observer)
    expert_demos_ex.run_commandline()


if __name__ == "__main__":
    main_console()
