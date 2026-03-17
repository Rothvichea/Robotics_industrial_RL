"""
KR6 R900 — PPO training script
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from kr6_env import KR6Env

LOG_DIR   = os.path.expanduser("~/Robotics_industrial_RL/rl_logs")
MODEL_DIR = os.path.expanduser("~/Robotics_industrial_RL/rl_models")
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

N_ENVS   = 8
env      = make_vec_env(KR6Env, n_envs=N_ENVS)
eval_env = Monitor(KR6Env())

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1,
)

checkpoint_cb = CheckpointCallback(
    save_freq=50_000,
    save_path=MODEL_DIR,
    name_prefix="kr6_ppo",
)

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=LOG_DIR,
    verbose=1,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)

print("=" * 50)
print("KR6 PPO Training")
print(f"  Envs:      {N_ENVS}")
print(f"  Obs:       {env.observation_space.shape}")
print(f"  Actions:   {env.action_space.shape}")
print(f"  Log dir:   {LOG_DIR}")
print(f"  Model dir: {MODEL_DIR}")
print("=" * 50)

model.learn(
    total_timesteps=500_000,
    callback=[eval_cb, checkpoint_cb],
    progress_bar=True,
)

model.save(os.path.join(MODEL_DIR, "kr6_ppo_final"))
print("Training complete — model saved.")
