"""
KR6 R900 — PPO evaluation script
Loads trained model and runs it on the environment
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from stable_baselines3 import PPO
from kr6_env import KR6Env, GOALS

MODEL_PATH = os.path.expanduser("~/Robotics_industrial_RL/rl_models/best_model.zip")

env   = KR6Env()
model = PPO.load(MODEL_PATH, env=env)

print("=" * 50)
print("KR6 PPO Evaluation")
print("=" * 50)

n_episodes  = 20
successes   = 0
total_steps = []
total_rewards = []
total_jerks   = []

for ep in range(n_episodes):
    obs, _ = env.reset()
    done   = False
    ep_reward = 0
    ep_steps  = 0
    ep_jerks  = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_steps  += 1
        ep_jerks.append(info["jerk"])
        done = terminated or truncated

    if info["reached"]:
        successes += 1

    total_steps.append(ep_steps)
    total_rewards.append(ep_reward)
    total_jerks.append(np.mean(ep_jerks))

    status = "✓ REACHED" if info["reached"] else "✗ TIMEOUT"
    print(f"  Ep {ep+1:2d}: {status} | steps={ep_steps:3d} | "
          f"reward={ep_reward:6.1f} | "
          f"final_dist={info['dist']:.4f} | "
          f"jerk={np.mean(ep_jerks):.4f}")

print("=" * 50)
print(f"Success rate:   {successes}/{n_episodes} ({100*successes/n_episodes:.0f}%)")
print(f"Avg steps:      {np.mean(total_steps):.1f}")
print(f"Avg reward:     {np.mean(total_rewards):.1f}")
print(f"Avg jerk:       {np.mean(total_jerks):.4f}")
print("=" * 50)
