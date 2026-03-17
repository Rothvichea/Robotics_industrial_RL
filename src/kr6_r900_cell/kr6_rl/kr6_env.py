"""
KR6 R900 — Gymnasium RL Environment
State:  joint positions (6) + joint velocities (6) + goal positions (6) = 18
Action: joint position deltas (6) — continuous, clipped to step size
Reward: goal reach bonus + smoothness − jerk − singularity penalty
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


JOINT_LIMITS = np.array([
    [-2.96706,  2.96706],
    [-3.31613,  0.78540],
    [-2.09440,  2.72271],
    [-3.22886,  3.22886],
    [-2.09440,  2.09440],
    [-6.10865,  6.10865],
])

MAX_VEL = np.array([6.283, 5.236, 6.283, 7.854, 7.854, 9.425])
HOME    = np.array([0.0, -1.5708, 1.5708, 0.0, 0.0, 0.0])

GOALS = [
    np.array([0.0,    -1.5708,  1.5708, 0.0,  0.0,  0.0]),
    np.array([0.785,  -1.0,     1.2,    0.5,  0.3,  0.0]),
    np.array([-0.785, -1.2,     1.5,   -0.5,  0.5,  0.0]),
    np.array([0.0,     0.0,     0.0,    0.0,  0.0,  0.0]),
    np.array([0.5,    -1.2,     1.8,    0.3, -0.3,  0.5]),
]

N_JOINTS    = 6
MAX_STEPS   = 200
STEP_SIZE   = 0.05
GOAL_THRESH = 0.05


class KR6Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_JOINTS * 3,), dtype=np.float32)
        self.joint_pos  = HOME.copy()
        self.joint_vel  = np.zeros(N_JOINTS)
        self.goal       = HOME.copy()
        self.prev_pos   = HOME.copy()
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.joint_pos  = HOME.copy() + np.random.uniform(-0.1, 0.1, N_JOINTS)
        self.joint_pos  = np.clip(self.joint_pos, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        self.joint_vel  = np.zeros(N_JOINTS)
        self.prev_pos   = self.joint_pos.copy()
        self.goal       = GOALS[np.random.randint(len(GOALS))].copy()
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        self.step_count += 1
        self.prev_pos   = self.joint_pos.copy()
        delta           = np.clip(action, -1.0, 1.0) * STEP_SIZE
        new_pos         = np.clip(self.joint_pos + delta, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        self.joint_vel  = new_pos - self.joint_pos
        self.joint_pos  = new_pos

        dist        = np.mean(np.abs(self.joint_pos - self.goal))
        jerk        = np.mean(np.abs(self.joint_vel - (self.prev_pos - self.joint_pos)))
        smoothness  = -0.1 * np.mean(np.abs(self.joint_vel))
        singularity = self._singularity_penalty()

        reward = (
            -2.0 * dist
            + smoothness
            - 0.5 * jerk
            - singularity
        )
        reached    = dist < GOAL_THRESH
        if reached:
            reward += 50.0

        terminated = bool(reached)
        truncated  = self.step_count >= MAX_STEPS
        info = {"dist": dist, "jerk": jerk, "singularity": singularity, "reached": reached}
        return self._obs(), reward, terminated, truncated, info

    def _obs(self):
        pos_range = JOINT_LIMITS[:, 1] - JOINT_LIMITS[:, 0]
        pos_mid   = (JOINT_LIMITS[:, 1] + JOINT_LIMITS[:, 0]) / 2
        pos_norm  = (self.joint_pos - pos_mid) / (pos_range / 2)
        vel_norm  = np.clip(self.joint_vel / (STEP_SIZE + 1e-6), -1, 1)
        goal_norm = (self.goal - pos_mid) / (pos_range / 2)
        return np.concatenate([pos_norm, vel_norm, goal_norm]).astype(np.float32)

    def _singularity_penalty(self):
        wrist_sing = max(0.0, 0.2 - abs(self.joint_pos[4])) * 2.0
        elbow_sing = max(0.0, 0.1 - abs(self.joint_pos[1] + self.joint_pos[2])) * 1.0
        return wrist_sing + elbow_sing
