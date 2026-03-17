"""
KR6 RL + Vision Feedback Loop
"""

import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from kr6_env import KR6Env, GOALS
from print_monitor import PrintMonitor
import cv2

MODEL_PATH      = os.path.expanduser("~/Robotics_industrial_RL/rl_models/best_model.zip")
CLEAN_IMG_PATH  = os.path.expanduser("~/smart-factory-agent/test_clean.jpg")
DEFECT_IMG_PATH = os.path.expanduser("~/smart-factory-agent/sample_data/screw_defect_01.png")
PRODUCT_TYPE    = "screw"

# load base frames once
_clean  = cv2.imread(CLEAN_IMG_PATH)
_defect = None
if os.path.exists(DEFECT_IMG_PATH):
    _defect = cv2.imread(DEFECT_IMG_PATH)

def get_frame(inject_defect: bool = False) -> np.ndarray:
    """Return a real image frame — clean or defective."""
    if inject_defect and _defect is not None:
        return cv2.resize(_defect, (320, 320))
    if _clean is not None:
        return cv2.resize(_clean, (320, 320))
    return np.ones((320, 320, 3), dtype=np.uint8) * 40


def run_rl_vision_loop(n_episodes: int = 10, defect_prob: float = 0.08):
    print("=" * 55)
    print("KR6 RL + Vision Feedback Loop")
    print(f"  Defect injection prob: {defect_prob*100:.0f}%")
    print("=" * 55)

    env   = KR6Env()
    model = PPO.load(MODEL_PATH, env=env)
    print("[RL] Model loaded")

    monitor = PrintMonitor(product_type=PRODUCT_TYPE, device="cuda")
    results_log = []

    for ep in range(n_episodes):
        obs, _    = env.reset()
        done      = False
        steps     = 0
        aborted   = False
        replans   = 0
        slowdowns = 0
        info      = {"reached": False, "dist": 999.0}

        print(f"\n── Episode {ep+1}/{n_episodes} | goal: {np.round(env.goal, 2)}")

        while not done:
            # inject defect randomly to simulate real print issues
            inject = np.random.random() < defect_prob
            frame  = get_frame(inject_defect=inject)
            insp   = monitor.inspect(frame)

            if insp.has_defect:
                print(f"   [DEFECT] step={steps} sev={insp.severity} "
                      f"conf={insp.confidence:.2f} action={insp.action}")

            if insp.action == "abort":
                aborted = True
                break
            elif insp.action == "stop_and_replan":
                replans += 1
                env.goal = GOALS[np.random.randint(len(GOALS))].copy()
                obs      = env._obs()
            elif insp.action == "slow_down":
                slowdowns += 1

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done   = terminated or truncated

        status = "ABORTED" if aborted else ("REACHED" if info["reached"] else "TIMEOUT")
        print(f"   → {status} | steps={steps} | dist={info['dist']:.4f} | "
              f"replans={replans} | slowdowns={slowdowns}")

        results_log.append({
            "status": status, "steps": steps,
            "dist": info["dist"], "replans": replans, "slowdowns": slowdowns,
        })

    print("\n" + "=" * 55)
    print("Summary")
    print("=" * 55)
    reached = sum(1 for r in results_log if r["status"] == "REACHED")
    aborted = sum(1 for r in results_log if r["status"] == "ABORTED")
    print(f"  Reached:         {reached}/{n_episodes}")
    print(f"  Aborted:         {aborted}/{n_episodes}")
    print(f"  Avg steps:       {np.mean([r['steps'] for r in results_log]):.1f}")
    print(f"  Avg dist:        {np.mean([r['dist']  for r in results_log]):.4f}")
    print(f"  Total replans:   {sum(r['replans']    for r in results_log)}")
    print(f"  Total slowdowns: {sum(r['slowdowns']  for r in results_log)}")
    print()
    for k, v in monitor.stats().items():
        print(f"  {k}: {v}")
    print("=" * 55)


if __name__ == "__main__":
    run_rl_vision_loop(n_episodes=10, defect_prob=0.08)
