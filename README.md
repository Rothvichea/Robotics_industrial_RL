# Robotics Industrial RL

![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?style=for-the-badge&logo=ros&logoColor=white)
![MoveIt2](https://img.shields.io/badge/MoveIt2-OMPL%20%2B%20KDL-orange?style=for-the-badge)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-00FFFF?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Ignition](https://img.shields.io/badge/Ignition-Gazebo%20Fortress-FF6600?style=for-the-badge)
![PPO](https://img.shields.io/badge/PPO-Stable--Baselines3-blueviolet?style=for-the-badge)

Fully autonomous **dual-arm robotic inspection and sorting cell** built on ROS 2 Humble, MoveIt 2, and Ignition Gazebo Fortress. Two KUKA KR6 R900 arms are mounted on opposite sides of an inspection table. A top-down camera feeds YOLOv8 defect classification; the two arms sort objects into separate bins in parallel — **4 objects sorted in 7–9 seconds**.

---

## Motivation

Industrial pick-and-place robots typically rely on hardcoded joint poses: brittle, manual to tune, and unable to adapt when object positions change. This project answers two questions:

1. **Can a robot learn smooth, singularity-free trajectories by itself?** — A PPO agent with shaped reward achieves **100% success** in 3 minutes of training.
2. **Can two arms collaborate on the same table without collision?** — A `table_lock` serialises access to the shared table zone while both arms fly to their bins simultaneously.

---

## Results

| Metric | Value |
|---|---|
| RL success rate | **100%** (20 / 20 goals) |
| Avg steps to goal | **17** / 200 max |
| Avg jerk | **0.036** |
| IK success rate | **100%** (after multi-seed retry + `avoid_collisions=False`) |
| Objects sorted (dual arm parallel) | **4 boxes in 7–9 s** |
| YOLOv8 defect confidence | **> 85%** |
| Arm-to-arm collisions | **0** |

---

## Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                     Ignition Gazebo World                        │
│                                                                  │
│  ARM 1 (0,0,0) yaw=0          Camera (0.6, 0, 1.0) → down      │
│  → picks DEFECT → REJECT bin                                     │
│                    Inspection Table (x≈0.6)                     │
│                    ├── box_defect_1  (bottle)                    │
│                    ├── box_defect_2  (screw)                     │
│                    ├── box_good_1    (bottle)                    │
│                    └── box_good_2    (screw)                     │
│  ARM 2 (1.2,0,0) yaw=π                                          │
│  → picks GOOD   → ACCEPT bin                                    │
└──────────────────────────────────────────────────────────────────┘
              │                         │
              ▼                         ▼
┌──────────────────────┐   ┌──────────────────────────┐
│   Phase 1 – Scan     │   │   Phase 2 – Sort          │
│                      │   │                           │
│  OpenCV contours     │   │  Nearest-neighbour order  │
│  → object presence   │   │                           │
│                      │   │  IK cascade (KDL):        │
│  YOLOv8 inference    │   │  pre_pick → approach      │
│  → DEFECT / GOOD     │   │  → pick → lift → place    │
│                      │   │                           │
│  pixel → world XY    │   │  table_lock: ARM1 / ARM2  │
│  (pinhole model)     │   │  share table zone safely  │
└──────────────────────┘   └──────────────────────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
          /arm_1_controller/follow_joint_trajectory
          /arm_2_controller/follow_joint_trajectory
          (parallel execution — no stop-start)
```

---

## Architecture

### Coordinate Frames

ARM 2 sits at `(1.2, 0, 0)` facing `−X` (yaw = π). All box positions are in the world frame. The transform to ARM 2's local frame:

```
local_x = 1.2 − world_x
local_y = −world_y
```

### IK Cascade

All four IK solutions are computed before any motion begins, each seeding the next. Approach always uses `pre_pick(lx, ly)` as seed — never the previous bin configuration — to prevent KDL from finding the "back-of-robot" mirror solution:

```
pre_pick  →  q_approach  →  q_pick  →  q_lift  →  q_place
```

`pre_pick` is computed analytically:

```
pre_pick(lx, ly) = [atan2(ly, lx), -0.9, 0.7, 0.0, -0.8, 0.0]
```

### Collision Avoidance Strategy

| Layer | Mechanism |
|---|---|
| Table zone | `table_lock` — only one arm in the table zone at a time |
| IK stage | `avoid_collisions=False` — MoveIt collision check disabled (physical safety via lock) |
| Bin zone | No lock needed — ARM 1 flies to reject bin (y=−0.45), ARM 2 to accept bin (y=+0.45), no spatial overlap |
| IK service | `ik_lock` — prevents concurrent async IK calls to `/compute_ik` |

### RL Environment

| Component | Detail |
|---|---|
| Observation | joint positions (6) + velocities (6) + goal (6) = 18-dim |
| Action | joint position deltas, clipped to ±0.05 rad/step |
| Reward | −2·dist + smoothness − 0.5·jerk − singularity_penalty + 50 (reach) |
| Algorithm | PPO · MlpPolicy · net [256, 256] · 8 parallel envs |
| Training | 500k steps · ~3 min · RTX 3060 |

---

## Development Phases

### Phase 1 — Single-Arm URDF + Gazebo Cell
Clean standalone ROS 2 package. KR6 R900 URDF, SRDF planning groups, `gz_ros2_control` verified with all 6 joints active.

### Phase 2 — MoveIt 2 Integration
KDL kinematics, OMPL planner, SRDF collision pairs, RViz MoveIt plugin for manual trajectory testing.

### Phase 3 — Dual Arm Cell
xacro macro with `yaw` parameter. ARM 1 at `(0,0,0)`, ARM 2 at `(1.2,0,0)` facing inward. Home: `joint_2 = −π/2`, all others = 0 (arms straight up). Two independent `JointTrajectoryController` instances.

### Phase 4 — PPO RL Trajectory Optimizer
Custom Gymnasium environment. **100% success, avg 17 steps, jerk = 0.036.** Training in 3 minutes.

### Phase 5a — RL + YOLO Feedback Loop
PPO agent connected to live YOLOv8 print-quality monitor. Defect severity triggers slow-down / replan / abort. 10/10 trials succeeded.

### Phase 5b — PyQt6 Live Dashboard
3-column Qt6 window: ARM 1/2 joint bars, status badges, vision QC counters, RL decision feed, timestamped event log.

### Phase 6 — Vision-Guided Pick & Place
`multi_box_inspector.py`: camera → OpenCV contours → pixel-to-world → YOLOv8 → IK cascade → parallel pick-and-place.

### Phase 7 — Suction Cups on Both Arms
`suction_cup.urdf.xacro` extended with `prefix` parameter. Mounted on both `arm_1_tool0` and `arm_2_tool0`. Box teleportation via `ign service /world/inspection_cell/set_pose`.

### Phase 8 — Collision Avoidance + Parallel Execution
`table_lock` for mutual exclusion in shared table zone. Both arms fly to bins simultaneously after lifting. `ik_lock` serialises `/compute_ik` service calls.

### Phase 9 — IK Robustness Fixes
11-seed retry loop with `atan2(y,x)` bias for `joint_1`. Name-based joint extraction (not `position[:6]`). `avoid_collisions=False` at IK stage.

### Phase 10 — Speed Optimisation
~40% reduction in total cycle time. 4 boxes sorted in **7–9 s** (down from 12–15 s).

---

## Project Structure

```
Robotics_industrial_RL/
├── src/kr6_r900_cell/
│   ├── urdf/
│   │   ├── kr6_r900_macro.xacro        # single-arm macro (yaw param)
│   │   ├── kr6_dual_arm.urdf.xacro     # ARM 1 + ARM 2 + suction cups
│   │   └── suction_cup.urdf.xacro      # suction cup (prefix param)
│   ├── srdf/
│   │   └── kr6_dual_arm.srdf           # arm_1, arm_2, dual_arm groups
│   ├── config/
│   │   ├── dual_arm_controllers.yaml   # two JointTrajectoryControllers
│   │   ├── kinematics.yaml             # KDL for arm_1 + arm_2
│   │   ├── ompl_planning.yaml          # OMPL + time parameterization
│   │   └── joint_limits.yaml           # per-joint velocity limits
│   ├── launch/
│   │   └── dual_arm_inspection.launch.py  # full-stack launch
│   ├── worlds/
│   │   └── inspection_cell.sdf         # Ignition world
│   ├── scripts/
│   │   ├── multi_box_inspector.py      # main orchestrator
│   │   └── dual_arm_dashboard.py       # PyQt6 live dashboard
│   └── kr6_rl/
│       ├── kr6_env.py                  # Gymnasium environment
│       ├── train_ppo.py                # PPO training
│       ├── eval_ppo.py                 # evaluation
│       └── rl_vision_loop.py           # RL + YOLO feedback
```

---

## Quick Start

```bash
# 1. Build
cd ~/Robotics_industrial_RL
colcon build --symlink-install
source install/setup.bash

# 2. Launch full stack
ros2 launch kr6_r900_cell dual_arm_inspection.launch.py
# wait for: "All is well! Everyone is happy!"

# 3. Run inspector
/usr/bin/python3 src/kr6_r900_cell/scripts/multi_box_inspector.py

# 4. Live dashboard (optional)
/usr/bin/python3 src/kr6_r900_cell/scripts/dual_arm_dashboard.py

# RL training (conda env)
conda activate industrial-ai
cd src/kr6_r900_cell/kr6_rl
python3 train_ppo.py    # ~3 min
python3 eval_ppo.py     # 100% success
```

---

## ROS 2 Topics

| Topic | Direction | Description |
|---|---|---|
| `/joint_states` | Subscribe | All 12 joint positions (both arms) |
| `/inspection_cam/image_raw` | Subscribe | RGB camera over table |
| `/arm_1_controller/follow_joint_trajectory` | Action | ARM 1 trajectory execution |
| `/arm_2_controller/follow_joint_trajectory` | Action | ARM 2 trajectory execution |
| `/compute_ik` | Service | MoveIt KDL IK solver |
| `/inspection/result` | Publish | JSON: verdict, confidence, action |
| `/arm_1/status` | Publish | IDLE / PICKING / PLACING / DONE |
| `/arm_2/status` | Publish | IDLE / PICKING / PLACING / DONE |
| `/rl/decision` | Publish | RL agent decision string |

---

## Technology Stack

| Component | Technology |
|---|---|
| Robot middleware | ROS 2 Humble |
| Motion planning | MoveIt 2 — OMPL + KDL IK |
| Simulation | Ignition Gazebo Fortress 6.x |
| Vision — defect | YOLOv8 (ultralytics ≥ 8.0) |
| Vision — presence | OpenCV contour detection |
| RL framework | Stable-Baselines3 ≥ 1.8 · PPO |
| Joint control | ros2_controllers — FollowJointTrajectory |
| Dashboard | PyQt6 ≥ 6.4 |
| Robot model | KUKA KR6 R900 (6-axis, 900 mm reach) × 2 |
| GPU | NVIDIA RTX 3060 · CUDA 12.8 |
| OS | Ubuntu 22.04 LTS |

---

## Key Engineering Decisions

**Why `avoid_collisions=False` at the IK stage?**
While ARM 2 is mid-trajectory, MoveIt's planning scene reflects its intermediate link positions. ARM 1 computing IK concurrently would have valid approach poses rejected as "colliding" with ARM 2's current position. Setting `avoid_collisions=False` at the IK stage disables this. Physical safety is guaranteed by `table_lock` — the two arms never enter the table zone simultaneously.

**Why `table_lock` instead of MoveIt collision checking?**
MoveIt collision avoidance plans around static obstacles. Moving arms are not static. A threading lock is simpler, faster, and guaranteed correct for this specific geometry.

**Why `pre_pick` as IK seed?**
KDL is seed-sensitive. For a suction cup pointing straight down, `joint_4` lies in the null space — any rotation is valid. Without a good seed, KDL converges to wrist-flip solutions. `pre_pick` biases the solver toward the "arm over table" configuration and eliminates 360° joint spins.

**Why direct `FollowJointTrajectory` instead of MoveIt execution?**
MoveIt's trajectory execution requires clock synchronization between `move_group` and Ignition (sim time vs wall time). Sending multi-waypoint trajectories directly to the controller bypasses this and produces smooth continuous motion.

---

## Roadmap

### Completed ✓
- [x] Single arm URDF + MoveIt 2 + RViz2
- [x] Trajectory planning + ghost visualization
- [x] Dual arm cell (xacro macro, opposite sides, yaw=π)
- [x] PPO RL optimizer — 100% success, jerk=0.036
- [x] YOLOv8 print monitor + RL feedback loop
- [x] PyQt6 live dashboard (joint bars, status, QC, RL feed)
- [x] Ignition inspection cell (textured boxes, bins, camera)
- [x] Vision-guided pick — pixel → world → IK → trajectory
- [x] Suction cups on both arms + box teleportation
- [x] Parallel dual-arm execution with table_lock
- [x] IK robustness (11-seed retry, name extraction, ik_lock)
- [x] Speed optimization — 4 boxes in 7–9 s

### Near-term 🔧
- [ ] Real suction physics (vacuum pressure feedback)
- [ ] Conveyor belt for continuous item flow
- [ ] ISO 13485 / EN 9100 inspection logging

### VLA Direction 🤖

The long-term goal is a **Vision-Language-Action model** replacing explicit motion planning:

- Natural language instructions → joint trajectories directly
- Zero-shot generalization to unseen object categories  
- Claude Vision API for semantic defect reasoning
- Fine-tune OpenVLA on simulated demonstrations
- Language-conditioned motion replacing KDL + OMPL

---

## Related Projects

| Project | Description |
|---|---|
| [smart-factory-agent](https://github.com/Rothvichea) | YOLOv8 + PPO + Claude VLM — end-to-end MLOps factory platform |
| [segformer-tensorrt](https://github.com/Rothvichea) | SegFormer-b0 TensorRT FP32 — 1.70× speedup, 138 FPS |
| [Safety Detection](https://github.com/Rothvichea) | PPE compliance + dynamic danger zones (ISO 10218-1) |
| [industrial-gnn-predictive-maintenance](https://github.com/Rothvichea/industrial-gnn-predictive-maintenance) | 1D-CNN + GraphSAGE bearing fault — 99.10% accuracy |

---
## Live Demonstrations

### Dual-Arm Inspection — Part 1

https://github.com/Rothvichea/Robotics_industrial_RL/raw/main/display_part1.mp4

### Dual-Arm Inspection — Part 2

https://github.com/Rothvichea/Robotics_industrial_RL/raw/main/display_part2.mp4

---
**Rothvichea CHEA** · Mechatronics Engineer · Lyon, France  
[Portfolio](https://rothvicheachea.netlify.app) · [LinkedIn](https://www.linkedin.com/in/chea-rothvichea-a96154227/) · [GitHub](https://github.com/Rothvichea)
