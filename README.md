# Robotics Industrial RL

![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?style=for-the-badge&logo=ros&logoColor=white)
![MoveIt2](https://img.shields.io/badge/MoveIt2-OMPL%20%2B%20KDL-orange?style=for-the-badge)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-00FFFF?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Ignition](https://img.shields.io/badge/Ignition-Gazebo%20Fortress-FF6600?style=for-the-badge)

**KUKA KR6 R900** autonomous quality-control cell combining real-time vision inspection with reinforcement-learning-optimised trajectory planning — built entirely in simulation using ROS 2, MoveIt 2, and Ignition Gazebo.

---

## Motivation

Industrial pick-and-place robots typically rely on hardcoded joint poses: brittle, manual to tune, and unable to adapt when objects shift. This project answers two questions:

1. **Can a robot learn smooth, singularity-free trajectories by itself?** — Using PPO reinforcement learning with a shaped reward (goal proximity − jerk − singularity penalty), the agent achieves **100% success** in 3 minutes of training.
2. **Can the robot decide where to pick based on what the camera sees?** — A YOLOv8 defect-detection pipeline classifies each item and the robot sorts it to the correct bin without any hardcoded object position.

---

## Results

| Metric | Value |
|---|---|
| RL success rate | **100%** (20 / 20 goals) |
| Avg steps to goal | **17.4** / 200 max |
| Avg jerk | **0.036** |
| Explained variance | 0.934 |
| Training time | **~3 min** (RTX 3060, 500k steps) |
| YOLO bottle defect confidence | **90.5%** |

---

## Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    Ignition Gazebo World                     │
│                                                              │
│   KUKA KR6 R900          Overhead Camera (z = 1.0 m)        │
│   + Suction Cup          ↓  /inspection_cam/image_raw       │
│   6-axis arm                                                 │
│   gz_ros2_control        Inspection Table                   │
│                          ├── box_defect_1  (bottle PNG)     │
│                          ├── box_defect_2  (screw  PNG)     │
│                          ├── box_good_1    (bottle PNG)     │
│                          └── box_good_2    (screw  PNG)     │
│                                                              │
│   Accept Bin (green)     Reject Bin (red)                   │
└──────────────────────────────────────────────────────────────┘
              │                        │
              ▼                        ▼
┌─────────────────────┐    ┌────────────────────────┐
│   Phase 1 – Scan    │    │   Phase 2 – Sort        │
│                     │    │                         │
│  OpenCV contours    │    │  Nearest-neighbour      │
│  → object presence  │    │  visit order            │
│                     │    │                         │
│  YOLOv8 inference   │    │  IK cascade:            │
│  → DEFECT / GOOD    │    │  approach → pick        │
│                     │    │  → lift → place         │
│  pixel → world XY   │    │                         │
│  (pinhole model)    │    │  Vacuum simulation      │
└─────────────────────┘    └────────────────────────┘
              │                        │
              └────────────┬───────────┘
                           ▼
              FollowJointTrajectory
              /joint_trajectory_controller
```

---

## Architecture

### Camera Geometry

The overhead camera is mounted at `(0.6, 0, 1.0)` m, pitched `π/2` (pointing straight down). Pixel-to-world conversion uses the pinhole model:

```
scale   = W / (2 · tan(FOV/2) · (z_cam − z_table))
world_y = (col − W/2) / scale
world_x = x_cam − (row − H/2) / scale
```

### IK Cascade

All four IK solutions are computed before any motion begins, with each seeding the next to prevent KDL wrist flips:

```
q_seed  →  q_approach  →  q_pick  →  q_lift  →  q_place
```

### RL Environment

| Component | Detail |
|---|---|
| Observation | joint positions (6) + velocities (6) + goal (6) = 18-dim |
| Action | joint position deltas, clipped to ±0.05 rad/step |
| Reward | −2·dist + smoothness − 0.5·jerk − singularity_penalty + 50 (on reach) |
| Termination | dist < 0.05 rad (success) or 200 steps (timeout) |
| Algorithm | PPO · MlpPolicy · net [256, 256] · 8 parallel envs |

---

## What Was Built

### Phase 1 — KR6 R900 URDF + MoveIt 2
Clean standalone ROS 2 package with self-contained URDF, SRDF planning groups (`manipulator`, `end_effector`, named states), KDL kinematics, and OMPL planning pipeline.

### Phase 2 — Trajectory Planning
MoveIt 2 action client for joint-goal execution. 5-pose sequence visualised as ghost trajectories in RViz2 via `DisplayTrajectory`.

### Phase 3 — Dual Arm Cell
xacro macro architecture. Two KR6 arms (`arm_1_`, `arm_2_`) placed 1.2 m apart with a combined `dual_arm` subgroup. Synchronized trajectory planning with inter-arm collision avoidance.

### Phase 4 — PPO RL Trajectory Optimizer
Custom Gymnasium environment. 500k-step PPO training in 3 minutes. **100% goal success, avg 17.4 steps, jerk = 0.036.**

### Phase 5 — Vision QC + Qt6 Dashboard
`PrintMonitor` (YOLOv8 defect detection on print frames) + RL vision feedback loop (slow down / replan / abort on defect severity) + live Qt6 dashboard with joint bars, RL metrics, and event log.

### Phase 6 — Ignition Gazebo Inspection Cell
Custom SDF world: inspection table, 4 PBR-textured boxes (real bottle/screw images on top face), green/red bins, overhead camera bridged to ROS 2. Full `gz_ros2_control` pipeline with suction cup URDF. Multi-box inspector: contour detection + per-box YOLO classification + nearest-neighbour pick-and-place sort.

---

## Project Structure

```
Robotics_industrial_RL/
├── src/kr6_r900_cell/
│   ├── urdf/
│   │   ├── kr6_r900_2.urdf.xacro      # main robot + ros2_control
│   │   ├── kr6_r900_macro.xacro        # reusable arm macro
│   │   ├── kr6_dual_arm.urdf.xacro     # dual arm cell
│   │   └── suction_cup.urdf.xacro      # vacuum end-effector
│   ├── srdf/
│   │   ├── kr6_r900_2.srdf             # single arm groups
│   │   └── kr6_dual_arm.srdf           # dual arm groups
│   ├── config/
│   │   ├── gazebo_controllers.yaml     # ros2_control config
│   │   ├── ompl_planning.yaml          # OMPL + time parameterization
│   │   └── moveit.rviz                 # saved RViz2 layout
│   ├── launch/
│   │   ├── display_moveit.launch.py    # RViz2 + MoveIt (no sim)
│   │   ├── dual_arm_moveit.launch.py   # dual arm cell
│   │   └── inspection_cell.launch.py   # full Gazebo cell
│   ├── worlds/
│   │   └── inspection_cell.sdf         # Ignition Gazebo world
│   ├── scripts/
│   │   ├── multi_box_inspector.py      # main inspection pipeline
│   │   ├── vision_pick_node.py         # IK-based vision-guided pick
│   │   ├── vacuum_controller.py        # suction cup controller
│   │   └── yolo_inspector_node.py      # YOLO ROS 2 node
│   └── kr6_rl/
│       ├── kr6_env.py                  # Gymnasium environment
│       ├── train_ppo.py                # PPO training
│       ├── eval_ppo.py                 # evaluation + metrics
│       ├── print_monitor.py            # print quality monitor
│       ├── rl_vision_loop.py           # RL + vision feedback
│       └── dashboard.py               # Qt6 live dashboard
```

---

## Quick Start

```bash
# Build
cd ~/Robotics_industrial_RL
colcon build --symlink-install
source install/setup.bash

# Launch simulation cell (Ignition + MoveIt 2 + controllers)
ros2 launch kr6_r900_cell inspection_cell.launch.py

# Wait ~13s for controllers to activate, then run inspection
/usr/bin/python3 src/kr6_r900_cell/scripts/multi_box_inspector.py

# RL training (requires conda env with PyTorch)
conda activate industrial-ai
cd src/kr6_r900_cell/kr6_rl
python3 train_ppo.py        # trains in ~3 min
python3 eval_ppo.py         # 100% success rate

# Qt6 live dashboard
python3 dashboard.py
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Robot middleware | ROS 2 Humble |
| Motion planning | MoveIt 2 — OMPL + KDL IK |
| Simulation | Ignition Gazebo Fortress |
| Vision — defect | YOLOv8 (ultralytics 8.4) |
| Vision — presence | OpenCV contour detection |
| RL framework | Stable-Baselines3 2.7 · PPO |
| Joint control | ros2_controllers — FollowJointTrajectory |
| Dashboard | PyQt6 |
| Robot model | KUKA KR6 R900 (6-axis, 900 mm reach) |
| GPU | NVIDIA RTX 3060 · CUDA 12.8 |
| OS | Ubuntu 22.04 LTS |

---

## Key Engineering Decisions

**Why direct FollowJointTrajectory instead of MoveIt execution?**
MoveIt's trajectory execution manager requires clock synchronization between `move_group` (wall time) and Ignition Gazebo (sim time). Sending multi-waypoint trajectories directly to the controller bypasses this entirely and produces smoother continuous motion — all waypoints interpolated in one action goal rather than stop-start single-point goals.

**Why contour detection for presence + YOLO for defect?**
YOLO defect models are trained to detect flaws, not object presence. A good (non-defective) item produces no YOLO detection. Contour detection catches it on the white mat and correctly labels it GOOD. Separating the two problems avoids false negatives on acceptable items.

**Why IK cascade seeding?**
KDL is an iterative Jacobian solver. For a suction cup pointing straight down, `joint_4` (forearm roll) lies in the null space — any rotation is valid. Without seeding each IK call from the previous solution, KDL can converge to a wrist-flip solution (joint_4 ± π from seed), causing a 360° unnecessary rotation. Cascaded seeding with 7 alternative seeds eliminates this in practice.

**Why PPO over SAC or TD3?**
The task is episodic with a clear terminal condition. PPO's on-policy rollouts are better suited than off-policy methods, and the discrete episode structure makes reward shaping straightforward. 8 parallel envs bring training to 3 minutes — fast enough for rapid iteration on reward design.

---

## Roadmap

### Completed ✓
- [x] KR6 R900 clean URDF + MoveIt 2 package
- [x] Single arm trajectory planning + RViz2 visualization
- [x] Dual arm cell with inter-arm collision avoidance
- [x] PPO RL optimizer — 100% success, jerk = 0.036
- [x] YOLOv8 print monitor + RL vision feedback loop
- [x] Qt6 live dashboard
- [x] Ignition Gazebo inspection cell — textured boxes, bins, camera
- [x] Vision-guided pick — pixel → world → IK → trajectory
- [x] Suction cup URDF + vacuum simulation
- [x] Multi-box inspection + nearest-neighbour sort

### Near-term 🔧
- [ ] Downward suction approach — wrist orientation fix, KDL null-space correction
- [ ] Real-time per-box YOLO from Gazebo camera (close-up crop approach)
- [ ] Conveyor belt for continuous item flow
- [ ] ISO 13485 / EN 9100 structured inspection logging

### VLA Direction 🤖

The long-term goal is to replace explicit motion planning with a **Vision-Language-Action model** that takes raw camera feed + natural language instructions and outputs joint trajectories directly:

- **"Pick up all defective screws"** → robot executes without any hardcoded poses
- Zero-shot generalization to unseen object categories
- Claude Vision API for semantic defect reasoning (partially implemented in the Smart Factory agent)
- Fine-tune OpenVLA or similar on simulated pick-and-place demonstrations
- Language-conditioned motion generation replacing KDL + OMPL

---

## Related Projects

| Project | Description |
|---|---|
| [smart-factory-agent](https://github.com/Rothvichea) | YOLOv8 + PPO + Claude VLM API — end-to-end MLOps factory platform |
| [segformer-tensorrt](https://github.com/Rothvichea) | SegFormer-b0 TensorRT FP32 — 1.70× speedup, 138 FPS, FP16 overflow analysis |
| [Safety Detection](https://github.com/Rothvichea) | Real-time PPE compliance + dynamic danger zone enforcement (ISO 10218-1) |
| [industrial-gnn-predictive-maintenance](https://github.com/Rothvichea/industrial-gnn-predictive-maintenance) | 1D-CNN + GraphSAGE bearing fault detection — 99.10% accuracy |

---

**Rothvichea CHEA** · Mechatronics Engineer · Lyon, France  
[Portfolio](https://rothvicheachea.netlify.app) · [LinkedIn](https://www.linkedin.com/in/chea-rothvichea-a96154227/) · [GitHub](https://github.com/Rothvichea)
