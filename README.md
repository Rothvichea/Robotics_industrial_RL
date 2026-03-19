# Robotics Industrial RL
### KR6 R900 Industrial Cell — Vision-Guided Inspection & Pick-and-Place

**ROS 2 Humble · MoveIt 2 · Ignition Gazebo Fortress · YOLOv8 · PPO RL · OpenCV · KDL IK**

---

## Overview

A fully autonomous industrial robotics simulation built around the **KUKA KR6 R900** 6-axis robot arm. The system combines real-time vision inspection with reinforcement-learning-optimized trajectory planning to implement a complete quality-control pick-and-place cell.

The pipeline operates in two sequential phases on each run:

1. **Phase 1 – Vision Scan.** The robot parks at HOME while an overhead camera captures a top-down view of the inspection table. OpenCV contour detection locates every object on the table. YOLOv8 defect-detection models then classify each object as **DEFECT** or **GOOD**.

2. **Phase 2 – Sort.** Objects are visited in greedy nearest-neighbour order to minimise arm travel. For each object the arm executes a smooth pick-and-place trajectory, descends with the suction tip facing down, simulates vacuum engagement, lifts clear of the table, and deposits the object into the **green accept bin** (GOOD) or the **red reject bin** (DEFECT).

---

## System Architecture

```
Ignition Gazebo
  ├── KUKA KR6 R900 (6-axis, gz_ros2_control)
  ├── Suction cup end-effector (tool0)
  ├── Inspection table + 4 textured boxes
  ├── Accept bin (green)  /  Reject bin (red)
  └── Overhead camera → /inspection_cam/image_raw

ROS 2 / MoveIt 2
  ├── move_group (OMPL planner + KDL IK)
  ├── joint_trajectory_controller
  └── /compute_ik service

Vision Pipeline
  ├── OpenCV contour detection (object presence)
  └── YOLOv8 (yolo_bottle.pt / yolo_screw.pt) → DEFECT / GOOD

RL Optimizer
  ├── Gymnasium environment (KR6Env)
  ├── PPO agent (Stable-Baselines3)
  └── Reward: smoothness + speed − jerk − singularity penalty
```

---

## What Was Built — Phase by Phase

### Phase 1 — KR6 R900 URDF + MoveIt2 ✓
Clean standalone ROS 2 package `kr6_r900_cell` with:
- Self-contained URDF (no external mesh dependencies)
- SRDF with `manipulator` and `end_effector` planning groups + named states
- KDL kinematics solver configuration
- OMPL planning pipeline
- RViz2 config saved for instant launch

### Phase 2 — Trajectory Planning + RViz2 Visualization ✓
- MoveIt2 action client for joint-goal trajectory execution
- 5-pose sequence: home → pose_a → pose_b → extended → home
- `DisplayTrajectory` publisher for ghost-path visualization in RViz2

### Phase 3 — Dual KR6 Arm Cell ✓
- xacro macro architecture for reusable arm definition
- Two KR6 arms (`arm_1_`, `arm_2_` prefixes) placed 1.2 m apart
- Three SRDF planning groups: `arm_1`, `arm_2`, `dual_arm` (subgroup)
- Synchronized trajectory planning with collision avoidance between arms

### Phase 4 — PPO RL Trajectory Optimizer ✓
- Custom Gymnasium environment with 18-dim observation space
  - Joint positions (6) + velocities (6) + goal positions (6)
- Continuous action space: joint position deltas
- Shaped reward: goal proximity − jerk − singularity penalty + reach bonus
- Training: 500k timesteps, 8 parallel envs, 3 minutes on RTX 3060
- **Results: 100% success rate, avg 17 steps to goal, jerk = 0.036**

### Phase 5 — Vision QC + Qt6 Dashboard ✓
- `PrintMonitor`: YOLOv8-based defect detection on simulated print frames
- `rl_vision_loop.py`: RL agent + vision feedback — slow down / replan / abort on defect
- `dashboard.py`: Qt6 live monitoring — joint bars, RL metrics, vision QC panel, event log

### Phase 6 — Ignition Gazebo Inspection Cell ✓
- Custom SDF world: inspection table, 4 textured boxes (PNG images on top face), green/red bins, overhead camera
- Gazebo camera bridged to ROS 2 via `ros_gz_bridge`
- `gz_ros2_control` with `joint_trajectory_controller` executing real trajectories
- Suction cup end-effector URDF attached to `tool0`
- Multi-box inspector: contour detection + YOLO classification per box + pick-and-place sort

---

## Technology Stack

| Component | Technology |
|---|---|
| Robot middleware | ROS 2 Humble |
| Motion planning | MoveIt 2 — OMPL + KDL IK |
| Simulation | Ignition Gazebo Fortress |
| Vision — defect | YOLOv8 (ultralytics) |
| Vision — presence | OpenCV contour detection |
| RL framework | Stable-Baselines3 PPO |
| Joint control | FollowJointTrajectory (ros2_controllers) |
| Dashboard | Qt6 (PyQt6) |
| Robot model | KUKA KR6 R900 |
| OS | Ubuntu 22.04 LTS |

---

## Project Structure

```
Robotics_industrial_RL/
├── src/
│   └── kr6_r900_cell/
│       ├── urdf/
│       │   ├── kr6_r900_2.urdf.xacro      # main robot URDF
│       │   ├── kr6_r900_macro.xacro        # reusable arm macro
│       │   ├── kr6_dual_arm.urdf.xacro     # dual arm cell
│       │   └── suction_cup.urdf.xacro      # suction gripper
│       ├── srdf/
│       │   ├── kr6_r900_2.srdf             # single arm planning groups
│       │   └── kr6_dual_arm.srdf           # dual arm planning groups
│       ├── config/
│       │   ├── gazebo_controllers.yaml     # ros2_control config
│       │   ├── kinematics.yaml             # KDL solver config
│       │   ├── ompl_planning.yaml          # OMPL planner config
│       │   └── moveit.rviz                 # saved RViz2 config
│       ├── launch/
│       │   ├── display_moveit.launch.py    # RViz2 + MoveIt (no Gazebo)
│       │   ├── dual_arm_moveit.launch.py   # dual arm cell
│       │   └── inspection_cell.launch.py   # full Gazebo inspection cell
│       ├── worlds/
│       │   └── inspection_cell.sdf         # Ignition Gazebo world
│       ├── scripts/
│       │   ├── multi_box_inspector.py      # main inspection pipeline
│       │   ├── vision_pick_node.py         # vision-guided pick (IK-based)
│       │   ├── vacuum_controller.py        # suction cup controller
│       │   ├── yolo_inspector_node.py      # YOLO ROS2 node
│       │   └── pick_place_node.py          # direct trajectory pick & place
│       └── kr6_rl/
│           ├── kr6_env.py                  # Gymnasium RL environment
│           ├── train_ppo.py                # PPO training script
│           ├── eval_ppo.py                 # evaluation script
│           ├── print_monitor.py            # YOLOv8 print quality monitor
│           ├── rl_vision_loop.py           # RL + vision feedback loop
│           └── dashboard.py               # Qt6 live dashboard
```

---

## Quick Start

```bash
# 1. Build
cd ~/Robotics_industrial_RL
colcon build --symlink-install
source install/setup.bash

# 2. Launch inspection cell (Ignition + MoveIt2 + controllers)
ros2 launch kr6_r900_cell inspection_cell.launch.py

# 3. Wait ~13s for controllers to activate, then run:
/usr/bin/python3 src/kr6_r900_cell/scripts/multi_box_inspector.py

# RL training (conda env)
conda activate industrial-ai
cd src/kr6_r900_cell/kr6_rl
python3 train_ppo.py

# Qt6 dashboard
python3 dashboard.py
```

---

## RL Results

| Metric | Value |
|---|---|
| Training timesteps | 500,000 |
| Training time | ~3 minutes (RTX 3060) |
| Success rate | **100%** (20/20) |
| Avg steps to goal | **17.4** |
| Avg jerk | **0.036** |
| Explained variance | 0.934 |

---

## Roadmap

### Completed ✓
- [x] KR6 R900 URDF + MoveIt2 clean package
- [x] Single arm trajectory planning + RViz2 visualization
- [x] Dual arm synchronized cell with collision avoidance
- [x] PPO RL trajectory optimizer (100% success, smooth motion)
- [x] YOLOv8 print quality monitor + RL vision feedback loop
- [x] Qt6 live dashboard
- [x] Ignition Gazebo inspection cell with textured boxes
- [x] Camera-to-ROS2 bridge + YOLO detection pipeline
- [x] Suction cup gripper URDF
- [x] Vision-guided pick (pixel → world → IK → trajectory)
- [x] Multi-box inspection + nearest-neighbour sort

### In Progress 🔧
- [ ] Proper suction physics (downward approach, wrist flip correction)
- [ ] Real-time YOLO detection from Gazebo camera feed (close-up per-box)
- [ ] Gripper force/contact simulation

### Future Work 🚀

#### Near-term
- [ ] **Gripper integration** — vacuum pressure feedback, contact confirmation
- [ ] **Conveyor belt** — continuous flow of items past the inspection station
- [ ] **Multi-camera setup** — side cameras for 3D defect localization
- [ ] **ISO 13485 / EN 9100 logging** — structured inspection records

#### RL Improvements
- [ ] **Curriculum learning** — progressively harder goals and obstacles
- [ ] **Multi-arm RL** — single PPO agent controlling both KR6 arms simultaneously
- [ ] **Sim-to-real transfer** — domain randomization for real KR6 deployment

#### VLA (Vision-Language-Action) Direction 🤖
The ultimate goal of this project is to replace the hardcoded inspection logic with a **Vision-Language-Action model** that:
- Takes raw camera feed as visual input
- Accepts natural language instructions ("pick up all defective screws")
- Outputs joint trajectories directly without explicit IK or motion planning
- Adapts to new object types without retraining YOLO models
- Integrates with **Claude API** for semantic reasoning about inspection results

Planned VLA integration:
- [ ] Connect Claude Vision API for intelligent defect reasoning (already partially done in Smart Factory agent)
- [ ] Fine-tune OpenVLA or similar on simulated pick-and-place demonstrations
- [ ] Replace hardcoded poses with language-conditioned motion generation
- [ ] Evaluate on zero-shot generalization to unseen object categories

---

## Author

**Rothvichea CHEA** — Mechatronics Engineer  
[Portfolio](https://rothvicheachea.netlify.app) · [LinkedIn](https://www.linkedin.com/in/chea-rothvichea-a96154227/) · [GitHub](https://github.com/Rothvichea)

*IMT Mines Alès — Engineering Degree in Industrial Performance and System Mechatronics*
