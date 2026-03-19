#!/usr/bin/env python3
"""
KR6 Vision-Guided Pick & Place
================================
Uses YOLO bbox to compute 3D pick position
Then uses MoveIt IK service to get joint angles
Then executes via joint_trajectory_controller
"""

import sys
import os
sys.path.insert(0, "/home/genji/miniconda3/envs/industrial-ai/lib/python3.10/site-packages")

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
import json
import math
import time
import threading


JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]

# camera params
CAM_X      = 0.6    # camera world x
CAM_Y      = 0.0    # camera world y
CAM_Z      = 1.4    # camera height
CAM_FOV    = 0.8    # horizontal FOV radians
IMG_W      = 640
IMG_H      = 640
TABLE_Z    = 0.55   # table surface height
BOX_Z      = 0.615  # table + box (pick height)
ABOVE_Z    = 0.78   # approach height above box

# known box positions in world frame (x, y) for 4 boxes
BOX_POSITIONS = {
    "box_defect_1": (0.75,  0.12),   # front left  — defect
    "box_defect_2": (0.75, -0.12),   # front right — defect
    "box_good_1":   (0.45,  0.12),   # back left   — good
    "box_good_2":   (0.45, -0.12),   # back right  — good
}

# fixed poses
HOME        = [0.0, -1.5708, 1.5708, 0.0,  0.0,  0.0]
PLACE_ACCEPT= [0.7, -1.0,    1.5,    0.0, -0.2,  0.0]
PLACE_REJECT= [-0.7,-1.0,    1.5,    0.0, -0.2,  0.0]
DURATIONS   = {"home": 4, "approach": 4, "pick": 4,
               "lift": 3, "place": 5, "retreat": 4}


def pixel_to_world(cx: int, cy: int):
    """
    Convert image pixel (cx, cy) to world (x, y) coordinates.
    Camera is overhead at (CAM_X, CAM_Y, CAM_Z) looking straight down.
    Camera is rotated 90deg around z (pitch=1.5708 in URDF).
    """
    fov_half  = CAM_FOV / 2.0
    half_size = CAM_Z - TABLE_Z  # distance from camera to table

    # angular offset from image center
    angle_x = ((cx - IMG_W / 2) / (IMG_W / 2)) * fov_half
    angle_y = ((cy - IMG_H / 2) / (IMG_H / 2)) * fov_half

    # world offset (camera pitched down, x=forward, y=left)
    offset_x = math.tan(angle_y) * half_size
    offset_y = math.tan(angle_x) * half_size

    world_x = CAM_X - offset_x
    world_y = CAM_Y - offset_y

    return world_x, world_y


class VisionPickNode(Node):

    def __init__(self):
        super().__init__("vision_pick_node")
        self._cb_group = ReentrantCallbackGroup()

        # trajectory controller
        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self._cb_group,
        )

        # IK service
        self._ik_client = self.create_client(
            GetPositionIK, '/compute_ik',
            callback_group=self._cb_group,
        )

        # inspection result subscriber
        self._sub = self.create_subscription(
            String, "/inspection/result",
            self.inspection_callback, 10,
            callback_group=self._cb_group,
        )

        self._busy = False
        self._lock = threading.Lock()
        self._processed = 0

        self.get_logger().info("Waiting for trajectory controller...")
        self._traj_client.wait_for_server()
        self.get_logger().info("Waiting for IK service...")
        self._ik_client.wait_for_service()
        self.get_logger().info("Vision Pick node ready!")

    def inspection_callback(self, msg: String):
        with self._lock:
            if self._busy:
                return
            try:
                result = json.loads(msg.data)
            except json.JSONDecodeError:
                return

            frame = result.get("frame", 0)
            if frame % 15 != 0:
                return
            if not result.get("has_defect") and result.get("verdict") == "GOOD":
                # only pick defects for reject, good items for accept
                pass

            self._processed += 1
            verdict = result.get("verdict", "GOOD")
            action  = result.get("action", "ACCEPT")
            cx      = result.get("cx", 320)
            cy      = result.get("cy", 320)
            conf    = result.get("confidence", 0.0)

            self.get_logger().info(
                f"[#{self._processed}] {verdict} conf={conf:.2f} "
                f"px=({cx},{cy}) → {action}")

            self._busy = True

        threading.Thread(
            target=self._run_sequence,
            args=(cx, cy, action),
            daemon=True,
        ).start()

    def _run_sequence(self, cx, cy, action):
        try:
            self.execute_vision_pick(cx, cy, action)
        finally:
            with self._lock:
                self._busy = False

    def execute_vision_pick(self, cx: int, cy: int, action: str):
        # compute 3D pick position from pixel
        wx, wy = pixel_to_world(cx, cy)
        self.get_logger().info(
            f"Object 3D position: x={wx:.3f} y={wy:.3f} z={BOX_Z:.3f}")

        # get IK for approach (above object)
        approach_joints = self.solve_ik(wx, wy, ABOVE_Z)
        pick_joints     = self.solve_ik(wx, wy, BOX_Z)

        if approach_joints is None or pick_joints is None:
            self.get_logger().error("IK failed — falling back to home")
            self.move_joints(HOME, 4)
            return

        place_joints = PLACE_ACCEPT if action == "ACCEPT" else PLACE_REJECT
        bin_name     = "ACCEPT ✓" if action == "ACCEPT" else "REJECT ✗"

        self.get_logger().info(f"Starting vision pick → {bin_name}")

        steps = [
            (HOME,           4, "home"),
            (approach_joints,4, "approach above object"),
            (pick_joints,    4, "pick object"),
            (approach_joints,3, "lift"),
            (place_joints,   5, f"place → {bin_name}"),
            (HOME,           4, "retreat"),
        ]

        for joints, dur, desc in steps:
            self.get_logger().info(f"  → {desc}")
            ok = self.move_joints(joints, dur)
            if not ok:
                self.get_logger().error(f"  ✗ Failed: {desc}")
                self.move_joints(HOME, 4)
                return
            time.sleep(0.3)

        self.get_logger().info(f"✓ Vision pick complete → {bin_name}")

    def solve_ik(self, x: float, y: float, z: float):
        """Call MoveIt IK service to get joint angles for (x,y,z)."""
        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = "manipulator"
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec = 5

        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        # tool pointing down
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 1.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0

        req.ik_request.pose_stamped = pose

        event  = threading.Event()
        result = [None]

        def cb(future):
            result[0] = future.result()
            event.set()

        self._ik_client.call_async(req).add_done_callback(cb)

        if not event.wait(timeout=10.0):
            self.get_logger().error(f"IK timeout for ({x:.2f},{y:.2f},{z:.2f})")
            return None

        resp = result[0]
        if resp.error_code.val != 1:
            self.get_logger().error(
                f"IK failed code={resp.error_code.val} "
                f"for ({x:.2f},{y:.2f},{z:.2f})")
            return None

        joints = list(resp.solution.joint_state.position[:6])
        self.get_logger().info(
            f"  IK solved: {[round(j,3) for j in joints]}")
        return joints

    def move_joints(self, positions, duration: int) -> bool:
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions       = list(positions)
        pt.velocities      = [0.0] * 6
        pt.time_from_start = Duration(sec=duration, nanosec=0)
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        event     = threading.Event()
        gh_holder = [None]
        res_holder= [None]

        def goal_cb(f):
            gh_holder[0] = f.result()
            event.set()

        self._traj_client.send_goal_async(goal).add_done_callback(goal_cb)

        if not event.wait(timeout=10.0):
            self.get_logger().error("Goal send timeout")
            return False

        gh = gh_holder[0]
        if not gh.accepted:
            self.get_logger().error("Goal rejected")
            return False

        event.clear()

        def result_cb(f):
            res_holder[0] = f.result()
            event.set()

        gh.get_result_async().add_done_callback(result_cb)

        if not event.wait(timeout=float(duration + 15)):
            self.get_logger().error("Execution timeout")
            return False

        code = res_holder[0].result.error_code
        if code == 0:
            return True
        else:
            self.get_logger().error(f"Motion error: {code}")
            return False


def main():
    rclpy.init()
    node = VisionPickNode()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
