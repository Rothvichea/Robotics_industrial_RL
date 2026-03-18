#!/usr/bin/env python3
"""
KR6 Pick & Place — direct controller, threaded executor
"""

import sys
import os
sys.path.insert(0, "/home/genji/miniconda3/envs/industrial-ai/lib/python3.10/site-packages")

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import json
import time
import threading


JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]

POSES = {
    "home":           [0.0,    -1.5708,  1.5708, 0.0,  0.0,  0.0],
    "approach_table": [0.0,    -1.0,     1.6,    0.0, -0.2,  0.0],
    "pick":           [0.0,    -0.8,     1.8,    0.0, -0.3,  0.0],
    "lift":           [0.0,    -1.0,     1.6,    0.0, -0.2,  0.0],
    "place_accept":   [0.7,    -1.0,     1.5,    0.0, -0.2,  0.0],
    "place_reject":   [-0.7,   -1.0,     1.5,    0.0, -0.2,  0.0],
    "retreat":        [0.0,    -1.5708,  1.5708, 0.0,  0.0,  0.0],
}

DURATIONS = {
    "home":           4,
    "approach_table": 4,
    "pick":           4,
    "lift":           3,
    "place_accept":   5,
    "place_reject":   5,
    "retreat":        4,
}


class PickPlaceNode(Node):

    def __init__(self):
        super().__init__("pick_place_node")

        self._cb_group = ReentrantCallbackGroup()

        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self._cb_group,
        )

        self._sub = self.create_subscription(
            String, "/inspection/result",
            self.inspection_callback, 10,
            callback_group=self._cb_group,
        )

        self._busy      = False
        self._processed = 0
        self._lock      = threading.Lock()

        self.get_logger().info("Waiting for joint_trajectory_controller...")
        self._client.wait_for_server()
        self.get_logger().info("Pick & Place node ready!")

    def inspection_callback(self, msg: String):
        with self._lock:
            if self._busy:
                return
            try:
                result = json.loads(msg.data)
            except json.JSONDecodeError:
                return

            frame = result.get("frame", 0)
            if frame % 10 != 0:
                return

            self._processed += 1
            verdict = result.get("verdict", "GOOD")
            action  = result.get("action",  "ACCEPT")
            conf    = result.get("confidence", 0.0)

            self.get_logger().info(
                f"[#{self._processed}] verdict={verdict} "
                f"action={action} conf={conf:.3f}")

            self._busy = True

        # run in thread to not block spin
        threading.Thread(
            target=self._run_sequence,
            args=(action,),
            daemon=True,
        ).start()

    def _run_sequence(self, action: str):
        try:
            self.execute_pick_place(action)
        finally:
            with self._lock:
                self._busy = False

    def execute_pick_place(self, action: str):
        place_pose = "place_accept" if action == "ACCEPT" else "place_reject"
        bin_name   = "ACCEPT BIN ✓" if action == "ACCEPT" else "REJECT BIN ✗"

        sequence = [
            ("home",           "Moving to home"),
            ("approach_table", "Approaching table"),
            ("pick",           "Picking box"),
            ("lift",           "Lifting"),
            (place_pose,       f"Placing → {bin_name}"),
            ("retreat",        "Retreating"),
        ]

        self.get_logger().info(f"Starting sequence → {bin_name}")

        for pose_name, desc in sequence:
            self.get_logger().info(f"  → {desc}")
            ok = self.move_to(pose_name)
            if not ok:
                self.get_logger().error(f"  ✗ Failed: {desc}")
                self.move_to("home")
                return
            time.sleep(0.3)

        self.get_logger().info(f"✓ Complete → {bin_name}")

    def move_to(self, pose_name: str) -> bool:
        positions = POSES[pose_name]
        duration  = DURATIONS[pose_name]

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        pt = JointTrajectoryPoint()
        pt.positions        = positions
        pt.velocities       = [0.0] * 6
        pt.time_from_start  = Duration(sec=duration, nanosec=0)
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # send goal and wait using event
        event      = threading.Event()
        gh_holder  = [None]
        res_holder = [None]

        def goal_cb(future):
            gh_holder[0] = future.result()
            event.set()

        self._client.send_goal_async(goal).add_done_callback(goal_cb)

        if not event.wait(timeout=10.0):
            self.get_logger().error(f"  Goal send timeout: {pose_name}")
            return False

        gh = gh_holder[0]
        if not gh.accepted:
            self.get_logger().error(f"  Goal rejected: {pose_name}")
            return False

        self.get_logger().info(f"  Executing {pose_name} ({duration}s)...")

        event.clear()

        def result_cb(future):
            res_holder[0] = future.result()
            event.set()

        gh.get_result_async().add_done_callback(result_cb)

        if not event.wait(timeout=float(duration + 20)):
            self.get_logger().error(f"  Result timeout: {pose_name}")
            return False

        code = res_holder[0].result.error_code
        if code == 0:
            self.get_logger().info(f"  ✓ {pose_name} reached!")
            return True
        else:
            self.get_logger().error(f"  ✗ {pose_name} error: {code}")
            return False


def main():
    rclpy.init()
    node = PickPlaceNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
