#!/usr/bin/env python3
"""
KR6 R900 — plan only mode, visualizes trajectory in RViz2
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    WorkspaceParameters,
    Constraints,
    JointConstraint,
    DisplayTrajectory,
    RobotState,
)
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Vector3
import time


JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6"
]

POSES = {
    "home":     [0.0,    -1.5708,  1.5708, 0.0,  0.0,  0.0],
    "pose_a":   [0.785,  -1.0,     1.2,    0.5,  0.3,  0.0],
    "pose_b":   [-0.785, -1.2,     1.5,   -0.5,  0.5,  0.0],
    "extended": [0.0,     0.0,     0.0,    0.0,  0.0,  0.0],
}


class TrajectoryNode(Node):

    def __init__(self):
        super().__init__("kr6_trajectory_node")
        self._client = ActionClient(self, MoveGroup, '/move_action')

        # publisher to visualize planned path in RViz2
        self._display_pub = self.create_publisher(
            DisplayTrajectory, '/display_planned_path', 10)

        self.get_logger().info("Waiting for move_group action server...")
        self._client.wait_for_server()
        self.get_logger().info("Connected to move_group!")

    def build_goal(self, joint_values, plan_only=True):
        constraints = Constraints()
        for name, value in zip(JOINT_NAMES, joint_values):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = value
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)

        request = MotionPlanRequest()
        request.group_name                      = "manipulator"
        request.num_planning_attempts           = 10
        request.allowed_planning_time           = 5.0
        request.max_velocity_scaling_factor     = 0.3
        request.max_acceleration_scaling_factor = 0.3
        request.goal_constraints.append(constraints)

        ws = WorkspaceParameters()
        ws.header.frame_id = "base_link"
        ws.min_corner = Vector3(x=-2.0, y=-2.0, z=-2.0)
        ws.max_corner = Vector3(x= 2.0, y= 2.0, z= 2.0)
        request.workspace_parameters = ws

        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only       = plan_only
        goal.planning_options.replan          = False
        goal.planning_options.replan_attempts = 0
        return goal

    def plan_to(self, name, joint_values):
        self.get_logger().info(f"── Planning to: {name}")
        goal   = self.build_goal(joint_values, plan_only=True)
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"Goal rejected for {name}")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        error = result.error_code.val
        if error == 1:
            self.get_logger().info(f"✓ Plan found for {name} — publishing to RViz2")
            # publish so RViz2 shows the planned path
            disp = DisplayTrajectory()
            disp.trajectory.append(result.planned_trajectory)
            disp.trajectory_start = result.trajectory_start
            self._display_pub.publish(disp)
            time.sleep(3.0)  # hold so you can see it in RViz2
        else:
            self.get_logger().error(f"✗ Planning failed for {name} — code: {error}")

        time.sleep(0.5)


def main():
    rclpy.init()
    node = TrajectoryNode()

    sequence = ["home", "pose_a", "pose_b", "extended", "home"]
    for pose_name in sequence:
        node.plan_to(pose_name, POSES[pose_name])

    node.get_logger().info("All plans complete!")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
