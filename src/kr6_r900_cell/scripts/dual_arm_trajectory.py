#!/usr/bin/env python3
"""
KR6 Dual Arm — synchronized trajectory planner
Plans arm_1 and arm_2 sequentially, publishes both to RViz2
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
)
from geometry_msgs.msg import Vector3
import time


ARM_1_JOINTS = [f"arm_1_joint_{i}" for i in range(1, 7)]
ARM_2_JOINTS = [f"arm_2_joint_{i}" for i in range(1, 7)]

# sequences: arm_1 reaches toward center, arm_2 mirrors it
SEQUENCE = [
    {
        "name": "both home",
        "arm_1": [0.0,    -1.5708,  1.5708, 0.0,  0.0,  0.0],
        "arm_2": [0.0,    -1.5708,  1.5708, 0.0,  0.0,  0.0],
    },
    {
        "name": "arm_1 reaches right, arm_2 reaches left",
        "arm_1": [ 0.8,   -1.0,     1.2,    0.0,  0.4,  0.0],
        "arm_2": [-0.8,   -1.0,     1.2,    0.0,  0.4,  0.0],
    },
    {
        "name": "arm_1 high, arm_2 low",
        "arm_1": [ 0.3,   -0.8,     0.8,    0.5,  0.3,  0.0],
        "arm_2": [ 0.3,   -1.8,     2.0,   -0.5,  0.3,  0.0],
    },
    {
        "name": "both home",
        "arm_1": [0.0,    -1.5708,  1.5708, 0.0,  0.0,  0.0],
        "arm_2": [0.0,    -1.5708,  1.5708, 0.0,  0.0,  0.0],
    },
]


class DualArmNode(Node):

    def __init__(self):
        super().__init__("dual_arm_trajectory_node")
        self._client = ActionClient(self, MoveGroup, '/move_action')
        self._display_pub = self.create_publisher(
            DisplayTrajectory, '/display_planned_path', 10)

        self.get_logger().info("Waiting for move_group...")
        self._client.wait_for_server()
        self.get_logger().info("Connected!")

    def build_goal(self, group_name, joint_names, joint_values):
        constraints = Constraints()
        for name, value in zip(joint_names, joint_values):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = value
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)

        request = MotionPlanRequest()
        request.group_name                      = group_name
        request.num_planning_attempts           = 10
        request.allowed_planning_time           = 5.0
        request.max_velocity_scaling_factor     = 0.3
        request.max_acceleration_scaling_factor = 0.3
        request.goal_constraints.append(constraints)

        ws = WorkspaceParameters()
        ws.header.frame_id = "world"
        ws.min_corner = Vector3(x=-2.5, y=-2.5, z=-2.5)
        ws.max_corner = Vector3(x= 2.5, y= 2.5, z= 2.5)
        request.workspace_parameters = ws

        goal = MoveGroup.Goal()
        goal.request                          = request
        goal.planning_options.plan_only       = True
        goal.planning_options.replan          = False
        goal.planning_options.replan_attempts = 0
        return goal

    def plan_arm(self, group, joint_names, joint_values):
        goal   = self.build_goal(group, joint_names, joint_values)
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"  Goal rejected for {group}")
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.error_code.val == 1:
            self.get_logger().info(f"  ✓ {group} plan found")
            return result
        else:
            self.get_logger().error(f"  ✗ {group} failed — code: {result.error_code.val}")
            return None

    def run_sequence(self):
        for step in SEQUENCE:
            self.get_logger().info(f"── Step: {step['name']}")

            # plan arm_1
            r1 = self.plan_arm("arm_1", ARM_1_JOINTS, step["arm_1"])
            # plan arm_2
            r2 = self.plan_arm("arm_2", ARM_2_JOINTS, step["arm_2"])

            # publish both trajectories to RViz2
            if r1 and r2:
                disp = DisplayTrajectory()
                disp.trajectory.append(r1.planned_trajectory)
                disp.trajectory.append(r2.planned_trajectory)
                disp.trajectory_start = r1.trajectory_start
                self._display_pub.publish(disp)
                self.get_logger().info("  → Published dual trajectory to RViz2")

            time.sleep(3.5)

        self.get_logger().info("Dual arm sequence complete!")


def main():
    rclpy.init()
    node = DualArmNode()
    node.run_sequence()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
