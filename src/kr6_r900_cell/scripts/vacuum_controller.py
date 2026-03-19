#!/usr/bin/env python3
"""
Vacuum Suction Controller
==========================
Simulates suction cup activation via Ignition contact + attach plugin
Publishes /vacuum/state and handles attach/detach of boxes
"""

import sys
import os
sys.path.insert(0, "/home/genji/miniconda3/envs/industrial-ai/lib/python3.10/site-packages")

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Pose
import threading
import time
import subprocess


class VacuumController(Node):

    def __init__(self):
        super().__init__("vacuum_controller")

        # vacuum state publisher
        self._state_pub = self.create_publisher(
            Bool, '/vacuum/active', 10)

        # command subscriber
        self.create_subscription(
            Bool, '/vacuum/cmd',
            self._cmd_cb, 10)

        # status publisher
        self._status_pub = self.create_publisher(
            String, '/vacuum/status', 10)

        self._active      = False
        self._attached    = False
        self._attached_box = None

        # publish state at 10Hz
        self.create_timer(0.1, self._publish_state)

        self.get_logger().info("Vacuum controller ready!")
        self.get_logger().info("  /vacuum/cmd  — Bool to activate/deactivate")
        self.get_logger().info("  /vacuum/active — current state")

    def _cmd_cb(self, msg: Bool):
        if msg.data and not self._active:
            self._activate()
        elif not msg.data and self._active:
            self._deactivate()

    def _activate(self):
        self._active = True
        self.get_logger().info("🔵 Vacuum ON — suction active")
        status = String()
        status.data = "VACUUM_ON"
        self._status_pub.publish(status)

    def _deactivate(self):
        self._active   = False
        self._attached = False
        self._attached_box = None
        self.get_logger().info("⚪ Vacuum OFF — suction released")
        status = String()
        status.data = "VACUUM_OFF"
        self._status_pub.publish(status)

    def _publish_state(self):
        msg = Bool()
        msg.data = self._active
        self._state_pub.publish(msg)


def main():
    rclpy.init()
    node = VacuumController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
