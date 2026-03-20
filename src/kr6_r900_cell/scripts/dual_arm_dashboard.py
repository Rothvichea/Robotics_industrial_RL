#!/usr/bin/env python3
"""
Dual Arm Industrial RL Dashboard
==================================
Live monitoring of:
  - Arm 1 + Arm 2 joint states from Gazebo
  - RL agent decisions (which arm, which action)
  - YOLO defect rate + per-box verdict
  - Arm status: IDLE / PICKING / PLACING / DONE
"""

import sys
import os
import threading
import time
import json
import numpy as np

sys.path.insert(0, "/home/genji/miniconda3/envs/industrial-ai/lib/python3.10/site-packages")

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QProgressBar, QFrame, QPushButton,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont


# ── shared state ──────────────────────────────────────────────────
class SystemState:
    def __init__(self):
        self._lock = threading.Lock()

        # joint states
        self.arm1_joints = [0.0] * 6
        self.arm2_joints = [0.0] * 6
        self.arm1_names  = [f"arm_1_joint_{i}" for i in range(1,7)]
        self.arm2_names  = [f"arm_2_joint_{i}" for i in range(1,7)]

        # arm status
        self.arm1_status = "IDLE"
        self.arm2_status = "IDLE"

        # inspection results
        self.box_results  = {}   # {box_name: {verdict, conf, action}}
        self.defect_count = 0
        self.good_count   = 0
        self.total_count  = 0

        # RL decisions
        self.rl_decisions = []   # list of strings
        self.current_arm  = None
        self.current_box  = None

        # log
        self.log_lines = []

    def update_joints(self, msg: JointState):
        with self._lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    if name.startswith("arm_1_"):
                        idx = int(name[-1]) - 1
                        if 0 <= idx < 6:
                            self.arm1_joints[idx] = msg.position[i]
                    elif name.startswith("arm_2_"):
                        idx = int(name[-1]) - 1
                        if 0 <= idx < 6:
                            self.arm2_joints[idx] = msg.position[i]

    def add_log(self, msg: str):
        with self._lock:
            ts = time.strftime("%H:%M:%S")
            self.log_lines.append(f"[{ts}] {msg}")
            if len(self.log_lines) > 15:
                self.log_lines = self.log_lines[-15:]

    def set_arm_status(self, arm: int, status: str):
        with self._lock:
            if arm == 1:
                self.arm1_status = status
            else:
                self.arm2_status = status

    def add_box_result(self, name: str, verdict: str, conf: float, action: str):
        with self._lock:
            self.box_results[name] = {
                "verdict": verdict,
                "conf": conf,
                "action": action,
            }
            self.total_count += 1
            if verdict == "DEFECT":
                self.defect_count += 1
            else:
                self.good_count += 1

    def add_rl_decision(self, msg: str):
        with self._lock:
            self.rl_decisions.append(msg)
            if len(self.rl_decisions) > 8:
                self.rl_decisions = self.rl_decisions[-8:]


STATE = SystemState()


# ── ROS2 node ─────────────────────────────────────────────────────
class DashboardNode(Node):
    def __init__(self):
        super().__init__("dual_arm_dashboard_node")
        cb = ReentrantCallbackGroup()

        self.create_subscription(
            JointState, "/joint_states",
            self._joint_cb, 10, callback_group=cb)

        self.create_subscription(
            String, "/inspection/result",
            self._inspection_cb, 10, callback_group=cb)

        self.create_subscription(
            String, "/arm/status",
            lambda m: STATE.set_arm_status(1, m.data), 10, callback_group=cb)

        self.create_subscription(
            String, "/rl/decision",
            self._rl_cb, 10, callback_group=cb)

        STATE.add_log("Dashboard node connected")
        STATE.add_log("Listening to /joint_states, /inspection/result")

    def _joint_cb(self, msg: JointState):
        STATE.update_joints(msg)

    def _inspection_cb(self, msg: String):
        try:
            result  = json.loads(msg.data)
            verdict = result.get("verdict", "GOOD")
            conf    = result.get("confidence", 0.0)
            action  = result.get("action", "ACCEPT")
            product = result.get("product", "?")
            frame   = result.get("frame", 0)
            STATE.add_box_result(f"obj_{frame}", verdict, conf, action)
            arm = 1 if action == "REJECT" else 2
            STATE.add_log(
                f"{product} {verdict} conf={conf:.2f} → ARM {arm} ({action})")
            STATE.add_rl_decision(
                f"ARM {arm}: {action}  {product}  {conf:.2f}")
        except Exception:
            pass

    def _rl_cb(self, msg: String):
        STATE.add_rl_decision(msg.data)
        STATE.add_log(f"RL: {msg.data}")


# ── UI helpers ────────────────────────────────────────────────────
STATUS_COLORS = {
    "IDLE":     "#666666",
    "PICKING":  "#44aaff",
    "PLACING":  "#ffaa00",
    "DONE":     "#44cc44",
    "ERROR":    "#ff4444",
}

JOINT_LIMITS = [
    (-2.967, 2.967), (-3.316, 0.785),
    (-2.094, 2.723), (-3.229, 3.229),
    (-2.094, 2.094), (-6.109, 6.109),
]


def mlabel(text, size=10, bold=False, color=None):
    lbl = QLabel(text)
    f = QFont("Monospace", size)
    f.setBold(bold)
    lbl.setFont(f)
    if color:
        lbl.setStyleSheet(f"color: {color};")
    return lbl


def hsep():
    l = QFrame()
    l.setFrameShape(QFrame.Shape.HLine)
    l.setStyleSheet("color: #333;")
    return l


# ── main window ───────────────────────────────────────────────────
class DualArmDashboard(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual Arm Industrial RL Dashboard")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet("background-color: #111122; color: #e0e0e0;")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        # col 1 — arm 1
        root.addLayout(self._build_arm_panel(1), 1)

        # col 2 — center (vision + RL + log)
        root.addLayout(self._build_center_panel(), 1)

        # col 3 — arm 2
        root.addLayout(self._build_arm_panel(2), 1)

        # timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(80)

    def _build_arm_panel(self, arm: int):
        v = QVBoxLayout()
        color = "#4466ff" if arm == 1 else "#ff6644"
        label = "ARM 1 — DEFECT → REJECT" if arm == 1 else "ARM 2 — GOOD → ACCEPT"

        v.addWidget(mlabel(label, 9, bold=True, color=color))
        v.addWidget(hsep())

        # status
        lbl_status = mlabel("IDLE", 20, bold=True,
                             color=STATUS_COLORS["IDLE"])
        lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(lbl_status)

        # joint bars
        bars = []
        lbls = []
        grid = QGridLayout()
        for i in range(6):
            grid.addWidget(mlabel(f"J{i+1}", 8), i, 0)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(50)
            bar.setTextVisible(False)
            bar.setFixedHeight(12)
            bar.setStyleSheet(
                f"QProgressBar{{background:#1a1a3a;border-radius:3px;}}"
                f"QProgressBar::chunk{{background:{color};border-radius:3px;}}")
            grid.addWidget(bar, i, 1)
            lbl = mlabel("  0.000", 8, color="#aaaaaa")
            grid.addWidget(lbl, i, 2)
            bars.append(bar)
            lbls.append(lbl)

        v.addLayout(grid)
        v.addStretch()

        if arm == 1:
            self.arm1_status_lbl = lbl_status
            self.arm1_bars = bars
            self.arm1_lbls = lbls
        else:
            self.arm2_status_lbl = lbl_status
            self.arm2_bars = bars
            self.arm2_lbls = lbls

        return v

    def _build_center_panel(self):
        v = QVBoxLayout()

        # vision QC
        v.addWidget(mlabel("VISION QC", 9, bold=True, color="#aaaaff"))
        v.addWidget(hsep())

        hv = QHBoxLayout()
        self.lbl_defects  = mlabel("0", 22, bold=True, color="#ff4444")
        self.lbl_good     = mlabel("0", 22, bold=True, color="#44cc44")
        self.lbl_total    = mlabel("0", 22, bold=True, color="#aaaaaa")
        for lbl, caption in [
            (self.lbl_defects, "DEFECT"),
            (self.lbl_good,    "GOOD"),
            (self.lbl_total,   "TOTAL"),
        ]:
            col = QVBoxLayout()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(lbl)
            col.addWidget(mlabel(caption, 8, color="#666666"))
            hv.addLayout(col)
        v.addLayout(hv)

        # defect rate bar
        v.addWidget(mlabel("Defect rate", 8, color="#888888"))
        self.defect_rate_bar = QProgressBar()
        self.defect_rate_bar.setRange(0, 100)
        self.defect_rate_bar.setValue(0)
        self.defect_rate_bar.setFixedHeight(14)
        self.defect_rate_bar.setStyleSheet(
            "QProgressBar{background:#1a1a3a;border-radius:4px;}"
            "QProgressBar::chunk{background:#ff4444;border-radius:4px;}")
        v.addWidget(self.defect_rate_bar)

        v.addSpacing(10)

        # RL decisions
        v.addWidget(mlabel("RL DECISIONS", 9, bold=True, color="#aaaaff"))
        v.addWidget(hsep())
        self.rl_lbl = QLabel("")
        self.rl_lbl.setFont(QFont("Monospace", 8))
        self.rl_lbl.setStyleSheet(
            "color:#88aaff; background:#0a0a1a; padding:6px; border-radius:4px;")
        self.rl_lbl.setWordWrap(True)
        self.rl_lbl.setFixedHeight(120)
        v.addWidget(self.rl_lbl)

        v.addSpacing(10)

        # event log
        v.addWidget(mlabel("EVENT LOG", 9, bold=True, color="#aaaaff"))
        v.addWidget(hsep())
        self.log_lbl = QLabel("")
        self.log_lbl.setFont(QFont("Monospace", 8))
        self.log_lbl.setStyleSheet(
            "color:#aaaaaa; background:#0a0a1a; padding:6px; border-radius:4px;")
        self.log_lbl.setWordWrap(True)
        self.log_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.log_lbl.setFixedHeight(160)
        v.addWidget(self.log_lbl)

        v.addStretch()
        return v

    def refresh(self):
        with STATE._lock:
            a1j = list(STATE.arm1_joints)
            a2j = list(STATE.arm2_joints)
            a1s = STATE.arm1_status
            a2s = STATE.arm2_status
            defects = STATE.defect_count
            good    = STATE.good_count
            total   = STATE.total_count
            rl      = list(STATE.rl_decisions)
            log     = list(STATE.log_lines)

        # arm 1 joints
        for i in range(6):
            lo, hi = JOINT_LIMITS[i]
            pct = int((a1j[i] - lo) / (hi - lo) * 100)
            pct = max(0, min(100, pct))
            self.arm1_bars[i].setValue(pct)
            self.arm1_lbls[i].setText(f"{a1j[i]:+.3f}")

        # arm 2 joints
        for i in range(6):
            lo, hi = JOINT_LIMITS[i]
            pct = int((a2j[i] - lo) / (hi - lo) * 100)
            pct = max(0, min(100, pct))
            self.arm2_bars[i].setValue(pct)
            self.arm2_lbls[i].setText(f"{a2j[i]:+.3f}")

        # arm status
        c1 = STATUS_COLORS.get(a1s, "#666666")
        c2 = STATUS_COLORS.get(a2s, "#666666")
        self.arm1_status_lbl.setText(a1s)
        self.arm1_status_lbl.setStyleSheet(
            f"color:{c1}; font-size:20px; font-weight:bold;")
        self.arm2_status_lbl.setText(a2s)
        self.arm2_status_lbl.setStyleSheet(
            f"color:{c2}; font-size:20px; font-weight:bold;")

        # vision
        self.lbl_defects.setText(str(defects))
        self.lbl_good.setText(str(good))
        self.lbl_total.setText(str(total))
        rate = int(defects / max(total, 1) * 100)
        self.defect_rate_bar.setValue(rate)

        # RL decisions
        self.rl_lbl.setText("\n".join(rl[-6:]))

        # log
        self.log_lbl.setText("\n".join(log))


def ros_spin(node):
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()


def main():
    rclpy.init()
    node = DashboardNode()

    ros_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DualArmDashboard()
    win.show()

    try:
        sys.exit(app.exec())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
