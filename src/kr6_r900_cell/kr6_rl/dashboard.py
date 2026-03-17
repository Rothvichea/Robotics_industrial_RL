"""
KR6 Industrial RL Dashboard — PyQt6
=====================================
Live monitoring of:
  - Joint states (6 joints, bar indicators)
  - RL agent status (episode, steps, reward, dist to goal)
  - Vision QC (defect rate, last detection, severity)
  - Arm status (RUNNING / REPLANNING / ABORTED / IDLE)
"""

import sys
import os
import numpy as np
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QProgressBar, QPushButton, QFrame,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPalette

from stable_baselines3 import PPO
from kr6_env import KR6Env, GOALS
from print_monitor import PrintMonitor


MODEL_PATH   = os.path.expanduser("~/Robotics_industrial_RL/rl_models/best_model.zip")
CLEAN_IMG    = os.path.expanduser("~/smart-factory-agent/test_clean.jpg")


# ── shared state updated by worker thread ─────────────────────────
class RobotState(QObject):
    updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.joint_pos    = np.zeros(6)
        self.joint_goal   = np.zeros(6)
        self.dist         = 0.0
        self.episode      = 0
        self.steps        = 0
        self.reward       = 0.0
        self.status       = "IDLE"
        self.defect_rate  = 0.0
        self.defect_count = 0
        self.total_frames = 0
        self.last_severity= "none"
        self.last_action  = "continue"
        self.log_lines    = []
        self._lock        = threading.Lock()

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
        self.updated.emit()

    def add_log(self, msg: str):
        with self._lock:
            ts = time.strftime("%H:%M:%S")
            self.log_lines.append(f"[{ts}] {msg}")
            if len(self.log_lines) > 12:
                self.log_lines = self.log_lines[-12:]
        self.updated.emit()


STATE = RobotState()


# ── worker thread: runs RL + vision loop ──────────────────────────
class RLWorker(threading.Thread):
    def __init__(self, n_episodes=20, defect_prob=0.08):
        super().__init__(daemon=True)
        self.n_episodes  = n_episodes
        self.defect_prob = defect_prob
        self._stop_flag  = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        import cv2

        env     = KR6Env()
        model   = PPO.load(MODEL_PATH, env=env)
        monitor = PrintMonitor(product_type="screw", device="cuda")
        STATE.add_log("RL agent + vision monitor ready")

        clean = cv2.imread(CLEAN_IMG) if os.path.exists(CLEAN_IMG) else None

        for ep in range(self.n_episodes):
            if self._stop_flag:
                break

            obs, _  = env.reset()
            done    = False
            steps   = 0
            aborted = False
            ep_rew  = 0.0
            info    = {"reached": False, "dist": 999.0}

            STATE.update(
                episode=ep+1,
                joint_goal=env.goal.copy(),
                status="RUNNING",
            )
            STATE.add_log(f"Ep {ep+1} — goal: {np.round(env.goal,2)}")

            while not done and not self._stop_flag:
                inject = np.random.random() < self.defect_prob
                if inject and clean is not None:
                    frame = cv2.resize(clean, (320, 320))
                else:
                    frame = np.ones((320,320,3), dtype=np.uint8) * 40

                insp = monitor.inspect(frame)

                if insp.has_defect:
                    STATE.add_log(
                        f"  DEFECT sev={insp.severity} "
                        f"conf={insp.confidence:.2f} → {insp.action}"
                    )

                if insp.action == "abort":
                    aborted = True
                    STATE.update(status="ABORTED")
                    STATE.add_log(f"  ABORTED at step {steps}")
                    break
                elif insp.action == "stop_and_replan":
                    STATE.update(status="REPLANNING")
                    env.goal = GOALS[np.random.randint(len(GOALS))].copy()
                    obs      = env._obs()
                    STATE.add_log(f"  Replanning → new goal")

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_rew += reward
                steps  += 1
                done    = terminated or truncated

                STATE.update(
                    joint_pos    = env.joint_pos.copy(),
                    joint_goal   = env.goal.copy(),
                    dist         = info["dist"],
                    steps        = steps,
                    reward       = ep_rew,
                    defect_rate  = monitor.defect_count / max(monitor.total_frames, 1),
                    defect_count = monitor.defect_count,
                    total_frames = monitor.total_frames,
                    last_severity= insp.severity,
                    last_action  = insp.action,
                    status       = "RUNNING" if not aborted else "ABORTED",
                )
                time.sleep(0.02)  # slow down for visualization

            if not aborted:
                status = "REACHED" if info["reached"] else "TIMEOUT"
                STATE.update(status=status)
                STATE.add_log(
                    f"  → {status} steps={steps} dist={info['dist']:.4f}"
                )
            time.sleep(0.5)

        STATE.update(status="IDLE")
        STATE.add_log("All episodes complete.")


# ── UI helpers ────────────────────────────────────────────────────
def make_label(text, size=11, bold=False, color=None):
    lbl = QLabel(text)
    f   = QFont("Monospace", size)
    f.setBold(bold)
    lbl.setFont(f)
    if color:
        lbl.setStyleSheet(f"color: {color};")
    return lbl


def separator():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #444;")
    return line


STATUS_COLORS = {
    "IDLE":       "#888888",
    "RUNNING":    "#44cc44",
    "REPLANNING": "#ffaa00",
    "ABORTED":    "#ff4444",
    "REACHED":    "#44ccff",
    "TIMEOUT":    "#ff8800",
}

SEVERITY_COLORS = {
    "none":   "#44cc44",
    "low":    "#ffff00",
    "medium": "#ffaa00",
    "high":   "#ff4444",
}


# ── main window ───────────────────────────────────────────────────
class Dashboard(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("KR6 Industrial RL Dashboard")
        self.setMinimumSize(900, 640)
        self.setStyleSheet("background-color: #1a1a2e; color: #e0e0e0;")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        # left column
        left = QVBoxLayout()
        left.addWidget(self._build_status_panel())
        left.addWidget(separator())
        left.addWidget(self._build_joint_panel())
        left.addStretch()
        root.addLayout(left, 1)

        # right column
        right = QVBoxLayout()
        right.addWidget(self._build_rl_panel())
        right.addWidget(separator())
        right.addWidget(self._build_vision_panel())
        right.addWidget(separator())
        right.addWidget(self._build_log_panel())
        right.addStretch()
        root.addLayout(right, 1)

        # bottom buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start")
        self.btn_stop  = QPushButton("■  Stop")
        self.btn_start.setStyleSheet(
            "background:#1a6b3a; color:white; padding:8px 20px; border-radius:4px;")
        self.btn_stop.setStyleSheet(
            "background:#6b1a1a; color:white; padding:8px 20px; border-radius:4px;")
        self.btn_start.clicked.connect(self.start_worker)
        self.btn_stop.clicked.connect(self.stop_worker)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(root)
        main_layout.addLayout(btn_layout)
        central.setLayout(main_layout)

        self.worker = None

        # connect state signal
        STATE.updated.connect(self.refresh)

        # timer fallback
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

    # ── panel builders ────────────────────────────────────────────
    def _build_status_panel(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(make_label("ARM STATUS", 10, bold=True, color="#aaaaff"))
        self.lbl_status = make_label("IDLE", 22, bold=True, color="#888888")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl_status)
        return w

    def _build_joint_panel(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(make_label("JOINT POSITIONS", 10, bold=True, color="#aaaaff"))
        self.joint_bars  = []
        self.joint_lbls  = []
        self.goal_lbls   = []
        grid = QGridLayout()
        for i in range(6):
            grid.addWidget(make_label(f"J{i+1}", 9), i, 0)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(50)
            bar.setTextVisible(False)
            bar.setFixedHeight(14)
            bar.setStyleSheet(
                "QProgressBar{background:#2a2a4a; border-radius:3px;}"
                "QProgressBar::chunk{background:#4466ff; border-radius:3px;}")
            grid.addWidget(bar, i, 1)
            lbl_v = make_label("0.000", 9, color="#cccccc")
            lbl_g = make_label("→ 0.000", 9, color="#44cc88")
            grid.addWidget(lbl_v, i, 2)
            grid.addWidget(lbl_g, i, 3)
            self.joint_bars.append(bar)
            self.joint_lbls.append(lbl_v)
            self.goal_lbls.append(lbl_g)
        v.addLayout(grid)
        return w

    def _build_rl_panel(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(make_label("RL AGENT", 10, bold=True, color="#aaaaff"))
        grid = QGridLayout()
        self.lbl_ep     = make_label("0", 11, color="#ffffff")
        self.lbl_steps  = make_label("0", 11, color="#ffffff")
        self.lbl_reward = make_label("0.0", 11, color="#ffffff")
        self.lbl_dist   = make_label("0.0000", 11, color="#ffffff")
        for i, (name, lbl) in enumerate([
            ("Episode",  self.lbl_ep),
            ("Steps",    self.lbl_steps),
            ("Reward",   self.lbl_reward),
            ("Dist",     self.lbl_dist),
        ]):
            grid.addWidget(make_label(name, 9, color="#888888"), i//2, (i%2)*2)
            grid.addWidget(lbl,                                  i//2, (i%2)*2+1)
        v.addLayout(grid)
        return w

    def _build_vision_panel(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(make_label("VISION QC", 10, bold=True, color="#aaaaff"))
        grid = QGridLayout()
        self.lbl_defect_rate  = make_label("0.0%",    11, color="#44cc44")
        self.lbl_defect_count = make_label("0/0",     11, color="#cccccc")
        self.lbl_severity     = make_label("none",    11, color="#44cc44")
        self.lbl_vaction      = make_label("continue",11, color="#cccccc")
        for i, (name, lbl) in enumerate([
            ("Defect rate",  self.lbl_defect_rate),
            ("Count",        self.lbl_defect_count),
            ("Severity",     self.lbl_severity),
            ("Action",       self.lbl_vaction),
        ]):
            grid.addWidget(make_label(name, 9, color="#888888"), i//2, (i%2)*2)
            grid.addWidget(lbl,                                  i//2, (i%2)*2+1)
        v.addLayout(grid)
        return w

    def _build_log_panel(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(make_label("EVENT LOG", 10, bold=True, color="#aaaaff"))
        self.log_lbl = QLabel("")
        self.log_lbl.setFont(QFont("Monospace", 8))
        self.log_lbl.setStyleSheet("color: #aaaaaa; background:#0d0d1a; padding:6px;")
        self.log_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.log_lbl.setWordWrap(True)
        self.log_lbl.setFixedHeight(160)
        v.addWidget(self.log_lbl)
        return w

    # ── refresh UI from state ─────────────────────────────────────
    def refresh(self):
        # status
        color = STATUS_COLORS.get(STATE.status, "#888888")
        self.lbl_status.setText(STATE.status)
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold;")

        # joints
        LIMITS_LOW  = np.array([-2.967, -3.316, -2.094, -3.229, -2.094, -6.109])
        LIMITS_HIGH = np.array([ 2.967,  0.785,  2.723,  3.229,  2.094,  6.109])
        for i in range(6):
            val   = STATE.joint_pos[i]
            goal  = STATE.joint_goal[i]
            lo, hi = LIMITS_LOW[i], LIMITS_HIGH[i]
            pct   = int((val - lo) / (hi - lo) * 100)
            pct   = np.clip(pct, 0, 100)
            self.joint_bars[i].setValue(pct)
            self.joint_lbls[i].setText(f"{val:+.3f}")
            self.goal_lbls[i].setText(f"→{goal:+.3f}")

        # RL
        self.lbl_ep.setText(str(STATE.episode))
        self.lbl_steps.setText(str(STATE.steps))
        self.lbl_reward.setText(f"{STATE.reward:.1f}")
        self.lbl_dist.setText(f"{STATE.dist:.4f}")

        # vision
        rate_pct = STATE.defect_rate * 100
        rate_color = "#44cc44" if rate_pct < 5 else "#ffaa00" if rate_pct < 15 else "#ff4444"
        self.lbl_defect_rate.setText(f"{rate_pct:.1f}%")
        self.lbl_defect_rate.setStyleSheet(f"color: {rate_color};")
        self.lbl_defect_count.setText(f"{STATE.defect_count}/{STATE.total_frames}")
        sev_color = SEVERITY_COLORS.get(STATE.last_severity, "#888888")
        self.lbl_severity.setText(STATE.last_severity)
        self.lbl_severity.setStyleSheet(f"color: {sev_color};")
        self.lbl_vaction.setText(STATE.last_action)

        # log
        with STATE._lock:
            lines = list(STATE.log_lines)
        self.log_lbl.setText("\n".join(lines))

    # ── buttons ───────────────────────────────────────────────────
    def start_worker(self):
        if self.worker and self.worker.is_alive():
            return
        self.worker = RLWorker(n_episodes=20, defect_prob=0.08)
        self.worker.start()
        STATE.add_log("Worker started.")

    def stop_worker(self):
        if self.worker:
            self.worker.stop()
        STATE.update(status="IDLE")
        STATE.add_log("Worker stopped.")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = Dashboard()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
