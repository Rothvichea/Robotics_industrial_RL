#!/usr/bin/env python3
"""
KR6 Multi-Box Inspection + Sort  (optimized)
=============================================
Fixes applied vs previous version:
  - ign service (not gz service) + ignition.msgs.Pose  ← boxes now actually move
  - Boxes are static in SDF so teleport is instant, no gravity fight
  - Pre-pick intermediate move keeps joint_5 near 0 → no 360° spin
  - IK seed = pre-pick joints so solver finds the clean "arm-over-table" solution
  - Smooth 3-step motion: HOME → PRE_PICK → APPROACH → PICK
"""

import sys, os, math, threading, subprocess, time
sys.path.insert(0, "/home/genji/miniconda3/envs/industrial-ai/lib/python3.10/site-packages")

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
from geometry_msgs.msg import PoseStamped
from ultralytics import YOLO
import cv2, numpy as np


# ─── Arm config — auto-detected in __init__ ───────────────────────────────────
# Single-arm fallback defaults (overwritten at runtime for dual-arm mode).
JOINT_NAMES      = [f"joint_{i}" for i in range(1, 7)]
CONTROLLER_NS    = "/joint_trajectory_controller/follow_joint_trajectory"
MOVEIT_GROUP     = "manipulator"
BASE_LINK        = "base_link"

# ─── YOLO models ──────────────────────────────────────────────────────────────
MODELS = {
    "bottle": os.path.expanduser("~/smart-factory-agent/models/yolo_bottle.pt"),
    "screw":  os.path.expanduser("~/smart-factory-agent/models/yolo_screw.pt"),
}

# ─── Geometry ─────────────────────────────────────────────────────────────────
TABLE_Z      = 0.45    # table top surface
BOX_HALF     = 0.03    # half of box height (0.06 m)
BOX_TOP_Z    = TABLE_Z + BOX_HALF * 2   # 0.51
BOX_CTR_Z    = TABLE_Z + BOX_HALF       # 0.48  (center)
TIP_OFFSET   = 0.065   # tool0 → suction tip distance when pointing down

# tool0 Z targets fed to IK  (tip = tool0 − TIP_OFFSET)
APPROACH_Z  = BOX_TOP_Z + 0.14 + TIP_OFFSET   # 0.715  (14 cm clearance)
PICK_Z      = BOX_TOP_Z + 0.004 + TIP_OFFSET   # 0.579  (tip just touching box)
LIFT_Z      = 0.82                              # carry height
PLACE_Z     = 0.48                              # drop over bin

# Bin world positions
ACCEPT_BIN = (0.6,  0.45)
REJECT_BIN = (0.6, -0.45)

# ─── Dual-arm config (defined after bins so they can be referenced) ───────────
# arm_2 is at world x=1.2 facing -x (yaw=π).
# World → arm_2 local:  lx = 1.2 - wx,  ly = -wy
ARM2_BASE_X = 1.2
ARM_CFG = {
    1: {
        "joints":    [f"arm_1_joint_{i}" for i in range(1, 7)],
        "ctrl":      "/arm_1_controller/follow_joint_trajectory",
        "group":     "arm_1",
        "base":      "arm_1_base_link",
        "verdict":   "DEFECT",
        "bin":       REJECT_BIN,
        "to_local":  lambda wx, wy: (wx, wy),
    },
    2: {
        "joints":    [f"arm_2_joint_{i}" for i in range(1, 7)],
        "ctrl":      "/arm_2_controller/follow_joint_trajectory",
        "group":     "arm_2",
        "base":      "arm_2_base_link",
        "verdict":   "GOOD",
        "bin":       ACCEPT_BIN,
        "to_local":  lambda wx, wy: (ARM2_BASE_X - wx, -wy),
    },
}

# Box center Z when "carried" at LIFT_Z and PLACE_Z
def carried_z(tool0_z):
    return tool0_z - TIP_OFFSET - BOX_HALF   # tip pos − half-box

# ─── Box definitions (match SDF) ──────────────────────────────────────────────
BOXES = [
    {"name": "box_defect_1", "product": "bottle", "wx": 0.655, "wy":  0.055},
    {"name": "box_defect_2", "product": "screw",  "wx": 0.655, "wy": -0.055},
    {"name": "box_good_1",   "product": "bottle", "wx": 0.545, "wy":  0.055},
    {"name": "box_good_2",   "product": "screw",  "wx": 0.545, "wy": -0.055},
]

# ─── Fixed joint configs ──────────────────────────────────────────────────────
HOME = [0.0, -1.5708, 0.0, 0.0, 0.0, 0.0]   # arm straight up

# Pre-pick: rough "arm over table" pose used as IK seed AND as an intermediate
# waypoint so joint_5 never needs to spin more than ~90°.
# joint_1 is adjusted per-box (atan2 of box position).
def pre_pick(wx, wy):
    j1 = math.atan2(wy, wx)   # base rotation toward box
    #         j1    j2     j3    j4     j5     j6
    return [  j1,  -0.9,  0.7,  0.0,  -0.8,  0.0 ]

# Ignition world name
IGN_WORLD = "inspection_cell"


# ─── Box teleporter (ign service, static boxes) ───────────────────────────────
def teleport_box(name: str, x: float, y: float, z: float):
    """Move a static box instantly via ign service set_pose."""
    req = (f'name: "{name}" '
           f'position {{ x: {x:.4f} y: {y:.4f} z: {z:.4f} }}')
    result = subprocess.run(
        ["ign", "service",
         "-s", f"/world/{IGN_WORLD}/set_pose",
         "--reqtype",  "ignition.msgs.Pose",
         "--reptype",  "ignition.msgs.Boolean",
         "--timeout",  "500",
         "--req",      req],
        capture_output=True, text=True, timeout=3.0,
    )
    if result.returncode != 0:
        print(f"[teleport] WARN: {result.stderr.strip()}")


# ─── Node ─────────────────────────────────────────────────────────────────────
class MultiBoxInspector(Node):

    def __init__(self):
        super().__init__("multi_box_inspector")
        cb = ReentrantCallbackGroup()

        # ── auto-detect single-arm vs dual-arm setup ──────────────────────────
        global JOINT_NAMES, CONTROLLER_NS, MOVEIT_GROUP, BASE_LINK
        try:
            _actions = subprocess.run(
                ["ros2", "action", "list"], capture_output=True, text=True, timeout=5
            ).stdout
        except Exception:
            _actions = ""

        self._dual_arm = "/arm_1_controller/follow_joint_trajectory" in _actions

        if self._dual_arm:
            JOINT_NAMES   = ARM_CFG[1]["joints"]
            CONTROLLER_NS = ARM_CFG[1]["ctrl"]
            MOVEIT_GROUP  = ARM_CFG[1]["group"]
            BASE_LINK     = ARM_CFG[1]["base"]
            self.get_logger().info("Auto-detected: DUAL ARM mode")
        else:
            self.get_logger().info("Auto-detected: SINGLE ARM mode")

        self._traj = ActionClient(
            self, FollowJointTrajectory,
            CONTROLLER_NS,
            callback_group=cb)

        # arm_2 client — only created in dual-arm mode
        self._traj2 = (
            ActionClient(self, FollowJointTrajectory,
                         ARM_CFG[2]["ctrl"], callback_group=cb)
            if self._dual_arm else None
        )

        self._ik = self.create_client(
            GetPositionIK, "/compute_ik", callback_group=cb)

        self._frame      = None
        self._frame_lock = threading.Lock()
        self.create_subscription(
            Image, "/inspection_cam/image_raw", self._cam_cb, 10,
            callback_group=cb)

        self._js      = None
        self._js_lock = threading.Lock()
        self.create_subscription(
            JointState, "/joint_states", self._js_cb, 10,
            callback_group=cb)

        self._models = {p: YOLO(m) for p, m in MODELS.items()}

        # dashboard publishers
        import json as _json
        self._json = _json
        self._pub_result = self.create_publisher(String, "/inspection/result", 10)
        self._pub_status = self.create_publisher(String, "/arm/status", 10)
        self._frame_count = 0

        # Shared mutex — only one arm may be in the table zone at a time.
        self._table_lock = threading.Lock()

        # The /compute_ik service client is NOT safe for concurrent async calls
        # from multiple threads — one client, one in-flight request at a time.
        self._ik_lock = threading.Lock()

        self.get_logger().info("Waiting for controller(s)…")
        self._traj.wait_for_server()
        if self._traj2:
            self._traj2.wait_for_server()
        self.get_logger().info("Waiting for IK service…")
        self._ik.wait_for_service()
        self.get_logger().info("Ready.")

        threading.Thread(target=self._run, daemon=True).start()

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _cam_cb(self, msg: Image):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        img  = data.reshape((msg.height, msg.width, 3))
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        with self._frame_lock:
            self._frame = img

    def _js_cb(self, msg: JointState):
        with self._js_lock:
            self._js = msg

    # ── inspection ────────────────────────────────────────────────────────────
    CAM_X   = 0.6
    CAM_Z   = 1.0
    CAM_FOV = 0.5   # radians
    MATCH_R = 0.12  # metres — max distance to attribute a detection to a box

    def _get_frame(self, timeout=6.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._frame_lock:
                if self._frame is not None:
                    return self._frame.copy()
            time.sleep(0.05)
        return None

    def _pixel_to_world(self, col, row, w, h):
        """Image (col, row) → world (wx, wy).
        Camera pitch=π/2, yaw=0:  col→+Y world,  row→−X world (inverted)
        """
        height = self.CAM_Z - TABLE_Z
        scale  = w / (2 * math.tan(self.CAM_FOV / 2) * height)
        wy = (col - w / 2) / scale
        wx = self.CAM_X - (row - h / 2) / scale
        return wx, wy

    def _nearest_sdf_name(self, wx, wy, used: set) -> str:
        """Return the SDF model name closest to (wx,wy), excluding already used names.
        Needed for ign service set_pose teleportation.
        Falls back to a generated name if all 4 known names are used (5th+ box).
        """
        best_name, best_dist = None, float("inf")
        for b in BOXES:
            if b["name"] in used:
                continue
            d = math.hypot(wx - b["wx"], wy - b["wy"])
            if d < best_dist:
                best_dist, best_name = d, b["name"]
        return best_name or f"box_extra_{len(used)}"

    def _find_and_classify(self) -> list:
        """Fully vision-driven detection — no hardcoded positions.
        1. Contour detection finds ALL objects on the white mat (any count).
        2. Pixel centre → world XY via camera model.
        3. YOLO classifies each object as DEFECT or GOOD.
        Returns list of dicts: {wx, wy, sdf_name, verdict, conf}
        """
        frame = self._get_frame()
        if frame is None:
            self.get_logger().error("No camera frame!")
            return []

        h, w = frame.shape[:2]

        # ── Step 1: find all objects via contour detection ────────────────────
        # Boxes are gray (~178/255) on a white mat (~242/255).
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Area filter: each box is 0.10×0.10 m → ~207k px² at this FOV/height
        MIN_AREA = 30_000
        MAX_AREA = 350_000

        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_AREA <= area <= MAX_AREA):
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            col = M["m10"] / M["m00"]
            row = M["m01"] / M["m00"]
            wx, wy = self._pixel_to_world(col, row, w, h)
            objects.append({"col": col, "row": row, "wx": wx, "wy": wy,
                             "verdict": "GOOD", "conf": 0.0})

        if not objects:
            self.get_logger().warn("No objects detected on table!")
            return []

        self.get_logger().info(f"  Contour detection: {len(objects)} object(s) found")

        # ── Step 2: YOLO on full frame → classify each detected object ────────
        frame_s = cv2.resize(frame, (640, 640))
        sx = w / 640.0
        sy = h / 640.0

        # Match radius in pixels: MATCH_R metres converted with camera scale.
        # Objects are ~456 px wide at this FOV; a detection center can be up to
        # ~228 px from the contour centroid, so we need a generous threshold.
        height    = self.CAM_Z - TABLE_Z
        scale_ppm = w / (2 * math.tan(self.CAM_FOV / 2) * height)
        thresh_px = self.MATCH_R * scale_ppm   # ~547 px  (matches old behaviour)

        for _, model in self._models.items():
            results = model.predict(source=frame_s, conf=0.15, verbose=False)
            if not results or results[0].boxes is None:
                continue
            for pred in results[0].boxes:
                conf = float(pred.conf[0].cpu())
                x1, y1, x2, y2 = pred.xyxy[0].cpu().numpy()
                u_det = (x1 + x2) / 2.0 * sx
                v_det = (y1 + y2) / 2.0 * sy

                # nearest contour object wins
                best_obj, best_px = None, float("inf")
                for obj in objects:
                    d = math.hypot(u_det - obj["col"], v_det - obj["row"])
                    if d < best_px:
                        best_px, best_obj = d, obj

                if best_obj and best_px < thresh_px and conf > best_obj["conf"]:
                    best_obj["verdict"] = "DEFECT"
                    best_obj["conf"]    = conf

        # ── Step 3: assign SDF names for teleportation ───────────────────────
        used_names = set()
        for obj in objects:
            name = self._nearest_sdf_name(obj["wx"], obj["wy"], used_names)
            obj["sdf_name"] = name
            used_names.add(name)

        # ── Log + publish to dashboard ────────────────────────────────────────
        for obj in objects:
            action = "→ REJECT ✗" if obj["verdict"] == "DEFECT" else "→ ACCEPT ✓"
            self.get_logger().info(
                f"  {obj['sdf_name']} at ({obj['wx']:.3f},{obj['wy']:.3f}) "
                f"{obj['verdict']} conf={obj['conf']:.2f}  {action}")
            self._frame_count += 1
            payload = self._json.dumps({
                "verdict":    obj["verdict"],
                "confidence": round(obj["conf"], 3),
                "action":     "REJECT" if obj["verdict"] == "DEFECT" else "ACCEPT",
                "product":    obj.get("product", "unknown"),
                "frame":      self._frame_count,
            })
            self._pub_result.publish(String(data=payload))

        return objects

    # ── IK ────────────────────────────────────────────────────────────────────
    def _ik_once(self, x, y, z, seed: list, group: str, base_link: str, joints: list):
        """Single raw IK call with a given seed.  Returns joint list or None."""
        req = GetPositionIK.Request()
        req.ik_request                  = PositionIKRequest()
        req.ik_request.group_name       = group
        req.ik_request.avoid_collisions = False  # table_lock handles physical safety
        req.ik_request.timeout.sec      = 2

        rs = RobotState()
        rs.joint_state.name     = joints
        rs.joint_state.position = seed
        req.ik_request.robot_state = rs

        pose = PoseStamped()
        pose.header.frame_id    = base_link
        pose.pose.position.x    = x
        pose.pose.position.y    = y
        pose.pose.position.z    = z
        # 180° around Y → tool Z points straight down
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 1.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0
        req.ik_request.pose_stamped = pose

        with self._ik_lock:   # one IK call in flight at a time
            done, result = threading.Event(), [None]
            self._ik.call_async(req).add_done_callback(
                lambda f: (result.__setitem__(0, f.result()), done.set()))
            if not done.wait(10.0):
                return None
            resp = result[0]
        if resp.error_code.val != 1:
            return None
        # The response contains the FULL robot state (all joints of all arms).
        # Extract only the joints that belong to THIS arm by matching names.
        sol_names = list(resp.solution.joint_state.name)
        sol_pos   = list(resp.solution.joint_state.position)
        name_to_pos = dict(zip(sol_names, sol_pos))
        extracted = [name_to_pos.get(jn) for jn in joints]
        if any(v is None for v in extracted):
            return None
        return extracted

    def _ik_solve(self, x, y, z, seed_joints=None, arm=1):
        """IK with automatic retry to avoid large joint jumps (wrist flips).
        x, y, z are already in the arm's LOCAL frame.
        """
        cfg     = ARM_CFG[arm] if self._dual_arm else {
            "group": MOVEIT_GROUP, "base": BASE_LINK, "joints": JOINT_NAMES}
        primary = seed_joints if seed_joints is not None else pre_pick(x, y)
        joints  = self._ik_once(x, y, z, primary,
                                cfg["group"], cfg["base"], cfg["joints"])

        if joints is None:
            self.get_logger().error(f"IK arm{arm} failed at ({x:.2f},{y:.2f},{z:.2f})")
            return None

        def _max_jump(sol):
            return max(abs(sol[i] - primary[i]) for i in range(6))

        best, best_jump = joints, _max_jump(joints)

        if best_jump > 1.5:
            j1_exp = math.atan2(y, x)
            retry_seeds = [
                (j1_exp,        0.0), (j1_exp,       0.3), (j1_exp,      -0.3),
                (j1_exp + 0.2,  0.0), (j1_exp - 0.2, 0.0),
                (j1_exp + 0.3,  0.3), (j1_exp - 0.3,-0.3),
                (j1_exp,       -0.9), (j1_exp,       0.9),  # varied j2 via seed
                (j1_exp + 0.1,  0.0), (j1_exp - 0.1, 0.0),
            ]
            for j1_bias, j4_bias in retry_seeds:
                alt_seed    = list(primary)
                alt_seed[0] = j1_bias
                alt_seed[3] = j4_bias
                alt = self._ik_once(x, y, z, alt_seed,
                                    cfg["group"], cfg["base"], cfg["joints"])
                if alt is None:
                    continue
                jump = _max_jump(alt)
                if jump < best_jump:
                    best_jump, best = jump, alt
                if best_jump < 1.0:
                    break

        self.get_logger().info(
            f"  ARM{arm} IK → {[round(j,3) for j in best]}  (max_jump={best_jump:.2f})")
        return best

    # ── motion ────────────────────────────────────────────────────────────────
    def _send_traj(self, waypoints: list, arm: int = 1) -> bool:
        """Send a multi-point trajectory for smooth continuous motion.
        waypoints = [(joint_positions, time_from_start_sec), ...]
        """
        j_names = (ARM_CFG[arm]["joints"] if self._dual_arm else JOINT_NAMES)
        client  = self._traj if arm == 1 else self._traj2

        traj             = JointTrajectory()
        traj.joint_names = j_names

        for positions, t in waypoints:
            pt               = JointTrajectoryPoint()
            pt.positions     = list(positions)
            pt.velocities    = [0.0] * 6
            pt.time_from_start = Duration(
                sec=int(t), nanosec=int((t % 1) * 1_000_000_000))
            traj.points.append(pt)

        goal            = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        total_t         = waypoints[-1][1]

        done      = threading.Event()
        gh_holder = [None]

        def _goal_cb(f):
            gh_holder[0] = f.result()
            done.set()

        client.send_goal_async(goal).add_done_callback(_goal_cb)
        if not done.wait(15.0):
            return False

        gh = gh_holder[0]
        if not gh.accepted:
            self.get_logger().error(f"ARM{arm} trajectory rejected")
            return False

        done.clear()
        gh.get_result_async().add_done_callback(lambda _: done.set())
        done.wait(float(total_t + 20))
        return True

    def _move(self, positions, duration: float, arm: int = 1) -> bool:
        """Single-point move — for HOME and simple transitions."""
        return self._send_traj([(positions, duration)], arm=arm)

    # ── nearest-neighbour ordering ────────────────────────────────────────────
    @staticmethod
    def _nearest_neighbour(objects: list, start_x: float, start_y: float) -> list:
        """Greedy nearest-neighbour sort.
        After placing each box the robot ends up at a bin; the next box chosen
        is the one closest (world XY) to that bin — minimises total travel.
        """
        remaining = list(objects)
        ordered   = []
        cx, cy    = start_x, start_y
        while remaining:
            nearest = min(remaining,
                          key=lambda o: math.hypot(o["wx"] - cx, o["wy"] - cy))
            ordered.append(nearest)
            remaining.remove(nearest)
            # next start = bin where this box will be dropped
            defect = nearest["verdict"] == "DEFECT"
            cx, cy = REJECT_BIN if defect else ACCEPT_BIN
        return ordered

    # ── pick & place ──────────────────────────────────────────────────────────
    def _pick_and_place(self, obj: dict, last_joints=None, arm: int = 1) -> list:
        """Pick obj with the given arm and drop in its bin.
        Returns place_j for IK seed cascading.

        Collision strategy — TABLE LOCK:
          Phase 1 (lock held):   pre_pick → approach → pick → lift
            Only ONE arm may be in this zone at a time.
          Phase 2 (lock free):   lift → place at bin
            Both arms can fly to their bins simultaneously — no overlap possible.

        arm=1 picks DEFECT → REJECT_BIN  (base at x=0,   faces +x)
        arm=2 picks GOOD   → ACCEPT_BIN  (base at x=1.2, faces -x)
        """
        cfg      = ARM_CFG[arm] if self._dual_arm else {
            "to_local": lambda wx, wy: (wx, wy),
            "bin": REJECT_BIN if obj["verdict"] == "DEFECT" else ACCEPT_BIN,
        }
        to_local = cfg["to_local"]

        wx,  wy    = obj["wx"], obj["wy"]
        bx_w, by_w = cfg["bin"]
        lx,  ly    = to_local(wx,   wy)       # box in arm-local frame
        blx, bly   = to_local(bx_w, by_w)     # bin  in arm-local frame
        label      = "REJECT ✗" if obj["verdict"] == "DEFECT" else "ACCEPT ✓"

        self.get_logger().info(
            f"▶ ARM{arm} {obj['sdf_name']} world({wx:.3f},{wy:.3f})"
            f" local({lx:.3f},{ly:.3f}) → {label}")

        # ── Pre-compute all IK before touching any lock ────────────────────────
        # Approach always uses pre_pick as seed — the bin place_j is a distant
        # configuration that KDL cannot converge from to the table approach pose.
        # Cascade (last_joints → approach → pick → lift → place) only applies
        # for pick/lift/place which are small incremental moves.
        approach_j = self._ik_solve(lx,  ly,  APPROACH_Z, pre_pick(lx, ly), arm=arm)
        if approach_j is None:
            self.get_logger().error(f"ARM{arm} IK failed (approach) — skipping")
            return last_joints
        pick_j  = self._ik_solve(lx,  ly,  PICK_Z,    approach_j, arm=arm)
        if pick_j is None:
            self.get_logger().error(f"ARM{arm} IK failed (pick) — skipping")
            return last_joints
        lift_j  = self._ik_solve(lx,  ly,  LIFT_Z,    pick_j,     arm=arm)
        if lift_j is None:
            self.get_logger().error(f"ARM{arm} IK failed (lift) — skipping")
            return last_joints
        place_j = self._ik_solve(blx, bly, PLACE_Z,   lift_j,     arm=arm)
        if place_j is None:
            self.get_logger().error(f"ARM{arm} IK failed (place) — skipping")
            return last_joints

        # ── Phase 1: TABLE ZONE — acquire lock ────────────────────────────────
        self.get_logger().info(f"  ARM{arm}: waiting for table access…")
        self._table_lock.acquire()
        self.get_logger().info(f"  ARM{arm}: table lock acquired — entering zone")
        self._pub_status.publish(String(data="PICKING"))

        try:
            if last_joints is None:
                self.get_logger().info(f"  ARM{arm} → pre_pick → approach → pick")
                self._send_traj([
                    (pre_pick(lx, ly), 0.8),
                    (approach_j,       1.6),
                    (pick_j,           2.4),
                ], arm=arm)
            else:
                self.get_logger().info(f"  ARM{arm} → approach → pick  (direct)")
                self._send_traj([
                    (approach_j, 1.0),
                    (pick_j,     1.8),
                ], arm=arm)
            time.sleep(0.10)

            self.get_logger().info(f"  ARM{arm} Vacuum ON")
            teleport_box(obj["sdf_name"], wx, wy, BOX_CTR_Z)

            # Lift straight up — still in the shared zone until clear of table
            self.get_logger().info(f"  ARM{arm} → lift")
            self._send_traj([(lift_j, 0.8)], arm=arm)

        finally:
            self._table_lock.release()
            self.get_logger().info(f"  ARM{arm}: table lock released — zone clear")

        # ── Phase 2: FLY TO BIN — no lock needed ──────────────────────────────
        self.get_logger().info(f"  ARM{arm} → place at bin  {label}")
        self._pub_status.publish(String(data="PLACING"))
        self._send_traj([(place_j, 1.2)], arm=arm)
        teleport_box(obj["sdf_name"], bx_w, by_w, carried_z(PLACE_Z))
        time.sleep(0.15)

        self.get_logger().info(f"  ARM{arm} Vacuum OFF — released")
        self._pub_status.publish(String(data="IDLE"))
        return place_j

    # ── arm worker ────────────────────────────────────────────────────────────
    def _run_arm(self, arm: int, objects: list):
        """Pick-and-place loop for one arm.  Called in its own thread."""
        if not objects:
            self.get_logger().info(f"ARM{arm}: nothing to pick.")
            return
        ordered = self._nearest_neighbour(objects, start_x=0.0, start_y=0.0)
        self.get_logger().info(
            f"ARM{arm}: {len(ordered)} item(s) — "
            f"{[o['sdf_name'] for o in ordered]}")
        last_j = None
        for obj in ordered:
            last_j = self._pick_and_place(obj, last_joints=last_j, arm=arm)
        self._move(HOME, 1.5, arm=arm)
        self.get_logger().info(f"ARM{arm}: done, returned home.")

    # ── main sequence ─────────────────────────────────────────────────────────
    def _run(self):
        time.sleep(3.0)
        self.get_logger().info("=== Inspection start ===")

        # Move both arms to HOME simultaneously
        if self._dual_arm:
            t1 = threading.Thread(target=self._move, args=(HOME, 1.5, 1))
            t2 = threading.Thread(target=self._move, args=(HOME, 1.5, 2))
            t1.start(); t2.start(); t1.join(); t2.join()
        else:
            self._move(HOME, 2.0)
        time.sleep(0.5)

        # Phase 1 — vision scan
        self.get_logger().info("--- Phase 1: Vision scan ---")
        objects = self._find_and_classify()

        if not objects:
            self.get_logger().warn("Nothing found on table — done.")
            return

        self.get_logger().info(f"Found {len(objects)} object(s).")

        if self._dual_arm:
            # Split: arm_1 handles DEFECT, arm_2 handles GOOD
            defects = [o for o in objects if o["verdict"] == "DEFECT"]
            goods   = [o for o in objects if o["verdict"] == "GOOD"]
            self.get_logger().info(
                f"--- Phase 2: ARM1 picks {len(defects)} DEFECT, "
                f"ARM2 picks {len(goods)} GOOD (parallel) ---")
            t1 = threading.Thread(target=self._run_arm, args=(1, defects), daemon=True)
            t2 = threading.Thread(target=self._run_arm, args=(2, goods),   daemon=True)
            t1.start(); t2.start(); t1.join(); t2.join()
        else:
            # Single-arm: pick everything with arm_1
            ordered = self._nearest_neighbour(objects, 0.0, 0.0)
            self.get_logger().info(
                f"--- Phase 2: Single arm picks {len(ordered)} object(s) ---")
            last_j = None
            for obj in ordered:
                last_j = self._pick_and_place(obj, last_joints=last_j, arm=1)
            self._move(HOME, 1.5, arm=1)

        self.get_logger().info("=== Done ===")


def main():
    rclpy.init()
    node = MultiBoxInspector()
    ex   = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
