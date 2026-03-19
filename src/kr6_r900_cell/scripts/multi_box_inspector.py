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
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
from geometry_msgs.msg import PoseStamped
from ultralytics import YOLO
import cv2, numpy as np


# ─── Joint names ──────────────────────────────────────────────────────────────
JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]

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

        self._traj = ActionClient(
            self, FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
            callback_group=cb)

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

        self.get_logger().info("Waiting for controller…")
        self._traj.wait_for_server()
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

        # ── Log ───────────────────────────────────────────────────────────────
        for obj in objects:
            action = "→ REJECT ✗" if obj["verdict"] == "DEFECT" else "→ ACCEPT ✓"
            self.get_logger().info(
                f"  {obj['sdf_name']} at ({obj['wx']:.3f},{obj['wy']:.3f}) "
                f"{obj['verdict']} conf={obj['conf']:.2f}  {action}")

        return objects

    # ── IK ────────────────────────────────────────────────────────────────────
    def _ik_once(self, x, y, z, seed: list):
        """Single raw IK call with a given seed.  Returns joint list or None."""
        req = GetPositionIK.Request()
        req.ik_request                  = PositionIKRequest()
        req.ik_request.group_name       = "manipulator"
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec      = 5

        rs = RobotState()
        rs.joint_state.name     = JOINT_NAMES
        rs.joint_state.position = seed
        req.ik_request.robot_state = rs

        pose = PoseStamped()
        pose.header.frame_id    = "base_link"
        pose.pose.position.x    = x
        pose.pose.position.y    = y
        pose.pose.position.z    = z
        # 180° around Y → tool Z points straight down
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 1.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0
        req.ik_request.pose_stamped = pose

        done, result = threading.Event(), [None]
        self._ik.call_async(req).add_done_callback(
            lambda f: (result.__setitem__(0, f.result()), done.set()))
        if not done.wait(10.0):
            return None
        resp = result[0]
        if resp.error_code.val != 1:
            return None
        return list(resp.solution.joint_state.position[:6])

    def _ik_solve(self, x, y, z, seed_joints=None):
        """IK with automatic retry to avoid large joint jumps (wrist flips).

        KDL can converge to any joint_4 value when the tool points straight
        down (null-space DOF).  We detect that and retry with seeds that bias
        joint_4 toward 0 so the solver stays near the current configuration.
        Returns the solution with the smallest max single-joint displacement.
        """
        primary = seed_joints if seed_joints is not None else pre_pick(x, y)
        joints  = self._ik_once(x, y, z, primary)

        if joints is None:
            self.get_logger().error(f"IK failed at ({x:.2f},{y:.2f},{z:.2f})")
            return None

        # Maximum jump of any single joint from the seed
        def _max_jump(sol):
            return max(abs(sol[i] - primary[i]) for i in range(6))

        best, best_jump = joints, _max_jump(joints)

        # If any joint jumped more than ~86°, try alternative seeds.
        # Two main culprits:
        #   j1 (base) — IK can find a "mirror" solution 180° away
        #   j4 (forearm roll) — null-space DOF when pointing straight down
        if best_jump > 1.5:
            j1_exp = math.atan2(y, x)   # expected base rotation toward target
            retry_seeds = [
                (j1_exp,        0.0),
                (j1_exp,        0.3),
                (j1_exp,       -0.3),
                (j1_exp + 0.2,  0.0),
                (j1_exp - 0.2,  0.0),
                (j1_exp + 0.2,  0.3),
                (j1_exp - 0.2, -0.3),
            ]
            for j1_bias, j4_bias in retry_seeds:
                alt_seed    = list(primary)
                alt_seed[0] = j1_bias
                alt_seed[3] = j4_bias
                alt = self._ik_once(x, y, z, alt_seed)
                if alt is None:
                    continue
                jump = _max_jump(alt)
                if jump < best_jump:
                    best_jump, best = jump, alt
                if best_jump < 1.0:
                    break   # good enough — stop early

        self.get_logger().info(
            f"  IK → {[round(j,3) for j in best]}  (max_jump={best_jump:.2f} rad)")
        return best

    # ── motion ────────────────────────────────────────────────────────────────
    def _send_traj(self, waypoints: list) -> bool:
        """Send a multi-point trajectory for smooth continuous motion.
        waypoints = [(joint_positions, time_from_start_sec), ...]
        All points travel as ONE trajectory — no stop between them.
        """
        traj             = JointTrajectory()
        traj.joint_names = JOINT_NAMES

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

        self._traj.send_goal_async(goal).add_done_callback(_goal_cb)
        if not done.wait(15.0):
            return False

        gh = gh_holder[0]
        if not gh.accepted:
            self.get_logger().error("Trajectory rejected")
            return False

        done.clear()

        def _res_cb(_):
            done.set()

        gh.get_result_async().add_done_callback(_res_cb)
        done.wait(float(total_t + 20))
        return True

    def _move(self, positions, duration: float) -> bool:
        """Single-point move — kept for HOME and simple transitions."""
        return self._send_traj([(positions, duration)])

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
    def _pick_and_place(self, obj: dict, last_joints=None) -> list:
        """Pick obj and drop in the correct bin.  Returns place_j for cascading.

        last_joints: joint config the arm is currently at (previous place_j).
          When provided the arm goes DIRECTLY to approach — no intermediate swing.
          When None (first box) a pre_pick intermediate is inserted so joint_5
          does not spin 360°.
        """
        wx, wy = obj["wx"], obj["wy"]
        defect = obj["verdict"] == "DEFECT"
        bx, by = REJECT_BIN if defect else ACCEPT_BIN
        label  = "REJECT ✗" if defect else "ACCEPT ✓"
        self.get_logger().info(f"▶ {obj['sdf_name']} at ({wx:.3f},{wy:.3f}) → {label}")

        # ── IK cascade: each step seeded from the previous solution ───────────
        # Seed for approach: use last place_j if available, else pre_pick()
        seed       = last_joints if last_joints is not None else pre_pick(wx, wy)
        approach_j = self._ik_solve(wx, wy, APPROACH_Z, seed)
        if approach_j is None:
            self.get_logger().error("IK failed (approach) — skipping")
            return last_joints
        pick_j = self._ik_solve(wx, wy, PICK_Z,    approach_j)
        if pick_j is None:
            self.get_logger().error("IK failed (pick) — skipping")
            return last_joints
        lift_j = self._ik_solve(wx, wy, LIFT_Z,    pick_j)
        if lift_j is None:
            self.get_logger().error("IK failed (lift) — skipping")
            return last_joints
        # Seed place from lift_j (arm's actual current config) not pre_pick.
        # pre_pick gives j1=atan2(by,bx) which is correct for direction but
        # ignores the full arm state — IK often finds a "mirror" back-solution.
        # lift_j already has the arm in the right region so the solver finds
        # the nearest valid bin pose without flipping.
        place_j = self._ik_solve(bx, by, PLACE_Z,  lift_j)
        if place_j is None:
            self.get_logger().error("IK failed (place) — skipping")
            return last_joints

        # ── Trajectory 1: → approach → pick ──────────────────────────────────
        # First box: insert pre_pick intermediate to avoid 360° joint_5 spin.
        # Subsequent boxes: arm already near the table — go straight to approach.
        if last_joints is None:
            self.get_logger().info("  → pre_pick → approach → pick")
            self._send_traj([
                (pre_pick(wx, wy), 3.0),
                (approach_j,       6.0),
                (pick_j,           8.0),
            ])
        else:
            self.get_logger().info("  → approach → pick  (direct, no swing)")
            self._send_traj([
                (approach_j, 3.5),
                (pick_j,     5.5),
            ])
        time.sleep(0.25)

        # ── Vacuum ON ─────────────────────────────────────────────────────────
        self.get_logger().info("  🔵 Vacuum ON")
        teleport_box(obj["sdf_name"], wx, wy, BOX_CTR_Z)

        # ── Trajectory 2: lift → place ────────────────────────────────────────
        self.get_logger().info(f"  → lift → {label}")
        self._send_traj([
            (lift_j,  2.5),
            (place_j, 6.0),
        ])
        teleport_box(obj["sdf_name"], bx, by, carried_z(PLACE_Z))
        time.sleep(0.3)

        self.get_logger().info("  ⚪ Vacuum OFF — released")
        return place_j   # ← caller uses this as seed for the next box

    # ── main sequence ─────────────────────────────────────────────────────────
    def _run(self):
        time.sleep(3.0)
        self.get_logger().info("=== Inspection start ===")
        self._move(HOME, 4)
        time.sleep(0.5)

        # Phase 1 — vision scan (robot at HOME, camera overhead)
        self.get_logger().info("--- Phase 1: Vision scan ---")
        objects = self._find_and_classify()

        if not objects:
            self.get_logger().warn("Nothing found on table — done.")
            return

        # Phase 2 — sort by nearest-neighbour from HOME (0,0)
        # then pick continuously, seeding each IK from the previous place pose
        ordered = self._nearest_neighbour(objects, start_x=0.0, start_y=0.0)
        self.get_logger().info(
            f"--- Phase 2: Sort {len(ordered)} object(s) "
            f"(order: {[o['sdf_name'] for o in ordered]}) ---")

        last_joints = None
        for obj in ordered:
            last_joints = self._pick_and_place(obj, last_joints=last_joints)

        # Single home return at the very end
        self.get_logger().info("  → home (all done)")
        self._move(HOME, 4.0)

        self.get_logger().info("=== Done ===")
        for obj in ordered:
            action = "REJECT" if obj["verdict"] == "DEFECT" else "ACCEPT"
            self.get_logger().info(
                f"  {obj['sdf_name']} ({obj['wx']:.3f},{obj['wy']:.3f}): "
                f"{obj['verdict']} → {action}")


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
