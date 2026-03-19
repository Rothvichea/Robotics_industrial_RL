import rclpy.parameter
#!/usr/bin/env python3
"""
YOLOv8 Inspection Node — no cv_bridge dependency
"""

import sys
import os

# force conda packages first
CONDA = "/home/genji/miniconda3/envs/industrial-ai/lib/python3.10/site-packages"
sys.path.insert(0, CONDA)
sys.path.insert(1, os.path.expanduser("~/smart-factory-agent"))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO


MODELS = {
    "bottle": os.path.expanduser("~/smart-factory-agent/models/yolo_bottle.pt"),
    "screw":  os.path.expanduser("~/smart-factory-agent/models/yolo_screw.pt"),
}

TEST_IMAGES = {
    "bottle_defect": os.path.expanduser("~/smart-factory-agent/data_test/bottle_defect_01.png"),
    "bottle_good":   os.path.expanduser("~/smart-factory-agent/data_test/bottle_good_01.png"),
    "screw_defect":  os.path.expanduser("~/smart-factory-agent/data_test/screw_defect_01.png"),
    "screw_good":    os.path.expanduser("~/smart-factory-agent/data_test/screw_defect_02.png"),
}

# cycle through all 4 images to simulate mixed good/bad boxes
TEST_CYCLE = ["bottle_defect", "bottle_good", "screw_defect", "bottle_good"]


def ros_image_to_cv2(msg: Image) -> np.ndarray:
    """Convert ROS Image msg to cv2 BGR without cv_bridge."""
    dtype  = np.uint8
    data   = np.frombuffer(msg.data, dtype=dtype)
    if msg.encoding == "rgb8":
        img = data.reshape((msg.height, msg.width, 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "bgr8":
        img = data.reshape((msg.height, msg.width, 3))
    elif msg.encoding == "mono8":
        img = data.reshape((msg.height, msg.width))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = data.reshape((msg.height, msg.width, 3))
    return img


def cv2_to_ros_image(frame: np.ndarray, header) -> Image:
    """Convert cv2 BGR to ROS Image msg without cv_bridge."""
    msg          = Image()
    msg.header   = header
    msg.height   = frame.shape[0]
    msg.width    = frame.shape[1]
    msg.encoding = "bgr8"
    msg.step     = frame.shape[1] * 3
    msg.data     = frame.tobytes()
    return msg


class YoloInspectorNode(Node):

    def __init__(self):
        super().__init__("yolo_inspector_node")

        self.declare_parameter("product_type",   "bottle")
        self.declare_parameter("conf_threshold",  0.25)
        self.declare_parameter("use_test_image",  False)
        self.declare_parameter("test_image_key",  "bottle_defect")

        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.product_type = self.get_parameter("product_type").value
        self.conf         = self.get_parameter("conf_threshold").value
        self.use_test_img = self.get_parameter("use_test_image").value
        self.test_img_key = self.get_parameter("test_image_key").value
        self.frame_count  = 0

        # load model
        model_path = MODELS.get(self.product_type)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = YOLO(model_path)

        # load test image
        self.test_frame = None
        if self.use_test_img:
            p = TEST_IMAGES.get(self.test_img_key)
            if p and Path(p).exists():
                self.test_frame = cv2.imread(p)
                self.get_logger().info(f"Test image loaded: {self.test_img_key}")

        # pubs / subs
        self.sub = self.create_subscription(
            Image, "/inspection_cam/image_raw",
            self.image_callback, 10)
        self.pub_result = self.create_publisher(String, "/inspection/result", 10)
        self.pub_image  = self.create_publisher(Image,  "/inspection/image_annotated", 10)

        # warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(source=dummy, verbose=False)

        cv2.namedWindow("YOLOv8 Live Inspection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv8 Live Inspection", 800, 800)

        self.get_logger().info(
            f"YoloInspectorNode ready | product={self.product_type} | "
            f"test_mode={self.use_test_img} | key={self.test_img_key}")

    def image_callback(self, msg: Image):
        self.frame_count += 1

        if self.use_test_img and self.test_frame is not None:
            frame = self.test_frame.copy()
        else:
            frame = ros_image_to_cv2(msg)

        frame = cv2.resize(frame, (640, 640))

        results    = self.model.predict(source=frame, conf=self.conf, verbose=False)
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                detections.append({
                    "confidence": round(float(box.conf[0].cpu()), 4),
                    "bbox":       box.xyxy[0].cpu().numpy().tolist(),
                })

        has_defect = len(detections) > 0
        verdict    = "DEFECT" if has_defect else "GOOD"
        action     = "REJECT" if has_defect else "ACCEPT"
        max_conf   = max((d["confidence"] for d in detections), default=0.0)

        # get best detection bbox
        best_bbox = [0, 0, 640, 640]
        best_cx   = 320
        best_cy   = 320
        if detections:
            best = max(detections, key=lambda x: x["confidence"])
            best_bbox = best["bbox"]
            best_cx = int((best_bbox[0] + best_bbox[2]) / 2)
            best_cy = int((best_bbox[1] + best_bbox[3]) / 2)

        result = {
            "frame":      self.frame_count,
            "product":    self.product_type,
            "verdict":    verdict,
            "action":     action,
            "has_defect": has_defect,
            "confidence": round(max_conf, 4),
            "detections": len(detections),
            "bbox":       best_bbox,
            "cx":         best_cx,
            "cy":         best_cy,
        }

        msg_out      = String()
        msg_out.data = json.dumps(result)
        self.pub_result.publish(msg_out)

        if not hasattr(self, "_last_verdict") or self._last_verdict != verdict:
            self._last_verdict = verdict
            self.get_logger().info(f"[{verdict}] conf={max_conf:.3f} → {action}")

        annotated = self._annotate(frame, result, detections)
        self.pub_image.publish(cv2_to_ros_image(annotated, msg.header))

        cv2.imshow("YOLOv8 Live Inspection", annotated)

    def _annotate(self, frame, result, detections):
        vis   = frame.copy()
        color = (0, 0, 255) if result["has_defect"] else (0, 255, 0)
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"defect {d['confidence']:.2f}",
                        (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(vis, (0, 0), (640, 40), color, -1)
        cv2.putText(vis, f"{result['verdict']} → {result['action']}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        return vis


def main():
    rclpy.init()
    node = YoloInspectorNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
