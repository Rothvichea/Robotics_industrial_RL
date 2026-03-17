"""
KR6 Print Quality Monitor
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

SMART_FACTORY = os.path.expanduser("~/smart-factory-agent")
sys.path.insert(0, SMART_FACTORY)

from ultralytics import YOLO

SEVERITY_LOW    = 0.25
SEVERITY_MEDIUM = 0.50
SEVERITY_HIGH   = 0.75


@dataclass
class InspectionResult:
    has_defect: bool
    confidence: float
    severity:   str
    action:     str
    bbox:       list  = field(default_factory=list)
    frame_id:   int   = 0
    timestamp:  float = field(default_factory=time.time)


class PrintMonitor:

    MODEL_PATHS = {
        "bottle":     f"{SMART_FACTORY}/models/yolo_bottle.pt",
        "screw":      f"{SMART_FACTORY}/models/yolo_screw.pt",
        "transistor": f"{SMART_FACTORY}/models/yolo_transistor.pt",
        "grid":       f"{SMART_FACTORY}/models/yolo_grid.pt",
    }

    def __init__(self, product_type: str = "screw", device: str = "cuda"):
        self.product_type  = product_type
        self.device        = device
        self.frame_id      = 0
        self.defect_count  = 0
        self.total_frames  = 0
        self.last_result   = None

        model_path = self.MODEL_PATHS.get(product_type)
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"[PrintMonitor] Loaded {product_type} on {device}")

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(source=dummy, verbose=False, device=device)
        print("[PrintMonitor] Warmed up.")

    def inspect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> InspectionResult:
        self.frame_id     += 1
        self.total_frames += 1

        results    = self.model.predict(
            source=frame, conf=conf_threshold,
            imgsz=640, verbose=False, device=self.device,
        )
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                detections.append({
                    "confidence": float(box.conf[0].cpu()),
                    "bbox":       box.xyxy[0].cpu().numpy().tolist(),
                })

        if not detections:
            result = InspectionResult(
                has_defect=False, confidence=0.0,
                severity="none", action="continue",
                frame_id=self.frame_id,
            )
        else:
            best       = max(detections, key=lambda x: x["confidence"])
            conf       = best["confidence"]
            sev, act   = self._classify(conf)
            self.defect_count += 1
            result = InspectionResult(
                has_defect=True, confidence=conf,
                severity=sev, action=act,
                bbox=best["bbox"], frame_id=self.frame_id,
            )

        self.last_result = result
        return result

    def _classify(self, confidence: float):
        if confidence >= SEVERITY_HIGH:
            return "high",   "abort"
        elif confidence >= SEVERITY_MEDIUM:
            return "medium", "stop_and_replan"
        else:
            return "low",    "slow_down"

    def stats(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "defect_count": self.defect_count,
            "defect_rate":  round(self.defect_count / max(self.total_frames, 1), 4),
            "product_type": self.product_type,
        }
