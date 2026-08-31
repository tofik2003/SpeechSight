"""
Step 2: Face Detection
Detects face bounding boxes and lip landmark coordinates for each frame.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from speechsight.core.models import BoundingBox, FaceLandmarks

logger = logging.getLogger("speechsight.pipeline.face_detector")


class FaceDetector:
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    def detect_faces_in_frame(self, frame: np.ndarray, frame_index: int = 0, timestamp: float = 0.0) -> List[Dict[str, Any]]:
        """
        Detects faces in an RGB image frame using skin-color segmentation & shape analysis with robust fallback.
        """
        h, w = frame.shape[:2]
        detections = []

        # Convert to YCrCb color space for skin detection
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        # Skin color range in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Morphological opening and closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (w * h * 0.015):  # At least 1.5% of frame
                x, y, fw, fh = cv2.boundingRect(cnt)
                aspect = fh / float(fw) if fw > 0 else 0
                if 0.7 <= aspect <= 2.2:
                    # Valid face candidate
                    mouth_roi_y = int(y + fh * 0.58)
                    mouth_roi_h = int(fh * 0.38)
                    mouth_roi_x = int(x + fw * 0.2)
                    mouth_roi_w = int(fw * 0.6)

                    mouth_left = [float(mouth_roi_x / w), float((mouth_roi_y + mouth_roi_h / 2) / h)]
                    mouth_right = [float((mouth_roi_x + mouth_roi_w) / w), float((mouth_roi_y + mouth_roi_h / 2) / h)]
                    mouth_top = [float((mouth_roi_x + mouth_roi_w / 2) / w), float(mouth_roi_y / h)]
                    mouth_bottom = [float((mouth_roi_x + mouth_roi_w / 2) / w), float((mouth_roi_y + mouth_roi_h) / h)]

                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    mouth_sub = gray[max(0, mouth_roi_y):min(h, mouth_roi_y+mouth_roi_h), max(0, mouth_roi_x):min(w, mouth_roi_x+mouth_roi_w)]
                    opening_val = float(np.std(mouth_sub) / 128.0) if mouth_sub.size > 0 else 0.1

                    landmarks = FaceLandmarks(
                        mouth_left=mouth_left,
                        mouth_right=mouth_right,
                        mouth_top=mouth_top,
                        mouth_bottom=mouth_bottom,
                        lip_opening=opening_val
                    )

                    bbox = BoundingBox(
                        x=float(x / w),
                        y=float(y / h),
                        width=float(fw / w),
                        height=float(fh / h)
                    )

                    detections.append({
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "bbox": bbox,
                        "landmarks": landmarks,
                        "confidence": 0.90,
                        "raw_rect": (x, y, fw, fh)
                    })

        # Sort detections by x-coordinate for consistent speaker 1 / speaker 2 left-to-right tracking
        detections.sort(key=lambda d: d["bbox"].x)

        # Fallback if no skin mask detected
        if not detections:
            cw, ch = int(w * 0.4), int(h * 0.45)
            cx, cy = int((w - cw) / 2), int((h - ch) / 3)
            mouth_roi_y = int(cy + ch * 0.6)
            mouth_roi_h = int(ch * 0.35)
            mouth_roi_x = int(cx + cw * 0.25)
            mouth_roi_w = int(cw * 0.5)

            landmarks = FaceLandmarks(
                mouth_left=[float(mouth_roi_x / w), float((mouth_roi_y + mouth_roi_h/2) / h)],
                mouth_right=[float((mouth_roi_x + mouth_roi_w) / w), float((mouth_roi_y + mouth_roi_h/2) / h)],
                mouth_top=[float((mouth_roi_x + mouth_roi_w/2) / w), float(mouth_roi_y / h)],
                mouth_bottom=[float((mouth_roi_x + mouth_roi_w/2) / w), float((mouth_roi_y + mouth_roi_h) / h)],
                lip_opening=0.15
            )

            bbox = BoundingBox(
                x=float(cx / w),
                y=float(cy / h),
                width=float(cw / w),
                height=float(ch / h)
            )

            detections.append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "bbox": bbox,
                "landmarks": landmarks,
                "confidence": 0.60,
                "raw_rect": (cx, cy, cw, ch)
            })

        return detections

    def process_video_frames(self, frames: List[np.ndarray], timestamps: List[float]) -> List[List[Dict[str, Any]]]:
        """Process all frames in the video and return per-frame detections."""
        all_frame_detections = []
        for idx, (frame, ts) in enumerate(zip(frames, timestamps)):
            dets = self.detect_faces_in_frame(frame, frame_index=idx, timestamp=ts)
            all_frame_detections.append(dets)
        return all_frame_detections
