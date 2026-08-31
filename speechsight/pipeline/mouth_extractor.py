"""
Step 4: Mouth Extraction
Crops, aligns, and normalizes mouth regions of interest (ROI) into standardized clips (e.g. 96x96 grayscale/RGB).
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from speechsight.core.models import FaceTrack
import logging

logger = logging.getLogger("speechsight.pipeline.mouth_extractor")


class MouthExtractor:
    def __init__(self, crop_size: int = 96, context_padding: float = 0.25):
        self.crop_size = crop_size
        self.context_padding = context_padding

    def extract_mouth_roi(self, frame: np.ndarray, face_track: FaceTrack) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts a normalized mouth crop from a frame given a FaceTrack.
        Returns (rgb_crop, gray_normalized_crop).
        """
        h, w = frame.shape[:2]
        bbox = face_track.bbox

        # Pixel coordinates of face box
        fx = int(bbox.x * w)
        fy = int(bbox.y * h)
        fw = int(bbox.width * w)
        fh = int(bbox.height * h)

        # Mouth ROI center and size estimation
        if face_track.landmarks and face_track.landmarks.mouth_top and face_track.landmarks.mouth_bottom:
            mx = int((face_track.landmarks.mouth_left[0] + face_track.landmarks.mouth_right[0]) * 0.5 * w)
            my = int((face_track.landmarks.mouth_top[1] + face_track.landmarks.mouth_bottom[1]) * 0.5 * h)
            mw = int(abs(face_track.landmarks.mouth_right[0] - face_track.landmarks.mouth_left[0]) * w * (1.0 + self.context_padding * 2))
            mh = int(abs(face_track.landmarks.mouth_bottom[1] - face_track.landmarks.mouth_top[1]) * h * (1.0 + self.context_padding * 2))
            mw = max(mw, int(fw * 0.45))
            mh = max(mh, int(fh * 0.35))
        else:
            mx = fx + int(fw * 0.5)
            my = fy + int(fh * 0.75)
            mw = int(fw * 0.55)
            mh = int(fh * 0.40)

        # Make crop square
        side = max(mw, mh)
        x1 = max(0, mx - side // 2)
        y1 = max(0, my - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)

        # Handle boundary adjustments
        if x2 - x1 < side and x1 > 0:
            x1 = max(0, x2 - side)
        if y2 - y1 < side and y1 > 0:
            y1 = max(0, y2 - side)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            # Fallback zero patch
            rgb_crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            gray_norm = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
            return rgb_crop, gray_norm

        rgb_crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting invariance
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray_equalized = clahe.apply(gray)
        gray_norm = (gray_equalized.astype(np.float32) - 128.0) / 128.0

        return rgb_crop, gray_norm

    def extract_track_mouth_sequences(
        self,
        frames: List[np.ndarray],
        per_frame_tracks: List[List[FaceTrack]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extracts temporal mouth crops for each tracked speaker across the video.
        Returns a dict mapping track_id -> { "rgb_crops": [...], "gray_tensors": np.ndarray, "timestamps": [...] }
        """
        track_sequences: Dict[int, Dict[str, Any]] = {}

        for frame_idx, (frame, tracks) in enumerate(zip(frames, per_frame_tracks)):
            for face_track in tracks:
                track_id = face_track.track_id
                if track_id not in track_sequences:
                    track_sequences[track_id] = {
                        "track_id": track_id,
                        "speaker_label": face_track.speaker_label,
                        "rgb_crops": [],
                        "gray_crops": [],
                        "timestamps": [],
                        "frame_indices": []
                    }

                rgb_crop, gray_norm = self.extract_mouth_roi(frame, face_track)
                track_sequences[track_id]["rgb_crops"].append(rgb_crop)
                track_sequences[track_id]["gray_crops"].append(gray_norm)
                track_sequences[track_id]["timestamps"].append(face_track.timestamp)
                track_sequences[track_id]["frame_indices"].append(frame_idx)

        # Convert gray crops to numpy arrays (T, H, W)
        for track_id, data in track_sequences.items():
            if data["gray_crops"]:
                data["tensor"] = np.stack(data["gray_crops"], axis=0)  # Shape (T, 96, 96)
            else:
                data["tensor"] = np.zeros((0, self.crop_size, self.crop_size), dtype=np.float32)

        logger.info(f"Mouth extraction complete: extracted sequences for {len(track_sequences)} speaker track(s).")
        return track_sequences
