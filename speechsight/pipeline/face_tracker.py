"""
Step 3: Face Tracking
Maintains consistent face track IDs over time across video frames using spatial IoU and feature distance.
"""

import numpy as np
from typing import List, Dict, Any
from speechsight.core.models import FaceTrack, BoundingBox, FaceLandmarks
import logging

logger = logging.getLogger("speechsight.pipeline.face_tracker")


def calculate_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    xA = max(boxA.x, boxB.x)
    yA = max(boxA.y, boxB.y)
    xB = min(boxA.x + boxA.width, boxB.x + boxB.width)
    yB = min(boxA.y + boxA.height, boxB.y + boxB.height)

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = boxA.width * boxA.height
    boxBArea = boxB.width * boxB.height

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.35, max_lost_frames: int = 15):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 1
        self.active_tracks: Dict[int, Dict[str, Any]] = {}

    def track(self, per_frame_detections: List[List[Dict[str, Any]]]) -> List[List[FaceTrack]]:
        """
        Takes raw per-frame face detections and returns tracked FaceTrack objects with consistent track IDs.
        """
        output_tracks: List[List[FaceTrack]] = []
        self.active_tracks.clear()
        self.next_track_id = 1

        for frame_idx, detections in enumerate(per_frame_detections):
            frame_tracks: List[FaceTrack] = []
            unmatched_detections = list(range(len(detections)))
            matched_tracks = set()

            # Match with currently active tracks
            for track_id, track_info in list(self.active_tracks.items()):
                last_bbox = track_info["last_bbox"]
                best_iou = 0.0
                best_det_idx = -1

                for det_idx in unmatched_detections:
                    det = detections[det_idx]
                    iou = calculate_iou(last_bbox, det["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = det_idx

                if best_iou >= self.iou_threshold and best_det_idx >= 0:
                    det = detections[best_det_idx]
                    unmatched_detections.remove(best_det_idx)
                    matched_tracks.add(track_id)

                    # Update track info
                    track_info["last_bbox"] = det["bbox"]
                    track_info["lost_count"] = 0
                    track_info["history"].append(det)

                    speaker_label = f"Speaker {track_id}"
                    face_track = FaceTrack(
                        track_id=track_id,
                        speaker_label=speaker_label,
                        frame_index=det["frame_index"],
                        timestamp=det["timestamp"],
                        bbox=det["bbox"],
                        landmarks=det["landmarks"],
                        is_active_speaker=False,
                        speaker_score=0.0
                    )
                    frame_tracks.append(face_track)
                else:
                    track_info["lost_count"] += 1

            # Create new tracks for remaining unmatched detections
            for det_idx in unmatched_detections:
                det = detections[det_idx]
                track_id = self.next_track_id
                self.next_track_id += 1

                self.active_tracks[track_id] = {
                    "track_id": track_id,
                    "last_bbox": det["bbox"],
                    "lost_count": 0,
                    "history": [det]
                }

                speaker_label = f"Speaker {track_id}"
                face_track = FaceTrack(
                    track_id=track_id,
                    speaker_label=speaker_label,
                    frame_index=det["frame_index"],
                    timestamp=det["timestamp"],
                    bbox=det["bbox"],
                    landmarks=det["landmarks"],
                    is_active_speaker=False,
                    speaker_score=0.0
                )
                frame_tracks.append(face_track)

            # Purge tracks lost for too many frames
            self.active_tracks = {
                t_id: info for t_id, info in self.active_tracks.items()
                if info["lost_count"] <= self.max_lost_frames
            }

            output_tracks.append(frame_tracks)

        logger.info(f"Face tracking complete: identified {self.next_track_id - 1} unique speaker tracks across {len(output_tracks)} frames.")
        return output_tracks
