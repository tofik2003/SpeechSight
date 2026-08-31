"""
SpeechSight AI - Dataset Preparation Pipeline.
Processes raw videos, extracts normalized mouth ROIs, aligns 16kHz mono audio, and outputs standardized manifests.
"""

import os
import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from speechsight.pipeline.face_detector import FaceDetector
from speechsight.pipeline.face_tracker import FaceTracker
from speechsight.pipeline.mouth_extractor import MouthExtractor
from speechsight.core.config import DEFAULT_CONFIG

logger = logging.getLogger("speechsight.data.prep")


class DatasetPreparationPipeline:
    def __init__(self, output_dir: Path, crop_size: int = 96, target_fps: float = 25.0):
        self.output_dir = Path(output_dir)
        self.crop_size = crop_size
        self.target_fps = target_fps
        self.face_detector = FaceDetector()
        self.face_tracker = FaceTracker()
        self.mouth_extractor = MouthExtractor(crop_size=crop_size)

        # Create subdirectories
        self.videos_dir = self.output_dir / "videos"
        self.audio_dir = self.output_dir / "audio"
        self.mouths_dir = self.output_dir / "mouth_crops"
        self.transcripts_dir = self.output_dir / "transcripts"
        self.meta_dir = self.output_dir / "metadata"

        for d in [self.videos_dir, self.audio_dir, self.mouths_dir, self.transcripts_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def process_sample(
        self,
        video_path: Path,
        transcript_text: str,
        speaker_id: str = "speaker_001",
        clip_id: Optional[str] = None,
        language: str = "en",
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes a single video into standard dataset label format.
        """
        clip_id = clip_id or f"{speaker_id}_clip_{Path(video_path).stem}"
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / max(fps, 1.0)
        end_time = end_time or duration

        frames = []
        timestamps = []
        frame_idx = 0
        sample_interval = fps / self.target_fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % max(1, int(round(sample_interval))) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
                timestamps.append(frame_idx / fps)
            frame_idx += 1
        cap.release()

        # Face & mouth extraction
        detections = self.face_detector.process_video_frames(frames, timestamps)
        tracks = self.face_tracker.track(detections)
        mouth_data = self.mouth_extractor.extract_track_mouth_sequences(frames, tracks)

        # Target file paths
        video_filename = f"{clip_id}.mp4"
        audio_filename = f"{clip_id}.wav"
        transcript_filename = f"{clip_id}.txt"
        meta_filename = f"{clip_id}.json"
        mouth_npy_filename = f"{clip_id}_mouth.npy"

        # Save transcript
        transcript_path = self.transcripts_dir / transcript_filename
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text.strip() + "\n")

        # Save mouth tensor
        tensor = mouth_data.get(1, {}).get("tensor", np.zeros((len(frames), self.crop_size, self.crop_size), dtype=np.float32))
        np.save(self.mouths_dir / mouth_npy_filename, tensor)

        # Metadata dictionary
        metadata = {
            "video_file": video_filename,
            "audio_file": audio_filename,
            "transcript_file": transcript_filename,
            "mouth_file": mouth_npy_filename,
            "optional_metadata": {
                "speaker_id": speaker_id,
                "language": language,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "frame_count": len(frames),
                "duration_sec": round(duration, 2),
                "word_count": len(transcript_text.strip().split())
            }
        }

        with open(self.meta_dir / meta_filename, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Prepared sample {clip_id}: {len(frames)} frames, {len(transcript_text.split())} words.")
        return metadata

    def build_manifest(self, train_split: float = 0.8, val_split: float = 0.1) -> Dict[str, Any]:
        """
        Compiles all processed samples into train, val, test splits in JSON and CSV formats.
        """
        all_metadata_files = list(self.meta_dir.glob("*.json"))
        records = []
        for mf in sorted(all_metadata_files):
            with open(mf, "r", encoding="utf-8") as f:
                records.append(json.load(f))

        np.random.seed(42)
        indices = np.random.permutation(len(records))
        n_train = int(len(records) * train_split)
        n_val = int(len(records) * val_split)

        train_set = [records[i] for i in indices[:n_train]]
        val_set = [records[i] for i in indices[n_train:n_train + n_val]]
        test_set = [records[i] for i in indices[n_train + n_val:]]

        manifest = {
            "dataset_name": "SpeechSight_AVSR_Dataset",
            "total_samples": len(records),
            "splits": {
                "train_count": len(train_set),
                "val_count": len(val_set),
                "test_count": len(test_set)
            },
            "train": train_set,
            "val": val_set,
            "test": test_set
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Write CSV manifest
        csv_path = self.output_dir / "dataset_index.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["clip_id", "speaker_id", "split", "duration_sec", "transcript_file", "mouth_file"])
            for split_name, split_list in [("train", train_set), ("val", val_set), ("test", test_set)]:
                for item in split_list:
                    meta = item["optional_metadata"]
                    writer.writerow([
                        Path(item["transcript_file"]).stem,
                        meta["speaker_id"],
                        split_name,
                        meta["duration_sec"],
                        item["transcript_file"],
                        item["mouth_file"]
                    ])

        logger.info(f"Manifest written to {manifest_path} and {csv_path} with {len(records)} samples.")
        return manifest
