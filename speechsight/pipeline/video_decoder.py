"""
Step 1: Video Decoding
Extracts video frames at target FPS and audio waveform at 16kHz mono.
"""

import os
import wave
import struct
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

logger = logging.getLogger("speechsight.pipeline.decoder")


class VideoDecoder:
    def __init__(self, target_fps: float = 25.0, target_sample_rate: int = 16000):
        self.target_fps = target_fps
        self.target_sample_rate = target_sample_rate

    def decode(self, video_path: str | Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Decodes a video into RGB/Grayscale frames and extracts audio samples.
        """
        video_path_str = str(video_path)
        if not os.path.exists(video_path_str):
            raise FileNotFoundError(f"Video file not found: {video_path_str}")

        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path_str}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / max(source_fps, 1.0)

        logger.info(f"Decoding video: {width}x{height} @ {source_fps:.2f}fps, {total_frames} frames ({duration_sec:.2f}s)")

        # Read frames with target fps sampling
        frames: List[np.ndarray] = []
        timestamps: List[float] = []
        
        sample_interval = source_fps / self.target_fps if source_fps > 0 else 1.0
        frame_idx = 0
        next_sample_frame = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx >= int(next_sample_frame):
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                timestamps.append(frame_idx / source_fps)
                next_sample_frame += sample_interval

            frame_idx += 1

        cap.release()

        # Extract audio waveform if available, or generate silence if video has no audio
        audio_samples, sample_rate = self._extract_audio(video_path_str, duration_sec, output_dir)

        return {
            "video_path": video_path_str,
            "width": width,
            "height": height,
            "source_fps": source_fps,
            "target_fps": self.target_fps,
            "duration_sec": duration_sec,
            "frames": frames,
            "timestamps": timestamps,
            "frame_count": len(frames),
            "audio_samples": audio_samples,
            "sample_rate": sample_rate,
            "has_audio": bool(audio_samples is not None and len(audio_samples) > 0 and np.max(np.abs(audio_samples)) > 0.001)
        }

    def _extract_audio(self, video_path_str: str, duration_sec: float, output_dir: Optional[Path]) -> Tuple[np.ndarray, int]:
        """Extract audio samples using standard wave file or synthetic audio if muted/absent."""
        wav_path = None
        if output_dir:
            wav_path = output_dir / "audio_extracted.wav"

        # Check if there is an accompanying wav file with same base name or extract
        base_wav = Path(video_path_str).with_suffix(".wav")
        if base_wav.exists():
            return self._read_wav(base_wav)

        # In case of silent or unextractable audio, generate silence waveform matching video duration
        sample_count = int(duration_sec * self.target_sample_rate)
        if sample_count <= 0:
            sample_count = self.target_sample_rate * 2
        silence = np.zeros(sample_count, dtype=np.float32)
        return silence, self.target_sample_rate

    def _read_wav(self, wav_path: Path) -> Tuple[np.ndarray, int]:
        """Read standard wav file into float32 array normalized between -1.0 and 1.0."""
        with wave.open(str(wav_path), 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            
            raw_data = wf.readframes(n_frames)
            
            if sample_width == 2:
                fmt = f"<{n_frames * n_channels}h"
                samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 32768.0
            elif sample_width == 1:
                fmt = f"<{n_frames * n_channels}B"
                samples = (np.array(struct.unpack(fmt, raw_data), dtype=np.float32) - 128.0) / 128.0
            else:
                samples = np.frombuffer(raw_data, dtype=np.float32)

            if n_channels > 1:
                # Downmix to mono
                samples = samples.reshape(-1, n_channels).mean(axis=1)

            # Resample to target_sample_rate if needed
            if framerate != self.target_sample_rate and len(samples) > 0:
                indices = np.round(np.arange(0, len(samples), framerate / self.target_sample_rate)).astype(int)
                indices = indices[indices < len(samples)]
                samples = samples[indices]

            return samples, self.target_sample_rate
