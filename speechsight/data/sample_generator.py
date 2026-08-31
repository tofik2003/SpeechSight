"""
SpeechSight Synthetic Sample and Demo Video Generator.
Generates multi-speaker video clips with animated lip movements and synchronized audio for offline tests and prototype demos.
"""

import os
import wave
import struct
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger("speechsight.data.sample_generator")


class SampleClipGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_synthetic_av_clip(
        self,
        filename: str,
        text: str,
        duration_sec: float = 3.0,
        speaker_id: str = "speaker_001",
        num_speakers: int = 1,
        is_silent: bool = False,
        is_noisy: bool = False,
        fps: int = 25,
        width: int = 640,
        height: int = 480
    ) -> Dict[str, Any]:
        """
        Creates a valid MP4 video and accompanying WAV audio file with animated speaker face(s).
        """
        video_path = self.output_dir / filename
        audio_path = video_path.with_suffix(".wav")
        txt_path = video_path.with_suffix(".txt")

        # Save transcript text
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")

        # Create Audio Track
        sample_rate = 16000
        total_samples = int(duration_sec * sample_rate)
        t_arr = np.linspace(0, duration_sec, total_samples, endpoint=False)

        if is_silent:
            audio_wave = np.zeros(total_samples, dtype=np.float32)
        else:
            # Formant synthesizer simulation: fundamental + harmonics
            f0 = 160.0 if "2" in speaker_id else 130.0
            base_signal = 0.4 * np.sin(2 * np.pi * f0 * t_arr) + 0.25 * np.sin(2 * np.pi * (f0 * 2) * t_arr) + 0.15 * np.sin(2 * np.pi * (f0 * 3) * t_arr)
            # Modulate with speech envelope (syllable bursts ~4 Hz)
            envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t_arr))
            audio_wave = (base_signal * envelope).astype(np.float32)

            if is_noisy:
                # Add background ambient noise
                noise = np.random.normal(0, 0.35, total_samples).astype(np.float32)
                audio_wave = audio_wave + noise

            # Normalize
            max_v = np.max(np.abs(audio_wave))
            if max_v > 0:
                audio_wave = audio_wave / max_v * 0.85

        # Write WAV file
        with wave.open(str(audio_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            int_samples = (audio_wave * 32767).astype(np.int16)
            wf.writeframes(int_samples.tobytes())

        # Create Video Frames with OpenCV VideoWriter
        total_frames = int(duration_sec * fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        if not out.isOpened():
            # Fallback to AVI if mp4v codec is not compiled in
            video_path = video_path.with_suffix(".avi")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for f_idx in range(total_frames):
            ts = f_idx / fps
            frame = np.full((height, width, 3), 32, dtype=np.uint8)  # dark background

            # Draw UI header / watermark
            cv2.putText(frame, f"SpeechSight AI - Test Clip ({speaker_id})", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"Time: {ts:.2f}s / {duration_sec:.2f}s", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # Draw Speaker 1
            spk1_x, spk1_y = (width // 4, height // 2) if num_speakers > 1 else (width // 2, height // 2)
            self._draw_speaker_face(frame, spk1_x, spk1_y, "Speaker 1", ts, is_speaking=(not is_silent and (ts < duration_sec/2 or num_speakers == 1)))

            # Draw Speaker 2 if multi-speaker
            if num_speakers > 1:
                spk2_x, spk2_y = (3 * width // 4, height // 2)
                self._draw_speaker_face(frame, spk2_x, spk2_y, "Speaker 2", ts, is_speaking=(not is_silent and ts >= duration_sec/2))

            out.write(frame)

        out.release()

        logger.info(f"Generated synthetic AV sample: {video_path} ({duration_sec}s, {total_frames} frames)")
        return {
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "transcript_path": str(txt_path),
            "transcript": text,
            "duration_sec": duration_sec,
            "speaker_id": speaker_id,
            "is_silent": is_silent,
            "is_noisy": is_noisy
        }

    def _draw_speaker_face(self, frame: np.ndarray, cx: int, cy: int, label: str, t: float, is_speaking: bool):
        """Draws an avatar face with dynamic mouth opening."""
        # Head contour
        cv2.ellipse(frame, (cx, cy), (75, 95), 0, 0, 360, (220, 190, 170), -1)
        cv2.ellipse(frame, (cx, cy), (75, 95), 0, 0, 360, (180, 140, 120), 2)

        # Eyes
        cv2.circle(frame, (cx - 28, cy - 25), 8, (60, 60, 60), -1)
        cv2.circle(frame, (cx + 28, cy - 25), 8, (60, 60, 60), -1)

        # Nose
        cv2.line(frame, (cx, cy - 10), (cx, cy + 15), (160, 120, 100), 2)

        # Mouth opening animation based on speaking state
        mouth_open = int(12 * np.abs(np.sin(2 * np.pi * 3.5 * t))) if is_speaking else 2
        mouth_y = cy + 45

        # Lips
        cv2.ellipse(frame, (cx, mouth_y), (26, max(3, mouth_open + 4)), 0, 0, 360, (80, 70, 190), -1)
        if mouth_open > 3:
            # Inner mouth cavity
            cv2.ellipse(frame, (cx, mouth_y), (20, mouth_open), 0, 0, 360, (30, 20, 40), -1)

        # Label tag
        cv2.putText(frame, label, (cx - 40, cy + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)

    def generate_all_presets(self) -> List[Dict[str, Any]]:
        """Generates all 4 standard evaluation and demo presets."""
        presets = [
            {
                "filename": "keynote_presentation.mp4",
                "text": "The meeting will start tomorrow.",
                "duration_sec": 2.8,
                "speaker_id": "speaker_001",
                "num_speakers": 1,
                "is_silent": False,
                "is_noisy": False
            },
            {
                "filename": "noisy_cafe_interview.mp4",
                "text": "SpeechSight combines speech recognition with visual lip movement analysis.",
                "duration_sec": 4.5,
                "speaker_id": "speaker_001",
                "num_speakers": 1,
                "is_silent": False,
                "is_noisy": True
            },
            {
                "filename": "silent_reading_demo.mp4",
                "text": "Visual only speech recognition when no audio is available.",
                "duration_sec": 3.8,
                "speaker_id": "speaker_001",
                "num_speakers": 1,
                "is_silent": True,
                "is_noisy": False
            },
            {
                "filename": "dialogue_two_speakers.mp4",
                "text": "We have finalized the roadmap. That sounds great let us proceed.",
                "duration_sec": 5.0,
                "speaker_id": "speaker_001",
                "num_speakers": 2,
                "is_silent": False,
                "is_noisy": False
            }
        ]

        results = []
        for p in presets:
            res = self.generate_synthetic_av_clip(**p)
            results.append(res)
        return results
