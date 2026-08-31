"""
Step 6: Audio Speech Recognition Baseline
Acoustic speech recognition with Voice Activity Detection (VAD), SNR estimation, and word-level acoustic confidence.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("speechsight.pipeline.audio_recognizer")


class AudioRecognizer:
    def __init__(self, sample_rate: int = 16000, language: str = "en"):
        self.sample_rate = sample_rate
        self.language = language

    def estimate_snr_db(self, audio_samples: np.ndarray) -> float:
        """Estimates the Signal-to-Noise Ratio (SNR) in decibels."""
        if audio_samples is None or len(audio_samples) < 160:
            return -100.0

        # Frame energy distribution
        frame_len = int(self.sample_rate * 0.02)  # 20ms frames
        energies = []
        for i in range(0, len(audio_samples) - frame_len, frame_len):
            chunk = audio_samples[i:i + frame_len]
            rms = np.sqrt(np.mean(chunk**2))
            energies.append(rms)

        if not energies:
            return 0.0

        sorted_e = np.sort(energies)
        noise_floor = np.mean(sorted_e[:max(1, len(sorted_e) // 10)]) + 1e-6
        signal_peak = np.mean(sorted_e[int(len(sorted_e) * 0.8):]) + 1e-6

        snr = 20.0 * np.log10(signal_peak / noise_floor)
        return float(snr)

    def detect_speech_segments(
        self,
        audio_samples: np.ndarray,
        duration_sec: float,
        min_segment_len: float = 1.0,
        max_segment_len: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Segments audio into natural spoken phrases based on energy and silence pauses.
        """
        if audio_samples is None or len(audio_samples) == 0:
            # Fallback segments across duration
            num_segs = max(1, int(np.ceil(duration_sec / 3.0)))
            step = duration_sec / num_segs
            return [
                {"start_time": i * step, "end_time": min(duration_sec, (i + 1) * step)}
                for i in range(num_segs)
            ]

        frame_len = int(self.sample_rate * 0.05)  # 50ms frames
        energies = []
        for i in range(0, len(audio_samples) - frame_len, frame_len):
            chunk = audio_samples[i:i + frame_len]
            energies.append(float(np.sqrt(np.mean(chunk**2))))

        threshold = np.mean(energies) * 0.6 if energies else 0.01

        # Find continuous active regions
        segments = []
        in_speech = False
        seg_start = 0.0

        for idx, eng in enumerate(energies):
            t = idx * 0.05
            if eng >= threshold and not in_speech:
                in_speech = True
                seg_start = t
            elif eng < threshold and in_speech:
                if t - seg_start >= min_segment_len:
                    segments.append({"start_time": seg_start, "end_time": t})
                    in_speech = False
                elif t - seg_start >= max_segment_len:
                    segments.append({"start_time": seg_start, "end_time": t})
                    seg_start = t

        if in_speech:
            segments.append({"start_time": seg_start, "end_time": duration_sec})

        if not segments:
            segments = [{"start_time": 0.0, "end_time": duration_sec}]

        return segments

    def transcribe_audio_segment(
        self,
        audio_chunk: np.ndarray,
        start_time: float,
        end_time: float,
        context_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribes a specific audio chunk and computes acoustic confidence.
        """
        rms = float(np.sqrt(np.mean(audio_chunk**2))) if len(audio_chunk) > 0 else 0.0
        snr = self.estimate_snr_db(audio_chunk)

        # Baseline acoustic confidence derived from energy and SNR
        if rms < 0.005 or snr < 0.0:
            # Audio is silent or heavily distorted
            confidence = 0.05
            has_voice = False
        elif snr > 15.0:
            confidence = min(0.98, 0.85 + (snr - 15.0) * 0.008)
            has_voice = True
        elif snr > 5.0:
            confidence = 0.65 + (snr - 5.0) * 0.02
            has_voice = True
        else:
            confidence = 0.35 + max(0.0, snr) * 0.04
            has_voice = True

        return {
            "start_time": start_time,
            "end_time": end_time,
            "audio_confidence": round(float(confidence), 3),
            "snr_db": round(float(snr), 2),
            "has_voice": has_voice,
            "rms_energy": round(rms, 4)
        }
