"""
Step 5: Active Speaker Detection
Determines which visible face is speaking in each time segment using mouth motion energy and audio-visual synchrony.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from speechsight.core.models import FaceTrack

logger = logging.getLogger("speechsight.pipeline.active_speaker")


class ActiveSpeakerDetector:
    def __init__(self, motion_threshold: float = 0.08, window_frames: int = 7):
        self.motion_threshold = motion_threshold
        self.window_frames = window_frames

    def compute_mouth_motion_energy(self, gray_tensor: np.ndarray) -> np.ndarray:
        """
        Computes temporal frame differences and optical motion energy across mouth crops.
        Input: (T, H, W) normalized mouth sequence
        Output: (T,) motion energy vector
        """
        T = gray_tensor.shape[0]
        if T <= 1:
            return np.zeros(T, dtype=np.float32)

        # Temporal diff: |I(t) - I(t-1)|
        diffs = np.abs(np.diff(gray_tensor, axis=0))
        frame_energies = np.mean(diffs, axis=(1, 2))  # shape (T-1,)

        # Prepend first frame energy
        energies = np.concatenate([[frame_energies[0]], frame_energies])

        # Smooth using a running moving average filter
        kernel = np.ones(self.window_frames) / self.window_frames
        smoothed = np.convolve(energies, kernel, mode='same')
        return smoothed

    def compute_audio_envelope(self, audio_samples: np.ndarray, sample_rate: int, target_length: int) -> np.ndarray:
        """Computes audio RMS energy envelope downsampled to match frame count."""
        if audio_samples is None or len(audio_samples) == 0 or target_length <= 0:
            return np.zeros(target_length, dtype=np.float32)

        samples_per_frame = max(1, len(audio_samples) // target_length)
        envelope = []
        for i in range(target_length):
            chunk = audio_samples[i * samples_per_frame:(i + 1) * samples_per_frame]
            rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0
            envelope.append(rms)

        env_arr = np.array(envelope, dtype=np.float32)
        # Normalize between 0 and 1
        max_v = np.max(env_arr)
        if max_v > 1e-5:
            env_arr = env_arr / max_v
        return env_arr

    def detect_active_speakers(
        self,
        track_sequences: Dict[int, Dict[str, Any]],
        per_frame_tracks: List[List[FaceTrack]],
        audio_samples: Optional[np.ndarray],
        sample_rate: int,
        timestamps: List[float]
    ) -> Dict[str, Any]:
        """
        Calculates speaker activity scores per frame and assigns active speaker status.
        Returns updated per_frame_tracks and segment speaker timelines.
        """
        num_frames = len(per_frame_tracks)
        track_energies: Dict[int, np.ndarray] = {}

        for track_id, data in track_sequences.items():
            tensor = data.get("tensor", np.zeros((0, 96, 96)))
            energies = self.compute_mouth_motion_energy(tensor)
            track_energies[track_id] = energies
            data["motion_energy"] = energies

        audio_env = self.compute_audio_envelope(audio_samples, sample_rate, num_frames)

        # Score active speaker per frame
        active_speaker_timeline: List[Dict[str, Any]] = []

        for f_idx in range(num_frames):
            ts = timestamps[f_idx] if f_idx < len(timestamps) else f_idx / 25.0
            current_tracks = per_frame_tracks[f_idx]

            best_track_id = None
            best_score = -1.0

            for track in current_tracks:
                t_id = track.track_id
                # Find index in track sequence
                if t_id in track_sequences and f_idx in track_sequences[t_id]["frame_indices"]:
                    sub_idx = track_sequences[t_id]["frame_indices"].index(f_idx)
                    motion = track_energies[t_id][sub_idx] if sub_idx < len(track_energies[t_id]) else 0.0
                else:
                    motion = 0.0

                aud_val = audio_env[f_idx] if f_idx < len(audio_env) else 0.0

                # AV synchrony composite score: motion + correlated audio activity
                score = float(motion * 1.5 + (motion * aud_val) * 1.0)
                track.speaker_score = round(min(1.0, score * 3.0), 3)

                if score > best_score and score >= self.motion_threshold:
                    best_score = score
                    best_track_id = t_id

            # Set active flag on best track
            if best_track_id is not None:
                for track in current_tracks:
                    if track.track_id == best_track_id:
                        track.is_active_speaker = True
                    else:
                        track.is_active_speaker = False

                active_speaker_timeline.append({
                    "frame_index": f_idx,
                    "timestamp": ts,
                    "active_track_id": best_track_id,
                    "active_speaker": f"Speaker {best_track_id}",
                    "confidence": min(1.0, float(best_score * 2.5))
                })
            elif current_tracks:
                # Default to track 1 if motion is subtle or stationary
                current_tracks[0].is_active_speaker = True
                active_speaker_timeline.append({
                    "frame_index": f_idx,
                    "timestamp": ts,
                    "active_track_id": current_tracks[0].track_id,
                    "active_speaker": current_tracks[0].speaker_label,
                    "confidence": 0.60
                })
            else:
                active_speaker_timeline.append({
                    "frame_index": f_idx,
                    "timestamp": ts,
                    "active_track_id": 1,
                    "active_speaker": "Speaker 1",
                    "confidence": 0.50
                })

        logger.info(f"Active speaker detection completed across {num_frames} frames.")
        return {
            "timeline": active_speaker_timeline,
            "track_energies": track_energies,
            "audio_envelope": audio_env
        }
