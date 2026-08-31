"""
Global configuration and hyperparameters for SpeechSight AI.
"""

from pydantic import BaseModel, Field
import os
from pathlib import Path


class SpeechSightConfig(BaseModel):
    # App Information
    app_name: str = "SpeechSight"
    app_version: str = "1.0.0"
    
    # Video & Audio parameters
    target_sample_rate: int = 16000  # 16 kHz mono audio
    target_fps: float = 25.0         # 25 fps normalized video
    mouth_crop_size: int = 96        # 96x96 pixels for mouth ROI
    mouth_context_padding: float = 0.25 # 25% padding around mouth landmarks
    max_clip_duration_seconds: float = 300.0 # 5 minutes MVP scope
    
    # Face & Active Speaker Detection
    face_detection_confidence: float = 0.5
    active_speaker_motion_threshold: float = 0.12
    smoothing_window_size: int = 5
    
    # Audio-Visual Fusion Weights
    default_audio_weight: float = 0.70
    default_visual_weight: float = 0.30
    noisy_snr_threshold_db: float = 10.0
    noisy_audio_weight: float = 0.40
    noisy_visual_weight: float = 0.60
    silent_visual_weight: float = 1.00
    
    # Confidence Thresholds
    high_confidence_threshold: float = 0.80
    medium_confidence_threshold: float = 0.55
    
    # Supported Languages
    initial_language: str = "en"
    supported_languages: list[str] = ["en", "hi", "mr", "es", "fr"]
    
    # Privacy & Storage
    temp_dir: Path = Field(default_factory=lambda: Path(os.getenv("SPEECH_SIGHT_TEMP_DIR", "/tmp/speechsight")))
    auto_delete_temp: bool = True
    encrypt_local_storage: bool = True


DEFAULT_CONFIG = SpeechSightConfig()
