"""
Core data models and schemas for SpeechSight AI.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ProcessingMode(str, Enum):
    AUDIO_VISUAL = "audio_visual"
    SILENT_VISUAL = "silent_visual"
    AUDIO_ONLY = "audio_only"


class ConfidenceStatus(str, Enum):
    HIGH_CONFIDENCE = "high_confidence"
    MEDIUM_CONFIDENCE = "medium_confidence"
    LOW_CONFIDENCE = "low_confidence"
    UNCERTAIN_VISUAL_ONLY = "uncertain_visual_only"


class PrivacyStatus(str, Enum):
    TEMPORARY_MEDIA_DELETED = "temporary_media_deleted"
    RETAINED_FOR_SESSION = "retained_for_session"
    WIPED = "wiped"


class BoundingBox(BaseModel):
    x: float = Field(..., description="Top-left x normalized (0-1) or pixel coordinate")
    y: float = Field(..., description="Top-left y normalized (0-1) or pixel coordinate")
    width: float = Field(..., description="Bounding box width")
    height: float = Field(..., description="Bounding box height")


class FaceLandmarks(BaseModel):
    mouth_left: Optional[List[float]] = None
    mouth_right: Optional[List[float]] = None
    mouth_top: Optional[List[float]] = None
    mouth_bottom: Optional[List[float]] = None
    lip_opening: Optional[float] = None


class FaceTrack(BaseModel):
    track_id: int
    speaker_label: str
    frame_index: int
    timestamp: float
    bbox: BoundingBox
    landmarks: Optional[FaceLandmarks] = None
    is_active_speaker: bool = False
    speaker_score: float = 0.0


class WordToken(BaseModel):
    word: str
    start_time: float
    end_time: float
    confidence: float
    visual_confidence: Optional[float] = None
    audio_confidence: Optional[float] = None


class TranscriptSegment(BaseModel):
    id: Optional[str] = None
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    speaker: str = Field("Speaker 1", description="Assigned speaker label")
    text: str = Field(..., description="Transcribed text")
    audio_confidence: float = Field(0.0, ge=0.0, le=1.0)
    visual_confidence: float = Field(0.0, ge=0.0, le=1.0)
    combined_confidence: float = Field(0.0, ge=0.0, le=1.0)
    status: ConfidenceStatus = Field(ConfidenceStatus.HIGH_CONFIDENCE)
    words: Optional[List[WordToken]] = None
    uncertainty_note: Optional[str] = None
    mouth_motion_energy: Optional[float] = None


class PipelineStepLog(BaseModel):
    step_number: int
    step_name: str
    duration_ms: float
    status: str
    details: Optional[str] = None


class PipelineMetrics(BaseModel):
    total_duration_sec: float
    video_duration_sec: float
    real_time_factor: float
    active_speakers_detected: int
    average_combined_confidence: float
    steps: List[PipelineStepLog] = []


class TranscriptResult(BaseModel):
    video_id: str
    language: str = "en"
    segments: List[TranscriptSegment]
    processing_mode: ProcessingMode = ProcessingMode.AUDIO_VISUAL
    privacy_status: PrivacyStatus = PrivacyStatus.TEMPORARY_MEDIA_DELETED
    metrics: Optional[PipelineMetrics] = None
    full_text: Optional[str] = None
    video_metadata: Optional[Dict[str, Any]] = None
