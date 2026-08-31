import pytest
from speechsight.pipeline.fusion_engine import PredictionFusionEngine
from speechsight.core.models import ProcessingMode, ConfidenceStatus


def test_audio_visual_fusion_high_snr():
    engine = PredictionFusionEngine()
    audio = {"start_time": 0.0, "end_time": 2.5, "audio_confidence": 0.95, "snr_db": 22.0}
    visual = {"start_time": 0.0, "end_time": 2.5, "visual_confidence": 0.80}

    seg = engine.fuse_segment(
        audio_result=audio,
        visual_result=visual,
        speaker_label="Speaker 1",
        text_hypothesis="The meeting will start tomorrow.",
        mode=ProcessingMode.AUDIO_VISUAL
    )

    assert seg.combined_confidence >= 0.85
    assert seg.status == ConfidenceStatus.HIGH_CONFIDENCE
    assert seg.speaker == "Speaker 1"
    assert len(seg.words) == 5


def test_silent_visual_mode():
    engine = PredictionFusionEngine()
    audio = {"start_time": 0.0, "end_time": 2.0, "audio_confidence": 0.0, "snr_db": -50.0}
    visual = {"start_time": 0.0, "end_time": 2.0, "visual_confidence": 0.70}

    seg = engine.fuse_segment(
        audio_result=audio,
        visual_result=visual,
        speaker_label="Speaker 1",
        text_hypothesis="Silent lip reading.",
        mode=ProcessingMode.SILENT_VISUAL
    )

    assert seg.combined_confidence == 0.70
    assert seg.status == ConfidenceStatus.UNCERTAIN_VISUAL_ONLY
    assert "Visual-only" in (seg.uncertainty_note or "")
