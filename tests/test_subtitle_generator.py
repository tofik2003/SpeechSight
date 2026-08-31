import pytest
from speechsight.pipeline.subtitle_generator import SubtitleGenerator, format_timestamp_srt, format_timestamp_vtt
from speechsight.core.models import TranscriptSegment, ConfidenceStatus


def test_timestamp_formatting():
    assert format_timestamp_srt(0.0) == "00:00:00,000"
    assert format_timestamp_srt(65.432) == "00:01:05,432"
    assert format_timestamp_vtt(65.432) == "00:01:05.432"


def test_subtitle_export_formats():
    segments = [
        TranscriptSegment(
            id="seg_001",
            start_time=0.0,
            end_time=2.8,
            speaker="Speaker 1",
            text="The meeting will start tomorrow.",
            audio_confidence=0.94,
            visual_confidence=0.72,
            combined_confidence=0.92,
            status=ConfidenceStatus.HIGH_CONFIDENCE
        )
    ]

    srt_out = SubtitleGenerator.to_srt(segments)
    assert "00:00:00,000 --> 00:00:02,800" in srt_out
    assert "[Speaker 1] The meeting will start tomorrow." in srt_out

    vtt_out = SubtitleGenerator.to_vtt(segments)
    assert "WEBVTT" in vtt_out
    assert "00:00:00.000 --> 00:00:02.800" in vtt_out

    txt_out = SubtitleGenerator.to_plain_text(segments)
    assert "[00:00 - 00:02] Speaker 1: The meeting will start tomorrow." in txt_out
