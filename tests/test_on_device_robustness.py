import tempfile
import numpy as np
from pathlib import Path

from speechsight.core.models import ProcessingMode, ConfidenceStatus
from speechsight.pipeline.fusion_engine import PredictionFusionEngine
from speechsight.pipeline.audio_recognizer import AudioRecognizer
from speechsight.pipeline.visual_recognizer import VisualSpeechRecognizer
from speechsight.pipeline.language_postprocessor import LanguagePostprocessor
from speechsight.pipeline.subtitle_generator import SubtitleGenerator, format_timestamp_srt, format_timestamp_vtt
from speechsight.training.models_numpy import AVSRNeuralModel


def test_fusion_engine_extreme_snr():
    engine = PredictionFusionEngine()

    # Extreme low SNR (-20 dB) -> should prioritize visual lip features
    audio_noisy = {"start_time": 0.0, "end_time": 2.0, "audio_confidence": 0.20, "snr_db": -20.0}
    visual_clear = {"visual_confidence": 0.85}

    seg_noisy = engine.fuse_segment(
        audio_result=audio_noisy,
        visual_result=visual_clear,
        speaker_label="Speaker 1",
        text_hypothesis="critical medical alert",
        mode=ProcessingMode.AUDIO_VISUAL
    )
    assert seg_noisy.combined_confidence > 0.50
    assert seg_noisy.status in [ConfidenceStatus.HIGH_CONFIDENCE, ConfidenceStatus.MEDIUM_CONFIDENCE]

    # Extreme high SNR (40 dB) -> strong audio confidence
    audio_clean = {"start_time": 0.0, "end_time": 2.0, "audio_confidence": 0.98, "snr_db": 40.0}
    visual_occluded = {"visual_confidence": 0.30}

    seg_clean = engine.fuse_segment(
        audio_result=audio_clean,
        visual_result=visual_occluded,
        speaker_label="Speaker 1",
        text_hypothesis="crystal clear audio",
        mode=ProcessingMode.AUDIO_VISUAL
    )
    assert seg_clean.combined_confidence >= 0.70


def test_audio_recognizer_empty_and_silence():
    rec = AudioRecognizer(sample_rate=16000)

    # Empty array
    snr_empty = rec.estimate_snr_db(np.array([]))
    assert snr_empty == -100.0

    segs = rec.detect_speech_segments(np.array([]), duration_sec=5.0)
    assert len(segs) >= 1

    # Complete zero silence
    silence = np.zeros(16000, dtype=np.float32)
    res = rec.transcribe_audio_segment(silence, start_time=0.0, end_time=1.0)
    assert res["audio_confidence"] <= 0.10
    assert res["has_voice"] is False


def test_visual_recognizer_empty_and_single_frame():
    rec = VisualSpeechRecognizer()

    # Empty tensor
    empty_tensor = np.zeros((0, 96, 96), dtype=np.float32)
    conf_empty = rec.calculate_visual_confidence(empty_tensor)
    assert conf_empty <= 0.20

    # Single frame tensor
    single_frame = np.ones((1, 96, 96), dtype=np.float32) * 0.5
    feats = rec.extract_visual_features(single_frame)
    assert feats.shape == (1, 5)


def test_language_postprocessor_edge_cases():
    lp = LanguagePostprocessor()

    assert lp.correct_text("") == ""
    assert lp.correct_text("   ") == ""
    assert lp.correct_text("hello world") == "Hello world."
    assert lp.correct_text("how are you?") == "How are you?"
    assert lp.correct_text("   multi   space   word   ") == "Multi space word."


def test_subtitle_timestamps_formatting():
    # Test short and long durations
    srt_0 = format_timestamp_srt(0.0)
    assert srt_0 == "00:00:00,000"

    srt_long = format_timestamp_srt(3723.456)
    assert srt_long == "01:02:03,456"

    vtt_long = format_timestamp_vtt(3723.456)
    assert vtt_long == "01:02:03.456"


def test_avsr_neural_model_forward_and_decode():
    model = AVSRNeuralModel()
    mouth = np.random.randn(10, 96, 96).astype(np.float32)
    audio = np.random.randn(8000).astype(np.float32)

    log_probs, cache = model.forward(mouth, audio)
    assert log_probs.shape == (10, model.vocab_size)
    assert np.all(log_probs <= 0.0)

    decoded = model.decode_greedy(log_probs)
    assert isinstance(decoded, str)
