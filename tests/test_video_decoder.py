import pytest
import numpy as np
from pathlib import Path
from speechsight.pipeline.video_decoder import VideoDecoder
from speechsight.data.sample_generator import SampleClipGenerator


@pytest.fixture
def sample_video(tmp_path):
    gen = SampleClipGenerator(tmp_path)
    info = gen.generate_synthetic_av_clip(
        filename="test_clip.mp4",
        text="Hello world test.",
        duration_sec=2.0,
        speaker_id="speaker_001"
    )
    return info


def test_video_decoder_decodes_frames(sample_video):
    decoder = VideoDecoder(target_fps=25.0)
    res = decoder.decode(sample_video["video_path"])

    assert res["frame_count"] > 0
    assert len(res["frames"]) == res["frame_count"]
    assert res["frames"][0].shape == (480, 640, 3)
    assert res["duration_sec"] >= 1.9
    assert bool(res["has_audio"]) is True
    assert len(res["audio_samples"]) > 0
