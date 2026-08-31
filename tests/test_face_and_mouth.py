import pytest
import numpy as np
from speechsight.pipeline.face_detector import FaceDetector
from speechsight.pipeline.face_tracker import FaceTracker
from speechsight.pipeline.mouth_extractor import MouthExtractor
from speechsight.data.sample_generator import SampleClipGenerator
from speechsight.pipeline.video_decoder import VideoDecoder


@pytest.fixture
def sample_frames(tmp_path):
    gen = SampleClipGenerator(tmp_path)
    info = gen.generate_synthetic_av_clip(
        filename="face_test.mp4",
        text="Testing face detection.",
        duration_sec=1.5
    )
    decoder = VideoDecoder()
    decoded = decoder.decode(info["video_path"])
    return decoded["frames"], decoded["timestamps"]


def test_face_detection_and_mouth_extraction(sample_frames):
    frames, timestamps = sample_frames
    detector = FaceDetector()
    tracker = FaceTracker()
    mouth_extractor = MouthExtractor(crop_size=96)

    # Face detection
    detections = detector.process_video_frames(frames, timestamps)
    assert len(detections) == len(frames)
    assert len(detections[0]) >= 1
    assert detections[0][0]["bbox"].width > 0

    # Face tracking
    tracks = tracker.track(detections)
    assert len(tracks) == len(frames)
    assert tracks[0][0].track_id == 1

    # Mouth extraction
    mouth_data = mouth_extractor.extract_track_mouth_sequences(frames, tracks)
    assert 1 in mouth_data
    tensor = mouth_data[1]["tensor"]
    assert tensor.shape == (len(frames), 96, 96)
    assert np.max(np.abs(tensor)) <= 1.5
