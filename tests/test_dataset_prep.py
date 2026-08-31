import pytest
from pathlib import Path
from speechsight.data.dataset_prep import DatasetPreparationPipeline
from speechsight.data.sample_generator import SampleClipGenerator


def test_dataset_preparation_pipeline(tmp_path):
    gen_dir = tmp_path / "raw"
    gen = SampleClipGenerator(gen_dir)
    clip_info = gen.generate_synthetic_av_clip(
        filename="clip_001.mp4",
        text="Testing dataset preparation.",
        duration_sec=1.5,
        speaker_id="speaker_001"
    )

    out_dir = tmp_path / "dataset"
    pipeline = DatasetPreparationPipeline(out_dir)

    meta = pipeline.process_sample(
        video_path=Path(clip_info["video_path"]),
        transcript_text=clip_info["transcript"],
        speaker_id="speaker_001"
    )

    assert meta["video_file"] == "speaker_001_clip_clip_001.mp4"
    assert meta["optional_metadata"]["speaker_id"] == "speaker_001"
    assert (out_dir / "transcripts" / meta["transcript_file"]).exists()
    assert (out_dir / "mouth_crops" / meta["mouth_file"]).exists()

    manifest = pipeline.build_manifest(train_split=1.0, val_split=0.0)
    assert manifest["total_samples"] == 1
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "dataset_index.csv").exists()
