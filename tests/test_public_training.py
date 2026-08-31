import tempfile
import json
from pathlib import Path
import numpy as np
from speechsight.data.public_data_loader import PublicCorpusLoader
from speechsight.training.train_advanced import AdvancedAVSRTrainer


def test_public_corpus_loader_merge():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create two sample manifests
        m1 = {
            "corpus_name": "GRID Test",
            "all_samples": [
                {"clip_id": "c1", "speaker_id": "grid_s01", "transcript": "set blue at nine"},
                {"clip_id": "c2", "speaker_id": "grid_s02", "transcript": "bin green by one"}
            ]
        }
        m2 = {
            "corpus_name": "LRS3 Test",
            "all_samples": [
                {"clip_id": "c3", "speaker_id": "speaker_001", "transcript": "natural sentence here"},
                {"clip_id": "c4", "speaker_id": "speaker_002", "transcript": "second sentence test"}
            ]
        }

        p1 = tmp_path / "m1.json"
        p2 = tmp_path / "m2.json"
        with open(p1, "w") as f:
            json.dump(m1, f)
        with open(p2, "w") as f:
            json.dump(m2, f)

        loader = PublicCorpusLoader()
        merged = loader.merge_corpora(
            manifest_paths=[p1, p2],
            output_dir=tmp_path / "merged",
            corpus_name="Test Combined",
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25
        )

        assert merged["total_samples"] == 4
        assert len(merged["train"]) == 2
        assert len(merged["val"]) >= 1
        assert (tmp_path / "merged" / "manifest.json").exists()


def test_advanced_trainer_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        crops_dir = tmp_path / "mouth_crops"
        transcripts_dir = tmp_path / "transcripts"
        crops_dir.mkdir(parents=True)
        transcripts_dir.mkdir(parents=True)

        # Generate dummy crop and transcript
        np.save(crops_dir / "clip_1_mouth.npy", np.random.randn(15, 96, 96).astype(np.float32))
        with open(transcripts_dir / "clip_1.txt", "w") as f:
            f.write("test speech")

        manifest = {
            "corpus_name": "Mini Corpus",
            "train": [
                {"clip_id": "c1", "mouth_file": "clip_1_mouth.npy", "transcript_file": "clip_1.txt"}
            ],
            "val": [
                {"clip_id": "c1", "mouth_file": "clip_1_mouth.npy", "transcript_file": "clip_1.txt"}
            ]
        }
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        trainer = AdvancedAVSRTrainer(output_dir=tmp_path / "checkpoints", use_augmentations=True)
        res = trainer.train_manifest(manifest_path=manifest_path, epochs=2, initial_lr=1e-3)

        assert res["status"] == "completed"
        assert res["epochs_trained"] == 2
        assert len(res["history"]) == 2
        assert (tmp_path / "checkpoints" / "speechsight_avsr_model.npz").exists()
        assert (tmp_path / "checkpoints" / "training_history.json").exists()
        assert (tmp_path / "checkpoints" / "model_card.json").exists()
