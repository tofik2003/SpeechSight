import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from speechsight.server.app import app
from speechsight.data.consent_manager import LegalDataManager
from speechsight.data.legal_data_loader import LegalDatasetLoader
from speechsight.training.trainer import AVSRTrainer

client = TestClient(app)


def test_consent_manager(tmp_path):
    mgr = LegalDataManager(tmp_path / "consent")
    records = mgr.get_all_consent_records()
    assert len(records) >= 20
    assert "speaker_001" in records
    
    check = mgr.verify_sample_legality("speaker_001")
    assert check["is_legal_to_train"] is True
    assert check["license"] == "CC-BY-4.0"


def test_legal_dataset_loader(tmp_path):
    loader = LegalDatasetLoader(tmp_path / "legal_corpus")
    summary = loader.prepare_full_legal_corpus(samples_per_speaker=1)
    assert summary["total_samples"] >= 20
    assert (tmp_path / "legal_corpus" / "processed" / "manifest.json").exists()


def test_training_api_endpoints():
    res1 = client.get("/api/train/datasets")
    assert res1.status_code == 200
    data1 = res1.json()
    assert "verified_datasets" in data1
    assert len(data1["verified_datasets"]) >= 3

    res2 = client.get("/api/train/status")
    assert res2.status_code == 200
    data2 = res2.json()
    assert "is_training" in data2
