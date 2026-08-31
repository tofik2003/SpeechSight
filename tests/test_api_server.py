import pytest
from fastapi.testclient import TestClient
from speechsight.server.app import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "capabilities" in data


def test_samples_endpoint():
    res = client.get("/api/samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) >= 4
    assert any(s["id"] == "keynote" for s in samples)


def test_transcribe_sample_endpoint():
    res = client.post("/api/transcribe/sample", json={
        "preset_id": "keynote",
        "mode": "audio_visual",
        "language": "en",
        "privacy_mode": True
    })
    assert res.status_code == 200
    data = res.json()
    assert "video_id" in data
    assert len(data["segments"]) > 0
    assert data["processing_mode"] == "audio_visual"


def test_export_endpoint():
    res = client.post("/api/export", json={
        "segments": [
            {
                "start_time": 0.0,
                "end_time": 2.8,
                "speaker": "Speaker 1",
                "text": "The meeting will start tomorrow.",
                "audio_confidence": 0.94,
                "visual_confidence": 0.72,
                "combined_confidence": 0.92,
                "status": "high_confidence"
            }
        ],
        "format": "srt",
        "include_speaker": True,
        "include_timestamps": True
    })
    assert res.status_code == 200
    data = res.json()
    assert data["format"] == "srt"
    assert "The meeting will start tomorrow" in data["content"]


def test_wipe_endpoint():
    res = client.post("/api/wipe")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["privacy_status"] == "wiped"
