import tempfile
import json
from pathlib import Path
import numpy as np
from speechsight.training.export_mobile import MobileModelExporter


def test_int8_quantization_and_dequantization():
    exporter = MobileModelExporter()
    weights = np.random.randn(64, 128).astype(np.float32)

    q, scale, zero_pt = exporter.quantize_int8(weights)

    assert q.dtype == np.int8
    assert np.all(q >= -127)
    assert np.all(q <= 127)

    dequantized = exporter.dequantize_int8(q, scale)
    max_error = np.max(np.abs(weights - dequantized))

    # Quantization error should be very low (< 0.05)
    assert max_error < 0.05


def test_mobile_model_export_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ckpt_path = tmp_path / "model.npz"
        assets_dir = tmp_path / "assets" / "models"

        # Create dummy checkpoint
        np.savez_compressed(
            ckpt_path,
            W_v=np.random.randn(96, 64).astype(np.float32),
            b_v=np.zeros(64, dtype=np.float32),
            W_a=np.random.randn(64, 64).astype(np.float32),
            b_a=np.zeros(64, dtype=np.float32),
            W_fuse=np.random.randn(128, 128).astype(np.float32),
            b_fuse=np.zeros(128, dtype=np.float32),
            W_h=np.random.randn(128, 128).astype(np.float32),
            W_x=np.random.randn(128, 128).astype(np.float32),
            b_h=np.zeros(128, dtype=np.float32),
            W_ctc=np.random.randn(128, 32).astype(np.float32),
            b_ctc=np.zeros(32, dtype=np.float32)
        )

        exporter = MobileModelExporter(checkpoint_path=ckpt_path, output_assets_dir=assets_dir)
        res = exporter.export_quantized_model()

        assert Path(res["quantized_npz"]).exists()
        assert Path(res["manifest_json"]).exists()
        assert res["compression_ratio"] > 0.50

        # Validate manifest content
        with open(res["manifest_json"], "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["model_format"] == "quantized_npz_int8"
        assert manifest["parameters"]["vocab_size"] == 32
        assert "Android (Compose / Room / Media3 / NNAPI)" in manifest["deployment_targets"]
