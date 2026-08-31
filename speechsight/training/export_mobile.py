"""
SpeechSight AI - Mobile Model Quantization and Export Pipeline.
Quantizes AVSR neural weights to INT8 / FP16 format for efficient on-device Android execution.
Outputs optimized model assets to `android/app/src/main/assets/models/`.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

logger = logging.getLogger("speechsight.training.export")


class MobileModelExporter:
    """
    Exports and quantizes neural weights for on-device mobile inference (Android TFLite / ONNX / NumPy embedded).
    """
    def __init__(
        self,
        checkpoint_path: Path = Path("./checkpoints/speechsight_avsr_model.npz"),
        output_assets_dir: Path = Path("./android/app/src/main/assets/models")
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_assets_dir = Path(output_assets_dir)
        self.output_assets_dir.mkdir(parents=True, exist_ok=True)

    def quantize_int8(self, array: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Symmetric INT8 quantization: quantized = round(array / scale).
        """
        max_val = np.max(np.abs(array))
        if max_val == 0:
            return np.zeros_like(array, dtype=np.int8), 1.0, 0
        scale = float(max_val / 127.0)
        q = np.clip(np.round(array / scale), -127, 127).astype(np.int8)
        return q, scale, 0

    def dequantize_int8(self, q_array: np.ndarray, scale: float) -> np.ndarray:
        """Dequantizes INT8 tensor back to FP32."""
        return (q_array.astype(np.float32) * scale)

    def export_quantized_model(self) -> Dict[str, Any]:
        """
        Loads checkpoint weights, quantizes all weight matrices, and saves assets for Android app.
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")

        raw_data = np.load(self.checkpoint_path)
        quantized_dict = {}
        scales_dict = {}
        original_size_bytes = 0
        quantized_size_bytes = 0

        for key in raw_data.files:
            tensor = raw_data[key]
            original_size_bytes += tensor.nbytes

            if key.startswith("W_") or key.startswith("b_"):
                q_tensor, scale, zero_point = self.quantize_int8(tensor)
                quantized_dict[key] = q_tensor
                scales_dict[f"{key}_scale"] = np.array([scale], dtype=np.float32)
                quantized_size_bytes += q_tensor.nbytes + 4
            else:
                quantized_dict[key] = tensor
                quantized_size_bytes += tensor.nbytes

        # Save quantized archive
        out_npz_path = self.output_assets_dir / "speechsight_avsr_quantized.npz"
        np.savez_compressed(out_npz_path, **quantized_dict, **scales_dict)

        actual_file_size = out_npz_path.stat().st_size
        compression_ratio = 1.0 - (actual_file_size / max(1, original_size_bytes))

        # Create mobile model manifest
        manifest = {
            "model_format": "quantized_npz_int8",
            "model_version": "2.0.0",
            "input_shapes": {
                "mouth_crops": [None, 96, 96],
                "audio_waveform": [None],
                "sample_rate_hz": 16000
            },
            "output_shapes": {
                "ctc_log_probs": [None, 32]
            },
            "parameters": {
                "d_visual": 64,
                "d_audio": 64,
                "d_fused": 128,
                "vocab_size": 32
            },
            "quantization": {
                "scheme": "symmetric_int8",
                "original_size_bytes": original_size_bytes,
                "compressed_size_bytes": actual_file_size,
                "compression_reduction_percent": round(compression_ratio * 100, 2)
            },
            "deployment_targets": [
                "Android (Compose / Room / Media3 / NNAPI)",
                "On-Device Termux / FastAPI Edge Server",
                "Browser WebAssembly / ONNX"
            ],
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S UTC")
        }

        manifest_file = self.output_assets_dir / "mobile_model_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Copy model card if exists
        model_card_src = self.checkpoint_path.parent / "model_card.json"
        if model_card_src.exists():
            with open(model_card_src, "r", encoding="utf-8") as f:
                card_data = json.load(f)
            with open(self.output_assets_dir / "model_card.json", "w", encoding="utf-8") as f:
                json.dump(card_data, f, indent=2)

        logger.info(
            f"Successfully exported mobile model to {out_npz_path} "
            f"({actual_file_size / 1024:.1f} KB, {manifest['quantization']['compression_reduction_percent']}% reduction)"
        )

        return {
            "quantized_npz": str(out_npz_path),
            "manifest_json": str(manifest_file),
            "size_kb": round(actual_file_size / 1024, 2),
            "compression_ratio": round(compression_ratio, 4)
        }


if __name__ == "__main__":
    exporter = MobileModelExporter()
    exporter.export_quantized_model()
