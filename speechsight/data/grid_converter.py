"""
GRID Corpus Ingestion and Converter.
Parses GRID dataset video/align files into SpeechSight format.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging
from speechsight.data.dataset_prep import DatasetPreparationPipeline

logger = logging.getLogger("speechsight.data.grid")


class GRIDConverter:
    def __init__(self, pipeline: DatasetPreparationPipeline):
        self.pipeline = pipeline

    def parse_align_file(self, align_path: Path) -> str:
        """Parses GRID .align file containing millisecond time markers and words."""
        words = []
        with open(align_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    w = parts[2]
                    if w not in ["sil", "sp"]:
                        words.append(w)
        return " ".join(words)

    def convert_directory(self, grid_root: Path, max_samples: int = 100) -> List[Dict[str, Any]]:
        """Converts GRID files into standard format."""
        grid_root = Path(grid_root)
        align_files = list(grid_root.rglob("*.align"))[:max_samples]
        processed = []

        for align_file in align_files:
            video_file = align_file.with_suffix(".mpg")
            if not video_file.exists():
                video_file = align_file.with_suffix(".mp4")
            if not video_file.exists():
                continue

            text = self.parse_align_file(align_file)
            if not text:
                continue

            speaker_id = align_file.parent.name
            clip_id = f"grid_{speaker_id}_{align_file.stem}"

            meta = self.pipeline.process_sample(
                video_path=video_file,
                transcript_text=text,
                speaker_id=speaker_id,
                clip_id=clip_id,
                language="en"
            )
            processed.append(meta)

        logger.info(f"Converted {len(processed)} GRID clips.")
        return processed
