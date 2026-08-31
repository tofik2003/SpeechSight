"""
LRS3 (Lip Reading Sentences 3) Dataset Ingestion and Converter.
Parses sentence-level alignments and video clips from Oxford BBC LRS3 format.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging
from speechsight.data.dataset_prep import DatasetPreparationPipeline

logger = logging.getLogger("speechsight.data.lrs3")


class LRS3Converter:
    def __init__(self, pipeline: DatasetPreparationPipeline):
        self.pipeline = pipeline

    def parse_alignment_file(self, txt_path: Path) -> Dict[str, Any]:
        """
        Parses LRS3 alignment file containing Text: ... and word timings.
        """
        text = ""
        words = []
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith("Text:"):
                text = line.replace("Text:", "").strip()
            elif line.startswith("Conf:"):
                continue
            elif line and not line.startswith("Word:"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        w, st, et = parts[0], float(parts[1]), float(parts[2])
                        words.append({"word": w, "start": st, "end": et})
                    except ValueError:
                        pass

        return {"text": text, "words": words}

    def convert_directory(self, lrs3_root: Path, max_samples: int = 100) -> List[Dict[str, Any]]:
        """Converts an LRS3 dataset folder into SpeechSight standard format."""
        lrs3_root = Path(lrs3_root)
        txt_files = list(lrs3_root.rglob("*.txt"))[:max_samples]
        processed = []

        for txt_file in txt_files:
            video_file = txt_file.with_suffix(".mp4")
            if not video_file.exists():
                continue

            parsed = self.parse_alignment_file(txt_file)
            if not parsed["text"]:
                continue

            speaker_id = txt_file.parent.name
            clip_id = f"lrs3_{speaker_id}_{txt_file.stem}"

            meta = self.pipeline.process_sample(
                video_path=video_file,
                transcript_text=parsed["text"],
                speaker_id=speaker_id,
                clip_id=clip_id,
                language="en"
            )
            processed.append(meta)

        logger.info(f"Converted {len(processed)} LRS3 clips.")
        return processed
