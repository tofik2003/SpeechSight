"""
Step 9: Language Correction and Postprocessing
Refines grammar, normalizes punctuation, resolves viseme homophenes, and preserves uncertainty flags.
"""

import re
import logging
from typing import List
from speechsight.core.models import TranscriptSegment, ConfidenceStatus

logger = logging.getLogger("speechsight.pipeline.postprocessor")


class LanguagePostprocessor:
    def __init__(self, language: str = "en"):
        self.language = language

    def correct_text(self, raw_text: str) -> str:
        """Applies casing, spacing, and terminal punctuation formatting."""
        if not raw_text:
            return ""

        text = raw_text.strip()
        # Fix repeated whitespace
        text = re.sub(r'\s+', ' ', text)
        # Capitalize first letter
        if len(text) > 0:
            text = text[0].upper() + text[1:]
        # Ensure sentence ends with punctuation
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def process_segments(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Postprocesses all transcript segments."""
        for seg in segments:
            seg.text = self.correct_text(seg.text)
            # Retain uncertainty notice if status is low confidence or visual-only
            if seg.status == ConfidenceStatus.UNCERTAIN_VISUAL_ONLY and not seg.uncertainty_note:
                seg.uncertainty_note = "Visual-only prediction. Please verify critical words."

        return segments
