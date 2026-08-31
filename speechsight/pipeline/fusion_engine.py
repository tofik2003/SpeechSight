"""
Step 8: Audio-Visual Prediction Fusion Engine
Dynamically balances audio and visual speech predictions based on acoustic SNR, visual clarity, and active speaker tracks.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional
from speechsight.core.models import (
    TranscriptSegment,
    ConfidenceStatus,
    ProcessingMode,
    WordToken
)
from speechsight.core.config import DEFAULT_CONFIG

logger = logging.getLogger("speechsight.pipeline.fusion")


class PredictionFusionEngine:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config

    def fuse_segment(
        self,
        audio_result: Dict[str, Any],
        visual_result: Dict[str, Any],
        speaker_label: str,
        text_hypothesis: str,
        mode: ProcessingMode = ProcessingMode.AUDIO_VISUAL
    ) -> TranscriptSegment:
        """
        Combines audio and visual predictions into a single timestamped TranscriptSegment.
        """
        start_t = audio_result.get("start_time", 0.0)
        end_t = audio_result.get("end_time", 0.0)
        a_conf = float(audio_result.get("audio_confidence", 0.0))
        v_conf = float(visual_result.get("visual_confidence", 0.0))
        snr = float(audio_result.get("snr_db", 20.0))

        if mode == ProcessingMode.SILENT_VISUAL or a_conf < 0.10:
            # Silent Lip Reading Mode
            w_audio = 0.0
            w_visual = 1.0
            combined_conf = v_conf
            status = ConfidenceStatus.UNCERTAIN_VISUAL_ONLY if v_conf < 0.75 else ConfidenceStatus.MEDIUM_CONFIDENCE
            note = "Visual-only lip-reading prediction (muted/silent audio). Verify high-stakes transcripts."
        elif mode == ProcessingMode.AUDIO_ONLY:
            # Audio Only Mode
            w_audio = 1.0
            w_visual = 0.0
            combined_conf = a_conf
            status = ConfidenceStatus.HIGH_CONFIDENCE if a_conf >= self.config.high_confidence_threshold else (
                ConfidenceStatus.MEDIUM_CONFIDENCE if a_conf >= self.config.medium_confidence_threshold else ConfidenceStatus.LOW_CONFIDENCE
            )
            note = None
        else:
            # Audio-Visual Multimodal Fusion
            if snr < self.config.noisy_snr_threshold_db:
                # Noisy environment -> rely heavily on visual lip movement
                w_audio = self.config.noisy_audio_weight
                w_visual = self.config.noisy_visual_weight
            else:
                w_audio = self.config.default_audio_weight
                w_visual = self.config.default_visual_weight

            combined_conf = (a_conf * w_audio) + (v_conf * w_visual)
            
            # Synergy bonus if both modalities are concordant
            if a_conf > 0.60 and v_conf > 0.60:
                combined_conf = min(0.99, combined_conf * 1.05)

            if combined_conf >= self.config.high_confidence_threshold:
                status = ConfidenceStatus.HIGH_CONFIDENCE
                note = None
            elif combined_conf >= self.config.medium_confidence_threshold:
                status = ConfidenceStatus.MEDIUM_CONFIDENCE
                note = "Moderate confidence; visual speech assisted noisy audio."
            else:
                status = ConfidenceStatus.LOW_CONFIDENCE
                note = "Low confidence segment; potential occlusion or background interference."

        # Break text hypothesis into timestamped word tokens
        words_list = text_hypothesis.strip().split()
        num_words = max(1, len(words_list))
        step = (end_t - start_t) / num_words
        word_tokens: List[WordToken] = []

        for i, w in enumerate(words_list):
            w_start = round(start_t + i * step, 2)
            w_end = round(start_t + (i + 1) * step, 2)
            # Slight random jitter on word-level confidence
            w_conf = round(float(np.clip(combined_conf + (np.sin(i * 1.7) * 0.04), 0.1, 0.99)), 2)
            word_tokens.append(WordToken(
                word=w,
                start_time=w_start,
                end_time=w_end,
                confidence=w_conf,
                audio_confidence=round(a_conf, 2),
                visual_confidence=round(v_conf, 2)
            ))

        return TranscriptSegment(
            start_time=round(start_t, 2),
            end_time=round(end_t, 2),
            speaker=speaker_label,
            text=text_hypothesis.strip(),
            audio_confidence=round(a_conf, 2),
            visual_confidence=round(v_conf, 2),
            combined_confidence=round(combined_conf, 2),
            status=status,
            words=word_tokens,
            uncertainty_note=note
        )

    def fuse_all_segments(
        self,
        segment_hypotheses: List[Dict[str, Any]],
        mode: ProcessingMode = ProcessingMode.AUDIO_VISUAL
    ) -> List[TranscriptSegment]:
        """Processes and fuses a batch of segment hypotheses."""
        fused_segments: List[TranscriptSegment] = []
        for idx, item in enumerate(segment_hypotheses):
            seg = self.fuse_segment(
                audio_result=item["audio"],
                visual_result=item["visual"],
                speaker_label=item.get("speaker", "Speaker 1"),
                text_hypothesis=item.get("text", ""),
                mode=mode
            )
            seg.id = f"seg_{idx+1:03d}"
            fused_segments.append(seg)
        return fused_segments
