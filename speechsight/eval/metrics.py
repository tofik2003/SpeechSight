"""
Evaluation metrics for SpeechSight AI:
- Word Error Rate (WER)
- Character Error Rate (CER)
- Active Speaker Detection Accuracy
- Expected Calibration Error (ECE) for confidence calibration
- Real-Time Factor (RTF)
"""

import numpy as np
from typing import List, Tuple, Dict, Any


def levenshtein_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[int, int, int, int]:
    """
    Computes Levenshtein distance between reference and hypothesis token sequences.
    Returns (substitutions, deletions, insertions, total_ref_tokens).
    """
    R = len(ref_tokens)
    H = len(hyp_tokens)

    d = np.zeros((R + 1, H + 1), dtype=int)
    for i in range(R + 1):
        d[i, 0] = i
    for j in range(H + 1):
        d[0, j] = j

    for i in range(1, R + 1):
        for j in range(1, H + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    # Backtrack to count S, D, I
    i, j = R, H
    substitutions, deletions, insertions = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and d[i, j] == d[i, j - 1] + 1:
            insertions += 1
            j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            deletions += 1
            i -= 1

    return substitutions, deletions, insertions, R


def compute_wer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Computes Word Error Rate (WER) = (S + D + I) / N
    """
    ref_words = [w.lower().strip(".,!?:;\"'()") for w in reference.split() if w.strip(".,!?:;\"'()")]
    hyp_words = [w.lower().strip(".,!?:;\"'()") for w in hypothesis.split() if w.strip(".,!?:;\"'()")]

    if not ref_words:
        wer = 1.0 if hyp_words else 0.0
        return {"wer": wer, "substitutions": 0, "deletions": 0, "insertions": len(hyp_words), "ref_words": 0}

    s, d, i, n = levenshtein_distance(ref_words, hyp_words)
    wer = min(1.0, (s + d + i) / n)
    return {
        "wer": round(float(wer), 4),
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "ref_words": n
    }


def compute_cer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Computes Character Error Rate (CER) = (S + D + I) / N
    """
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))

    if not ref_chars:
        cer = 1.0 if hyp_chars else 0.0
        return {"cer": cer, "substitutions": 0, "deletions": 0, "insertions": len(hyp_chars), "ref_chars": 0}

    s, d, i, n = levenshtein_distance(ref_chars, hyp_chars)
    cer = min(1.0, (s + d + i) / n)
    return {
        "cer": round(float(cer), 4),
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "ref_chars": n
    }


def compute_active_speaker_accuracy(ground_truth_speakers: List[str], predicted_speakers: List[str]) -> float:
    """Computes categorical accuracy for active speaker predictions."""
    if not ground_truth_speakers or not predicted_speakers:
        return 1.0
    matches = sum(1 for gt, pred in zip(ground_truth_speakers, predicted_speakers) if gt.lower() == pred.lower())
    return round(matches / min(len(ground_truth_speakers), len(predicted_speakers)), 4)


def compute_calibration_error(confidences: List[float], correct_flags: List[bool], num_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    """
    if not confidences or not correct_flags:
        return 0.0

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(num_bins):
        bin_lower, bin_upper = bins[i], bins[i + 1]
        in_bin = [
            (c, flag) for c, flag in zip(confidences, correct_flags)
            if bin_lower <= c < bin_upper or (i == num_bins - 1 and c == bin_upper)
        ]
        if in_bin:
            bin_size = len(in_bin)
            avg_confidence = np.mean([c for c, _ in in_bin])
            avg_accuracy = np.mean([1.0 if flag else 0.0 for _, flag in in_bin])
            ece += (bin_size / total) * abs(avg_accuracy - avg_confidence)

    return round(float(ece), 4)
