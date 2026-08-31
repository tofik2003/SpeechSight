import pytest
from speechsight.eval.metrics import (
    compute_wer,
    compute_cer,
    compute_active_speaker_accuracy,
    compute_calibration_error
)


def test_wer_calculation():
    # Exact match
    res1 = compute_wer("the meeting will start tomorrow", "the meeting will start tomorrow")
    assert res1["wer"] == 0.0

    # 1 substitution in 5 words -> WER = 0.2
    res2 = compute_wer("the meeting will start tomorrow", "the meeting will begin tomorrow")
    assert res2["wer"] == 0.2
    assert res2["substitutions"] == 1


def test_cer_calculation():
    res = compute_cer("speech", "speed")
    assert res["cer"] > 0.0


def test_active_speaker_accuracy():
    gt = ["Speaker 1", "Speaker 1", "Speaker 2", "Speaker 2"]
    pred = ["Speaker 1", "Speaker 1", "Speaker 2", "Speaker 1"]
    acc = compute_active_speaker_accuracy(gt, pred)
    assert acc == 0.75


def test_expected_calibration_error():
    confs = [0.9, 0.8, 0.6, 0.4]
    flags = [True, True, True, False]
    ece = compute_calibration_error(confs, flags)
    assert ece >= 0.0
