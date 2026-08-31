import numpy as np
from speechsight.training.augmentations import VisualAugmentor, AcousticAugmentor, augment_sample


def test_visual_augmentor_shape_and_range():
    aug = VisualAugmentor(blur_prob=1.0, occlusion_prob=1.0)
    mouth_tensor = np.ones((10, 96, 96), dtype=np.float32) * 0.5

    augmented = aug.augment(mouth_tensor)

    assert augmented.shape == (10, 96, 96)
    assert np.all(augmented >= 0.0)
    assert np.all(augmented <= 1.0)
    # Ensure some variation was introduced
    assert not np.array_equal(augmented, mouth_tensor)


def test_acoustic_augmentor_noise_and_masking():
    aug = AcousticAugmentor(mask_prob=1.0, max_mask_length_ms=50.0)
    audio = np.sin(np.linspace(0, 100, 16000)).astype(np.float32) * 0.5

    augmented = aug.augment(audio, sample_rate=16000)

    assert augmented.shape == (16000,)
    assert np.all(augmented >= -1.0)
    assert np.all(augmented <= 1.0)
    # Check if masking or noise occurred
    assert not np.array_equal(augmented, audio)


def test_augment_sample_multimodal():
    mouth = np.random.uniform(0.2, 0.8, size=(12, 96, 96)).astype(np.float32)
    audio = np.random.uniform(-0.5, 0.5, size=8000).astype(np.float32)

    aug_mouth, aug_audio = augment_sample(mouth, audio)

    assert aug_mouth.shape == (12, 96, 96)
    assert aug_audio.shape == (8000,)
    assert aug_mouth.dtype == np.float32
    assert aug_audio.dtype == np.float32
