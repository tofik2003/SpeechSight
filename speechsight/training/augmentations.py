"""
SpeechSight AI - Multimodal Data Augmentation Engine for Audio-Visual Speech Recognition.
Provides visual spatiotemporal augmentations (jitter, brightness/contrast, Gaussian blur, synthetic occlusion)
and acoustic augmentations (SpecAugment, background noise injection, gain perturbation).
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import cv2


class VisualAugmentor:
    """
    Visual data augmentor operating on normalized (T, H, W) mouth sequence tensors.
    """
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.85, 1.15),
        contrast_range: Tuple[float, float] = (0.80, 1.20),
        crop_jitter_ratio: float = 0.08,
        blur_prob: float = 0.30,
        occlusion_prob: float = 0.25
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.crop_jitter_ratio = crop_jitter_ratio
        self.blur_prob = blur_prob
        self.occlusion_prob = occlusion_prob

    def augment(self, mouth_tensor: np.ndarray) -> np.ndarray:
        """
        Applies random visual transformations across the temporal sequence.
        Input: (T, H, W) numpy array in float32 [0.0, 1.0]
        Output: (T, H, W) augmented numpy array
        """
        if mouth_tensor is None or len(mouth_tensor) == 0:
            return mouth_tensor

        T, H, W = mouth_tensor.shape
        augmented = mouth_tensor.copy()

        # 1. Random Brightness & Contrast
        alpha = np.random.uniform(*self.contrast_range)
        beta = np.random.uniform(self.brightness_range[0] - 1.0, self.brightness_range[1] - 1.0)
        augmented = np.clip(augmented * alpha + beta, 0.0, 1.0)

        # 2. Random Gaussian Blur (Simulate defocus or fast movement)
        if np.random.rand() < self.blur_prob:
            ksize = 3
            for t in range(T):
                augmented[t] = cv2.GaussianBlur(augmented[t], (ksize, ksize), sigmaX=0.8)

        # 3. Random Synthetic Mouth Occlusion (Simulate mic, hand, beard, moustache)
        if np.random.rand() < self.occlusion_prob:
            occ_h = int(H * np.random.uniform(0.15, 0.30))
            occ_w = int(W * np.random.uniform(0.20, 0.40))
            y0 = np.random.randint(0, max(1, H - occ_h))
            x0 = np.random.randint(0, max(1, W - occ_w))
            occ_val = np.random.uniform(0.1, 0.4)
            augmented[:, y0:y0+occ_h, x0:x0+occ_w] = occ_val

        # 4. Spatial Jitter / Sub-crop Rescale
        if np.random.rand() < 0.5:
            pad_h = int(H * self.crop_jitter_ratio)
            pad_w = int(W * self.crop_jitter_ratio)
            y_start = np.random.randint(0, pad_h + 1)
            x_start = np.random.randint(0, pad_w + 1)
            
            # Crop slightly and resize back to (H, W)
            cropped_seq = []
            for t in range(T):
                c = augmented[t, y_start:H-pad_h+y_start, x_start:W-pad_w+x_start]
                resized = cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR)
                cropped_seq.append(resized)
            augmented = np.array(cropped_seq, dtype=np.float32)

        return augmented.astype(np.float32)


class AcousticAugmentor:
    """
    Acoustic data augmentor operating on 16kHz mono audio waveforms.
    """
    def __init__(
        self,
        noise_snr_db_range: Tuple[float, float] = (5.0, 25.0),
        gain_db_range: Tuple[float, float] = (-6.0, 6.0),
        mask_prob: float = 0.35,
        max_mask_length_ms: float = 120.0
    ):
        self.noise_snr_db_range = noise_snr_db_range
        self.gain_db_range = gain_db_range
        self.mask_prob = mask_prob
        self.max_mask_length_ms = max_mask_length_ms

    def augment(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Applies acoustic perturbations (noise injection, gain scaling, temporal masking).
        """
        if audio_samples is None or len(audio_samples) == 0:
            return audio_samples

        augmented = audio_samples.copy().astype(np.float32)
        N = len(augmented)

        # 1. Random Gain Scaling
        gain_db = np.random.uniform(*self.gain_db_range)
        gain_factor = 10.0 ** (gain_db / 20.0)
        augmented = augmented * gain_factor

        # 2. Additive Background Noise Injection
        if np.random.rand() < 0.7:
            target_snr_db = np.random.uniform(*self.noise_snr_db_range)
            signal_power = np.mean(augmented ** 2) + 1e-10
            noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))
            noise = np.random.normal(0, np.sqrt(noise_power), size=N).astype(np.float32)
            augmented = augmented + noise

        # 3. SpecAugment-style Temporal Masking (zeroing out random chunks)
        if np.random.rand() < self.mask_prob:
            mask_len = int((self.max_mask_length_ms / 1000.0) * sample_rate)
            if N > mask_len:
                mask_start = np.random.randint(0, N - mask_len)
                augmented[mask_start:mask_start + mask_len] = 0.0

        return np.clip(augmented, -1.0, 1.0).astype(np.float32)


def augment_sample(
    mouth_tensor: Optional[np.ndarray],
    audio_samples: Optional[np.ndarray],
    visual_augmentor: Optional[VisualAugmentor] = None,
    acoustic_augmentor: Optional[AcousticAugmentor] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Applies unified multimodal data augmentations.
    """
    v_aug = visual_augmentor or VisualAugmentor()
    a_aug = acoustic_augmentor or AcousticAugmentor()

    aug_mouth = v_aug.augment(mouth_tensor) if mouth_tensor is not None else None
    aug_audio = a_aug.augment(audio_samples) if audio_samples is not None else None

    return aug_mouth, aug_audio
