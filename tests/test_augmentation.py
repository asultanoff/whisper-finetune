from __future__ import annotations

import numpy as np

from whisper_finetune.augmentation import WaveformAugmenter
from whisper_finetune.config import AudioAugmentationConfig


def _waveform(seconds: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    time_axis = np.arange(int(seconds * sample_rate), dtype=np.float32) / float(sample_rate)
    return (0.15 * np.sin(2.0 * np.pi * 220.0 * time_axis)).astype(np.float32)


def _noise_config() -> AudioAugmentationConfig:
    return AudioAugmentationConfig.from_dict(
        {
            "enabled": True,
            "seed": 12345,
            "train_only": True,
            "mode": "deterministic_per_sample",
            "datasets": {
                "demo": {
                    "enabled": True,
                    "profile": "phone",
                    "splits": ["train"],
                    "apply_p": 1.0,
                }
            },
            "profiles": {
                "phone": {
                    "noise": {
                        "p": 1.0,
                        "snr_db": [10.0, 10.0],
                    }
                }
            },
        }
    )


def test_waveform_augmenter_applies_matching_train_policy() -> None:
    augmenter = WaveformAugmenter(_noise_config())
    waveform = _waveform()

    augmented = augmenter.maybe_augment(
        waveform,
        sample_id="sample-1",
        dataset_name="demo",
        dataset_repo_id="org/demo",
        dataset_split="train",
        sample_rate=16000,
    )

    assert augmented.shape == waveform.shape
    assert not np.allclose(augmented, waveform)


def test_waveform_augmenter_skips_eval_when_train_only_is_enabled() -> None:
    augmenter = WaveformAugmenter(_noise_config())
    waveform = _waveform()

    augmented = augmenter.maybe_augment(
        waveform,
        sample_id="sample-1",
        dataset_name="demo",
        dataset_repo_id="org/demo",
        dataset_split="validation",
        sample_rate=16000,
    )

    np.testing.assert_allclose(augmented, waveform)


def test_waveform_augmenter_is_deterministic_per_sample() -> None:
    augmenter = WaveformAugmenter(_noise_config())
    waveform = _waveform()

    out_a = augmenter.maybe_augment(
        waveform,
        sample_id="sample-1",
        dataset_name="demo",
        dataset_repo_id="org/demo",
        dataset_split="train",
        sample_rate=16000,
    )
    out_b = augmenter.maybe_augment(
        waveform,
        sample_id="sample-1",
        dataset_name="demo",
        dataset_repo_id="org/demo",
        dataset_split="train",
        sample_rate=16000,
    )
    out_c = augmenter.maybe_augment(
        waveform,
        sample_id="sample-2",
        dataset_name="demo",
        dataset_repo_id="org/demo",
        dataset_split="train",
        sample_rate=16000,
    )

    np.testing.assert_allclose(out_a, out_b)
    assert not np.allclose(out_a, out_c)


def test_waveform_augmenter_can_match_by_repo_id() -> None:
    augmenter = WaveformAugmenter(
        AudioAugmentationConfig.from_dict(
            {
                "enabled": True,
                "seed": 7,
                "datasets": {
                    "org/demo": {
                        "enabled": True,
                        "profile": "phone",
                        "splits": ["train"],
                        "apply_p": 1.0,
                    }
                },
                "profiles": {"phone": {"noise": {"p": 1.0, "snr_db": [15.0, 15.0]}}},
            }
        )
    )
    waveform = _waveform()

    augmented = augmenter.maybe_augment(
        waveform,
        sample_id="sample-1",
        dataset_name="alias-demo",
        dataset_repo_id="org/demo",
        dataset_split="train",
        sample_rate=16000,
    )

    assert augmented.shape == waveform.shape
    assert not np.allclose(augmented, waveform)
