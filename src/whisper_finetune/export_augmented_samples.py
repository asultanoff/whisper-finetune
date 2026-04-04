from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from .augmentation import build_waveform_augmenter
from .config import AppConfig, DatasetConfig, load_config
from .data import _canonicalize_columns, _load_hf_split, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export clean and augmented audio samples for manual evaluation.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset selector. Matches alias/source name first, then repo_id. Defaults to the first configured dataset.",
    )
    parser.add_argument("--num-samples", type=int, default=100, help="Number of augmented samples to export.")
    parser.add_argument(
        "--output-dir",
        default="manual-eval/augmented-samples",
        help="Directory where clean/augmented WAVs and manifest will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Shuffle seed. Defaults to experiment.seed from config.",
    )
    parser.add_argument(
        "--allow-unchanged",
        action="store_true",
        help="Include rows even when the configured augmentation leaves them unchanged.",
    )
    return parser.parse_args()


def _find_dataset_config(config: AppConfig, selector: str | None) -> DatasetConfig:
    if selector is None:
        return config.data.datasets[0]
    for dataset_config in config.data.datasets:
        if selector in {dataset_config.source_name, dataset_config.repo_id, dataset_config.alias}:
            return dataset_config
    raise ValueError(f"Dataset not found in config: {selector}")


def _audio_duration_seconds(audio: dict[str, Any]) -> float:
    return len(audio["array"]) / float(audio["sampling_rate"])


def _filter_example(example: dict[str, Any], config: AppConfig) -> bool:
    text = normalize_text(example["text"], config.data.text_normalization)
    if not text:
        return False
    duration_seconds = _audio_duration_seconds(example["audio"])
    if duration_seconds < config.data.min_audio_seconds:
        return False
    if config.data.max_audio_seconds is not None and duration_seconds > config.data.max_audio_seconds:
        return False
    return True


def _write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())


def export_samples(config: AppConfig, *, dataset_selector: str | None, num_samples: int, output_dir: str, seed: int | None, allow_unchanged: bool) -> Path:
    augmenter = build_waveform_augmenter(config.data.audio_augmentation)
    if augmenter is None:
        raise RuntimeError("Audio augmentation is disabled in the config. Enable data.audio_augmentation first.")

    dataset_config = _find_dataset_config(config, dataset_selector)
    split_name = dataset_config.train_split
    dataset = _load_hf_split(dataset_config, split_name, default_cache_dir=config.cache.dataset_dir)
    dataset = _canonicalize_columns(
        dataset,
        dataset_config=dataset_config,
        split_name=split_name,
        sampling_rate=config.data.audio_sampling_rate,
    )
    dataset = dataset.filter(
        lambda example: _filter_example(example, config),
        num_proc=config.data.preprocessing_num_workers,
        desc=f"Filtering export candidates for {dataset_config.source_name}",
    )
    dataset = dataset.shuffle(seed=seed if seed is not None else config.experiment.seed)

    base_output_dir = Path(output_dir)
    target_dir = base_output_dir / dataset_config.source_name
    clean_dir = target_dir / "clean"
    augmented_dir = target_dir / "augmented"
    clean_dir.mkdir(parents=True, exist_ok=True)
    augmented_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "manifest.jsonl"

    exported = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in dataset:
            audio = row["audio"]
            clean_waveform = np.asarray(audio["array"], dtype=np.float32)
            augmented_waveform = augmenter.maybe_augment(
                clean_waveform,
                sample_id=str(row["__sample_id"]),
                dataset_name=str(row["__source_dataset"]),
                dataset_repo_id=str(row["__source_repo_id"]),
                dataset_split=str(row["__source_split"]),
                sample_rate=int(audio["sampling_rate"]),
            )
            changed = not np.allclose(clean_waveform, augmented_waveform)
            if not allow_unchanged and not changed:
                continue

            stem = f"{exported:04d}"
            clean_name = f"{stem}_clean.wav"
            augmented_name = f"{stem}_aug.wav"
            _write_wav(clean_dir / clean_name, clean_waveform, audio["sampling_rate"])
            _write_wav(augmented_dir / augmented_name, augmented_waveform, audio["sampling_rate"])
            manifest.write(
                json.dumps(
                    {
                        "index": exported,
                        "sample_id": row["__sample_id"],
                        "dataset_name": row["__source_dataset"],
                        "dataset_repo_id": row["__source_repo_id"],
                        "dataset_split": row["__source_split"],
                        "text": normalize_text(row["text"], config.data.text_normalization),
                        "duration_seconds": _audio_duration_seconds(audio),
                        "changed": changed,
                        "clean_wav": f"clean/{clean_name}",
                        "augmented_wav": f"augmented/{augmented_name}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            exported += 1
            if exported >= num_samples:
                break

    if exported < num_samples:
        raise RuntimeError(
            f"Only exported {exported} samples from {dataset_config.source_name}. "
            "Relax the filters or rerun with --allow-unchanged."
        )
    return target_dir


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)
    output_path = export_samples(
        config,
        dataset_selector=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        allow_unchanged=args.allow_unchanged,
    )
    print(output_path)


if __name__ == "__main__":
    main()
