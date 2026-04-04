from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .augmentation import WaveformAugmenter
from .config import TextNormalizationConfig
from .data import normalize_text


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    text_normalization: TextNormalizationConfig
    remove_encoder_input_length_restriction: bool = False
    augmenter: WaveformAugmenter | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("features must not be empty")

        audio_items = [feature["audio"] for feature in features]
        sampling_rates = {audio["sampling_rate"] for audio in audio_items}
        if len(sampling_rates) != 1:
            raise ValueError(f"Expected a single sampling rate in batch, found: {sorted(sampling_rates)}")
        sampling_rate = audio_items[0]["sampling_rate"]

        audio_arrays = []
        for index, (feature, audio) in enumerate(zip(features, audio_items, strict=True)):
            waveform = np.asarray(audio["array"], dtype=np.float32)
            if self.augmenter is not None:
                waveform = self.augmenter.maybe_augment(
                    waveform,
                    sample_id=str(feature.get("__sample_id", f"sample-{index}")),
                    dataset_name=str(feature.get("__source_dataset", "")),
                    dataset_repo_id=(
                        None
                        if feature.get("__source_repo_id") is None
                        else str(feature.get("__source_repo_id"))
                    ),
                    dataset_split=str(feature.get("__source_split", "train")),
                    sample_rate=sampling_rate,
                )
            audio_arrays.append(waveform)

        padding = "longest" if self.remove_encoder_input_length_restriction else "max_length"
        batch = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            padding=padding,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        label_features = [
            {
                "input_ids": self.processor.tokenizer(
                    normalize_text(feature["text"], self.text_normalization)
                ).input_ids
            }
            for feature in features
        ]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
