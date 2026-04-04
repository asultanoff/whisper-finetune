from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .config import TextNormalizationConfig
from .data import normalize_text


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    text_normalization: TextNormalizationConfig
    remove_encoder_input_length_restriction: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("features must not be empty")

        audios = [feature["audio"] for feature in features]
        sampling_rates = {audio["sampling_rate"] for audio in audios}
        if len(sampling_rates) != 1:
            raise ValueError(f"Expected a single sampling rate in batch, found: {sorted(sampling_rates)}")

        padding = "longest" if self.remove_encoder_input_length_restriction else "max_length"
        batch = self.processor.feature_extractor(
            [audio["array"] for audio in audios],
            sampling_rate=audios[0]["sampling_rate"],
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
