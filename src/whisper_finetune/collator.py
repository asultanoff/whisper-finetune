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
    task: str
    text_normalization: TextNormalizationConfig
    remove_encoder_input_length_restriction: bool = False
    augmenter: WaveformAugmenter | None = None

    def _prompt_language(self, feature: dict[str, Any]) -> str | None:
        language = feature.get("__prompt_language")
        if language is None:
            return None
        normalized = str(language).strip()
        return normalized or None

    def _decoder_prompt_ids(self, language: str | None) -> list[int]:
        prompt_ids = [self.decoder_start_token_id]
        prompt_ids.extend(
            token_id
            for _, token_id in self.processor.get_decoder_prompt_ids(language=language, task=self.task)
        )
        return prompt_ids

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

        prompt_features = []
        prompt_languages = []
        label_features = []
        eos_token_id = self.processor.tokenizer.eos_token_id
        for feature in features:
            prompt_language = self._prompt_language(feature)
            prompt_languages.append(prompt_language)
            prompt_ids = self._decoder_prompt_ids(prompt_language)
            prompt_features.append({"input_ids": prompt_ids})
            text_ids = self.processor.tokenizer(
                normalize_text(feature["text"], self.text_normalization),
                add_special_tokens=False,
            ).input_ids
            label_ids = list(prompt_ids) + list(text_ids)
            if eos_token_id is not None:
                label_ids.append(eos_token_id)
            label_features.append({"input_ids": label_ids})

        generation_prompts = self.processor.tokenizer.pad(prompt_features, return_tensors="pt")
        batch["generation_decoder_input_ids"] = generation_prompts["input_ids"]
        batch["generation_decoder_attention_mask"] = generation_prompts["attention_mask"]
        if all(language is not None for language in prompt_languages):
            batch["generation_language"] = prompt_languages

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
