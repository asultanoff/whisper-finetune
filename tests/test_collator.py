from __future__ import annotations

from types import SimpleNamespace

import torch

from whisper_finetune.collator import DataCollatorSpeechSeq2SeqWithPadding
from whisper_finetune.config import TextNormalizationConfig


class FakeFeatureExtractor:
    def __init__(self) -> None:
        self.calls = []

    def __call__(
        self,
        audio_arrays,
        sampling_rate,
        padding,
        truncation,
        return_attention_mask,
        return_tensors,
    ):
        frame_lengths = [len(audio) // 160 for audio in audio_arrays]
        max_frames = max(frame_lengths) if padding == "longest" else 3000
        attention_mask = torch.zeros(len(audio_arrays), max_frames, dtype=torch.long)
        for idx, frame_length in enumerate(frame_lengths):
            attention_mask[idx, :frame_length] = 1

        self.calls.append(
            {
                "sampling_rate": sampling_rate,
                "padding": padding,
                "truncation": truncation,
                "return_attention_mask": return_attention_mask,
                "return_tensors": return_tensors,
                "frame_lengths": frame_lengths,
                "audio_sums": [float(sum(audio)) for audio in audio_arrays],
            }
        )
        return {
            "input_features": torch.zeros(len(audio_arrays), 80, max_frames),
            "attention_mask": attention_mask,
        }


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=True):
        ids = [len(str(text))]
        if add_special_tokens:
            return SimpleNamespace(input_ids=[1, 2] + ids + [self.eos_token_id])
        return SimpleNamespace(input_ids=ids)

    def pad(self, label_features, return_tensors):
        max_length = max(len(item["input_ids"]) for item in label_features)
        input_ids = []
        attention_mask = []
        for item in label_features:
            ids = list(item["input_ids"])
            padding = [self.pad_token_id] * (max_length - len(ids))
            input_ids.append(ids + padding)
            attention_mask.append([1] * len(ids) + [0] * len(padding))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class FakeProcessor:
    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def get_decoder_prompt_ids(self, language=None, task=None):
        prompt = []
        if language == "uz":
            prompt.append((1, 11))
        elif language == "en":
            prompt.append((1, 12))
        if task == "transcribe":
            prompt.append((len(prompt) + 1, 21))
        prompt.append((len(prompt) + 1, 31))
        return prompt


def test_collator_computes_features_on_the_fly_with_longest_padding() -> None:
    feature_extractor = FakeFeatureExtractor()
    processor = FakeProcessor(feature_extractor=feature_extractor, tokenizer=FakeTokenizer())
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=1,
        task="transcribe",
        text_normalization=TextNormalizationConfig(lowercase=True),
        remove_encoder_input_length_restriction=True,
    )

    batch = collator(
        [
            {"audio": {"array": [0.0] * 16000, "sampling_rate": 16000}, "text": " Hello "},
            {
                "audio": {"array": [0.0] * 32000, "sampling_rate": 16000},
                "text": "WORLD",
                "__prompt_language": "uz",
            },
        ]
    )

    assert batch["input_features"].shape == (2, 80, 200)
    assert batch["attention_mask"].shape == (2, 200)
    assert batch["attention_mask"].sum(dim=-1).tolist() == [100, 200]
    assert batch["labels"].tolist() == [
        [21, 31, 5, 99, -100],
        [11, 21, 31, 5, 99],
    ]
    assert batch["generation_decoder_input_ids"].tolist() == [
        [1, 21, 31, 0],
        [1, 11, 21, 31],
    ]
    assert batch["generation_decoder_attention_mask"].tolist() == [
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]
    assert "generation_language" not in batch
    assert feature_extractor.calls[-1]["padding"] == "longest"
    assert feature_extractor.calls[-1]["truncation"] is False


def test_collator_passes_generation_languages_when_all_examples_have_language() -> None:
    feature_extractor = FakeFeatureExtractor()
    processor = FakeProcessor(feature_extractor=feature_extractor, tokenizer=FakeTokenizer())
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=1,
        task="transcribe",
        text_normalization=TextNormalizationConfig(),
    )

    batch = collator(
        [
            {
                "audio": {"array": [0.0] * 16000, "sampling_rate": 16000},
                "text": "a",
                "__prompt_language": "uz",
            },
            {
                "audio": {"array": [0.0] * 16000, "sampling_rate": 16000},
                "text": "b",
                "__prompt_language": "en",
            },
        ]
    )

    assert batch["generation_language"] == ["uz", "en"]


def test_collator_uses_max_length_padding_when_encoder_patch_is_disabled() -> None:
    feature_extractor = FakeFeatureExtractor()
    processor = FakeProcessor(feature_extractor=feature_extractor, tokenizer=FakeTokenizer())
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=1,
        task="transcribe",
        text_normalization=TextNormalizationConfig(),
        remove_encoder_input_length_restriction=False,
    )

    batch = collator(
        [
            {"audio": {"array": [0.0] * 16000, "sampling_rate": 16000}, "text": "a"},
            {"audio": {"array": [0.0] * 32000, "sampling_rate": 16000}, "text": "bb"},
        ]
    )

    assert batch["input_features"].shape == (2, 80, 3000)
    assert batch["attention_mask"].shape == (2, 3000)
    assert feature_extractor.calls[-1]["padding"] == "max_length"


def test_collator_applies_audio_augmentation_before_feature_extraction() -> None:
    class FakeAugmenter:
        def __init__(self) -> None:
            self.calls = []

        def maybe_augment(
            self,
            waveform,
            *,
            sample_id,
            dataset_name,
            dataset_repo_id,
            dataset_split,
            sample_rate,
        ):
            self.calls.append(
                {
                    "sample_id": sample_id,
                    "dataset_name": dataset_name,
                    "dataset_repo_id": dataset_repo_id,
                    "dataset_split": dataset_split,
                    "sample_rate": sample_rate,
                }
            )
            return waveform + 1.0

    feature_extractor = FakeFeatureExtractor()
    augmenter = FakeAugmenter()
    processor = FakeProcessor(feature_extractor=feature_extractor, tokenizer=FakeTokenizer())
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=1,
        task="transcribe",
        text_normalization=TextNormalizationConfig(),
        augmenter=augmenter,
    )

    collator(
        [
            {
                "audio": {"array": [0.0] * 1600, "sampling_rate": 16000},
                "text": "a",
                "__sample_id": "sample-1",
                "__source_dataset": "demo",
                "__source_repo_id": "org/demo",
                "__source_split": "train",
            }
        ]
    )

    assert augmenter.calls == [
        {
            "sample_id": "sample-1",
            "dataset_name": "demo",
            "dataset_repo_id": "org/demo",
            "dataset_split": "train",
            "sample_rate": 16000,
        }
    ]
    assert feature_extractor.calls[-1]["audio_sums"] == [1600.0]
