from __future__ import annotations

from types import SimpleNamespace

from whisper_finetune.config import AppConfig
from whisper_finetune.train import _audio_duration_seconds, _filter_example, _resolve_max_label_tokens


def _config() -> AppConfig:
    return AppConfig.from_dict(
        {
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "min_audio_seconds": 0.1,
                "max_audio_seconds": 30.0,
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ],
            },
        }
    )


class MetadataAudio:
    metadata = SimpleNamespace(duration_seconds_from_header=12.5)

    def __getitem__(self, key):
        raise AssertionError(f"audio payload should not be decoded for key={key}")


def test_audio_duration_prefers_header_metadata_without_decoding_audio() -> None:
    assert _audio_duration_seconds(MetadataAudio()) == 12.5


def test_filter_example_uses_header_metadata_without_decoding_audio() -> None:
    assert _filter_example({"audio": MetadataAudio(), "text": "valid text"}, _config()) is True


def test_audio_duration_falls_back_to_array_length_for_dict_audio() -> None:
    audio = {"array": [0.0] * 32000, "sampling_rate": 16000}

    assert _audio_duration_seconds(audio) == 2.0


def test_filter_example_skips_audio_decode_errors() -> None:
    class BrokenAudio:
        metadata = None

        def __getitem__(self, key):
            raise RuntimeError("corrupt audio")

    assert _filter_example({"audio": BrokenAudio(), "text": "valid text"}, _config()) is False


def test_filter_example_drops_overlong_text() -> None:
    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[1] * len(text.split()))

    processor = SimpleNamespace(tokenizer=FakeTokenizer())

    assert (
        _filter_example(
            {
                "audio": MetadataAudio(),
                "text": "one two three four five",
            },
            _config(),
            processor=processor,
            max_label_tokens=4,
        )
        is False
    )


def test_resolve_max_label_tokens_uses_model_decoder_capacity() -> None:
    config = _config()
    model = SimpleNamespace(config=SimpleNamespace(max_target_positions=448))

    _resolve_max_label_tokens(config, model)

    assert config.data.max_label_tokens == 444
