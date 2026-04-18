import random
from types import SimpleNamespace

from whisper_finetune.config import DatasetConfig, TextNormalizationConfig
from whisper_finetune.data import (
    _canonicalize_columns,
    _shuffle_dataset,
    add_length_grouping_column,
    normalize_text,
    split_train_for_validation,
)


class FakeDataset:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    @property
    def column_names(self):
        if not self.values:
            return []
        return list(self.values[0].keys())

    def map(self, function, batched=False, num_proc=None, desc=None):
        if not batched:
            raise AssertionError("FakeDataset only supports batched=True in this test suite")
        batch = {column: [row[column] for row in self.values] for column in self.column_names}
        updates = function(batch)
        mapped = []
        for index, row in enumerate(self.values):
            new_row = dict(row)
            for key, values in updates.items():
                new_row[key] = values[index]
            mapped.append(new_row)
        return FakeDataset(mapped)

    def train_test_split(self, test_size, seed, shuffle):
        values = list(self.values)
        if shuffle:
            random.Random(seed).shuffle(values)
        test_count = max(1, int(round(len(values) * test_size)))
        split_index = len(values) - test_count
        return {
            "train": FakeDataset(values[:split_index]),
            "test": FakeDataset(values[split_index:]),
        }

    def shuffle(self, seed):
        values = list(self.values)
        random.Random(seed).shuffle(values)
        return FakeDataset(values)

    def rename_column(self, original_column_name, new_column_name):
        if new_column_name in self.column_names:
            raise ValueError(f"New column name {new_column_name} already in dataset")
        return FakeDataset(
            [
                {
                    (new_column_name if key == original_column_name else key): value
                    for key, value in row.items()
                }
                for row in self.values
            ]
        )

    def remove_columns(self, column_names):
        columns = set(column_names)
        return FakeDataset(
            [
                {key: value for key, value in row.items() if key not in columns}
                for row in self.values
            ]
        )

    def add_column(self, name, column):
        return FakeDataset(
            [
                {**row, name: column[index]}
                for index, row in enumerate(self.values)
            ]
        )

    def cast_column(self, column, feature):
        return self


def test_canonicalize_columns_allows_configured_text_to_replace_existing_text_column() -> None:
    dataset = FakeDataset(
        [
            {
                "audio": {"array": [0.0], "sampling_rate": 16000},
                "text": "wrong transcript",
                "source_sentence": "correct transcript",
                "aligned_sentence": "metadata",
            }
        ]
    )
    config = DatasetConfig(
        repo_id="org/uzbooks",
        alias="uzbooks",
        train_split="train",
        audio_column="audio",
        text_column="source_sentence",
    )

    canonicalized = _canonicalize_columns(
        dataset,
        dataset_config=config,
        split_name="train",
        sampling_rate=16000,
        prompt_language="uz",
    )

    assert canonicalized.values[0]["text"] == "correct transcript"
    assert "source_sentence" not in canonicalized.column_names
    assert canonicalized.values[0]["__source_dataset"] == "uzbooks"
    assert canonicalized.values[0]["__prompt_language"] == "uz"


def test_split_train_for_validation_uses_ratio() -> None:
    dataset = FakeDataset([1, 2, 3, 4, 5])
    config = DatasetConfig(
        repo_id="org/dataset",
        train_split="train",
        validation_from_train_ratio=0.2,
        audio_column="audio",
        text_column="text",
    )

    train_dataset, eval_dataset = split_train_for_validation(dataset, config)

    assert len(train_dataset) == 4
    assert len(eval_dataset) == 1


def test_split_train_for_validation_returns_none_without_ratio() -> None:
    dataset = FakeDataset([1, 2, 3])
    config = DatasetConfig(
        repo_id="org/dataset",
        train_split="train",
        validation_split="validation",
        audio_column="audio",
        text_column="text",
    )

    train_dataset, eval_dataset = split_train_for_validation(dataset, config)

    assert train_dataset is dataset
    assert eval_dataset is None


def test_normalize_text_applies_requested_rules() -> None:
    normalized = normalize_text(
        "  HeLLo   world  ",
        TextNormalizationConfig(lowercase=True, strip=True, collapse_whitespace=True),
    )

    assert normalized == "hello world"


def test_shuffle_dataset_uses_seeded_shuffle() -> None:
    dataset = FakeDataset([1, 2, 3, 4, 5])

    shuffled = _shuffle_dataset(dataset, seed=42)

    assert shuffled.values == [4, 2, 3, 5, 1]


def test_add_length_grouping_column_uses_audio_array_length() -> None:
    dataset = FakeDataset(
        [
            {"audio": {"array": [0.0] * 16000, "sampling_rate": 16000}, "text": "a"},
            {"audio": {"array": [0.0] * 32000, "sampling_rate": 16000}, "text": "bb"},
        ]
    )

    annotated = add_length_grouping_column(
        dataset,
        length_grouping_key="audio",
        text_normalization=TextNormalizationConfig(),
        split_name="train",
    )

    assert [row["__audio_length"] for row in annotated.values] == [16000, 32000]


def test_add_length_grouping_column_uses_tokenized_text_length() -> None:
    class FakeTokenizer:
        def __call__(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            return {"input_ids": [[1] * len(text.split()) for text in texts]}

    dataset = FakeDataset(
        [
            {"audio": {"array": [0.0] * 16000, "sampling_rate": 16000}, "text": "  hello   world  "},
            {"audio": {"array": [0.0] * 16000, "sampling_rate": 16000}, "text": "one"},
        ]
    )

    annotated = add_length_grouping_column(
        dataset,
        length_grouping_key="text",
        text_normalization=TextNormalizationConfig(strip=True, collapse_whitespace=True),
        processor=SimpleNamespace(tokenizer=FakeTokenizer()),
        split_name="train",
    )

    assert [row["__text_length"] for row in annotated.values] == [2, 1]
