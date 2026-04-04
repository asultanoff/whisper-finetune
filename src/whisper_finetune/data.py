from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig, DatasetConfig, TextNormalizationConfig

LENGTH_COLUMN_NAMES = {
    "audio": "__audio_length",
    "text": "__text_length",
}


@dataclass(slots=True)
class DatasetSummary:
    name: str
    train_examples: int
    eval_examples: int


@dataclass(slots=True)
class DatasetBundle:
    train_dataset: Any
    eval_dataset: Any | None
    summaries: list[DatasetSummary]


def normalize_text(text: Any, config: TextNormalizationConfig) -> str:
    normalized = "" if text is None else str(text)
    if config.strip:
        normalized = normalized.strip()
    if config.collapse_whitespace:
        normalized = " ".join(normalized.split())
    if config.lowercase:
        normalized = normalized.lower()
    return normalized


def split_train_for_validation(dataset: Any, dataset_config: DatasetConfig) -> tuple[Any, Any | None]:
    if dataset_config.validation_from_train_ratio is None:
        return dataset, None
    split = dataset.train_test_split(
        test_size=dataset_config.validation_from_train_ratio,
        seed=dataset_config.validation_from_train_seed,
        shuffle=dataset_config.shuffle_before_split,
    )
    return split["train"], split["test"]


def _select_limit(dataset: Any | None, limit: int | None) -> Any | None:
    if dataset is None or limit is None:
        return dataset
    if limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def _datasets_module() -> Any:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Run `uv sync` inside whisper-finetune first."
        ) from exc
    return datasets


def _load_hf_split(dataset_config: DatasetConfig, split_name: str, default_cache_dir: str | None) -> Any:
    datasets = _datasets_module()
    kwargs = {
        "path": dataset_config.repo_id,
        "name": dataset_config.config_name,
        "split": split_name,
        "revision": dataset_config.revision,
        "cache_dir": dataset_config.cache_dir or default_cache_dir,
        "trust_remote_code": dataset_config.trust_remote_code,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return datasets.load_dataset(**kwargs)


def _canonicalize_columns(dataset: Any, dataset_config: DatasetConfig, split_name: str, sampling_rate: int) -> Any:
    datasets = _datasets_module()

    missing = [
        column
        for column in (dataset_config.audio_column, dataset_config.text_column)
        if column not in dataset.column_names
    ]
    if missing:
        raise ValueError(
            f"{dataset_config.source_name} split '{split_name}' is missing required columns: {', '.join(missing)}"
        )

    if dataset_config.audio_column != "audio":
        dataset = dataset.rename_column(dataset_config.audio_column, "audio")
    if dataset_config.text_column != "text":
        dataset = dataset.rename_column(dataset_config.text_column, "text")

    size = len(dataset)
    dataset = dataset.add_column("__source_dataset", [dataset_config.source_name] * size)
    dataset = dataset.add_column("__source_repo_id", [dataset_config.repo_id] * size)
    dataset = dataset.add_column("__source_split", [split_name] * size)
    dataset = dataset.add_column(
        "__sample_id",
        [f"{dataset_config.source_name}:{split_name}:{index}" for index in range(size)],
    )
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=sampling_rate))
    return dataset


def _concat_or_single(parts: list[Any]) -> Any | None:
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    datasets = _datasets_module()
    return datasets.concatenate_datasets(parts)


def _shuffle_dataset(dataset: Any | None, seed: int) -> Any | None:
    if dataset is None:
        return None
    if hasattr(dataset, "shuffle"):
        return dataset.shuffle(seed=seed)
    return dataset


def _length_column_name(length_grouping_key: str) -> str:
    if length_grouping_key not in LENGTH_COLUMN_NAMES:
        raise ValueError(f"Unsupported length grouping key: {length_grouping_key}")
    return LENGTH_COLUMN_NAMES[length_grouping_key]


def add_length_grouping_column(
    dataset: Any | None,
    *,
    length_grouping_key: str | None,
    text_normalization: TextNormalizationConfig,
    processor: Any | None = None,
    num_proc: int | None = None,
    split_name: str = "dataset",
) -> Any | None:
    if dataset is None or length_grouping_key is None:
        return dataset

    column_name = _length_column_name(length_grouping_key)
    if column_name in getattr(dataset, "column_names", []):
        return dataset

    if length_grouping_key == "audio":
        return dataset.map(
            lambda examples: {
                column_name: [len(audio["array"]) for audio in examples["audio"]],
            },
            batched=True,
            num_proc=num_proc,
            desc=f"Computing {length_grouping_key} lengths for {split_name}",
        )

    if processor is None:
        raise ValueError("processor is required when length_grouping_key='text'")

    def compute_text_lengths(examples: dict[str, list[Any]]) -> dict[str, list[int]]:
        texts = [normalize_text(text, text_normalization) for text in examples["text"]]
        tokenized = processor.tokenizer(texts)
        return {column_name: [len(ids) for ids in tokenized["input_ids"]]}

    return dataset.map(
        compute_text_lengths,
        batched=True,
        num_proc=num_proc,
        desc=f"Computing {length_grouping_key} lengths for {split_name}",
    )


def load_dataset_bundle(config: AppConfig) -> DatasetBundle:
    train_parts: list[Any] = []
    eval_parts: list[Any] = []
    summaries: list[DatasetSummary] = []

    for dataset_config in config.data.datasets:
        train_split = _load_hf_split(
            dataset_config,
            dataset_config.train_split,
            default_cache_dir=config.cache.dataset_dir,
        )
        train_split, eval_split = split_train_for_validation(train_split, dataset_config)

        if eval_split is None and dataset_config.validation_split:
            eval_split = _load_hf_split(
                dataset_config,
                dataset_config.validation_split,
                default_cache_dir=config.cache.dataset_dir,
            )

        train_split = _canonicalize_columns(
            train_split,
            dataset_config=dataset_config,
            split_name=dataset_config.train_split,
            sampling_rate=config.data.audio_sampling_rate,
        )
        train_split = _select_limit(train_split, dataset_config.max_train_samples)

        eval_examples = 0
        if eval_split is not None:
            eval_name = (
                dataset_config.validation_split
                if dataset_config.validation_split
                else f"{dataset_config.train_split}:ratio"
            )
            eval_split = _canonicalize_columns(
                eval_split,
                dataset_config=dataset_config,
                split_name=eval_name,
                sampling_rate=config.data.audio_sampling_rate,
            )
            eval_split = _select_limit(eval_split, dataset_config.max_validation_samples)
            eval_parts.append(eval_split)
            eval_examples = len(eval_split)

        train_parts.append(train_split)
        summaries.append(
            DatasetSummary(
                name=dataset_config.source_name,
                train_examples=len(train_split),
                eval_examples=eval_examples,
            )
        )

    train_dataset = _shuffle_dataset(_concat_or_single(train_parts), seed=config.experiment.seed)
    eval_dataset = _concat_or_single(eval_parts)
    return DatasetBundle(train_dataset=train_dataset, eval_dataset=eval_dataset, summaries=summaries)
