from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from .augmentation import build_waveform_augmenter
from .collator import DataCollatorSpeechSeq2SeqWithPadding
from .config import AppConfig, load_config, save_config_artifacts
from .data import (
    LENGTH_COLUMN_NAMES,
    DatasetPart,
    _concat_or_single,
    _shuffle_dataset,
    add_length_grouping_column,
    load_dataset_bundle,
    normalize_text,
)
from .hub import upload_final_artifacts_to_hub
from .metrics import word_error_rate
from .patches import enable_whisper_encoder_input_length_patch
from .prompted_trainer import WhisperPromptedSeq2SeqTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune Whisper with multiple Hugging Face datasets.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--local_rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--local-rank", dest="local_rank", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _trainer_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The 'transformers' package is required. Run `uv sync` inside whisper-finetune first."
        ) from exc

    return Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor, set_seed


def _build_training_arguments(config: AppConfig, has_eval_dataset: bool) -> Any:
    _, Seq2SeqTrainingArguments, _, _, _ = _trainer_dependencies()
    fields = set(Seq2SeqTrainingArguments.__dataclass_fields__)
    args: dict[str, Any] = {
        "output_dir": config.experiment.output_dir,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "warmup_steps": config.training.warmup_steps,
        "warmup_ratio": config.training.warmup_ratio,
        "num_train_epochs": config.training.num_train_epochs,
        "max_steps": config.training.max_steps,
        "logging_steps": config.training.logging_steps,
        "save_steps": config.training.save_steps,
        "save_total_limit": config.training.save_total_limit,
        "save_on_each_node": False,
        "save_safetensors": True,
        "predict_with_generate": config.training.predict_with_generate,
        "generation_max_length": config.model.generation_max_length,
        "generation_num_beams": config.training.generation_num_beams,
        "fp16": config.training.fp16,
        "bf16": config.training.bf16,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "dataloader_num_workers": config.training.dataloader_num_workers,
        "train_sampling_strategy": config.training.train_sampling_strategy,
        "length_column_name": (
            LENGTH_COLUMN_NAMES[config.training.length_grouping_key]
            if config.training.length_grouping_key is not None
            else None
        ),
        "report_to": config.training.report_to,
        "load_best_model_at_end": config.training.load_best_model_at_end and has_eval_dataset,
        "metric_for_best_model": config.training.metric_for_best_model,
        "greater_is_better": config.training.greater_is_better,
        "optim": config.training.optim,
        "weight_decay": config.training.weight_decay,
        "label_smoothing_factor": config.training.label_smoothing_factor,
        "max_grad_norm": config.training.max_grad_norm,
        "remove_unused_columns": False,
        "run_name": config.experiment.run_name,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "push_to_hub": False,
        "deepspeed": config.training.deepspeed_config,
    }

    eval_strategy_key = "eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"
    args[eval_strategy_key] = "steps" if has_eval_dataset else "no"
    if has_eval_dataset:
        args["eval_steps"] = config.training.eval_steps

    return Seq2SeqTrainingArguments(**{key: value for key, value in args.items() if key in fields and value is not None})


def _build_run_suffix(now: datetime | None = None, token: str | None = None) -> str:
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{token or uuid4().hex[:8]}"


def _with_suffix(path: Path, suffix: str) -> Path:
    return path.parent / f"{path.name}-{suffix}"


def _resolve_tensorboard_dir(base_output_dir: Path, base_tensorboard_dir: Path, resolved_output_dir: Path, suffix: str) -> Path:
    try:
        relative = base_tensorboard_dir.relative_to(base_output_dir)
    except ValueError:
        return _with_suffix(base_tensorboard_dir, suffix)
    return resolved_output_dir / relative


def _resolve_tensorboard_dir_for_output(base_output_dir: Path, base_tensorboard_dir: Path, resolved_output_dir: Path) -> Path:
    try:
        relative = base_tensorboard_dir.relative_to(base_output_dir)
    except ValueError:
        return base_tensorboard_dir
    return resolved_output_dir / relative


def _shared_run_suffix(config: AppConfig) -> str:
    if _distributed_world_size() <= 1:
        return _build_run_suffix()

    base_output_dir = Path(config.experiment.output_dir)
    suffix_file = base_output_dir.parent / f".{base_output_dir.name}.run_suffix"
    suffix_file.parent.mkdir(parents=True, exist_ok=True)
    if _is_rank_zero():
        suffix = _build_run_suffix()
        suffix_file.write_text(suffix, encoding="utf-8")
        return suffix

    start = time.monotonic()
    while not suffix_file.exists():
        if time.monotonic() - start > 600:
            raise TimeoutError(f"Timed out waiting for shared run suffix: {suffix_file}")
        time.sleep(1.0)
    return suffix_file.read_text(encoding="utf-8").strip()


def _resolve_experiment_paths(
    config: AppConfig,
    *,
    now: datetime | None = None,
    token: str | None = None,
) -> None:
    base_output_dir = Path(config.experiment.output_dir)
    if config.experiment.run_name is None:
        config.experiment.run_name = base_output_dir.name or "whisper-finetune"

    if not config.experiment.unique_output_dir:
        return

    suffix = _build_run_suffix(now=now, token=token) if now is not None or token is not None else _shared_run_suffix(config)
    base_tensorboard_dir = Path(config.experiment.tensorboard_dir)
    resolved_output_dir = _with_suffix(base_output_dir, suffix)

    config.experiment.output_dir = str(resolved_output_dir)
    config.experiment.tensorboard_dir = str(
        _resolve_tensorboard_dir(base_output_dir, base_tensorboard_dir, resolved_output_dir, suffix)
    )
    config.experiment.run_name = f"{config.experiment.run_name}-{suffix}"


def _resolve_resume_from_output_dir(config: AppConfig) -> str | None:
    if not config.training.resume_from_output_dir:
        return None

    from transformers.trainer_utils import get_last_checkpoint

    resume_path = Path(config.training.resume_from_output_dir)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume output directory not found: {resume_path}")

    if resume_path.is_dir() and resume_path.name.startswith("checkpoint-"):
        checkpoint_dir = resume_path
        output_dir = resume_path.parent
    else:
        checkpoint = get_last_checkpoint(str(resume_path))
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint-* directories found in: {resume_path}")
        checkpoint_dir = Path(checkpoint)
        output_dir = resume_path

    base_output_dir = Path(config.experiment.output_dir)
    base_tensorboard_dir = Path(config.experiment.tensorboard_dir)
    config.experiment.output_dir = str(output_dir)
    config.experiment.tensorboard_dir = str(
        _resolve_tensorboard_dir_for_output(base_output_dir, base_tensorboard_dir, output_dir)
    )
    config.experiment.unique_output_dir = False

    return str(checkpoint_dir)


def _prepare_model(config: AppConfig) -> tuple[Any, Any]:
    _, _, WhisperForConditionalGeneration, WhisperProcessor, _ = _trainer_dependencies()

    if config.model.remove_encoder_input_length_restriction:
        enable_whisper_encoder_input_length_patch()

    load_source = config.model.load_source
    from_pretrained_kwargs: dict[str, Any] = {}
    if not Path(load_source).exists():
        from_pretrained_kwargs["cache_dir"] = config.cache.model_dir

    processor = WhisperProcessor.from_pretrained(
        load_source,
        **from_pretrained_kwargs,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        load_source,
        **from_pretrained_kwargs,
    )

    model.generation_config.language = None
    model.generation_config.task = config.model.task
    model.generation_config.forced_decoder_ids = None

    if config.model.freeze_encoder:
        model.freeze_encoder()

    if config.training.gradient_checkpointing:
        model.config.use_cache = False

    return processor, model


def _resolve_max_label_tokens(config: AppConfig, model: Any) -> None:
    if config.data.max_label_tokens is not None:
        return
    max_target_positions = getattr(model.config, "max_target_positions", None)
    if max_target_positions is None:
        return
    # Labels contain language/task/notimestamps plus EOS after the decoder-start token is stripped.
    config.data.max_label_tokens = max(1, int(max_target_positions) - 4)
    logging.info("Resolved data.max_label_tokens=%s from model max_target_positions=%s", config.data.max_label_tokens, max_target_positions)


def _validate_deepspeed_config(config: AppConfig) -> None:
    if not config.training.deepspeed_config:
        return

    deepspeed_path = Path(config.training.deepspeed_config)
    if not deepspeed_path.is_file():
        raise FileNotFoundError(f"DeepSpeed config not found: {deepspeed_path}")

    try:
        import deepspeed  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "DeepSpeed is enabled in config but the package is not installed. Run `uv sync` in whisper-finetune."
        ) from exc


def _configure_single_process_deepspeed_env(config: AppConfig) -> None:
    if not config.training.deepspeed_config:
        return

    required_env = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    if all(name in os.environ for name in required_env):
        return

    import torch

    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_gpu_count > 1:
        raise RuntimeError(
            "DeepSpeed is enabled but distributed launcher environment variables are missing. "
            "For multi-GPU runs, launch with `torchrun --nproc_per_node=<N>` or `deepspeed --num_gpus=<N>`."
        )

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    logging.info("Configured single-process DeepSpeed environment variables for this run.")


def _metadata_value(metadata: Any, key: str) -> Any:
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get(key)
    return getattr(metadata, key, None)


def _audio_duration_seconds(audio: Any) -> float:
    metadata = audio.get("metadata") if isinstance(audio, dict) else getattr(audio, "metadata", None)
    for key in ("duration_seconds_from_header", "duration_seconds"):
        value = _metadata_value(metadata, key)
        if value is not None:
            return float(value)

    return len(audio["array"]) / float(audio["sampling_rate"])


def _filter_example(
    example: dict[str, Any],
    config: AppConfig,
    processor: Any | None = None,
    max_label_tokens: int | None = None,
) -> bool:
    text = normalize_text(example["text"], config.data.text_normalization)
    if not text:
        return False
    if max_label_tokens is not None:
        if processor is None:
            raise ValueError("processor is required when max_label_tokens is set")
        label_token_count = len(processor.tokenizer(text, add_special_tokens=False).input_ids)
        if label_token_count > max_label_tokens:
            return False

    try:
        duration_seconds = _audio_duration_seconds(example["audio"])
    except Exception:
        return False
    if duration_seconds < config.data.min_audio_seconds:
        return False
    if config.data.max_audio_seconds is not None and duration_seconds > config.data.max_audio_seconds:
        return False
    return True


def _safe_cache_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "dataset"


def _preprocess_cache_file(
    config: AppConfig,
    part: DatasetPart,
    *,
    split_name: str,
    stage: str,
) -> str:
    cache_root = Path(config.cache.dataset_dir) / "preprocess"
    cache_root.mkdir(parents=True, exist_ok=True)
    dataset = part.dataset
    payload = {
        "stage": stage,
        "dataset_name": part.name,
        "source_split": part.split,
        "prepared_split": split_name,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_rows": len(dataset),
        "dataset_columns": list(getattr(dataset, "column_names", [])),
        "audio_sampling_rate": config.data.audio_sampling_rate,
        "min_audio_seconds": config.data.min_audio_seconds,
        "max_audio_seconds": config.data.max_audio_seconds,
        "max_label_tokens": config.data.max_label_tokens,
        "text_normalization": {
            "lowercase": config.data.text_normalization.lowercase,
            "strip": config.data.text_normalization.strip,
            "collapse_whitespace": config.data.text_normalization.collapse_whitespace,
        },
        "length_grouping_key": config.training.length_grouping_key,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return str(cache_root / f"{_safe_cache_name(part.name)}-{_safe_cache_name(part.split)}-{stage}-{digest}.arrow")


def _combined_preprocess_cache_file(
    config: AppConfig,
    parts: list[DatasetPart],
    *,
    stage: str,
) -> str:
    cache_root = Path(config.cache.dataset_dir) / "preprocess"
    cache_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "seed": config.experiment.seed,
        "parts": [
            {
                "name": part.name,
                "split": part.split,
                "fingerprint": getattr(part.dataset, "_fingerprint", None),
                "rows": len(part.dataset),
            }
            for part in parts
        ],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return str(cache_root / f"combined-{stage}-{digest}.arrow")


def _distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_rank_zero() -> bool:
    return _distributed_rank() == 0


@contextmanager
def _file_lock(lock_path: Path, *, poll_seconds: float = 2.0, timeout_seconds: float = 24 * 60 * 60):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} rank={_distributed_rank()}\n".encode("utf-8"))
        except FileExistsError:
            if time.monotonic() - start > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for preprocessing lock: {lock_path}")
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _preprocessing_lock_path(config: AppConfig, split_name: str) -> Path:
    return Path(config.cache.dataset_dir) / "preprocess" / f"{_safe_cache_name(split_name)}.lock"


def _prepare_split(part: DatasetPart, config: AppConfig, split_name: str, processor: Any) -> Any | None:
    dataset = part.dataset
    if dataset is None:
        return None

    workers = config.data.preprocessing_num_workers
    max_label_tokens = config.data.max_label_tokens
    dataset = dataset.filter(
        lambda example: _filter_example(
            example,
            config,
            processor=processor,
            max_label_tokens=max_label_tokens,
        ),
        num_proc=workers,
        desc=f"Filtering {split_name} examples",
        cache_file_name=_preprocess_cache_file(config, part, split_name=split_name, stage="filter"),
        load_from_cache_file=True,
    )
    return add_length_grouping_column(
        dataset,
        length_grouping_key=config.training.length_grouping_key,
        text_normalization=config.data.text_normalization,
        processor=processor,
        num_proc=workers,
        split_name=split_name,
        cache_file_name=_preprocess_cache_file(
            config,
            DatasetPart(name=part.name, split=part.split, dataset=dataset),
            split_name=split_name,
            stage=f"length-{config.training.length_grouping_key}",
        ),
        load_from_cache_file=True,
    )


def _prepare_split_parts(
    parts: list[DatasetPart],
    config: AppConfig,
    split_name: str,
    processor: Any,
    *,
    shuffle: bool,
) -> Any | None:
    if _distributed_world_size() <= 1:
        return _prepare_split_parts_on_current_rank(parts, config, split_name, processor, shuffle=shuffle)
    with _file_lock(_preprocessing_lock_path(config, split_name)):
        return _prepare_split_parts_on_current_rank(parts, config, split_name, processor, shuffle=shuffle)


def _prepare_split_parts_on_current_rank(
    parts: list[DatasetPart],
    config: AppConfig,
    split_name: str,
    processor: Any,
    *,
    shuffle: bool,
) -> Any | None:
    prepared_parts = []
    total_parts = len(parts)
    for index, part in enumerate(parts, start=1):
        prepared = _prepare_split(part, config, f"{split_name} part {index}/{total_parts}", processor)
        if prepared is not None:
            prepared_parts.append(prepared)

    dataset = _concat_or_single(prepared_parts)
    if shuffle:
        dataset = _shuffle_dataset(
            dataset,
            seed=config.experiment.seed,
            indices_cache_file_name=_combined_preprocess_cache_file(
                config,
                [DatasetPart(name=part.name, split=part.split, dataset=prepared) for part, prepared in zip(parts, prepared_parts, strict=True)],
                stage=f"{split_name}-shuffle",
            ),
        )
    return dataset


def _build_compute_metrics(processor: Any, config: AppConfig):
    def compute_metrics(prediction_output: Any) -> dict[str, float]:
        prediction_ids = prediction_output.predictions
        label_ids = prediction_output.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        predictions = processor.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        references = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        predictions = [normalize_text(text, config.data.text_normalization) for text in predictions]
        references = [normalize_text(text, config.data.text_normalization) for text in references]

        return {"wer": word_error_rate(predictions, references)}

    return compute_metrics


def _build_trainer(
    config: AppConfig,
    model: Any,
    processor: Any,
    train_dataset: Any,
    eval_dataset: Any | None,
    training_args: Any,
) -> Any:
    Seq2SeqTrainer, _, _, _, _ = _trainer_dependencies()
    PromptedTrainer = WhisperPromptedSeq2SeqTrainer.bind(Seq2SeqTrainer)
    augmenter = build_waveform_augmenter(config.data.audio_augmentation)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        task=config.model.task,
        text_normalization=config.data.text_normalization,
        remove_encoder_input_length_restriction=config.model.remove_encoder_input_length_restriction,
        augmenter=augmenter,
    )
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": _build_compute_metrics(processor, config) if eval_dataset is not None else None,
    }

    signature = inspect.signature(PromptedTrainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = processor

    return PromptedTrainer(**trainer_kwargs)


def _log_bundle(bundle: Any) -> None:
    for summary in bundle.summaries:
        logging.info(
            "dataset=%s train_examples=%s eval_examples=%s",
            summary.name,
            summary.train_examples,
            summary.eval_examples,
        )


def run_training(config: AppConfig, *, source_config_path: str | None = None) -> None:
    _, _, _, _, set_seed = _trainer_dependencies()
    set_seed(config.experiment.seed)
    resume_from_checkpoint = _resolve_resume_from_output_dir(config)
    _validate_deepspeed_config(config)
    _configure_single_process_deepspeed_env(config)
    _resolve_experiment_paths(config)

    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(config.experiment.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(config.cache.model_dir).mkdir(parents=True, exist_ok=True)
    Path(config.cache.dataset_dir).mkdir(parents=True, exist_ok=True)
    os.environ["TENSORBOARD_LOGGING_DIR"] = config.experiment.tensorboard_dir
    logging.info("run_name=%s output_dir=%s", config.experiment.run_name, config.experiment.output_dir)

    processor, model = _prepare_model(config)
    _resolve_max_label_tokens(config, model)
    if config.experiment.save_config_snapshot:
        save_config_artifacts(config, output_dir, source_config_path=source_config_path)

    bundle = load_dataset_bundle(config)
    _log_bundle(bundle)

    train_dataset = _prepare_split_parts(bundle.train_parts, config, "train", processor, shuffle=True)
    eval_dataset = _prepare_split_parts(bundle.eval_parts, config, "eval", processor, shuffle=False)

    training_args = _build_training_arguments(config, has_eval_dataset=eval_dataset is not None)
    trainer = _build_trainer(
        config=config,
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    if trainer.is_world_process_zero():
        processor.save_pretrained(output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    eval_metrics = None
    if eval_dataset is not None and config.training.final_eval:
        eval_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    upload_final_artifacts_to_hub(
        config=config,
        trainer=trainer,
        processor=processor,
        train_metrics=train_result.metrics,
        eval_metrics=eval_metrics,
        dataset_summaries=bundle.summaries,
        source_config_path=source_config_path,
    )


def main() -> None:
    load_dotenv()
    _configure_logging()
    args = parse_args()
    config = load_config(args.config)
    run_training(config, source_config_path=args.config)


if __name__ == "__main__":
    main()
