from __future__ import annotations

import argparse
import inspect
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from .augmentation import build_waveform_augmenter
from .collator import DataCollatorSpeechSeq2SeqWithPadding
from .config import AppConfig, load_config, save_config_artifacts
from .data import LENGTH_COLUMN_NAMES, add_length_grouping_column, load_dataset_bundle, normalize_text
from .hub import upload_final_artifacts_to_hub
from .metrics import word_error_rate
from .patches import enable_whisper_encoder_input_length_patch


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

    suffix = _build_run_suffix(now=now, token=token)
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

    if config.model.language:
        processor.tokenizer.set_prefix_tokens(language=config.model.language, task=config.model.task)
        model.generation_config.language = config.model.language
    model.generation_config.task = config.model.task
    model.generation_config.forced_decoder_ids = None

    if config.model.freeze_encoder:
        model.freeze_encoder()

    if config.training.gradient_checkpointing:
        model.config.use_cache = False

    return processor, model


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


def _prepare_split(dataset: Any | None, config: AppConfig, split_name: str, processor: Any) -> Any | None:
    if dataset is None:
        return None

    workers = config.data.preprocessing_num_workers
    dataset = dataset.filter(
        lambda example: _filter_example(example, config),
        num_proc=workers,
        desc=f"Filtering {split_name} examples",
    )
    return add_length_grouping_column(
        dataset,
        length_grouping_key=config.training.length_grouping_key,
        text_normalization=config.data.text_normalization,
        processor=processor,
        num_proc=workers,
        split_name=split_name,
    )


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
    augmenter = build_waveform_augmenter(config.data.audio_augmentation)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
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

    signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = processor

    return Seq2SeqTrainer(**trainer_kwargs)


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

    if config.experiment.save_config_snapshot:
        save_config_artifacts(config, output_dir, source_config_path=source_config_path)

    processor, model = _prepare_model(config)
    bundle = load_dataset_bundle(config)
    _log_bundle(bundle)

    train_dataset = _prepare_split(bundle.train_dataset, config, "train", processor)
    eval_dataset = _prepare_split(bundle.eval_dataset, config, "eval", processor)

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
    processor.save_pretrained(output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    eval_metrics = None
    if eval_dataset is not None:
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
