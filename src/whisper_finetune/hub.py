from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .config import AppConfig, save_config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _build_model_card(config: AppConfig, dataset_summaries: list[Any], train_metrics: dict[str, Any], eval_metrics: dict[str, Any] | None) -> str:
    dataset_lines = "\n".join(
        f"- `{summary.name}`: train={summary.train_examples}, eval={summary.eval_examples}"
        for summary in dataset_summaries
    )
    train_lines = "\n".join(f"- `{key}`: {value}" for key, value in sorted(train_metrics.items()))
    eval_lines = (
        "\n".join(f"- `{key}`: {value}" for key, value in sorted(eval_metrics.items()))
        if eval_metrics
        else "- No evaluation dataset configured"
    )
    return f"""# Whisper Finetune Export

Base model: `{config.model.name_or_path}`

## Training datasets
{dataset_lines}

## Train metrics
{train_lines}

## Eval metrics
{eval_lines}

## Notes
- Final model and processor artifacts only.
- Interim training checkpoints are intentionally excluded from this repo export.
"""


def upload_final_artifacts_to_hub(
    *,
    config: AppConfig,
    trainer: Any,
    processor: Any,
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any] | None,
    dataset_summaries: list[Any],
) -> None:
    if not config.hub.enabled:
        return
    if not trainer.is_world_process_zero():
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face Hub upload is enabled but huggingface_hub is not installed."
        ) from exc

    output_dir = Path(config.experiment.output_dir)
    token = os.getenv(config.hub.token_env_var)
    delete_patterns = "*" if config.hub.replace_existing_repo_files else None

    api = HfApi(token=token)
    api.create_repo(
        repo_id=config.hub.repo_id,
        private=config.hub.private,
        exist_ok=True,
        repo_type="model",
    )

    with TemporaryDirectory(prefix=f"{config.hub.export_subdir}-", dir=output_dir) as tmpdir:
        export_dir = Path(tmpdir)

        trainer.save_model(str(export_dir))
        processor.save_pretrained(str(export_dir))
        save_config(config, export_dir / "resolved-config.yaml")

        _write_json(export_dir / "train_results.json", train_metrics)
        if eval_metrics is not None:
            _write_json(export_dir / "eval_results.json", eval_metrics)
        trainer.state.save_to_json(str(export_dir / "trainer_state.json"))
        (export_dir / "README.md").write_text(
            _build_model_card(config, dataset_summaries, train_metrics, eval_metrics),
            encoding="utf-8",
        )

        # Preserve a local copy of the curated export alongside normal trainer outputs.
        local_export_dir = output_dir / config.hub.export_subdir
        if local_export_dir.exists():
            shutil.rmtree(local_export_dir)
        shutil.copytree(export_dir, local_export_dir)

        api.upload_folder(
            repo_id=config.hub.repo_id,
            repo_type="model",
            folder_path=export_dir,
            commit_message=config.hub.commit_message,
            delete_patterns=delete_patterns,
        )
