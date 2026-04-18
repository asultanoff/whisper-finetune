import os
from datetime import datetime
from pathlib import Path

import pytest

from whisper_finetune.config import AppConfig, ConfigError, load_config, save_config_artifacts
from whisper_finetune.train import (
    _configure_single_process_deepspeed_env,
    _resolve_experiment_paths,
    _resolve_resume_from_output_dir,
)


def test_dataset_config_rejects_both_validation_modes() -> None:
    with pytest.raises(ConfigError):
        AppConfig.from_dict(
            {
                "model": {"name_or_path": "openai/whisper-small"},
                "data": {
                    "datasets": [
                        {
                            "repo_id": "org/dataset",
                            "train_split": "train",
                            "validation_split": "validation",
                            "validation_from_train_ratio": 0.1,
                            "audio_column": "audio",
                            "text_column": "text",
                        }
                    ]
                },
            }
        )


def test_dataset_config_accepts_ratio_validation() -> None:
    config = AppConfig.from_dict(
        {
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "language": "uz",
                        "train_split": "train",
                        "validation_from_train_ratio": 0.1,
                        "audio_column": "audio",
                        "text_column": "transcript",
                    }
                ]
            },
        }
    )

    dataset_config = config.data.datasets[0]
    assert dataset_config.language == "uz"
    assert dataset_config.validation_split is None
    assert dataset_config.validation_from_train_ratio == 0.1


def test_data_config_accepts_max_label_tokens() -> None:
    config = AppConfig.from_dict(
        {
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "max_label_tokens": 444,
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

    assert config.data.max_label_tokens == 444


def test_load_config_resolves_relative_deepspeed_path(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    ds_dir = config_dir / "deepspeed"
    ds_dir.mkdir(parents=True)
    (ds_dir / "zero1.json").write_text("{}", encoding="utf-8")
    config_path = config_dir / "train.yaml"
    config_path.write_text(
        """
model:
  name_or_path: openai/whisper-small
data:
  datasets:
    - repo_id: org/dataset
      train_split: train
      audio_column: audio
      text_column: text
training:
  deepspeed_config: deepspeed/zero1.json
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.training.deepspeed_config == str((ds_dir / "zero1.json").resolve())


def test_load_config_resolves_relative_resume_and_init_paths(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    previous_run = tmp_path / "outputs" / "previous-run"
    previous_run.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    config_path = config_dir / "train.yaml"
    config_path.write_text(
        """
model:
  init_from_output_dir: ../outputs/previous-run
data:
  datasets:
    - repo_id: org/dataset
      train_split: train
      audio_column: audio
      text_column: text
training:
  resume_from_output_dir: ../outputs/previous-run
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    expected_path = str(previous_run.resolve())
    assert config.model.init_from_output_dir == expected_path
    assert config.training.resume_from_output_dir == expected_path


def test_cache_config_uses_expected_default_subdirectories() -> None:
    config = AppConfig.from_dict(
        {
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ]
            },
            "cache": {"root_dir": ".cache"},
        }
    )

    assert config.cache.root_dir == ".cache"
    assert config.cache.model_dir == ".cache/models"
    assert config.cache.dataset_dir == ".cache/datasets"


def test_experiment_config_defaults_tensorboard_dir_from_output_dir() -> None:
    config = AppConfig.from_dict(
        {
            "experiment": {"output_dir": "outputs/run-a"},
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ]
            },
        }
    )

    assert config.experiment.tensorboard_dir == "outputs/run-a/tensorboard"


def test_experiment_config_defaults_unique_output_dir_to_true() -> None:
    config = AppConfig.from_dict(
        {
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ]
            },
        }
    )

    assert config.experiment.unique_output_dir is True


def test_training_config_accepts_random_sampling_strategy() -> None:
    config = AppConfig.from_dict(
        {
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ]
            },
            "training": {"train_sampling_strategy": "random"},
        }
    )

    assert config.training.train_sampling_strategy == "random"


def test_model_config_accepts_init_from_output_dir_without_base_name() -> None:
    config = AppConfig.from_dict(
        {
            "model": {"init_from_output_dir": "/tmp/previous-output"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ]
            },
        }
    )

    assert config.model.init_from_output_dir == "/tmp/previous-output"
    assert config.model.load_source == "/tmp/previous-output"


def test_training_config_accepts_group_by_length_text_key() -> None:
    config = AppConfig.from_dict(
        {
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ]
            },
            "training": {
                "train_sampling_strategy": "group_by_length",
                "length_grouping_key": "text",
            },
        }
    )

    assert config.training.train_sampling_strategy == "group_by_length"
    assert config.training.length_grouping_key == "text"


def test_training_config_uses_zero_warmup_steps_when_only_warmup_ratio_is_set() -> None:
    config = AppConfig.from_dict(
        {
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ]
            },
            "training": {"warmup_ratio": 0.15},
        }
    )

    assert config.training.warmup_ratio == 0.15
    assert config.training.warmup_steps == 0


def test_data_config_accepts_audio_augmentation_profiles_and_policies() -> None:
    config = AppConfig.from_dict(
        {
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "audio_augmentation": {
                    "enabled": True,
                    "seed": 123,
                    "datasets": {
                        "org/dataset": {
                            "enabled": True,
                            "profile": "phone",
                            "splits": ["train"],
                            "apply_p": 0.5,
                        }
                    },
                    "profiles": {
                        "phone": {
                            "codec": {
                                "p": 0.5,
                                "backend": "fast",
                                "modes": {"mulaw_narrowband": 1.0},
                            },
                            "noise": {"p": 1.0, "snr_db": [10.0, 20.0]},
                            "clipping": {"p": 0.1, "threshold": [0.8, 0.95]},
                        }
                    },
                },
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

    assert config.data.audio_augmentation.enabled is True
    assert config.data.audio_augmentation.datasets["org/dataset"].profile == "phone"
    assert config.data.audio_augmentation.profiles["phone"].codec is not None
    assert config.data.audio_augmentation.profiles["phone"].codec.backend == "fast"
    assert config.data.audio_augmentation.profiles["phone"].noise is not None


def test_data_config_rejects_unknown_codec_backend() -> None:
    with pytest.raises(ConfigError):
        AppConfig.from_dict(
            {
                "model": {"name_or_path": "openai/whisper-small"},
                "data": {
                    "audio_augmentation": {
                        "enabled": True,
                        "datasets": {
                            "org/dataset": {
                                "enabled": True,
                                "profile": "phone",
                            }
                        },
                        "profiles": {
                            "phone": {
                                "codec": {
                                    "p": 1.0,
                                    "backend": "invalid",
                                    "modes": {"mulaw_narrowband": 1.0},
                                }
                            }
                        },
                    },
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


def test_data_config_rejects_unknown_audio_augmentation_profile_reference() -> None:
    with pytest.raises(ConfigError):
        AppConfig.from_dict(
            {
                "model": {"name_or_path": "openai/whisper-small"},
                "data": {
                    "audio_augmentation": {
                        "enabled": True,
                        "datasets": {
                            "org/dataset": {
                                "enabled": True,
                                "profile": "missing",
                            }
                        },
                        "profiles": {},
                    },
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


def test_training_config_rejects_unknown_sampling_strategy() -> None:
    with pytest.raises(ConfigError):
        AppConfig.from_dict(
            {
                "model": {"name_or_path": "openai/whisper-small"},
                "data": {
                    "datasets": [
                        {
                            "repo_id": "org/dataset",
                            "train_split": "train",
                            "audio_column": "audio",
                            "text_column": "text",
                        }
                    ]
                },
                "training": {"train_sampling_strategy": "invalid"},
            }
        )


def test_training_config_rejects_group_by_length_without_key() -> None:
    with pytest.raises(ConfigError):
        AppConfig.from_dict(
            {
                "model": {"name_or_path": "openai/whisper-small"},
                "data": {
                    "datasets": [
                        {
                            "repo_id": "org/dataset",
                            "train_split": "train",
                            "audio_column": "audio",
                            "text_column": "text",
                        }
                    ]
                },
                "training": {"train_sampling_strategy": "group_by_length"},
            }
        )


def test_training_config_accepts_resume_from_output_dir() -> None:
    config = AppConfig.from_dict(
        {
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ]
            },
            "training": {"resume_from_output_dir": "/tmp/previous-output"},
        }
    )

    assert config.training.resume_from_output_dir == "/tmp/previous-output"


def test_resolve_experiment_paths_creates_unique_run_output_dir() -> None:
    config = AppConfig.from_dict(
        {
            "experiment": {
                "output_dir": "outputs/run-a",
                "tensorboard_dir": "outputs/run-a/tensorboard",
                "run_name": "run-a",
                "unique_output_dir": True,
            },
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ]
            },
        }
    )

    _resolve_experiment_paths(
        config,
        now=datetime(2026, 4, 4, 4, 30, 15),
        token="abc12345",
    )

    assert config.experiment.output_dir == "outputs/run-a-20260404-043015-abc12345"
    assert config.experiment.tensorboard_dir == "outputs/run-a-20260404-043015-abc12345/tensorboard"
    assert config.experiment.run_name == "run-a-20260404-043015-abc12345"


def test_resolve_experiment_paths_suffixes_external_tensorboard_dir() -> None:
    config = AppConfig.from_dict(
        {
            "experiment": {
                "output_dir": "outputs/run-a",
                "tensorboard_dir": "logs/tensorboard",
                "run_name": "run-a",
                "unique_output_dir": True,
            },
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ]
            },
        }
    )

    _resolve_experiment_paths(
        config,
        now=datetime(2026, 4, 4, 4, 30, 15),
        token="abc12345",
    )

    assert config.experiment.output_dir == "outputs/run-a-20260404-043015-abc12345"
    assert config.experiment.tensorboard_dir == "logs/tensorboard-20260404-043015-abc12345"


def test_resolve_resume_from_output_dir_uses_latest_checkpoint_and_disables_unique_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "previous-run"
    (output_dir / "checkpoint-50").mkdir(parents=True)
    (output_dir / "checkpoint-100").mkdir(parents=True)

    config = AppConfig.from_dict(
        {
            "experiment": {
                "output_dir": "outputs/new-run",
                "tensorboard_dir": "outputs/new-run/tensorboard",
                "unique_output_dir": True,
            },
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ]
            },
            "training": {"resume_from_output_dir": str(output_dir)},
        }
    )

    checkpoint_path = _resolve_resume_from_output_dir(config)

    assert checkpoint_path == str(output_dir / "checkpoint-100")
    assert config.experiment.output_dir == str(output_dir)
    assert config.experiment.tensorboard_dir == str(output_dir / "tensorboard")
    assert config.experiment.unique_output_dir is False


def test_single_process_deepspeed_env_bootstrap(monkeypatch) -> None:
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)

    config = AppConfig.from_dict(
        {
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ]
            },
            "training": {"deepspeed_config": "configs/deepspeed/zero1.json"},
        }
    )

    _configure_single_process_deepspeed_env(config)

    assert os.environ["RANK"] == "0"
    assert os.environ["WORLD_SIZE"] == "1"
    assert os.environ["LOCAL_RANK"] == "0"
    assert os.environ["MASTER_ADDR"] == "127.0.0.1"
    assert os.environ["MASTER_PORT"] == "29500"


def test_save_config_artifacts_writes_resolved_and_original_training_config(tmp_path: Path) -> None:
    config = AppConfig.from_dict(
        {
            "model": {"name_or_path": "openai/whisper-small"},
            "data": {
                "datasets": [
                    {
                        "repo_id": "org/dataset",
                        "train_split": "train",
                        "audio_column": "audio",
                        "text_column": "text",
                    }
                ]
            },
        }
    )
    source_config = tmp_path / "train.yaml"
    source_config.write_text("model:\n  name_or_path: openai/whisper-small\n", encoding="utf-8")
    target_dir = tmp_path / "artifacts"

    save_config_artifacts(config, target_dir, source_config_path=source_config)

    assert (target_dir / "resolved-config.yaml").is_file()
    assert (target_dir / "training-config.yaml").read_text(encoding="utf-8") == source_config.read_text(encoding="utf-8")
