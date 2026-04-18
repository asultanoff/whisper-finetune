from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _unknown_keys(raw: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"Unknown keys in {context}: {', '.join(unknown)}")


def _positive_optional(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise ConfigError(f"{field_name} must be > 0 when provided")


def _validate_probability(value: float, field_name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{field_name} must be in the interval [0, 1]")


def _parse_float_range(value: Any, field_name: str, *, minimum: float) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        parsed = float(value)
        if parsed < minimum:
            raise ConfigError(f"{field_name} must be >= {minimum}")
        return parsed, parsed
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{field_name} must be a number or a [min, max] pair")
    start = float(value[0])
    end = float(value[1])
    if start < minimum or end < start:
        raise ConfigError(f"{field_name} must have {minimum} <= min <= max")
    return start, end


def _parse_int_range(value: Any, field_name: str, *, minimum: int) -> tuple[int, int]:
    if isinstance(value, int):
        if value < minimum:
            raise ConfigError(f"{field_name} must be >= {minimum}")
        return value, value
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{field_name} must be an int or a [min, max] pair")
    start = int(value[0])
    end = int(value[1])
    if start < minimum or end < start:
        raise ConfigError(f"{field_name} must have {minimum} <= min <= max")
    return start, end


@dataclass(slots=True)
class TextNormalizationConfig:
    lowercase: bool = False
    strip: bool = True
    collapse_whitespace: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TextNormalizationConfig":
        if raw is None:
            return cls()
        _unknown_keys(raw, {"lowercase", "strip", "collapse_whitespace"}, "data.text_normalization")
        return cls(
            lowercase=bool(raw.get("lowercase", False)),
            strip=bool(raw.get("strip", True)),
            collapse_whitespace=bool(raw.get("collapse_whitespace", True)),
        )


@dataclass(slots=True)
class CodecAugmentationConfig:
    p: float = 0.0
    backend: str = "fast"
    modes: dict[str, float] = field(default_factory=dict)
    sample_rate: int = 8000
    highpass_hz: float = 350.0
    lowpass_hz: float = 3000.0
    roundtrips: int = 2

    def __post_init__(self) -> None:
        _validate_probability(self.p, "data.audio_augmentation.profiles[].codec.p")
        if self.backend not in {"fast", "ffmpeg"}:
            raise ConfigError("data.audio_augmentation.profiles[].codec.backend must be one of: fast, ffmpeg")
        if self.sample_rate < 4000:
            raise ConfigError("data.audio_augmentation.profiles[].codec.sample_rate must be >= 4000")
        if self.highpass_hz < 0.0:
            raise ConfigError("data.audio_augmentation.profiles[].codec.highpass_hz must be >= 0")
        if self.lowpass_hz <= self.highpass_hz:
            raise ConfigError(
                "data.audio_augmentation.profiles[].codec.lowpass_hz must be greater than highpass_hz"
            )
        if self.roundtrips <= 0:
            raise ConfigError("data.audio_augmentation.profiles[].codec.roundtrips must be > 0")
        allowed_modes = {"mulaw_narrowband", "gsm_narrowband"}
        unknown_modes = sorted(set(self.modes) - allowed_modes)
        if unknown_modes:
            raise ConfigError(
                "data.audio_augmentation.profiles[].codec.modes contains unsupported values: "
                + ", ".join(unknown_modes)
            )
        for name, weight in self.modes.items():
            if weight <= 0:
                raise ConfigError(
                    f"data.audio_augmentation.profiles[].codec.modes.{name} must be > 0"
                )
        if self.p > 0.0 and not self.modes:
            raise ConfigError("data.audio_augmentation.profiles[].codec.modes is required when codec.p > 0")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "CodecAugmentationConfig | None":
        if raw is None:
            return None
        allowed = {"p", "backend", "modes", "sample_rate", "highpass_hz", "lowpass_hz", "roundtrips"}
        _unknown_keys(raw, allowed, "data.audio_augmentation.profiles[].codec")
        modes_raw = raw.get("modes", {})
        if not isinstance(modes_raw, dict):
            raise ConfigError("data.audio_augmentation.profiles[].codec.modes must be a mapping")
        return cls(
            p=float(raw.get("p", 0.0)),
            backend=str(raw.get("backend", "fast")),
            modes={str(name): float(weight) for name, weight in modes_raw.items()},
            sample_rate=int(raw.get("sample_rate", 8000)),
            highpass_hz=float(raw.get("highpass_hz", 350.0)),
            lowpass_hz=float(raw.get("lowpass_hz", 3000.0)),
            roundtrips=int(raw.get("roundtrips", 2)),
        )


@dataclass(slots=True)
class PacketLossAugmentationConfig:
    p: float = 0.0
    burst_ms: tuple[int, int] = (20, 80)
    bursts: tuple[int, int] = (1, 2)
    fill: str = "zero"

    def __post_init__(self) -> None:
        _validate_probability(self.p, "data.audio_augmentation.profiles[].packet_loss.p")
        if self.fill not in {"zero", "hold"}:
            raise ConfigError("data.audio_augmentation.profiles[].packet_loss.fill must be one of: zero, hold")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PacketLossAugmentationConfig | None":
        if raw is None:
            return None
        allowed = {"p", "burst_ms", "bursts", "fill"}
        _unknown_keys(raw, allowed, "data.audio_augmentation.profiles[].packet_loss")
        return cls(
            p=float(raw.get("p", 0.0)),
            burst_ms=_parse_int_range(raw.get("burst_ms", [20, 80]), "data.audio_augmentation.profiles[].packet_loss.burst_ms", minimum=1),
            bursts=_parse_int_range(raw.get("bursts", [1, 2]), "data.audio_augmentation.profiles[].packet_loss.bursts", minimum=1),
            fill=str(raw.get("fill", "zero")),
        )


@dataclass(slots=True)
class ClippingAugmentationConfig:
    p: float = 0.0
    threshold: tuple[float, float] = (0.75, 0.95)

    def __post_init__(self) -> None:
        _validate_probability(self.p, "data.audio_augmentation.profiles[].clipping.p")
        if self.threshold[0] <= 0.0 or self.threshold[1] > 1.0:
            raise ConfigError(
                "data.audio_augmentation.profiles[].clipping.threshold must be within (0, 1]"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ClippingAugmentationConfig | None":
        if raw is None:
            return None
        allowed = {"p", "threshold"}
        _unknown_keys(raw, allowed, "data.audio_augmentation.profiles[].clipping")
        return cls(
            p=float(raw.get("p", 0.0)),
            threshold=_parse_float_range(
                raw.get("threshold", [0.75, 0.95]),
                "data.audio_augmentation.profiles[].clipping.threshold",
                minimum=1e-4,
            ),
        )


@dataclass(slots=True)
class NoiseAugmentationConfig:
    p: float = 0.0
    snr_db: tuple[float, float] = (18.0, 30.0)

    def __post_init__(self) -> None:
        _validate_probability(self.p, "data.audio_augmentation.profiles[].noise.p")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "NoiseAugmentationConfig | None":
        if raw is None:
            return None
        allowed = {"p", "snr_db"}
        _unknown_keys(raw, allowed, "data.audio_augmentation.profiles[].noise")
        return cls(
            p=float(raw.get("p", 0.0)),
            snr_db=_parse_float_range(
                raw.get("snr_db", [18.0, 30.0]),
                "data.audio_augmentation.profiles[].noise.snr_db",
                minimum=0.0,
            ),
        )


@dataclass(slots=True)
class ReverbAugmentationConfig:
    p: float = 0.0
    rt60_ms: tuple[float, float] = (80.0, 220.0)

    def __post_init__(self) -> None:
        _validate_probability(self.p, "data.audio_augmentation.profiles[].reverb.p")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ReverbAugmentationConfig | None":
        if raw is None:
            return None
        allowed = {"p", "rt60_ms"}
        _unknown_keys(raw, allowed, "data.audio_augmentation.profiles[].reverb")
        return cls(
            p=float(raw.get("p", 0.0)),
            rt60_ms=_parse_float_range(
                raw.get("rt60_ms", [80.0, 220.0]),
                "data.audio_augmentation.profiles[].reverb.rt60_ms",
                minimum=1.0,
            ),
        )


@dataclass(slots=True)
class AudioAugmentationProfileConfig:
    codec: CodecAugmentationConfig | None = None
    packet_loss: PacketLossAugmentationConfig | None = None
    clipping: ClippingAugmentationConfig | None = None
    noise: NoiseAugmentationConfig | None = None
    reverb: ReverbAugmentationConfig | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AudioAugmentationProfileConfig":
        allowed = {"codec", "packet_loss", "clipping", "noise", "reverb"}
        _unknown_keys(raw, allowed, "data.audio_augmentation.profiles[]")
        return cls(
            codec=CodecAugmentationConfig.from_dict(raw.get("codec")),
            packet_loss=PacketLossAugmentationConfig.from_dict(raw.get("packet_loss")),
            clipping=ClippingAugmentationConfig.from_dict(raw.get("clipping")),
            noise=NoiseAugmentationConfig.from_dict(raw.get("noise")),
            reverb=ReverbAugmentationConfig.from_dict(raw.get("reverb")),
        )


@dataclass(slots=True)
class DatasetAudioAugmentationPolicyConfig:
    enabled: bool = False
    profile: str | None = None
    splits: tuple[str, ...] = ("train",)
    apply_p: float = 1.0

    def __post_init__(self) -> None:
        _validate_probability(self.apply_p, "data.audio_augmentation.datasets[].apply_p")
        if self.enabled and not self.profile:
            raise ConfigError("data.audio_augmentation.datasets[].profile is required when enabled is true")
        if not self.splits:
            raise ConfigError("data.audio_augmentation.datasets[].splits must contain at least one split")

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, train_only: bool) -> "DatasetAudioAugmentationPolicyConfig":
        allowed = {"enabled", "profile", "splits", "apply_p"}
        _unknown_keys(raw, allowed, "data.audio_augmentation.datasets[]")
        splits_raw = raw.get("splits")
        if splits_raw is None:
            splits = ("train",) if train_only else ("train", "validation", "test")
        else:
            if not isinstance(splits_raw, (list, tuple)) or not splits_raw:
                raise ConfigError("data.audio_augmentation.datasets[].splits must be a non-empty list")
            splits = tuple(str(split) for split in splits_raw if str(split))
        return cls(
            enabled=bool(raw.get("enabled", False)),
            profile=(None if raw.get("profile") is None else str(raw.get("profile"))),
            splits=splits,
            apply_p=float(raw.get("apply_p", 1.0)),
        )


@dataclass(slots=True)
class AudioAugmentationConfig:
    enabled: bool = False
    seed: int = 0
    train_only: bool = True
    mode: str = "deterministic_per_sample"
    ffmpeg_bin: str = "ffmpeg"
    datasets: dict[str, DatasetAudioAugmentationPolicyConfig] = field(default_factory=dict)
    profiles: dict[str, AudioAugmentationProfileConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode != "deterministic_per_sample":
            raise ConfigError(
                "data.audio_augmentation.mode must be 'deterministic_per_sample'"
            )
        if self.enabled:
            if not self.datasets:
                raise ConfigError("data.audio_augmentation.datasets is required when augmentation is enabled")
            if not self.profiles:
                raise ConfigError("data.audio_augmentation.profiles is required when augmentation is enabled")
        for dataset_name, policy in self.datasets.items():
            if policy.enabled and policy.profile not in self.profiles:
                raise ConfigError(
                    f"data.audio_augmentation.datasets.{dataset_name}.profile references unknown profile {policy.profile!r}"
                )

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "AudioAugmentationConfig":
        if raw is None:
            return cls()
        allowed = {"enabled", "seed", "train_only", "mode", "ffmpeg_bin", "datasets", "profiles"}
        _unknown_keys(raw, allowed, "data.audio_augmentation")
        train_only = bool(raw.get("train_only", True))
        datasets_raw = raw.get("datasets", {})
        profiles_raw = raw.get("profiles", {})
        if not isinstance(datasets_raw, dict):
            raise ConfigError("data.audio_augmentation.datasets must be a mapping")
        if not isinstance(profiles_raw, dict):
            raise ConfigError("data.audio_augmentation.profiles must be a mapping")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            seed=int(raw.get("seed", 0)),
            train_only=train_only,
            mode=str(raw.get("mode", "deterministic_per_sample")),
            ffmpeg_bin=str(raw.get("ffmpeg_bin", "ffmpeg")),
            datasets={
                str(name): DatasetAudioAugmentationPolicyConfig.from_dict(value, train_only=train_only)
                for name, value in datasets_raw.items()
            },
            profiles={
                str(name): AudioAugmentationProfileConfig.from_dict(value)
                for name, value in profiles_raw.items()
            },
        )


@dataclass(slots=True)
class DatasetConfig:
    repo_id: str
    config_name: str | None = None
    alias: str | None = None
    language: str | None = None
    revision: str | None = None
    cache_dir: str | None = None
    trust_remote_code: bool = False
    train_split: str = "train"
    validation_split: str | None = None
    validation_from_train_ratio: float | None = None
    validation_from_train_seed: int = 42
    shuffle_before_split: bool = True
    audio_column: str = "audio"
    text_column: str = "text"
    max_train_samples: int | None = None
    max_validation_samples: int | None = None

    def __post_init__(self) -> None:
        if not self.repo_id:
            raise ConfigError("data.datasets[].repo_id is required")
        if not self.train_split:
            raise ConfigError(f"{self.repo_id}: train_split is required")
        if not self.audio_column:
            raise ConfigError(f"{self.repo_id}: audio_column is required")
        if not self.text_column:
            raise ConfigError(f"{self.repo_id}: text_column is required")
        if self.audio_column == self.text_column:
            raise ConfigError(f"{self.repo_id}: audio_column and text_column must differ")
        if self.validation_split and self.validation_from_train_ratio is not None:
            raise ConfigError(
                f"{self.repo_id}: validation_split and validation_from_train_ratio are mutually exclusive"
            )
        if self.validation_from_train_ratio is not None:
            if not 0.0 < self.validation_from_train_ratio < 1.0:
                raise ConfigError(
                    f"{self.repo_id}: validation_from_train_ratio must be in the open interval (0, 1)"
                )
        _positive_optional(self.max_train_samples, f"{self.repo_id}.max_train_samples")
        _positive_optional(self.max_validation_samples, f"{self.repo_id}.max_validation_samples")

    @property
    def source_name(self) -> str:
        if self.alias:
            return self.alias
        if self.config_name:
            return f"{self.repo_id}:{self.config_name}"
        return self.repo_id

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetConfig":
        allowed = {
            "repo_id",
            "config_name",
            "alias",
            "language",
            "revision",
            "cache_dir",
            "trust_remote_code",
            "train_split",
            "validation_split",
            "validation_from_train_ratio",
            "validation_from_train_seed",
            "shuffle_before_split",
            "audio_column",
            "text_column",
            "max_train_samples",
            "max_validation_samples",
        }
        _unknown_keys(raw, allowed, "data.datasets[]")
        return cls(
            repo_id=str(raw.get("repo_id", "")),
            config_name=raw.get("config_name"),
            alias=raw.get("alias"),
            language=(None if raw.get("language") is None else str(raw.get("language"))),
            revision=raw.get("revision"),
            cache_dir=raw.get("cache_dir"),
            trust_remote_code=bool(raw.get("trust_remote_code", False)),
            train_split=str(raw.get("train_split", "train")),
            validation_split=raw.get("validation_split"),
            validation_from_train_ratio=raw.get("validation_from_train_ratio"),
            validation_from_train_seed=int(raw.get("validation_from_train_seed", 42)),
            shuffle_before_split=bool(raw.get("shuffle_before_split", True)),
            audio_column=str(raw.get("audio_column", "audio")),
            text_column=str(raw.get("text_column", "text")),
            max_train_samples=raw.get("max_train_samples"),
            max_validation_samples=raw.get("max_validation_samples"),
        )


@dataclass(slots=True)
class DataConfig:
    datasets: list[DatasetConfig]
    audio_sampling_rate: int = 16000
    preprocessing_num_workers: int | None = None
    min_audio_seconds: float = 0.0
    max_audio_seconds: float | None = 30.0
    text_normalization: TextNormalizationConfig = field(default_factory=TextNormalizationConfig)
    audio_augmentation: AudioAugmentationConfig = field(default_factory=AudioAugmentationConfig)

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ConfigError("data.datasets must contain at least one dataset")
        if self.audio_sampling_rate <= 0:
            raise ConfigError("data.audio_sampling_rate must be > 0")
        if self.min_audio_seconds < 0.0:
            raise ConfigError("data.min_audio_seconds must be >= 0")
        if self.max_audio_seconds is not None and self.max_audio_seconds <= self.min_audio_seconds:
            raise ConfigError("data.max_audio_seconds must be greater than data.min_audio_seconds")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataConfig":
        allowed = {
            "datasets",
            "audio_sampling_rate",
            "preprocessing_num_workers",
            "min_audio_seconds",
            "max_audio_seconds",
            "text_normalization",
            "audio_augmentation",
        }
        _unknown_keys(raw, allowed, "data")
        datasets_raw = raw.get("datasets")
        if not isinstance(datasets_raw, list):
            raise ConfigError("data.datasets must be a list")
        return cls(
            datasets=[DatasetConfig.from_dict(item) for item in datasets_raw],
            audio_sampling_rate=int(raw.get("audio_sampling_rate", 16000)),
            preprocessing_num_workers=raw.get("preprocessing_num_workers"),
            min_audio_seconds=float(raw.get("min_audio_seconds", 0.0)),
            max_audio_seconds=(
                None if raw.get("max_audio_seconds") is None else float(raw.get("max_audio_seconds"))
            ),
            text_normalization=TextNormalizationConfig.from_dict(raw.get("text_normalization")),
            audio_augmentation=AudioAugmentationConfig.from_dict(raw.get("audio_augmentation")),
        )


@dataclass(slots=True)
class ModelConfig:
    name_or_path: str | None = None
    init_from_output_dir: str | None = None
    language: str | None = None
    task: str = "transcribe"
    freeze_encoder: bool = False
    generation_max_length: int = 225
    remove_encoder_input_length_restriction: bool = False

    def __post_init__(self) -> None:
        if not self.name_or_path and not self.init_from_output_dir:
            raise ConfigError("One of model.name_or_path or model.init_from_output_dir is required")
        if not self.task:
            raise ConfigError("model.task is required")
        if self.generation_max_length <= 0:
            raise ConfigError("model.generation_max_length must be > 0")

    @property
    def load_source(self) -> str:
        if self.init_from_output_dir:
            return self.init_from_output_dir
        if self.name_or_path:
            return self.name_or_path
        raise ConfigError("No model load source configured")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        allowed = {
            "name_or_path",
            "init_from_output_dir",
            "language",
            "task",
            "freeze_encoder",
            "generation_max_length",
            "remove_encoder_input_length_restriction",
        }
        _unknown_keys(raw, allowed, "model")
        return cls(
            name_or_path=(None if raw.get("name_or_path") is None else str(raw.get("name_or_path"))),
            init_from_output_dir=raw.get("init_from_output_dir"),
            language=raw.get("language"),
            task=str(raw.get("task", "transcribe")),
            freeze_encoder=bool(raw.get("freeze_encoder", False)),
            generation_max_length=int(raw.get("generation_max_length", 225)),
            remove_encoder_input_length_restriction=bool(
                raw.get("remove_encoder_input_length_restriction", False)
            ),
        )


@dataclass(slots=True)
class CacheConfig:
    root_dir: str = ".cache"
    model_dir: str | None = None
    dataset_dir: str | None = None

    def __post_init__(self) -> None:
        root = Path(self.root_dir)
        if self.model_dir is None:
            self.model_dir = str(root / "models")
        if self.dataset_dir is None:
            self.dataset_dir = str(root / "datasets")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "CacheConfig":
        if raw is None:
            return cls()
        allowed = {"root_dir", "model_dir", "dataset_dir"}
        _unknown_keys(raw, allowed, "cache")
        return cls(
            root_dir=str(raw.get("root_dir", ".cache")),
            model_dir=raw.get("model_dir"),
            dataset_dir=raw.get("dataset_dir"),
        )


@dataclass(slots=True)
class HubConfig:
    enabled: bool = False
    repo_id: str | None = None
    private: bool = True
    token_env_var: str = "HF_TOKEN"
    commit_message: str = "Upload final Whisper finetune artifacts"
    replace_existing_repo_files: bool = True
    export_subdir: str = "hf-export"

    def __post_init__(self) -> None:
        if self.enabled and not self.repo_id:
            raise ConfigError("hub.repo_id is required when hub.enabled is true")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "HubConfig":
        if raw is None:
            return cls()
        allowed = {
            "enabled",
            "repo_id",
            "private",
            "token_env_var",
            "commit_message",
            "replace_existing_repo_files",
            "export_subdir",
        }
        _unknown_keys(raw, allowed, "hub")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            repo_id=raw.get("repo_id"),
            private=bool(raw.get("private", True)),
            token_env_var=str(raw.get("token_env_var", "HF_TOKEN")),
            commit_message=str(raw.get("commit_message", "Upload final Whisper finetune artifacts")),
            replace_existing_repo_files=bool(raw.get("replace_existing_repo_files", True)),
            export_subdir=str(raw.get("export_subdir", "hf-export")),
        )


@dataclass(slots=True)
class TrainingConfig:
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    warmup_ratio: float = 0.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    lr_scheduler_type: str = "linear"
    logging_steps: int = 25
    eval_steps: int = 250
    save_steps: int = 250
    save_total_limit: int = 3
    generation_num_beams: int = 1
    predict_with_generate: bool = True
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    train_sampling_strategy: str = "random"
    length_grouping_key: str | None = None
    resume_from_output_dir: str | None = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    optim: str = "adamw_torch"
    weight_decay: float = 0.0
    label_smoothing_factor: float = 0.0
    max_grad_norm: float = 1.0
    deepspeed_config: str | None = None

    def __post_init__(self) -> None:
        numeric_fields = {
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "generation_num_beams": self.generation_num_beams,
            "dataloader_num_workers": self.dataloader_num_workers,
        }
        for name, value in numeric_fields.items():
            if value <= 0:
                raise ConfigError(f"training.{name} must be > 0")
        if self.warmup_steps < 0:
            raise ConfigError("training.warmup_steps must be >= 0")
        if self.train_sampling_strategy not in {"random", "sequential", "group_by_length"}:
            raise ConfigError(
                "training.train_sampling_strategy must be one of: random, sequential, group_by_length"
            )
        if self.length_grouping_key not in {None, "audio", "text"}:
            raise ConfigError("training.length_grouping_key must be one of: audio, text")
        if self.train_sampling_strategy == "group_by_length" and self.length_grouping_key is None:
            raise ConfigError("training.length_grouping_key is required when train_sampling_strategy=group_by_length")
        if self.resume_from_output_dir is not None and not self.resume_from_output_dir:
            raise ConfigError("training.resume_from_output_dir must not be empty")
        if self.learning_rate <= 0:
            raise ConfigError("training.learning_rate must be > 0")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ConfigError("training.warmup_ratio must be in the interval [0, 1)")
        if self.num_train_epochs <= 0:
            raise ConfigError("training.num_train_epochs must be > 0")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ConfigError("training.max_steps must be -1 or a positive integer")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TrainingConfig":
        if raw is None:
            return cls()
        allowed = {
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "warmup_steps",
            "warmup_ratio",
            "num_train_epochs",
            "max_steps",
            "lr_scheduler_type",
            "logging_steps",
            "eval_steps",
            "save_steps",
            "save_total_limit",
            "generation_num_beams",
            "predict_with_generate",
            "report_to",
            "fp16",
            "bf16",
            "gradient_checkpointing",
            "dataloader_num_workers",
            "train_sampling_strategy",
            "length_grouping_key",
            "resume_from_output_dir",
            "load_best_model_at_end",
            "metric_for_best_model",
            "greater_is_better",
            "optim",
            "weight_decay",
            "label_smoothing_factor",
            "max_grad_norm",
            "deepspeed_config",
        }
        _unknown_keys(raw, allowed, "training")
        warmup_ratio = float(raw.get("warmup_ratio", 0.0))
        warmup_steps_raw = raw.get("warmup_steps")
        warmup_steps = (
            int(warmup_steps_raw)
            if warmup_steps_raw is not None
            else (0 if warmup_ratio > 0.0 else 500)
        )
        return cls(
            per_device_train_batch_size=int(raw.get("per_device_train_batch_size", 8)),
            per_device_eval_batch_size=int(raw.get("per_device_eval_batch_size", 8)),
            gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
            learning_rate=float(raw.get("learning_rate", 1e-5)),
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=float(raw.get("num_train_epochs", 3.0)),
            max_steps=int(raw.get("max_steps", -1)),
            lr_scheduler_type=str(raw.get("lr_scheduler_type", "linear")),
            logging_steps=int(raw.get("logging_steps", 25)),
            eval_steps=int(raw.get("eval_steps", 250)),
            save_steps=int(raw.get("save_steps", 250)),
            save_total_limit=int(raw.get("save_total_limit", 3)),
            generation_num_beams=int(raw.get("generation_num_beams", 1)),
            predict_with_generate=bool(raw.get("predict_with_generate", True)),
            report_to=list(raw.get("report_to", ["tensorboard"])),
            fp16=bool(raw.get("fp16", False)),
            bf16=bool(raw.get("bf16", False)),
            gradient_checkpointing=bool(raw.get("gradient_checkpointing", False)),
            dataloader_num_workers=int(raw.get("dataloader_num_workers", 4)),
            train_sampling_strategy=str(raw.get("train_sampling_strategy", "random")),
            length_grouping_key=raw.get("length_grouping_key"),
            resume_from_output_dir=raw.get("resume_from_output_dir"),
            load_best_model_at_end=bool(raw.get("load_best_model_at_end", True)),
            metric_for_best_model=str(raw.get("metric_for_best_model", "wer")),
            greater_is_better=bool(raw.get("greater_is_better", False)),
            optim=str(raw.get("optim", "adamw_torch")),
            weight_decay=float(raw.get("weight_decay", 0.0)),
            label_smoothing_factor=float(raw.get("label_smoothing_factor", 0.0)),
            max_grad_norm=float(raw.get("max_grad_norm", 1.0)),
            deepspeed_config=raw.get("deepspeed_config"),
        )


@dataclass(slots=True)
class ExperimentConfig:
    output_dir: str = "outputs/whisper-finetune"
    tensorboard_dir: str | None = None
    seed: int = 42
    run_name: str | None = None
    unique_output_dir: bool = True
    save_config_snapshot: bool = True

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ConfigError("experiment.output_dir is required")
        if self.tensorboard_dir is None:
            self.tensorboard_dir = str(Path(self.output_dir) / "tensorboard")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ExperimentConfig":
        if raw is None:
            return cls()
        allowed = {"output_dir", "tensorboard_dir", "seed", "run_name", "unique_output_dir", "save_config_snapshot"}
        _unknown_keys(raw, allowed, "experiment")
        return cls(
            output_dir=str(raw.get("output_dir", "outputs/whisper-finetune")),
            tensorboard_dir=raw.get("tensorboard_dir"),
            seed=int(raw.get("seed", 42)),
            run_name=raw.get("run_name"),
            unique_output_dir=bool(raw.get("unique_output_dir", True)),
            save_config_snapshot=bool(raw.get("save_config_snapshot", True)),
        )


@dataclass(slots=True)
class AppConfig:
    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppConfig":
        _unknown_keys(raw, {"experiment", "model", "data", "cache", "hub", "training"}, "root config")
        if "model" not in raw:
            raise ConfigError("model section is required")
        if "data" not in raw:
            raise ConfigError("data section is required")
        return cls(
            experiment=ExperimentConfig.from_dict(raw.get("experiment")),
            model=ModelConfig.from_dict(raw["model"]),
            data=DataConfig.from_dict(raw["data"]),
            cache=CacheConfig.from_dict(raw.get("cache")),
            hub=HubConfig.from_dict(raw.get("hub")),
            training=TrainingConfig.from_dict(raw.get("training")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping")
    config = AppConfig.from_dict(raw)
    if config.model.init_from_output_dir:
        init_path = Path(config.model.init_from_output_dir)
        if not init_path.is_absolute():
            config.model.init_from_output_dir = str((config_path.parent / init_path).resolve())
    if config.training.resume_from_output_dir:
        resume_path = Path(config.training.resume_from_output_dir)
        if not resume_path.is_absolute():
            config.training.resume_from_output_dir = str((config_path.parent / resume_path).resolve())
    if config.training.deepspeed_config:
        deepspeed_path = Path(config.training.deepspeed_config)
        if not deepspeed_path.is_absolute():
            config.training.deepspeed_config = str((config_path.parent / deepspeed_path).resolve())
    return config


def save_config(config: AppConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)


def save_config_artifacts(
    config: AppConfig,
    target_dir: str | Path,
    *,
    source_config_path: str | Path | None = None,
) -> None:
    destination = Path(target_dir)
    destination.mkdir(parents=True, exist_ok=True)
    save_config(config, destination / "resolved-config.yaml")

    if source_config_path is None:
        return

    source = Path(source_config_path)
    if not source.is_file():
        raise FileNotFoundError(f"Training config source file not found: {source}")
    shutil.copy2(source, destination / "training-config.yaml")
