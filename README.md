# Whisper Finetune

Config-driven Whisper finetuning in a standalone `uv` repo.

The project keeps the useful DeepSpeed-oriented training shape from the `icefall`
Whisper recipe, but uses a Hugging Face-native stack so it can train against
multiple Hugging Face dataset repos with per-dataset schema and split rules.

## Features

- YAML config for experiment, model, training, and datasets
- `uv` for dependency management
- `.env` loading via `python-dotenv`
- shared cache config for model and dataset downloads
- multiple Hugging Face dataset repos in one run
- per-dataset `audio_column` and `text_column`
- per-dataset validation from either:
  - explicit validation split
  - ratio carved from the train split
- on-the-fly mel extraction in the collator
- optional Whisper patch for variable encoder lengths up to 30 seconds
- correct encoder self-attention and decoder cross-attention masking for padded batch tails
- optional on-the-fly waveform corruption before mel extraction
- TensorBoard logging
- DeepSpeed support
- grouped batching by similar audio lengths or text lengths
- cleaned final model export and optional private Hugging Face upload
- resume training from a previous output folder
- initialize a fresh new run from a previous output folder
- unique output directories per run by default
- export clean/augmented review samples for manual listening

## Install

```bash
cd whisper-finetune
cp .env.example .env
uv sync --extra dev
```

If you need private Hugging Face access:

```dotenv
HF_TOKEN=hf_...
```

If you enable codec-style telephony augmentation, make sure `ffmpeg` is
available on the machine that runs training.

## Run

Without DeepSpeed:

```bash
uv run whisper-finetune-train --config configs/multi-dataset.example.yaml
```

With DeepSpeed:

```bash
uv run deepspeed --num_gpus 1 --module whisper_finetune.train --config configs/train.whisper-small.akmalsultanov.yaml
```

TensorBoard:

```bash
uv run tensorboard --logdir outputs
```

Export augmented review samples:

```bash
uv run whisper-finetune-export-augments \
  --config configs/train.whisper-small.akmalsultanov.yaml \
  --dataset lb_tg_raw_4k \
  --num-samples 100
```

## Config Overview

Example configs:

- [multi-dataset.example.yaml](/home/asultanov/drives/jobs/hanny_ai/training/whisper-finetune/configs/multi-dataset.example.yaml)
- [train.whisper-small.akmalsultanov.yaml](/home/asultanov/drives/jobs/hanny_ai/training/whisper-finetune/configs/train.whisper-small.akmalsultanov.yaml)

### `experiment`

```yaml
experiment:
  output_dir: outputs/my-run
  tensorboard_dir: outputs/my-run/tensorboard
  run_name: my-run
  unique_output_dir: true
```

`unique_output_dir: true` appends a timestamped suffix so each run writes to a
new folder.

### `cache`

```yaml
cache:
  root_dir: .cache
  model_dir: .cache/models
  dataset_dir: .cache/datasets
```

### `model`

Start from a base model:

```yaml
model:
  name_or_path: openai/whisper-small
```

Start a fresh new stage from a previous local training output:

```yaml
model:
  init_from_output_dir: outputs/previous-run-20260404-123456-abcd1234
```

Other useful model options:

```yaml
model:
  task: transcribe
  language: en
  freeze_encoder: false
  generation_max_length: 225
  remove_encoder_input_length_restriction: true
```

`remove_encoder_input_length_restriction: true` enables dynamic encoder lengths
for clips up to 30 seconds. Mel features are still computed on the fly, and
batch padding is masked correctly in encoder self-attention and decoder
cross-attention.

### `data`

Each dataset entry can define its own repo, split strategy, and schema:

```yaml
data:
  audio_sampling_rate: 16000
  preprocessing_num_workers: 4
  min_audio_seconds: 0.1
  max_audio_seconds: 30.0
  text_normalization:
    lowercase: false
    strip: true
    collapse_whitespace: true
  datasets:
    - repo_id: mozilla-foundation/common_voice_17_0
      config_name: en
      alias: common_voice_en
      train_split: train
      validation_split: validation
      audio_column: audio
      text_column: sentence
    - repo_id: speechcolab/gigaspeech
      alias: gigaspeech_ratio_val
      train_split: train
      validation_from_train_ratio: 0.02
      validation_from_train_seed: 42
      audio_column: audio
      text_column: text
```

`validation_split` and `validation_from_train_ratio` are mutually exclusive.

Optional waveform corruption is configured under `data.audio_augmentation` and
is applied in the collator before Whisper feature extraction:

```yaml
data:
  audio_augmentation:
    enabled: true
    seed: 42
    train_only: true
    mode: deterministic_per_sample
    datasets:
      my_dataset_alias:
        enabled: true
        profile: phone_like
        splits: [train]
        apply_p: 0.95
    profiles:
      phone_like:
        codec:
          p: 0.95
          modes:
            mulaw_narrowband: 0.7
            gsm_narrowband: 0.3
          sample_rate: 8000
          highpass_hz: 320.0
          lowpass_hz: 2700.0
          roundtrips: 2
        noise:
          p: 0.15
          snr_db: [18.0, 28.0]
        clipping:
          p: 0.2
          threshold: [0.65, 0.82]
        packet_loss:
          p: 0.35
          burst_ms: [30, 120]
          bursts: [1, 3]
          fill: hold
```

Dataset policy keys match the dataset alias if one is configured, otherwise the
dataset repo id. For telephony-style training, codec roundtrips plus packet
loss and clipping are much more useful than broadband white noise. Codec
corruption requires `ffmpeg`.

### `training`

Typical training section:

```yaml
training:
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 2
  learning_rate: 1.0e-4
  warmup_ratio: 0.15
  num_train_epochs: 5
  lr_scheduler_type: linear
  logging_steps: 25
  eval_steps: 750
  save_steps: 750
  save_total_limit: 3
  report_to:
    - tensorboard
  bf16: true
  gradient_checkpointing: false
  dataloader_num_workers: 16
  train_sampling_strategy: group_by_length
  length_grouping_key: text
  metric_for_best_model: wer
  greater_is_better: false
  deepspeed_config: deepspeed/zero1.json
```

Grouped batching options:

- `train_sampling_strategy: random`
- `train_sampling_strategy: sequential`
- `train_sampling_strategy: group_by_length`

If `train_sampling_strategy: group_by_length`, then set:

- `length_grouping_key: audio`
- `length_grouping_key: text`

`group_by_length` is still shuffled each epoch. It does not do a fixed global
sort. It randomizes first and then groups similar lengths together.

Resume a previous run including optimizer, scheduler, and trainer state:

```yaml
training:
  resume_from_output_dir: outputs/previous-run-20260404-123456-abcd1234
```

This resumes from the latest `checkpoint-*` found in that folder and keeps
writing into the same run directory.

DeepSpeed is enabled by pointing at a JSON config file such as
[zero1.json](/home/asultanov/drives/jobs/hanny_ai/training/whisper-finetune/configs/deepspeed/zero1.json).

## Current Akmal Config

The current training config in
[train.whisper-small.akmalsultanov.yaml](/home/asultanov/drives/jobs/hanny_ai/training/whisper-finetune/configs/train.whisper-small.akmalsultanov.yaml)
is set to:

- `openai/whisper-small`
- on-the-fly mel extraction
- telephony-heavy waveform corruption enabled on train splits
- dynamic encoder-length patch enabled
- grouped batching by `text` length
- TensorBoard logging
- DeepSpeed launch
- unique output directory per run

## Hugging Face Export

To upload the final cleaned model repo after training:

```yaml
hub:
  enabled: true
  repo_id: your-name/your-private-whisper-repo
  private: true
  token_env_var: HF_TOKEN
  replace_existing_repo_files: true
  export_subdir: hf-export
```

The export includes:

- final model artifacts
- processor/tokenizer files
- `resolved-config.yaml`
- `train_results.json`
- `eval_results.json` when available
- `trainer_state.json`
- `README.md`

Interim `checkpoint-*` directories are intentionally excluded from the curated
export.

## Notes

- If no configured dataset produces a validation set, training runs without evaluation.
- Train datasets are merged together before training. Eval datasets are merged together before evaluation.
- The merged train set is shuffled once before training starts, and the Trainer reshuffles sampling each epoch.
- Dynamic encoder lengths only support clips up to 30 seconds. This repo does not implement long-form training chunking.
