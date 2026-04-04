# Whisper Finetune

Standalone Whisper finetuning scaffold built in `whisper-finetune`.

It takes the useful high-level idea from the `icefall` Whisper recipe, but it
switches to a Hugging Face native stack because the main requirement here is
configurable training over multiple Hugging Face dataset repos with per-dataset
schema and split rules.

## What it supports

- YAML config for experiment, model, training, and datasets
- shared cache config for Hugging Face model and dataset downloads
- explicit TensorBoard log directory config
- multiple Hugging Face dataset repos in one run
- per-dataset `audio_column` and `text_column`
- per-dataset validation via:
  - explicit validation split
  - ratio split carved out from the train split
- on-the-fly mel extraction in the training collator
- optional Whisper encoder patch to avoid padding every batch to 30 seconds
- optional `.env` loading for secrets such as `HF_TOKEN`
- DeepSpeed support via config, similar to the `icefall` Whisper recipe
- optional post-training upload of a cleaned final model repo to Hugging Face
- `uv` as the dependency manager

## Quick start

```bash
cd whisper-finetune
cp .env.example .env
uv sync --extra dev
uv run whisper-finetune-train --config configs/multi-dataset.example.yaml
uv run tensorboard --logdir outputs/whisper-small-multiset/tensorboard
```

If you need a Hugging Face token for private datasets or models, put it in
`.env`:

```dotenv
HF_TOKEN=hf_...
```

## Config shape

See [configs/multi-dataset.example.yaml](/home/asultanov/drives/jobs/hanny_ai/training/whisper-finetune/configs/multi-dataset.example.yaml).

The relevant dataset fields are:

- `repo_id`: Hugging Face dataset repo
- `config_name`: optional dataset config/subset
- `train_split`: split used for training
- `validation_split`: explicit validation split to load
- `validation_from_train_ratio`: optional ratio to carve from `train_split`
- `audio_column`: source audio column name
- `text_column`: source transcript column name

`validation_split` and `validation_from_train_ratio` are mutually exclusive.

Use the `cache` section to control where model and dataset downloads go:

```yaml
cache:
  root_dir: .cache
  model_dir: .cache/models
  dataset_dir: .cache/datasets
```

TensorBoard logs go to `experiment.tensorboard_dir`:

```yaml
experiment:
  output_dir: outputs/my-run
  tensorboard_dir: outputs/my-run/tensorboard
```

DeepSpeed is enabled by pointing `training.deepspeed_config` at a JSON file.
The repo includes [configs/deepspeed/zero1.json](/home/asultanov/drives/jobs/hanny_ai/training/whisper-finetune/configs/deepspeed/zero1.json), which mirrors the zero-stage-1 setup used in the `icefall` recipe.

To upload a cleaned final model repo after training, use:

```yaml
hub:
  enabled: true
  repo_id: your-name/your-private-whisper-repo
  private: true
  token_env_var: HF_TOKEN
  replace_existing_repo_files: true
  export_subdir: hf-export
```

The staged export includes final model artifacts, processor files,
`resolved-config.yaml`, `train_results.json`, `eval_results.json` when present,
`trainer_state.json`, and a small model card. Interim `checkpoint-*`
directories are intentionally excluded.

To remove the encoder-side 30-second padding requirement for clips shorter than
30 seconds, set:

```yaml
model:
  remove_encoder_input_length_restriction: true
```

This changes the collator to compute mel features from raw audio with
`padding="longest"` instead of pre-padding every sample to Whisper's fixed
30-second window.

## Notes

- If none of the configured datasets produce a validation set, training falls
  back to no evaluation automatically.
- The repo currently concatenates all train datasets together and all
  validation datasets together. Weighted sampling is not implemented yet.
- The dynamic encoder-length option only supports clips up to 30 seconds. It
  removes wasteful padding for shorter clips; it does not add long-form
  training support.
