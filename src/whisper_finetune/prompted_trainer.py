from __future__ import annotations

import contextlib
from typing import Any

import torch


GENERATION_DECODER_INPUT_IDS_KEY = "generation_decoder_input_ids"
GENERATION_DECODER_ATTENTION_MASK_KEY = "generation_decoder_attention_mask"
GENERATION_LANGUAGE_KEY = "generation_language"


def split_generation_prompt_inputs(inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_inputs = {
        key: value
        for key, value in inputs.items()
        if key
        not in {
            GENERATION_DECODER_INPUT_IDS_KEY,
            GENERATION_DECODER_ATTENTION_MASK_KEY,
            GENERATION_LANGUAGE_KEY,
        }
    }
    generation_prompt_inputs = {}
    if GENERATION_DECODER_INPUT_IDS_KEY in inputs:
        generation_prompt_inputs["decoder_input_ids"] = inputs[GENERATION_DECODER_INPUT_IDS_KEY]
    if GENERATION_DECODER_ATTENTION_MASK_KEY in inputs:
        generation_prompt_inputs["decoder_attention_mask"] = inputs[GENERATION_DECODER_ATTENTION_MASK_KEY]
    if GENERATION_LANGUAGE_KEY in inputs:
        generation_prompt_inputs["language"] = inputs[GENERATION_LANGUAGE_KEY]
    return model_inputs, generation_prompt_inputs


def _is_deepspeed_zero3_enabled(trainer: Any) -> bool:
    accelerator = getattr(trainer, "accelerator", None)
    state = getattr(accelerator, "state", None)
    plugin = getattr(state, "deepspeed_plugin", None)
    if plugin is None:
        return False
    zero_stage = getattr(plugin, "zero_stage", None)
    if zero_stage is not None:
        return int(zero_stage) == 3
    hf_ds_config = getattr(plugin, "hf_ds_config", None)
    config = getattr(hf_ds_config, "config", None)
    if isinstance(config, dict):
        return int(config.get("zero_optimization", {}).get("stage", 0)) == 3
    return False


class WhisperPromptedSeq2SeqTrainer:
    """Mixin-like wrapper around Seq2SeqTrainer semantics for generation prompt inputs."""

    @classmethod
    def bind(cls, base_cls):
        if getattr(base_cls, "_whisper_prompted_trainer_bound", False):
            return base_cls

        class BoundWhisperPromptedSeq2SeqTrainer(base_cls):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                model_inputs, _ = split_generation_prompt_inputs(inputs)
                return super().compute_loss(model, model_inputs, return_outputs=return_outputs, **kwargs)

            def prediction_step(
                self,
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=None,
                **gen_kwargs,
            ):
                if not self.args.predict_with_generate or prediction_loss_only:
                    model_inputs, _ = split_generation_prompt_inputs(inputs)
                    return super().prediction_step(
                        model,
                        model_inputs,
                        prediction_loss_only=prediction_loss_only,
                        ignore_keys=ignore_keys,
                    )

                from torch.distributed.fsdp import FullyShardedDataParallel

                has_labels = "labels" in inputs
                inputs = self._prepare_inputs(inputs)
                model_inputs, generation_prompt_inputs = split_generation_prompt_inputs(inputs)

                if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
                    gen_kwargs = self._gen_kwargs.copy()
                if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
                    gen_kwargs.pop("num_beams")
                if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
                    gen_kwargs.pop("max_length")

                default_synced_gpus = _is_deepspeed_zero3_enabled(self) or isinstance(self.model, FullyShardedDataParallel)
                gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)

                generation_inputs = model_inputs.copy()
                generation_inputs.update(generation_prompt_inputs)

                summon_full_params_context = (
                    FullyShardedDataParallel.summon_full_params(self.model)
                    if torch.distributed.is_available() and isinstance(self.model, FullyShardedDataParallel)
                    else contextlib.nullcontext()
                )

                with summon_full_params_context:
                    generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

                if self.model.generation_config._from_model_config:
                    self.model.generation_config._from_model_config = False

                gen_config = self.model.generation_config
                default_gen_config = gen_config._get_default_generation_params()
                gen_config.update(**default_gen_config, defaults_only=True)
                if generated_tokens.shape[-1] < gen_config.max_length:
                    generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
                elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
                    generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

                with torch.no_grad():
                    if has_labels:
                        with self.compute_loss_context_manager():
                            outputs = model(**model_inputs)
                        if self.label_smoother is not None:
                            loss = self.label_smoother(outputs, model_inputs["labels"]).detach().mean()
                        else:
                            loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).detach().mean()
                    else:
                        loss = None

                if self.args.prediction_loss_only:
                    return loss, None, None

                if has_labels:
                    labels = model_inputs["labels"]
                    if labels.shape[-1] < gen_config.max_length:
                        labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
                    elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                        labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
                else:
                    labels = None

                return loss, generated_tokens, labels

        BoundWhisperPromptedSeq2SeqTrainer._whisper_prompted_trainer_bound = True
        return BoundWhisperPromptedSeq2SeqTrainer
