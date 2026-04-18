from __future__ import annotations

import torch

from whisper_finetune.prompted_trainer import (
    GENERATION_DECODER_ATTENTION_MASK_KEY,
    GENERATION_DECODER_INPUT_IDS_KEY,
    split_generation_prompt_inputs,
)


def test_split_generation_prompt_inputs_separates_model_and_generation_tensors() -> None:
    inputs = {
        "input_features": torch.zeros(2, 80, 100),
        "labels": torch.ones(2, 10, dtype=torch.long),
        GENERATION_DECODER_INPUT_IDS_KEY: torch.ones(2, 4, dtype=torch.long),
        GENERATION_DECODER_ATTENTION_MASK_KEY: torch.ones(2, 4, dtype=torch.long),
    }

    model_inputs, generation_inputs = split_generation_prompt_inputs(inputs)

    assert set(model_inputs) == {"input_features", "labels"}
    assert set(generation_inputs) == {"decoder_input_ids", "decoder_attention_mask"}
    assert generation_inputs["decoder_input_ids"].shape == (2, 4)
