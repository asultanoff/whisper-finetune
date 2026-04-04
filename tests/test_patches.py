from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from whisper_finetune.patches import enable_whisper_encoder_input_length_patch


@dataclass
class FakeBaseModelOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None

    def __getitem__(self, index: int):
        values = (self.last_hidden_state, self.hidden_states, self.attentions)
        return values[index]


@dataclass
class FakeSeq2SeqModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: object | None = None
    decoder_hidden_states: tuple[torch.Tensor, ...] | None = None
    decoder_attentions: tuple[torch.Tensor, ...] | None = None
    cross_attentions: tuple[torch.Tensor, ...] | None = None
    encoder_last_hidden_state: torch.Tensor | None = None
    encoder_hidden_states: tuple[torch.Tensor, ...] | None = None
    encoder_attentions: tuple[torch.Tensor, ...] | None = None


def _identity_decorator(fn):
    return fn


def _build_fake_modeling_whisper():
    class AttentionMaskMixin:
        @property
        def device(self) -> torch.device:
            return next(self.parameters()).device

        @property
        def dtype(self) -> torch.dtype:
            return next(param.dtype for param in self.parameters() if param.is_floating_point())

        def invert_attention_mask(self, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            else:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)
            return (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

    class FakeEncoderLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_attention_mask: torch.Tensor | None = None

        def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None, **kwargs):
            self.seen_attention_mask = attention_mask.clone() if attention_mask is not None else None
            return hidden_states

    class FakeWhisperEncoder(AttentionMaskMixin, nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(max_source_positions=1500)
            self.conv1 = nn.Conv1d(80, 4, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(4, 4, kernel_size=3, stride=2, padding=1)
            self.embed_positions = nn.Embedding(1500, 4)
            self.layers = nn.ModuleList([FakeEncoderLayer()])
            self.layer_norm = nn.LayerNorm(4)
            self.dropout = 0.0
            self.layerdrop = 0.0

        def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
            return (input_lengths - 1) // 2 + 1

    class FakeWhisperDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_encoder_attention_mask: torch.Tensor | None = None

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            position_ids=None,
            use_cache=None,
            **kwargs,
        ):
            self.seen_encoder_attention_mask = (
                encoder_attention_mask.clone() if encoder_attention_mask is not None else None
            )
            batch_size = encoder_hidden_states.shape[0]
            target_length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            hidden_size = encoder_hidden_states.shape[-1]
            return SimpleNamespace(
                last_hidden_state=torch.zeros(batch_size, target_length, hidden_size),
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
            )

    class FakeWhisperModel(AttentionMaskMixin, nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(apply_spec_augment=False)
            self.encoder = FakeWhisperEncoder()
            self.decoder = FakeWhisperDecoder()

        def _mask_input_features(self, input_features: torch.Tensor, attention_mask=None):
            return input_features

    return SimpleNamespace(
        WhisperEncoder=FakeWhisperEncoder,
        WhisperModel=FakeWhisperModel,
        merge_with_config_defaults=lambda fn: fn,
        capture_outputs=lambda fn: fn,
        can_return_tuple=lambda fn: fn,
        auto_docstring=lambda fn: fn,
        BaseModelOutput=FakeBaseModelOutput,
        Seq2SeqModelOutput=FakeSeq2SeqModelOutput,
    )


def test_encoder_patch_accepts_shorter_inputs_and_rejects_longer_ones() -> None:
    fake_modeling_whisper = _build_fake_modeling_whisper()

    enable_whisper_encoder_input_length_patch(fake_modeling_whisper)
    encoder = fake_modeling_whisper.WhisperEncoder()

    short_output = encoder(torch.randn(2, 80, 100))
    exact_output = encoder(torch.randn(1, 80, 3000))

    assert short_output.last_hidden_state.shape == (2, 50, 4)
    assert exact_output.last_hidden_state.shape == (1, 1500, 4)

    with pytest.raises(ValueError):
        encoder(torch.randn(1, 80, 3200))


def test_encoder_patch_rebuilds_attention_mask_for_variable_lengths() -> None:
    fake_modeling_whisper = _build_fake_modeling_whisper()

    enable_whisper_encoder_input_length_patch(fake_modeling_whisper)
    encoder = fake_modeling_whisper.WhisperEncoder()
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    encoder(torch.randn(2, 80, 10), attention_mask=attention_mask)
    seen_attention_mask = encoder.layers[0].seen_attention_mask

    assert seen_attention_mask is not None
    assert seen_attention_mask.shape == (2, 1, 1, 5)
    assert seen_attention_mask[0, 0, 0, :4].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert seen_attention_mask[0, 0, 0, 4].item() == torch.finfo(seen_attention_mask.dtype).min
    assert seen_attention_mask[1, 0, 0, :].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_model_patch_passes_rebuilt_mask_to_decoder_cross_attention() -> None:
    fake_modeling_whisper = _build_fake_modeling_whisper()

    enable_whisper_encoder_input_length_patch(fake_modeling_whisper)
    model = fake_modeling_whisper.WhisperModel()
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    outputs = model(
        input_features=torch.randn(2, 80, 10),
        attention_mask=attention_mask,
        decoder_input_ids=torch.ones(2, 3, dtype=torch.long),
    )

    assert outputs.last_hidden_state.shape == (2, 3, 4)
    assert model.decoder.seen_encoder_attention_mask is not None
    assert model.decoder.seen_encoder_attention_mask.shape == (2, 1, 1, 5)
    assert torch.equal(model.decoder.seen_encoder_attention_mask, model.encoder.layers[0].seen_attention_mask)


def test_model_patch_rebuilds_cross_attention_mask_when_encoder_outputs_are_precomputed() -> None:
    fake_modeling_whisper = _build_fake_modeling_whisper()

    enable_whisper_encoder_input_length_patch(fake_modeling_whisper)
    model = fake_modeling_whisper.WhisperModel()
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)
    encoder_outputs = fake_modeling_whisper.BaseModelOutput(last_hidden_state=torch.randn(1, 5, 4))

    model(
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        decoder_input_ids=torch.ones(1, 2, dtype=torch.long),
    )

    assert model.decoder.seen_encoder_attention_mask is not None
    assert model.decoder.seen_encoder_attention_mask.shape == (1, 1, 1, 5)
    assert model.decoder.seen_encoder_attention_mask[0, 0, 0, :4].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert model.decoder.seen_encoder_attention_mask[0, 0, 0, 4].item() == torch.finfo(
        model.decoder.seen_encoder_attention_mask.dtype
    ).min
