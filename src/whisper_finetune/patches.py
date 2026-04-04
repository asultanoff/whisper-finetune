from __future__ import annotations

import torch
from torch import nn


def _downsample_attention_mask(module, attention_mask: torch.Tensor | None, output_length: int) -> torch.Tensor | None:
    if attention_mask is None:
        return None

    device = getattr(module, "device", attention_mask.device)
    attention_mask = attention_mask.to(device=device, dtype=torch.long)
    input_lengths = attention_mask.sum(dim=-1)
    output_lengths = module._get_feat_extract_output_lengths(input_lengths).clamp(max=output_length)
    frame_ids = torch.arange(output_length, device=attention_mask.device).unsqueeze(0)
    return frame_ids < output_lengths.unsqueeze(1)


def _prepare_encoder_attention_mask(module, attention_mask: torch.Tensor | None, output_length: int):
    reduced_attention_mask = _downsample_attention_mask(module, attention_mask, output_length)
    if reduced_attention_mask is None:
        return None, None
    return reduced_attention_mask, module.invert_attention_mask(reduced_attention_mask)


def enable_whisper_encoder_input_length_patch(modeling_whisper=None) -> None:
    if modeling_whisper is None:
        from transformers.models.whisper import modeling_whisper

    if getattr(modeling_whisper.WhisperEncoder, "_whisper_finetune_patch_enabled", False):
        return

    @modeling_whisper.merge_with_config_defaults
    @modeling_whisper.capture_outputs
    def forward(self, input_features, attention_mask=None, **kwargs):
        expected_seq_length = (
            self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        )
        if input_features.shape[-1] > expected_seq_length:
            raise ValueError(
                f"Whisper expects mel input features with length <= {expected_seq_length}, "
                f"but found {input_features.shape[-1]}. Split or crop audio longer than 30 seconds."
            )

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.embed_positions(position_ids)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        _, encoder_attention_mask = _prepare_encoder_attention_mask(self, attention_mask, hidden_states.shape[1])

        for encoder_layer in self.layers:
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True

            if not to_drop:
                hidden_states = encoder_layer(
                    hidden_states,
                    encoder_attention_mask,
                    **kwargs,
                )

        hidden_states = self.layer_norm(hidden_states)
        return modeling_whisper.BaseModelOutput(last_hidden_state=hidden_states)

    @modeling_whisper.can_return_tuple
    def model_forward(
        self,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = None,
        past_key_values=None,
        decoder_inputs_embeds: tuple[torch.FloatTensor] | None = None,
        decoder_position_ids: tuple[torch.LongTensor] | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        encoder_attention_mask = None
        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)
            encoder_outputs = self.encoder(
                input_features,
                attention_mask=attention_mask,
                **kwargs,
            )
        elif not isinstance(encoder_outputs, modeling_whisper.BaseModelOutput):
            encoder_outputs = modeling_whisper.BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if attention_mask is not None:
            _, encoder_attention_mask = _prepare_encoder_attention_mask(
                self.encoder,
                attention_mask,
                encoder_outputs.last_hidden_state.shape[1],
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            **kwargs,
        )

        return modeling_whisper.Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    @modeling_whisper.merge_with_config_defaults
    @modeling_whisper.capture_outputs
    def decoder_forward(
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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = (
                modeling_whisper.EncoderDecoderCache(
                    modeling_whisper.DynamicCache(config=self.config),
                    modeling_whisper.DynamicCache(config=self.config),
                )
                if encoder_hidden_states is not None or self.config.is_encoder_decoder
                else modeling_whisper.DynamicCache(config=self.config)
            )

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_key_values_length
            position_ids = position_ids.unsqueeze(0).repeat(inputs_embeds.shape[0], 1)

        if input_ids is not None:
            positions = self.embed_positions(
                input_ids,
                past_key_values_length=past_key_values_length,
                position_ids=position_ids,
            )
        else:
            positions = self.embed_positions(
                inputs_embeds,
                past_key_values_length=past_key_values_length,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        causal_mask = modeling_whisper.create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        for decoder_layer in self.layers:
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            hidden_states = decoder_layer(
                hidden_states,
                causal_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.layer_norm(hidden_states)

        return modeling_whisper.BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    modeling_whisper.WhisperEncoder.forward = forward
    if hasattr(modeling_whisper, "WhisperDecoder"):
        modeling_whisper.WhisperDecoder.forward = decoder_forward
    modeling_whisper.WhisperModel.forward = model_forward
    modeling_whisper.WhisperEncoder._whisper_finetune_patch_enabled = True
