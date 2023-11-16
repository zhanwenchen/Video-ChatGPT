from undecorated import undecorated
from types import MethodType
from warnings import warn
from typing import List, Optional, Tuple, Union
import torch
from torch import stack as torch_stack, no_grad as torch_no_grad, cat as torch_cat, equal as torch_equal
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
RATIO = 256/100


class VisionConfig:
    def __init__(self):
        self.frame_size = 224
        self.patch_size = 14
        self.hidden_size = 1024
        self.use_vid_start_end = None
        self.vid_start_token = None
        self.vid_end_token = None
        self.vid_patch_token = None


class VideoChatGPTConfig(LlamaConfig):
    model_type = "VideoChatGPT"


class VideoChatGPTLlamaModel(LlamaModel):
    config_class = VideoChatGPTConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):  # TODO: Remove unused params
        super(VideoChatGPTLlamaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_config = VisionConfig()

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def initialize_vision_modules(self, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        vision_config = self.vision_config
        num_patches = (vision_config.frame_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            video_token_len=num_patches,
            vision_config=vision_config
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (input_ids.shape[1] != 1 or self.training) and video_spatio_temporal_features is not None:

            video_features = self.mm_projector(video_spatio_temporal_features)
            dummy_video_features = torch.zeros(video_features.shape[1], 1024, device=inputs_embeds.device,
                                               dtype=inputs_embeds.dtype)
            dummy_video_features = self.mm_projector(dummy_video_features)

            new_input_embeds = []
            cur_video_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.vision_config.vid_patch_token).sum() == 0:
                    # Multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_video_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_video_idx += 1
                    continue
                if self.vision_config.use_vid_start_end:
                    if (cur_input_ids == self.vision_config.vid_start_token).sum() != (
                            cur_input_ids == self.vision_config.vid_end_token).sum():
                        raise ValueError("The number of video start tokens and video end tokens should be the same.")
                    video_start_tokens = torch.where(cur_input_ids == self.vision_config.vid_start_token)[0]
                    for video_start_token_pos in video_start_tokens:
                        cur_video_features = video_features[cur_video_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_video_features.shape[0]
                        if cur_input_ids[video_start_token_pos + num_patches + 1] != self.vision_config.vid_end_token:
                            raise ValueError("The video end token should follow the video start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos].detach(),
                                                              cur_input_embeds[
                                                              video_start_token_pos:video_start_token_pos + 1],
                                                              cur_video_features, cur_input_embeds[
                                                                                  video_start_token_pos + num_patches
                                                                                  + 1:video_start_token_pos
                                                                                  + num_patches + 2],
                                                              cur_input_embeds[
                                                              video_start_token_pos + num_patches + 2:].detach()),
                                                             dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos + 1],
                                                              cur_video_features,
                                                              cur_input_embeds[video_start_token_pos
                                                                               + num_patches + 1:]), dim=0)
                        cur_video_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_video_features = video_features[cur_video_idx]
                    num_patches = cur_video_features.shape[0]
                    if (cur_input_ids == self.vision_config.vid_patch_token).sum() != num_patches:
                        raise ValueError(
                            "The number of video patch tokens should be the same as the number of video patches.")
                    masked_indices = torch.where(cur_input_ids == self.vision_config.vid_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patches,
                                                       device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The video patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(),
                                                          cur_video_features,
                                                          cur_input_embeds[mask_index_start + num_patches:].detach()),
                                                         dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_video_features,
                                                          cur_input_embeds[mask_index_start + num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_video_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(VideoChatGPTLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class VideoChatGPTLlamaForCausalLM(LlamaForCausalLM):
    config_class = VideoChatGPTConfig

    def __init__(self, config, sequence_bias_sequence_ids: list, bias: float):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VideoChatGPTLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.bias = bias

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            token_ids=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            video_spatio_temporal_features=video_spatio_temporal_features
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states) # torch.Size([1, 434, 32006])
        if self.training and token_ids is not None:
            # TODO: Should I do this for all sequences or just the last ones?
            # preds = logits.argmax(dim=-1)
            # logits_original = logits[:, -1, :]
            num_tokens = len(token_ids)
            # logits_old = logits.detach().clone()
            bias = self.bias
            for idx, token in enumerate(token_ids):
                logits[:, -(num_tokens-idx)-1, token] += bias
            # print(torch_equal(logits_old, logits))
            # print(torch_equal(logits_old.argmax(-1), logits.argmax(-1)))
            # breakpoint()
            # logits_new = token_ids(preds, logits_original) # torch.Size([1, 32006])
            # logits_new = token_ids(preds, logits[0, ...]) # torch.Size([1, 32006])
            # torch_equal(logits_original, logits_new)
            # torch_equal(logits[0], logits_new)
            # breakpoint()
            # logits[:, -1, :] = logits_new
            # if not torch_equal(logits_original, logits_new):
            #     print('logits_original != logits_new')
            # sequence_ids = list(token_ids.sequence_bias.keys())[0]
            # Check logits_original
            # Check self.sequence_bias_sequence_ids
            # Check labels.
        #     # pre-process distribution
        #     token_ids.prepared_bias_variables
        #     token_ids.length_1_bias
        #     next_token_logits = logits[:, -1, :]
        #     sequence_bias = token_ids.sequence_bias
        #     sequence_ids = list(sequence_bias.keys())[0]
        #     next_token_logits[:, list(self.sequence_bias.keys())[0]]
        #     next_token_logits_new = token_ids(input_ids, next_token_logits)
        #     next_token_logits_new[:, list(sequence_bias.keys())[0]]
        #     logits[:, -1, :] = token_ids(input_ids, logits[:, -1, :]) # torch.Size([1, 32006])

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # breakpoint()
            shift_logits = logits[..., :-1, :].contiguous() # torch.Size([1, 433, 32006])
            # if token_ids is not None:
            #     # TODO: Should I do this for all sequences or just the last ones?
            #     # shift_logits_original = shift_logits
            #     for i in range(shift_logits.size(1)):
            #         shift_logits_original = shift_logits[:, i, :]
            #         shift_logits_new = token_ids(input_ids, shift_logits_original)
            #         shift_logits[:, i, :] = shift_logits_new
            #         if not torch_equal(logits_original, logits_new):
            #             print('logits_original != logits_new')
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "video_spatio_temporal_features": kwargs.get("video_spatio_temporal_features", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_vid_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_config = self.get_model().vision_config
        vision_config.use_vid_start_end = mm_use_vid_start_end
        tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=None)

        if mm_use_vid_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]



def zero_video_spatio_temporal_features_at_t(video_spatio_temporal_features, t):
    # [100, 1024]
    video_spatio_temporal_features_clone = video_spatio_temporal_features.detach().clone()
    temporal_features = video_spatio_temporal_features_clone[:, :100, :]
    # [256, 1024]
    spatial_features = video_spatio_temporal_features_clone[:, 100:, :]
    temporal_features_t = temporal_features[:, t, :]  # [1024]
    temporal_features[:, t, :].zero_()
    spatial_features -= temporal_features_t * RATIO  # Approximation
    return torch_cat((temporal_features, spatial_features), dim=1)


class VideoChatGPTLlamaForCausalLMLoo(VideoChatGPTLlamaForCausalLM):
    def __init__(self, config, sequence_bias_dicts: list, num_frames: int, bias: float):
        self.sequence_bias_sequence_ids = sequence_bias_sequence_ids = [sequence_id for sequence_id in sequence_bias_dicts]
        super().__init__(config, sequence_bias_sequence_ids, bias)
        # https://discuss.huggingface.co/t/how-to-output-loss-from-model-generate/16999/2?u=zhanwenchen
        # generate_with_grad = undecorated(self.generate)
        # self.generate_with_grad = MethodType(generate_with_grad, self)
        # self.sequence_bias_dicts = sequence_bias_dicts
        self.num_frames = num_frames
        # sequence_bias_dict = self.sequence_bias_dicts[frame_idx]
        # self.sequence_bias_processors = [SequenceBiasLogitsProcessor(sequence_bias=sequence_bias_dict) for sequence_bias_dict in sequence_bias_dicts]


    @torch_no_grad()
    # @torch_autocast('cuda', enabled=True)
    def get_idx(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ):
        # training = self.training
        num_clips = video_spatio_temporal_features.size(0)
        # warn(f'num_clips={num_clips}, video_spatio_temporal_features.size()={video_spatio_temporal_features.size()}')
        assert num_clips == 1
        num_frames_per_clip = self.num_frames
        forward = super().forward
        # self.eval()
        output = forward(
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            video_spatio_temporal_features,
            return_dict=False,
        )
        assert len(output) != 1
        loss_original = biggest_loss = output[0]
        frame_idx_to_remove_biggest = None
        for frame_idx_to_remove in range(num_frames_per_clip):
            video_spatio_temporal_features_zero_t = zero_video_spatio_temporal_features_at_t(video_spatio_temporal_features, frame_idx_to_remove)
            output = forward(
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                video_spatio_temporal_features_zero_t,
                return_dict=False,
            )
            loss_zero_t = output[0]
            # print(f't={frame_idx_to_remove} type(output)={type(output)}, loss_zero_t.size()={loss_zero_t.size()}, loss_zero_t={loss_zero_t}')
            if loss_zero_t > biggest_loss:
                biggest_loss = loss_zero_t
                frame_idx_to_remove_biggest = frame_idx_to_remove
                # print(f'At t={frame_idx_to_remove}, loss increases from loss_original={loss_original} to {loss_zero_t}. The biggest loss is now biggest_loss={biggest_loss} at t={frame_idx_to_remove_biggest}')
            # if loss_zero_t < loss_original:
            #     print(f'At t={frame_idx_to_remove}, loss drops from loss_original={loss_original} to {loss_zero_t}')
        # if frame_idx_to_remove_biggest:
        #     print(f'At t={frame_idx_to_remove_biggest}, biggest loss increases from loss_original={loss_original} to {biggest_loss}')
        # else:
        #     print(f'No t drops loss from loss_original={loss_original}')
        if frame_idx_to_remove_biggest is None:
            print(f'No t drops loss from loss_original={loss_original}')
        # if training:
        #     self.train()
        # logits = map_frame_idx_to_logits(frame_idx_to_remove_biggest)# Change logits to match the frame idx's embedding.
        # return forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        #     video_spatio_temporal_features=video_spatio_temporal_features,
        # )
        return frame_idx_to_remove_biggest


    # @torch_autocast('cuda', enabled=False)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ):
        frame_idx = self.get_idx(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            video_spatio_temporal_features=video_spatio_temporal_features,
        )
        if frame_idx is not None:
            token_ids = self.sequence_bias_sequence_ids[frame_idx]
        else:
            token_ids = None
        # https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.SequenceBiasLogitsProcessor
        # next_token_logits = outputs.logits[:, -1, :]

        # # pre-process distribution
        # next_token_scores = token_ids(input_ids, next_token_logits)

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            video_spatio_temporal_features=video_spatio_temporal_features,
            return_dict=return_dict,
            token_ids=token_ids,
        )
        return output



AutoConfig.register("VideoChatGPT", VideoChatGPTConfig)
AutoModelForCausalLM.register(VideoChatGPTConfig, VideoChatGPTLlamaForCausalLM)
