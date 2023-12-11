from __future__ import annotations

from dataclasses import dataclass

from compel import Compel, DownweightMode, BaseTextualInversionManager
from compel.embeddings_provider import EmbeddingsProvider, EmbeddingsProviderMulti, ReturnedEmbeddingsType
from typing import Optional, Union, Tuple, List, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import CLIPTokenizer, CLIPTextModel
    from torch import Tensor

__all__ = ["Prompt", "EncodedPrompt", "EncodedPrompts", "PromptEncoder"]

def default_get_dtype_for_device(device: torch.device) -> torch.dtype:
    import torch
    return torch.float32

class PromptEncoder(Compel):
    """
    Extend compel slightly to permit multiple CLIP skip levels and make encoding more generic
    """
    def __init__(self,
         tokenizer: Union[CLIPTokenizer, List[CLIPTokenizer]],
         text_encoder: Union[CLIPTextModel, List[CLIPTextModel]],
         textual_inversion_manager: Optional[BaseTextualInversionManager] = None,
         dtype_for_device_getter: Callable[[torch.device], torch.dtype] = default_get_dtype_for_device,
         truncate_long_prompts: bool = True,
         padding_attention_mask_value: int = 1,
         downweight_mode: DownweightMode = DownweightMode.MASK,
         returned_embeddings_type: ReturnedEmbeddingsType = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
         requires_pooled: Union[bool, List[bool]] = False,
         device: Optional[str] = None
     ) -> None:
        """
        Copied from https://github.com/damian0815/compel/blob/main/src/compel/compel.py
        Modified slightly to change EmbeddingsProvider to FlexibleEmbeddingsProvider
        """
        if isinstance(tokenizer, (tuple, list)) and not isinstance(text_encoder, (tuple, list)):
            raise ValueError("Cannot provide list of tokenizers, but not of text encoders.")
        elif not isinstance(tokenizer, (tuple, list)) and isinstance(text_encoder, (tuple, list)):
            raise ValueError("Cannot provide list of text encoders, but not of tokenizers.")
        elif isinstance(tokenizer, (tuple, list)) and isinstance(text_encoder, (tuple, list)):
            self.conditioning_provider = EmbeddingsProviderMulti(tokenizers=tokenizer,
                text_encoders=text_encoder,
                textual_inversion_manager=textual_inversion_manager,
                dtype_for_device_getter=dtype_for_device_getter,
                truncate=truncate_long_prompts,
                padding_attention_mask_value = padding_attention_mask_value,
                downweight_mode=downweight_mode,
                returned_embeddings_type=returned_embeddings_type,
                requires_pooled_mask = requires_pooled
            )
        else:
            self.conditioning_provider = FlexibleEmbeddingsProvider(tokenizer=tokenizer,
                text_encoder=text_encoder,
                textual_inversion_manager=textual_inversion_manager,
                dtype_for_device_getter=dtype_for_device_getter,
                truncate=truncate_long_prompts,
                padding_attention_mask_value = padding_attention_mask_value,
                downweight_mode=downweight_mode,
                returned_embeddings_type=returned_embeddings_type,
                device=device
            )
        self._device = device
        self.requires_pooled = requires_pooled

    @property
    def clip_skip(self) -> int:
        """
        Passes clip-skip through to conditioning provider
        """
        return getattr(self.conditioning_provider, "clip_skip", 0)

    @clip_skip.setter
    def clip_skip(self, skip: int) -> None:
        """
        Passes clip-skip through to conditioning provider
        """
        setattr(self.conditioning_provider, "clip_skip", skip)

class FlexibleEmbeddingsProvider(EmbeddingsProvider):
    """
    Extend compel slightly to permit multiple CLIP skip levels and make encoding more generic
    """
    clip_skip: int = 0

    def _encode_token_ids_to_embeddings(self, token_ids: Tensor, attention_mask: Optional[Tensor]=None) -> Tensor:
        """
        Extends compels functionality to permit any level of clip skip
        """
        needs_hidden_states = (
            self.returned_embeddings_type == ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED or
            self.returned_embeddings_type == ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
        )
        text_encoder_output = self.text_encoder(
            token_ids,
            attention_mask,
            output_hidden_states=needs_hidden_states,
            return_dict=True
        )
        if self.returned_embeddings_type is ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED:
            penultimate_hidden_state = text_encoder_output.hidden_states[-(self.clip_skip + 2)]
            return penultimate_hidden_state
        elif self.returned_embeddings_type is ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED:
            penultimate_hidden_state = text_encoder_output.hidden_states[-(self.clip_skip + 1)]
            return self.text_encoder.text_model.final_layer_norm(penultimate_hidden_state)
        elif self.returned_embeddings_type is ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED:
            # already normalized
            return text_encoder_output.last_hidden_state

        assert False, f"unrecognized ReturnEmbeddingsType: {self.returned_embeddings_type}"

    def get_pooled_embeddings(
        self,
        texts: List[str],
        attention_mask: Optional[Tensor]=None,
        device: Optional[str]=None
    ) -> Optional[Tensor]:
        """
        Uses the generic way to get pooled embeddings
        """
        import torch
        device = device or self.device

        token_ids = self.get_token_ids(texts, padding="max_length", truncation_override=True)
        token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)

        return self.text_encoder(token_ids, attention_mask)[0]


@dataclass(frozen=True)
class Prompt:
    """
    This class holds, at a minimum, a prompt string.
    It can also contain a start frame, end frame, and weight.
    """
    positive: Optional[str] = None
    negative: Optional[str] = None
    positive_2: Optional[str] = None
    negative_2: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    weight: Optional[float] = None

    def get_frame_overlap(self, frames: List[int]) -> float:
        """
        Gets the frame overlap ratio for this prompt
        """
        if self.start is None:
            return 1.0
        end = self.end
        if end is None:
            end = max(frames)

        prompt_frame_list = list(range(self.start, end))
        return len(set(prompt_frame_list).intersection(set(frames))) / len(frames)

    def __str__(self) -> str:
        if self.positive is None:
            return "(none)"
        return self.positive

@dataclass
class EncodedPrompt:
    """
    After encoding a prompt, this class holds the tensors and provides
    methods for accessing the encoded tensors.
    """
    prompt: Union[Prompt, str]
    embeds: Tensor
    negative_embeds: Optional[Tensor]
    pooled_embeds: Optional[Tensor]
    negative_pooled_embeds: Optional[Tensor]

    def __str__(self) -> str:
        return str(self.prompt)

    def check_get_tensor(
        self,
        frames: Optional[List[int]],
        tensor: Optional[Tensor]
    ) -> Tuple[Optional[Tensor], Union[float, int]]:
        """
        Checks if a tensor exists and should be returned and should be scaled.
        """
        if frames is None or isinstance(self.prompt, str) or tensor is None:
            return tensor, 1.0
        weight = 1.0 if self.prompt.weight is None else self.prompt.weight
        if frames is None or self.prompt.start is None:
            return tensor * weight, weight
        overlap = self.prompt.get_frame_overlap(frames)
        if overlap == 0 or weight == 0:
            return None, 0
        weight *= overlap
        return tensor * weight, weight

    def get_embeds(
        self,
        frames: Optional[List[int]] = None
    ) -> Tuple[Optional[Tensor], Union[float, int]]:
        """
        Gets the encoded embeds.
        """
        return self.check_get_tensor(frames, self.embeds)

    def get_negative_embeds(
        self,
        frames: Optional[List[int]] = None
    ) -> Tuple[Optional[Tensor], Union[float, int]]:
        """
        Gets the encoded negative embeds.
        """
        return self.check_get_tensor(frames, self.negative_embeds)

    def get_pooled_embeds(
        self,
        frames: Optional[List[int]] = None
    ) -> Tuple[Optional[Tensor], Union[float, int]]:
        """
        Gets the encoded pooled embeds.
        """
        return self.check_get_tensor(frames, self.pooled_embeds)
    
    def get_negative_pooled_embeds(
        self,
        frames: Optional[List[int]] = None
    ) -> Tuple[Optional[Tensor], Union[float, int]]:
        """
        Gets the encoded negative pooled embeds.
        """
        return self.check_get_tensor(frames, self.negative_pooled_embeds)

    @property
    def dtype(self) -> torch.dtype:
        """
        Gets the dtype of the encoded prompt.
        """
        return self.embeds.dtype

if TYPE_CHECKING:
    PromptGetterCallable = Callable[
        [EncodedPrompt, Optional[List[int]]],
        Tuple[Optional[Tensor], Union[float, int]]
    ]

@dataclass
class EncodedImagePrompt:
    """
    Holds an encoded image prompt when using IP adapter
    """
    prompt_embeds: Tensor
    uncond_embeds: Tensor
    scale: float

@dataclass
class EncodedPrompts:
    """
    Holds any number of encoded prompts.
    """
    prompts: List[EncodedPrompt]
    is_sdxl: bool
    do_classifier_free_guidance: bool
    image_prompt_embeds: Optional[Tensor] # input, frames, batch, tokens, embeds
    image_uncond_prompt_embeds: Optional[Tensor] # input, frames, batch, tokens, embeds

    def get_stacked_tensor(
        self,
        frames: Optional[List[int]],
        getter: PromptGetterCallable
    ) -> Optional[Tensor]:
        """
        Gets a tensor from prompts using a callable.
        """
        import torch
        return_tensor = None
        for prompt in self.prompts:
            tensor, weight = getter(prompt, frames)
            if tensor is not None and weight is not None and weight > 0:
                if return_tensor is None:
                    return_tensor = tensor * weight
                else:
                    return_tensor = torch.cat([return_tensor, tensor * weight], dim=1) # type: ignore[unreachable]
        return return_tensor

    def get_mean_tensor(
        self,
        frames: Optional[List[int]],
        getter: PromptGetterCallable
    ) -> Optional[Tensor]:
        """
        Gets a tensor from prompts using a callable.
        """
        import torch
        return_tensor = None
        total_weight = 0.0
        for prompt in self.prompts:
            tensor, weight = getter(prompt, frames)
            if tensor is not None and weight is not None and weight > 0:
                total_weight += weight
                if return_tensor is None:
                    return_tensor = (tensor * weight).unsqueeze(0)
                else:
                    return_tensor = torch.cat([return_tensor, (tensor * weight).unsqueeze(0)]) # type: ignore[unreachable]
        if return_tensor is not None:
            return torch.sum(return_tensor, 0) / total_weight
        return None

    def get_image_prompt_embeds(
        self,
        frames: Optional[List[int]]=None
    ) -> Tensor:
        """
        Gets image prompt embeds.
        """
        if self.image_prompt_embeds is None:
            raise RuntimeError("get_image_prompt_embeds called, but no image prompt embeds present.")
        import torch
        return_tensor: Optional[Tensor] = None
        for image_embeds in self.image_prompt_embeds:
            if frames is None:
                image_embeds = image_embeds[0]
            else:
                frame_length = image_embeds.shape[0]
                if frames[-1] <= frames[0]:
                    # Wraparound
                    image_embeds = torch.cat([
                        image_embeds[frames[0]:frame_length],
                        image_embeds[:frames[-1]]
                    ])
                else:
                    image_embeds = image_embeds[frames]
                # Collapse along frames
                image_embeds = image_embeds.mean(0)

            if return_tensor is None:
                return_tensor = image_embeds
            else:
                return_tensor = torch.cat(
                    [return_tensor, image_embeds],
                    dim=1
                )
        if return_tensor is None:
            raise RuntimeError("Prompt embeds could not be retrieved.")
        return return_tensor

    def get_image_uncond_prompt_embeds(
        self,
        frames: Optional[List[int]]=None
    ) -> Tensor:
        """
        Gets image unconditioning prompt embeds.
        """
        if self.image_uncond_prompt_embeds is None:
            raise RuntimeError("get_image_prompt_embeds called, but no image prompt embeds present.")
        import torch
        return_tensor: Optional[Tensor] = None
        for uncond_embeds in self.image_uncond_prompt_embeds:
            if frames is None:
                uncond_embeds = uncond_embeds[0]
            else:
                frame_length = uncond_embeds.shape[0]
                if frames[-1] <= frames[0]:
                    # Wraparound
                    uncond_embeds = torch.cat([
                        uncond_embeds[frames[0]:frame_length],
                        uncond_embeds[:frames[-1]]
                    ])
                else:
                    uncond_embeds = uncond_embeds[frames]
                # Collapse along frames
                uncond_embeds = uncond_embeds.mean(0)

            if return_tensor is None:
                return_tensor = uncond_embeds
            else:
                return_tensor = torch.cat(
                    [return_tensor, uncond_embeds],
                    dim=1
                )
        if return_tensor is None:
            raise RuntimeError("Prompt embeds could not be retrieved.")
        return return_tensor

    def get_embeds(self, frames: Optional[List[int]] = None) -> Optional[Tensor]:
        """
        Gets the encoded embeds.
        """
        import torch
        get_embeds: PromptGetterCallable = lambda prompt, frames: prompt.get_embeds(frames)
        method = self.get_mean_tensor if self.is_sdxl else self.get_stacked_tensor
        result = method(frames, get_embeds)
        if result is None:
            return None
        if self.is_sdxl and self.image_prompt_embeds is not None:
            result = torch.cat([result, self.get_image_prompt_embeds(frames)], dim=1)
        if self.is_sdxl and self.do_classifier_free_guidance:
            negative_result = self.get_negative_embeds(frames)
            if negative_result is None:
                negative_result = torch.zeros_like(result)
            result = torch.cat([negative_result, result], dim=0)
        elif not self.is_sdxl and self.image_prompt_embeds is not None and result is not None:
            if self.do_classifier_free_guidance:
                negative, positive = result.chunk(2)
            else:
                negative, positive = None, result
            positive = torch.cat([positive, self.get_image_prompt_embeds(frames)], dim=1)
            if self.do_classifier_free_guidance and negative is not None and self.image_uncond_prompt_embeds is not None:
                negative = torch.cat([negative, self.get_image_uncond_prompt_embeds(frames)], dim=1)
                return torch.cat([negative, positive], dim=0)
            else:
                return positive
        return result

    def get_negative_embeds(self, frames: Optional[List[int]] = None) -> Optional[Tensor]:
        """
        Gets the encoded negative embeds.
        """
        if not self.is_sdxl:
            return None
        import torch
        get_embeds: PromptGetterCallable = lambda prompt, frames: prompt.get_negative_embeds(frames)
        method = self.get_mean_tensor if self.is_sdxl else self.get_stacked_tensor
        result = method(frames, get_embeds)
        if self.is_sdxl and self.image_uncond_prompt_embeds is not None and result is not None:
            return torch.cat([result, self.get_image_uncond_prompt_embeds(frames)], dim=1)
        elif self.image_uncond_prompt_embeds is not None and result is not None:
            if self.do_classifier_free_guidance:
                negative, positive = result.chunk(2)
            else:
                negative, positive = result, None
            negative = torch.cat([negative, self.get_image_uncond_prompt_embeds(frames)], dim=1)
            return negative
        return result

    def get_pooled_embeds(self, frames: Optional[List[int]] = None) -> Optional[Tensor]:
        """
        Gets the encoded pooled embeds.
        """
        if not self.is_sdxl:
            return None
        get_embeds: PromptGetterCallable = lambda prompt, frames: prompt.get_pooled_embeds(frames)
        return self.get_mean_tensor(frames, get_embeds)

    def get_negative_pooled_embeds(self, frames: Optional[List[int]] = None) -> Optional[Tensor]:
        """
        Gets the encoded negative pooled embeds.
        """
        if not self.is_sdxl:
            return None
        get_embeds: PromptGetterCallable = lambda prompt, frames: prompt.get_negative_pooled_embeds(frames)
        return self.get_mean_tensor(frames, get_embeds)

    def get_add_text_embeds(self, frames: Optional[List[int]] = None) -> Optional[Tensor]:
        """
        Gets added text embeds for SDXL.
        """
        if not self.is_sdxl:
            return None
        import torch
        pooled_embeds = self.get_pooled_embeds(frames)
        if self.do_classifier_free_guidance and pooled_embeds is not None:
            negative_pooled_embeds = self.get_negative_pooled_embeds()
            if negative_pooled_embeds is None:
                negative_pooled_embeds = torch.zeros_like(pooled_embeds)
            pooled_embeds = torch.cat([negative_pooled_embeds, pooled_embeds], dim=0)
        return pooled_embeds

    @property
    def dtype(self) -> torch.dtype:
        """
        Gets the dtype of the encoded prompt.
        """
        if not self.prompts:
            raise ValueError("No prompts, cannot determine dtype.")
        return self.prompts[0].dtype
