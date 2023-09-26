from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, dtype

__all__ = ["Prompt", "EncodedPrompt", "EncodedPrompts"]

@dataclass
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
    def dtype(self) -> dtype:
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
class EncodedPrompts:
    """
    Holds any number of encoded prompts.
    """
    prompts: List[EncodedPrompt]
    is_sdxl: bool
    do_classifier_free_guidance: bool
    image_prompt_embeds: Optional[Tensor]
    image_uncond_prompt_embeds: Optional[Tensor]

    def get_prompt_tensor(
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
            if tensor is not None:
                total_weight += weight
                if return_tensor is None:
                    return_tensor = tensor.unsqueeze(0)
                else:
                    return_tensor = torch.cat([return_tensor, tensor.unsqueeze(0)], dim=0) # type: ignore[unreachable]
        if return_tensor is not None:
            return torch.sum(return_tensor, 0) / total_weight
        return None

    def get_embeds(self, frames: Optional[List[int]] = None) -> Optional[Tensor]:
        """
        Gets the encoded embeds.
        """
        import torch
        get_embeds: PromptGetterCallable = lambda prompt, frames: prompt.get_embeds(frames)
        result = self.get_prompt_tensor(frames, get_embeds)
        if result is None:
            return None
        if self.is_sdxl and self.image_prompt_embeds is not None:
            return torch.cat([result, self.image_prompt_embeds], dim=1)
        elif self.image_prompt_embeds is not None and result is not None:
            if self.do_classifier_free_guidance:
                negative, positive = result.chunk(2)
            else:
                negative, positive = None, result
            positive = torch.cat([positive, self.image_prompt_embeds], dim=1)
            if self.do_classifier_free_guidance and negative is not None and self.image_uncond_prompt_embeds is not None:
                negative = torch.cat([negative, self.image_uncond_prompt_embeds], dim=1)
                return torch.cat([negative, positive])
            else:
                return positive
        return result

    def get_negative_embeds(self, frames: Optional[List[int]] = None) -> Optional[Tensor]:
        """
        Gets the encoded negative embeds.
        """
        import torch
        get_embeds: PromptGetterCallable = lambda prompt, frames: prompt.get_negative_embeds(frames)
        result = self.get_prompt_tensor(frames, get_embeds)
        if self.is_sdxl and self.image_uncond_prompt_embeds is not None and result is not None:
            return torch.cat([result, self.image_uncond_prompt_embeds], dim=1)
        elif self.image_uncond_prompt_embeds is not None and result is not None:
            if self.do_classifier_free_guidance:
                negative, positive = result.chunk(2)
            else:
                negative, positive = result, None
            negative = torch.cat([negative, self.image_uncond_prompt_embeds], dim=1)
            return negative
        return result

    def get_pooled_embeds(self, frames: Optional[List[int]] = None) -> Optional[Tensor]:
        """
        Gets the encoded pooled embeds.
        """
        get_embeds: PromptGetterCallable = lambda prompt, frames: prompt.get_pooled_embeds(frames)
        return self.get_prompt_tensor(frames, get_embeds)

    def get_negative_pooled_embeds(self, frames: Optional[List[int]] = None) -> Optional[Tensor]:
        """
        Gets the encoded negative pooled embeds.
        """
        get_embeds: PromptGetterCallable = lambda prompt, frames: prompt.get_negative_pooled_embeds(frames)
        return self.get_prompt_tensor(frames, get_embeds)

    @property
    def dtype(self) -> dtype:
        """
        Gets the dtype of the encoded prompt.
        """
        if not self.prompts:
            raise ValueError("No prompts, cannot determine dtype.")
        return self.prompts[0].dtype
