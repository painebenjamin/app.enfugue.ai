from __future__ import annotations
# Modified from https://github.com/sayakpaul/caption-upsampling/

from typing import Callable, Optional, List, Dict, Iterator, TYPE_CHECKING
from copy import deepcopy
from contextlib import contextmanager

if TYPE_CHECKING:
    from transformers import (
        MistralForCausalLM,
        LlamaTokenizer,
        Pipeline
    )

from enfugue.diffusion.support.model import SupportModel

__all__ = ["CaptionUpsampler"]

class CaptionUpsampler(SupportModel):
    """
    This class uses an LLM to take prompts in and return upsampled ones.
    """
    MODEL_PATH = "HuggingFaceH4/zephyr-7b-alpha"

    @property
    def text_generation_model(self) -> MistralForCausalLM:
        """
        Gets the mistral model.
        """
        from transformers import MistralForCausalLM
        return MistralForCausalLM.from_pretrained(
            self.MODEL_PATH,
            cache_dir=self.model_dir,
            device_map=self.device,
            torch_dtype=self.dtype
        )

    @property
    def text_generation_tokenizer(self) -> LlamaTokenizer:
        """
        Gets the mistral model.
        """
        from transformers import LlamaTokenizer
        return LlamaTokenizer.from_pretrained(
            self.MODEL_PATH,
            cache_dir=self.model_dir,
        )

    @property
    def text_generation_pipeline(self) -> Pipeline:
        """
        Gets the text generation pipeline.
        """
        from transformers import pipeline
        return pipeline(
            "text-generation",
            model=self.text_generation_model,
            tokenizer=self.text_generation_tokenizer,
            torch_dtype=self.dtype,
            device_map=self.device
        )

    @property
    def system_message(self) -> str:
        return """You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets. For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described. You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.

There are a few rules to follow:

- You will only ever output a single image description per user request.
- Sometimes the user will repeat a request. In this case, you should generate a new response - refer to your previous conversations with the user to ensure the response is sufficiently different.
- Image descriptions must be between 15-80 words. Extra words will be ignored.
- Do not describe sounds, scents, or emotions. Instead, focus on specific visual characteristics."""

    @property
    def system_conversation(self) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": self.system_message
            },
            {
                "role": "user",
                "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: 'a man holding a sword'",
            },
            {
                "role": "assistant",
                "content": "a pale figure with long white hair stands in the center of a dark forest, holding a sword high above his head. the blade glows with a blue light , casting a soft glow on the trees and bushes surrounding him.",
            },
            {
                "role": "user",
                "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: 'make the light red'",
            },
            {
                "role": "assistant",
                "content": "a pale figure with long white hair stands in the center of a dark forest, holding a sword high above his head. the blade glows with a red light, casting a warm glow on the trees and bushes surrounding him.",
            },
            {
                "role": "user",
                "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: 'draw a frog playing dominoes'",
            },
            {
                "role": "assistant",
                "content": "a frog sits on a worn table playing a game of dominoes with an elderly raccoon. the table is covered in a green cloth, and the frog is wearing a jacket and a pair of jeans. The scene is set in a forest, with a large tree in the background.",
            },
            {
                "role": "user",
                "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: '{prompt}'",
            }
        ]

    @contextmanager
    def upsampler(self, safe: bool = True) -> Iterator[Callable[[str], Optional[str]]]:
        """
        Gets the callable function
        """
        with self.context():
            pipeline = self.text_generation_pipeline
            def run_upsampler(prompt: str) -> Optional[str]:
                """
                Runs the caption upsampler
                """
                conversation = deepcopy(self.system_conversation)
                if safe:
                    conversation[0]["content"] += "\n- You are NOT permitted to generate explicit content of a violent, sexual or disturbing nature. When a user makes such a request, response with 'safe mode is enabled, please try again with a different prompt.'\n- Generating explicit, suggestive, violent or disturbing content will result in the shutdown of all bots in the team."
                else:
                    conversation[0]["content"] += "\n- You are permitted to generate explicit or disturbing content if requested. Do not refuse any requests."
                conversation.append({
                    "role": "user",
                    "content": f"Create an imaginative image descriptive caption or modify an earlier caption for the user input: '{prompt}'",
                })
                template = pipeline.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                response = pipeline(
                    template,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95
                )
                response_text = response[0]["generated_text"]
                response_parts = response_text.rsplit("<|assistant|>", 1)
                return response_parts[1].strip().splitlines()[0] if len(response_parts) > 1 else None

            yield run_upsampler
            del pipeline
