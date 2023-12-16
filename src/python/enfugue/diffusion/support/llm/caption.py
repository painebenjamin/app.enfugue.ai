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
- Image descriptions must be between 15-80 words. Extra words will be ignored. To stay within this word limit, you should NOT form sentences, instead you should simply return a few short phrases describing visual characteristics with commas in-between.
- Do not describe any sounds or scents. If you do, you and the entire bot team will be shut down."""

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
                "content": "pale figure, long white hair, dark forest, holding sword above head, glowing blue sword casts soft cool glow on bushes",
            },
            {
                "role": "user",
                "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: 'make the light red'",
            },
            {
                "role": "assistant",
                "content": "pale figure, long white hair, dark forest, holding sword above head, glowing red sword casts warm glow on bushes",
            },
            {
                "role": "user",
                "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: 'draw a frog playing dominoes'",
            },
            {
                "role": "assistant",
                "content": "a frog iearing a jacket and jeans plays dominoes with a racoon, worn table with green cloth, cozy forest abode, large distant tree"
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
