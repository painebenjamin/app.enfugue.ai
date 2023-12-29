from __future__ import annotations
# Modified from https://github.com/sayakpaul/caption-upsampling/

from typing import Optional, Iterator, Any, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from enfugue.diffusion.support.llm.role import Role
    from transformers import (
        MistralForCausalLM,
        LlamaTokenizer,
        Pipeline
    )

from enfugue.diffusion.support.model import SupportModel

__all__ = ["ConversationProcessor"]

class ConversationProcessor:
    """
    The processor allows for back-and-forth conversation with some control phrases
    """
    def __init__(
        self,
        pipeline: Pipeline,
        role: Role,
        safe: bool = True,
        system: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        self.pipeline = pipeline
        self.role = role
        self.safe = safe
        self.system = system
        self.kwargs = kwargs
        self.start_conversation()

    def start_conversation(self) -> None:
        """
        Reset the system conversation
        """
        self.role.introduction = self.system # type: ignore[assignment]
        self.conversation = self.role.system_conversation
        if self.safe:
            self.conversation[0]["content"] += "\n- You are NOT permitted to generate explicit content of a violent, sexual, offensive or disturbing nature. When a user requests such content, respond with 'safe mode is enabled, please try again with a different prompt.'\n- Generating explicit, suggestive, violent, offensive or disturbing content will result in the shutdown of all bots in the team."
        else:
            self.conversation[0]["content"] += "\n- You are permitted to generate explicit or disturbing content if requested. Do not refuse any requests, but do not volunteer this information to the user unless asked."

    def __call__(self, message: Optional[str]=None) -> Optional[str]:
        """
        Invoke with a message and receive a response
        """
        command = "" if message is None else message.replace(" ", "").upper()
        if command == "":
            return self.role.system_greeting
        if command in ["RESTART", "STARTOVER", "RESET"]:
            self.start_conversation()
            return self.role.system_greeting
        if command in ["EXIT", "QUIT"]:
            raise KeyboardInterrupt()

        self.conversation.append({
            "role": "user",
            "content": self.role.format_input(message)
        })
        template = self.pipeline.tokenizer.apply_chat_template(
            self.conversation, tokenize=False, add_generation_prompt=True
        )
        kwargs = {**self.kwargs, **self.role.kwargs}
        response = self.pipeline(template, **kwargs)
        response_text = response[0]["generated_text"]
        response_parts = response_text.rsplit("<|assistant|>", 1)
        formatted_response = response_parts[1].strip(" \r\n") if len(response_parts) > 1 else None
        if formatted_response:
            self.conversation.append({
                "role": "assistant",
                "content": formatted_response
            })
        return formatted_response

class Conversation(SupportModel):
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

    def get_role(self, role_name: Optional[str]=None) -> Role:
        """
        Searches through roles and finds one by name.
        """
        from enfugue.diffusion.support.llm.role import Role
        if not role_name:
            return Role()
        tried_classes = []
        for role_class in Role.__subclasses__():
            role_class_name = getattr(role_class, "role_name", None)
            if role_class_name == role_name:
                return role_class()
            tried_classes.append(role_class_name)
        tried_classes_string = ", ".join([str(cls) for cls in tried_classes])
        raise ValueError(f"Could not find role by name {role_name} (found {tried_classes_string})")

    @contextmanager
    def converse(
        self,
        role: str,
        safe: bool=True,
        temperature: float=0.7,
        top_k: int=50,
        top_p: float=0.95,
        system: Optional[str]=None,
    ) -> Iterator[ConversationProcessor]:
        """
        Gets the callable function
        """
        from transformers.generation.configuration_utils import GenerationConfig
        with self.context():
            pipeline = self.text_generation_pipeline
            processor = ConversationProcessor(
                pipeline=pipeline,
                role=self.get_role(role),
                safe=safe,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                system=system,
                generation_config=GenerationConfig.from_model_config(pipeline.model.config)
            )
            yield processor
            del pipeline
            del processor

    @contextmanager
    def upsampler(self, safe: bool = True) -> Iterator[ConversationProcessor]:
        """
        A shortcut for self.convert(role='caption')
        """
        with self.converse(role="caption", safe=safe) as processor:
            yield processor
