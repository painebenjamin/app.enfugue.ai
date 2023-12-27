from __future__ import annotations

from typing import List, Dict, TypedDict, Any, Optional

__all__ = ["Role", "MessageDict"]

class MessageDict(TypedDict):
    role: str
    content: str

class Role:
    """
    This class allows for defining roles for LLMs to adopt
    """
    role_name = "default"

    def format_input(self, message: Optional[str]) -> str:
        """
        Given user input, format the message to the bot
        """
        return "" if message is None else message

    @property
    def system_greeting(self) -> str:
        """
        The greeting displayed at the start of conversations
        """
        return "Hello, I am your personal assistant. How can I help today?"

    @property
    def system_message(self) -> str:
        """
        The message told to the bot at the beginning instructing it
        """
        return "You are a general-purpose assistant bot. A user will prompt you with various questions or instructions, and you are to respond as helpfully as possible to the best of your ability. At times, the user may request that you perform a different function. You should interpret this request and comply by operating along the user's provided guidelines."

    @property
    def max_new_tokens(self) -> int:
        """
        The maximum length of the response
        """
        return 2**16

    @property
    def kwargs(self) -> Dict[str, Any]:
        """
        The arguments passed to the invocation
        """
        return {
            "max_new_tokens": self.max_new_tokens
        }

    @property
    def system_examples(self) -> List[MessageDict]:
        """
        Examples given to the system to refine behavior
        """
        return []

    @property
    def system_conversation(self) -> List[MessageDict]:
        """
        The conversation to begin the invocation
        """
        return [{
            "role": "system",
            "content": self.system_message
        }] + self.system_examples # type: ignore[return-value]
