from typing import Dict, Union, Iterator, Tuple

__all__ = ["TokenMerger"]


class TokenMerger:
    """
    This class allows for merging prompts in a weighted manner.
    """

    tokens: Dict[str, Union[int, float]]

    def __init__(self, *initial_phrases: str) -> None:
        self.tokens = {}
        for phrase in initial_phrases:
            self.add(phrase)

    def add(self, phrase: str, weight: Union[int, float] = 1) -> None:
        """
        Adds a token to the weights.
        """
        tokens = phrase.split(",")
        token_count = len(tokens)
        for i, token in enumerate(tokens):
            token_weight_add = min(token.count("("), token.count(")"))
            token_weight_remove = min(token.count("["), token.count("]"))
            token_weight_mult = token_weight_add - token_weight_remove
            token_immediacy_weight = max((token_count - i) / token_count, 0.5)

            if token_weight_mult < 1:
                weight /= (token_weight_mult * -1) + 1
            elif token_weight_mult > 1:
                weight *= token_weight_mult + 1
            weight *= token_immediacy_weight
            token = token.strip(" (){}[]")
            if token not in self.tokens:
                self.tokens[token] = weight
            self.tokens[token] += weight

    def __iter__(self) -> Iterator[Tuple[str, Union[int, float]]]:
        """
        Iterates over tokens in-order.
        """
        total_weight = sum(self.tokens.values())
        for key, value in reversed(sorted(self.tokens.items(), key=lambda kv: kv[1])):
            yield (key, value / total_weight)

    def __str__(self) -> str:
        """
        Stringifies the tokens.
        """
        return ",".join([token for token, weight in iter(self)])
