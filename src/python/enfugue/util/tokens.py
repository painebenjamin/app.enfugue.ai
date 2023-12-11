__all__ = ["merge_tokens"]

def merge_tokens(*prompts_no_weight: str, **prompt_weights: float) -> str:
    """
    Merges any number of tokens for the compel library to parse later.
    """
    return "".join([
        f"({prompt}){weight:.2f}"
        for prompt, weight in prompt_weights.items()
    ] + [
        f"({prompt})1.0"
        for prompt in prompts_no_weight
    ])
