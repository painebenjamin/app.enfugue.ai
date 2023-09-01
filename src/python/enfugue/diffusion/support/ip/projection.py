import torch

__all__ = [
    "ImageProjectionModel"
]

class ImageProjectionModel(torch.nn.Module):
    """
    An NN module for projection/normalizing embeds
    """
    def __init__(
        self,
        cross_attention_dim: int = 768,
        clip_embeddings_dim: int = 1024,
        clip_extra_context_tokens: int = 4
    ) -> None:
        super(ImageProjectionModel, self).__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        On forward, grab embeds
        """
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

