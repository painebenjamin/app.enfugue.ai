import torch

__all__ = [
    "ImageProjectionModel",
    "MLPProjectionModel",
    "FaceIDMLPProjectionModel",
    "ProjectionPlusModel",
]

class MLPProjectionModel(torch.nn.Module):
    """
    SD model with image prompt
    """
    def __init__(
        self,
        cross_attention_dim: int = 1024,
        clip_embeddings_dim: int = 1024
    ) -> None:
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

class FaceIDMLPProjectionModel(torch.nn.Module):
    """
    SD model with image prompt
    """
    def __init__(
        self,
        cross_attention_dim: int = 768,
        id_embeddings_dim: int = 512,
        num_tokens: int = 4
    ) -> None:
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


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

class ProjectionPlusModel(torch.nn.Module):
    def __init__(
        self,
        cross_attention_dim: int=768,
        id_embeddings_dim: int=512,
        clip_embeddings_dim: int=1280,
        num_tokens: int=4
    ) -> None:
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        from enfugue.diffusion.support.ip.resampler import FacePerceiverResampler # type: ignore
        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )
        
    def forward(
        self,
        id_embeds: torch.Tensor,
        clip_embeds: torch.Tensor,
        shortcut: bool=False,
        scale: float=1.0
    ) -> torch.Tensor:
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out
