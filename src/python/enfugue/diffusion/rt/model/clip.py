import torch

from typing import List, Dict, Tuple

from onnx.onnx_ml_pb2 import ModelProto as ONNXModel

from enfugue.diffusion.rt.model.base import BaseModel
from enfugue.diffusion.rt.optimizer import Optimizer

__all__ = ["CLIP"]


class CLIP(BaseModel):
    """
    This is the model for CLiP (text embeddings)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        use_fp16: bool = False,
        device: str = "cuda",
        max_batch_size: int = 16,
        embedding_dim: int = 768,
        text_maxlen: int = 77,
    ):
        super(CLIP, self).__init__(
            model,
            use_fp16=False,  # Override
            text_maxlen=77,  # Override
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        )
        self.name = "CLIP"

    def get_model_key(self) -> str:
        """
        Gets the unique key for this model
        """
        return "clip"

    def get_input_names(self) -> List[str]:
        """
        Gets the input names that will be passed into the graph.
        """
        return ["input_ids"]

    def get_output_names(self) -> List[str]:
        """
        Gets the output names that will come from the graph.
        """
        return ["text_embeddings", "pooled_output"]

    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """
        Gets dynamic axes that can be shared.
        """
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ) -> Dict[str, List[Tuple[int, ...]]]:
        """
        Gets the optimal dimensions for all inputs
        """
        self.check_dims(batch_size, image_height, image_height)
        (min_batch, max_batch), _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        return {
            "input_ids": [
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        }

    def get_shape_dict(self, batch_size: int, image_height: int, image_width: int) -> Dict[str, Tuple[int, ...]]:
        self.check_dims(batch_size, image_height, image_width)
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }

    def get_sample_input(self, batch_size: int, image_height: int, image_width: int) -> torch.Tensor:
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph: ONNXModel) -> ONNXModel:
        return Optimizer.run(
            onnx_graph,
            select_inputs=[0],
            select_outputs=[0],
            select_output_names=["text_embeddings"],
        )
