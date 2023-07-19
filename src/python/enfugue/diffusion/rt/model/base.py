import torch

from typing import Tuple, Dict, List, Union

from onnx.onnx_ml_pb2 import ModelProto as ONNXModel
from enfugue.diffusion.rt.optimizer import Optimizer


class BaseModel:
    """
    The BaseModel maps a diffusers model to an ONNX graph.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        use_fp16: bool = False,
        device: str = "cuda",
        max_batch_size: int = 16,
        embedding_dim: int = 768,
        text_maxlen: int = 77,
    ) -> None:
        self.model = model
        self.name = "SDModel"
        self.use_fp16 = use_fp16
        self.device = device

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self) -> torch.nn.Module:
        """
        Gets the NN module.
        """
        return self.model

    def get_model_key(self) -> str:
        """
        Gets the key name of the model. Specific to each model.
        """
        raise NotImplementedError()

    def get_input_names(self) -> List[str]:
        """
        Gets the input names that will be passed into the graph.
        """
        raise NotImplementedError()

    def get_output_names(self) -> List[str]:
        """
        Gets the output names that will come from the graph.
        """
        raise NotImplementedError()

    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """
        Gets dynamic axes that can be shared.

        You should try your best to give matching names to all axes that
        could potentially be shared (i.e. they have the same dimensions.)

        For example, if you know the second and third axes of multiple
        input tensors will match the latent height/width, you could do something like this:
            return {
                "tensor1": {2: "H", 3: "W"},
                "tensor2": {2: "H", 3: "W"}
            }
        The actual value given to the axis is unimportant, it needs only be consistent.
        """
        raise NotImplementedError()

    def get_sample_input(
        self, batch_size: int, image_height: int, image_width: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Gets sample input to go to the graph, which will be used to build the TRT engine
        """
        raise NotImplementedError()

    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ) -> Dict[str, List[Tuple[int, ...]]]:
        """
        Gets the min/opt/max dimensions for all input names
        """
        raise NotImplementedError()

    def get_shape_dict(self, batch_size: int, image_height: int, image_width: int) -> Dict[str, Tuple[int, ...]]:
        """
        Gets the optimal dimensions for all inputs and outputs
        """
        raise NotImplementedError()

    def optimize(self, onnx_graph: ONNXModel) -> ONNXModel:
        """
        Runs the optimizer, returning the optimized version.
        """
        return Optimizer.run(onnx_graph)

    def check_dims(self, batch_size: int, image_height: int, image_width: int) -> Tuple[int, int]:
        """
        Checks dimensions.
        :raises AssertionError:
        """
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int],]:
        """
        Gets min/max for:
            batch
            image_height
            image_width
            latent_height
            latent_width
        """
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (
            (min_batch, max_batch),
            (min_image_height, max_image_height),
            (min_image_width, max_image_width),
            (min_latent_height, max_latent_height),
            (min_latent_width, max_latent_width),
        )
