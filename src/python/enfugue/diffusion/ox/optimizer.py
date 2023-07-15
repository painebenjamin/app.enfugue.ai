from typing import Optional, List

import torch
import onnx

from polygraphy.backend.onnx.loader import fold_constants

from onnx_graphsurgeon import import_onnx, export_onnx

from onnx_graphsurgeon.ir.graph import Graph
from onnx.onnx_ml_pb2 import ModelProto as ONNXModel
from onnx.shape_inference import infer_shapes

from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import optimize_model

class Optimizer:
    graph: Graph

    def __init__(self, onnx_model: ONNXModel) -> None:
        self.graph = import_onnx(onnx_model)

    def select_outputs(self, to_select: List[int], names: Optional[List[str]] = None) -> None:
        """
        Selects and reindexes  graph outputs.
        """
        self.graph.outputs = [self.graph.outputs[i] for i in to_select]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def cleanup(self, return_onnx: bool = False) -> Optional[ONNXModel]:
        """
        Cleans up the graph and re-sorts.
        """
        self.graph.cleanup().toposort()
        if return_onnx:
            return export_onnx(self.graph)
        return None

    def fold_constants(self, return_onnx: bool = False) -> Optional[ONNXModel]:
        """
        Folds the constants from the graph, then re-imports.
        """
        onnx_graph = fold_constants(export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph
        return None

    def infer_shapes(self, return_onnx: bool = False) -> Optional[ONNXModel]:
        """
        Runs shape inference on the graph and re-imports.
        """
        onnx_graph = export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = infer_shapes(onnx_graph)

        self.graph = import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph
        return None

    @staticmethod
    def run(
        onnx_graph: str,
        select_inputs: Optional[List[int]] = None,
        select_outputs: Optional[List[int]] = None,
        select_output_names: Optional[List[str]] = None,
        use_fp16: bool = False
    ) -> ONNXModel:
        """
        Creates an optimizer, runs it, and returns the optimized model
        """
        if torch.cuda.is_available():
            optimizer = Optimizer(onnx.load(onnx_graph))
            if select_inputs:
                optimizer.select_outputs(select_inputs)
            optimizer.cleanup()
            optimizer.fold_constants()
            optimizer.infer_shapes()
            if select_outputs:
                optimizer.select_outputs(select_outputs, select_output_names)
            return optimizer.cleanup(return_onnx=True)  # type: ignore
        else:
            optimization_options = FusionOptions(None)
            optimization_options.enable_group_norm = False
            optimization_options.enable_nhwc_conv = False
            optimization_options.enable_qordered_matmul = False
            optimizer = optimize_model(
                input=onnx_graph,
                opt_level=0,
                optimization_options=optimization_options,
                use_gpu=False,
                only_onnxruntime=False
            )

            if use_fp16:
                optimizer.convert_float_to_float16( # type: ignore
                    keep_io_types=True, disable_shape_infer=True, op_block_list=['RandomNormalLike']
                )

            optimizer.topological_sort() # type: ignore
            return optimizer.model # type: ignore
