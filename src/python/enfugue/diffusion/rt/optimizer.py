from typing import Optional, List

from polygraphy.backend.onnx.loader import fold_constants

from onnx_graphsurgeon import import_onnx, export_onnx

from onnx_graphsurgeon.ir.graph import Graph
from onnx.onnx_ml_pb2 import ModelProto as ONNXModel
from onnx.shape_inference import infer_shapes


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
        onnx_graph: ONNXModel,
        select_inputs: Optional[List[int]] = None,
        select_outputs: Optional[List[int]] = None,
        select_output_names: Optional[List[str]] = None,
    ) -> ONNXModel:
        """
        Creates an optimizer, runs it, and returns the optimized model
        """
        optimizer = Optimizer(onnx_graph)
        if select_inputs:
            optimizer.select_outputs(select_inputs)
        optimizer.cleanup()
        optimizer.fold_constants()
        optimizer.infer_shapes()
        if select_outputs:
            optimizer.select_outputs(select_outputs, select_output_names)
        return optimizer.cleanup(return_onnx=True)  # type: ignore
