from __future__ import annotations

import os
import numpy as np

from typing import Tuple, Any, Dict, Iterable

from glob import glob
from numpy.linalg import norm

from insightface.model_zoo import model_zoo as ModelZoo
from insightface.app.common import Face

__all__ = ['FaceAnalyzer']

class FaceAnalyzer:
    def __init__(self, model_dir: str, **kwargs: Any) -> None:
        self.models: Dict[str, Any] = {}
        self.model_dir = model_dir
        onnx_files = glob(os.path.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)

        for onnx_file in onnx_files:
            model = ModelZoo.get_model(onnx_file, **kwargs)
            assert model is not None, "ONNX model {onnx_file} not expected"
            assert model.taskname not in self.models, "ONNX model {onnx_file} provides an already-provided for task {model.taskname}"
            self.models[model.taskname] = model
        assert "detection" in self.models, "Detection model missing!"

    @property
    def det_model(self) -> Any:
        """
        Returns the detection model
        """
        return self.models["detection"]

    def prepare(
        self,
        ctx_id: int,
        det_thresh: float=0.5,
        det_size: Tuple[int, int]=(640, 640)
    ) -> None:
        """
        Prepares inference context
        """
        self.det_thresh = det_thresh
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == "detection":
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img: np.ndarray, max_num: int=0) -> Iterable[Face]:
        """
        Gets detected faces
        """
        bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric="default")
        if bboxes.shape[0] == 0:
            return []

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None

            if kpss is not None:
                kps = kpss[i]

            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == "detection":
                    continue
                model.get(img, face)
            yield face
