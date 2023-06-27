import os
import glob
import PIL
import PIL.Image
import shutil

from typing import List, Dict, Any
from webob import Request, Response

from pibble.api.exceptions import BadRequestError, NotFoundError
from pibble.util.files import load_json
from pibble.ext.user.server.base import UserExtensionHandlerRegistry

from enfugue.api.controller.base import EnfugueAPIControllerBase
from enfugue.database.models import DiffusionModel
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.plan import DiffusionPlan, DiffusionStep, DiffusionNode

__all__ = ["EnfugueAPIModelsController"]


class EnfugueAPIModelsController(EnfugueAPIControllerBase):
    handlers = UserExtensionHandlerRegistry()

    @handlers.path("^/api/checkpoints$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured()
    def get_checkpoints(self, request: Request, response: Response) -> List[str]:
        """
        Gets installed checkpoints.
        """
        checkpoints = []
        checkpoints_dir = os.path.join(self.engine_root, "checkpoint")
        if os.path.exists(checkpoints_dir):
            checkpoints = os.listdir(checkpoints_dir)
        return checkpoints

    @handlers.path("^/api/lora$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured()
    def get_lora(self, request: Request, response: Response) -> List[str]:
        """
        Gets installed lora.
        """
        lora = []
        lora_dir = os.path.join(self.engine_root, "lora")
        if os.path.exists(lora_dir):
            lora = os.listdir(lora_dir)
        return lora

    @handlers.path("^/api/inversions$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured()
    def get_inversions(self, request: Request, response: Response) -> List[str]:
        """
        Gets installed textual inversions.
        """
        inversions = []
        inversions_dir = os.path.join(self.engine_root, "inversion")
        if os.path.exists(inversions_dir):
            inversions = os.listdir(inversions_dir)
        return inversions

    @handlers.path("^/api/tensorrt$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured("DiffusionModel", "read")
    def get_tensorrt_engines(self, request: Request, response: Response) -> List[Dict[str, Any]]:
        """
        Finds engines in the model directory and determines their metadata and status.
        """
        engines = []
        for engine in glob.glob(f"{self.engine_root}/**/engine.plan", recursive=True):
            engine_dir = os.path.abspath(os.path.dirname(engine))
            engine_type = os.path.basename(os.path.dirname(engine_dir))
            engine_key = os.path.basename(os.path.dirname(engine))
            engine_model = os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(engine_dir)))
            )
            engine_model_name = engine_model
            engine_metadata_path = os.path.join(engine_dir, "metadata.json")
            engine_used = False
            engine_used_by = []
            engine_lora = []
            engine_inversion = []
            engine_metadata = {}
            engine_size = 512

            if os.path.exists(engine_metadata_path):
                engine_metadata = load_json(engine_metadata_path)
                engine_lora = engine_metadata.get("lora", [])
                engine_inversion = engine_metadata.get("inversion", [])

                engine_lora_dict: Dict[str, float] = dict(
                    [(str(part[0]), float(part[1])) for part in engine_lora]
                )

                engine_size = engine_metadata.get("size", engine_size)
                engine_used = False
                maybe_name, _, maybe_inpainting = engine_model.rpartition("-")
                if maybe_inpainting == "inpainting":
                    engine_model_name = maybe_name

                possible_models = (
                    self.database.query(self.orm.DiffusionModel)
                    .filter(
                        (self.orm.DiffusionModel.model == f"{engine_model_name}.ckpt")
                        | (self.orm.DiffusionModel.model == f"{engine_model_name}.safetensors")
                    )
                    .filter(self.orm.DiffusionModel.size == engine_size)
                    .all()
                )

                for model in possible_models:
                    mismatched = False
                    matched_lora = []
                    matched_inversion = []
                    for lora in model.lora:
                        lora_name, ext = os.path.splitext(lora.model)
                        if engine_lora_dict.get(lora_name, None) != lora.weight:
                            mismatched = True
                            continue
                        else:
                            matched_lora.append(lora_name)
                    for inversion in model.inversion:
                        inversion_name, ext = os.path.splitext(inversion.model)
                        if inversion_name not in engine_inversion:
                            mismatched = True
                            continue
                        else:
                            matched_inversion.append(inversion_name)
                    if (
                        len(matched_lora) == len(engine_lora_dict.keys())
                        and len(matched_inversion) == len(engine_inversion)
                        and not mismatched
                    ):
                        engine_used_by.append(model.name)
                        engine_used = True

            engines.append(
                {
                    "key": engine_key,
                    "type": engine_type,
                    "model": engine_model,
                    "lora": engine_lora,
                    "inversion": engine_inversion,
                    "used": engine_used,
                    "used_by": list(set(engine_used_by)),
                    "size": engine_size,
                    "bytes": os.path.getsize(engine),
                }
            )
        return engines

    @handlers.path(
        "^/api/tensorrt/(?P<model_name>[^\/]+)/(?P<engine_type>[a-z]+)/(?P<engine_key>[a-zA-Z0-9]+)$"
    )
    @handlers.methods("DELETE")
    @handlers.format()
    @handlers.secured("DiffusionModel", "update")
    def delete_tensorrt_engine(
        self,
        request: Request,
        response: Response,
        model_name: str,
        engine_type: str,
        engine_key: str,
    ) -> None:
        """
        Removes an individual tensorrt engine.
        """
        engine_dir = os.path.join(
            self.engine_root, "models", model_name, "tensorrt", engine_type, engine_key
        )
        if not os.path.exists(engine_dir):
            raise NotFoundError(
                f"Couldn't find {engine_type} TensorRT engine for {model_name} with key {engine_key}"
            )
        shutil.rmtree(engine_dir)

    @handlers.path("^/api/models/(?P<model_name>[^\/]+)/tensorrt$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured("DiffusionModel", "read")
    def get_model_tensorrt_status(
        self, request: Request, response: Response, model_name: str
    ) -> Dict[str, Any]:
        """
        Gets TensorRT status for a particular model
        """
        model = (
            self.database.query(self.orm.DiffusionModel)
            .filter(self.orm.DiffusionModel.name == model_name)
            .one_or_none()
        )
        if not model:
            raise NotFoundError(f"No model named {model_name}")

        return DiffusionPipelineManager.get_tensorrt_status(
            self.engine_root,
            model.model,
            model.size,
            [(lora.model, lora.weight) for lora in model.lora],
            [inversion.model for inversion in model.inversion],
        )

    @handlers.path("^/api/models/(?P<model_name>[^\/]+)/tensorrt/(?P<network_name>[^\/]+)$")
    @handlers.methods("POST")
    @handlers.format()
    @handlers.secured("DiffusionModel", "update")
    def create_model_tensorrt_engine(
        self, request: Request, response: Response, model_name: str, network_name: str
    ) -> Dict[str, Any]:
        """
        Issues a job to create an engine.
        """
        plan = DiffusionPlan(**self.get_plan_kwargs_from_model(model_name, include_prompts=False))
        plan.build_tensorrt = True

        step = DiffusionStep(
            prompt="a green field, blue sky, outside", width=plan.size, height=plan.size
        )
        network_name = network_name.lower()

        if network_name == "inpaint_unet":
            step.image = PIL.Image.new("RGB", (plan.size, plan.size))
            step.mask = PIL.Image.new("RGB", (plan.size, plan.size))
        elif network_name == "controlled_unet":
            step.control_image = PIL.Image.new("RGB", (plan.size, plan.size))
            step.controlnet = "canny"  # Have to pick one, go with the first choice
        elif network_name != "unet":
            raise BadRequestError(f"Unknown or unsupported network {network_name}")

        build_metadata = {"model": model_name, "network": network_name}
        plan.nodes = [DiffusionNode([(0, 0), (plan.size, plan.size)], step)]
        plan.image_callback_steps = None  # Disable decoding
        return self.invoke(
            request.token.user.id,
            plan,
            save=False,
            disable_intermediate_decoding=True,
            communication_timeout=3600,
            metadata={"tensorrt_build": build_metadata},
        ).format()

    @handlers.path("^/api/models/(?P<model_name>[^\/]+)$")
    @handlers.methods("PUT")
    @handlers.format()
    @handlers.secured("DiffusionModel", "update")
    def modify_model(self, request: Request, response: Response, model_name: str) -> DiffusionModel:
        """
        Asks the pipeline manager for information about models.
        """
        model = (
            self.database.query(self.orm.DiffusionModel)
            .filter(self.orm.DiffusionModel.name == model_name)
            .one_or_none()
        )

        if not model:
            raise NotFoundError(f"No model named {model_name}")

        model.name = request.parsed.get("name", model.name)
        model.model = request.parsed.get("checkpoint", model.model)
        model.size = request.parsed.get("size", model.size)
        model.prompt = request.parsed.get("prompt", model.prompt)
        model.negative_prompt = request.parsed.get("negative_prompt", model.negative_prompt)

        for existing_lora in model.lora:
            self.database.delete(existing_lora)

        for existing_inversion in model.inversion:
            self.database.delete(existing_inversion)

        self.database.commit()

        for lora in request.parsed.get("lora", []):
            new_lora = self.orm.DiffusionModelLora(
                diffusion_model_name=model.name, model=lora["model"], weight=lora["weight"]
            )
            self.database.add(new_lora)

        for inversion in request.parsed.get("inversion", []):
            new_inversion = self.orm.DiffusionModelInversion(
                diffusion_model_name=model.name,
                model=inversion,
            )
            self.database.add(new_inversion)

        self.database.commit()
        return model

    @handlers.path("^/api/models/(?P<model_name>.+)$")
    @handlers.methods("DELETE")
    @handlers.secured("DiffusionModel", "delete")
    def delete_model(self, request: Request, response: Response, model_name: str) -> None:
        """
        Asks the pipeline manager for information about models.
        """
        model = (
            self.database.query(self.orm.DiffusionModel)
            .filter(self.orm.DiffusionModel.name == model_name)
            .one_or_none()
        )
        if not model:
            raise NotFoundError(f"No model named {model_name}")

        for lora in model.lora:
            self.database.delete(lora)
        for inversion in model.inversion:
            self.database.delete(inversion)

        self.database.commit()
        self.database.delete(model)
        self.database.commit()

    @handlers.path("^/api/models$")
    @handlers.methods("POST")
    @handlers.format()
    @handlers.secured("DiffusionModel", "create")
    def create_model(self, request: Request, response: Response) -> DiffusionModel:
        """
        Creates a new model.
        """
        try:
            new_model = self.orm.DiffusionModel(
                name=request.parsed["name"],
                model=request.parsed["checkpoint"],
                size=request.parsed.get("size", 512),
                prompt=request.parsed.get("prompt", ""),
                negative_prompt=request.parsed.get("negative_prompt", ""),
            )
            self.database.add(new_model)
            self.database.commit()
            for lora in request.parsed.get("lora", []):
                new_lora = self.orm.DiffusionModelLora(
                    diffusion_model_name=new_model.name, model=lora["model"], weight=lora["weight"]
                )
                self.database.add(new_lora)
                self.database.commit()
            for inversion in request.parsed.get("inversion", []):
                new_inversion = self.orm.DiffusionModelInversion(
                    diffusion_model_name=new_model.name, model=inversion
                )
                self.database.add(new_inversion)
                self.database.commit()
            return new_model
        except KeyError as ex:
            raise BadRequestError(f"Missing required parameter {ex}")
