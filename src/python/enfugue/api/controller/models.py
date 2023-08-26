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

from enfugue.util import find_files_in_directory
from enfugue.api.controller.base import EnfugueAPIControllerBase
from enfugue.database.models import DiffusionModel
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.plan import DiffusionPlan, DiffusionStep, DiffusionNode
from enfugue.diffusion.constants import (
    DEFAULT_MODEL,
    DEFAULT_INPAINTING_MODEL,
    DEFAULT_SDXL_MODEL,
    DEFAULT_SDXL_REFINER
)

__all__ = ["EnfugueAPIModelsController"]


class EnfugueAPIModelsController(EnfugueAPIControllerBase):
    handlers = UserExtensionHandlerRegistry()

    MODEL_DEFAULT_FIELDS = [
        "width",
        "height",
        "chunking_size",
        "chunking_blur",
        "num_inference_steps",
        "guidance_scale",
        "refiner_denoising_strength",
        "refiner_guidance_scale",
        "refiner_aesthetic_score",
        "refiner_negative_aesthetic_score",
        "prompt_2",
        "negative_prompt_2",
        "upscale",
        "outscale",
        "upscale_iterative",
        "upscale_pipeline",
        "upscale_method",
        "upscale_diffusion",
        "upscale_diffusion_prompt",
        "upscale_diffusion_prompt_2",
        "upscale_diffusion_negative_prompt",
        "upscale_diffusion_negative_prompt_2",
        "upscale_diffusion_controlnet",
        "upscale_diffusion_steps",
        "upscale_diffusion_strength",
        "upscale_diffusion_guidance_scale",
        "upscale_diffusion_chunking_size",
        "upscale_diffusion_scale_chunking_size",
        "upscale_diffusion_scale_chunking_blur"
    ]

    DEFAULT_CHECKPOINTS = [
        os.path.basename(DEFAULT_MODEL),
        os.path.basename(DEFAULT_INPAINTING_MODEL),
        os.path.basename(DEFAULT_SDXL_MODEL),
        os.path.basename(DEFAULT_SDXL_REFINER),
    ]

    @handlers.path("^/api/checkpoints$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured()
    def get_checkpoints(self, request: Request, response: Response) -> List[str]:
        """
        Gets installed checkpoints.
        """
        checkpoints_dir = self.configuration.get(
            "enfugue.engine.checkpoint", os.path.join(self.engine_root, "checkpoint")
        )
        checkpoints = [
            os.path.basename(filename)
            for filename in find_files_in_directory(checkpoints_dir)
        ]
        for checkpoint in self.DEFAULT_CHECKPOINTS:
            if checkpoint not in checkpoints:
                checkpoints.append(checkpoint)
        return checkpoints

    @handlers.path("^/api/lora$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured()
    def get_lora(self, request: Request, response: Response) -> List[str]:
        """
        Gets installed lora.
        """
        lora_dir = self.configuration.get("enfugue.engine.lora", os.path.join(self.engine_root, "lora"))
        lora = [
            os.path.basename(filename)
            for filename in find_files_in_directory(lora_dir)
        ]
        return lora

    @handlers.path("^/api/lycoris$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured()
    def get_lycoris(self, request: Request, response: Response) -> List[str]:
        """
        Gets installed lycoris/locon
        """
        lycoris = []
        lycoris_dir = self.configuration.get("enfugue.engine.lycoris", os.path.join(self.engine_root, "lycoris"))
        lycoris = [
            os.path.basename(filename)
            for filename in find_files_in_directory(lycoris_dir)
        ]
        return lycoris

    @handlers.path("^/api/inversions$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured()
    def get_inversions(self, request: Request, response: Response) -> List[str]:
        """
        Gets installed textual inversions.
        """
        inversions_dir = self.configuration.get("enfugue.engine.inversion", os.path.join(self.engine_root, "inversion"))
        inversions = [
            os.path.basename(filename)
            for filename in find_files_in_directory(inversions_dir)
        ]
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
        for engine in glob.glob(f"{self.engine_root}/tensorrt/**/engine.plan", recursive=True):
            engine_dir = os.path.abspath(os.path.dirname(engine))
            engine_type = os.path.basename(os.path.dirname(engine_dir))
            engine_key = os.path.basename(os.path.dirname(engine))
            engine_model = os.path.basename(os.path.dirname(os.path.dirname(engine_dir)))
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
                engine_lycoris = engine_metadata.get("lycoris", [])
                engine_inversion = engine_metadata.get("inversion", [])

                engine_lora_dict: Dict[str, float] = dict([(str(part[0]), float(part[1])) for part in engine_lora])
                engine_lycoris_dict: Dict[str, float] = dict(
                    [(str(part[0]), float(part[1])) for part in engine_lycoris]
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
                    matched_lycoris = []
                    matched_inversion = []
                    for lora in model.lora:
                        lora_name, ext = os.path.splitext(lora.model)
                        if engine_lora_dict.get(lora_name, None) != lora.weight:
                            mismatched = True
                            continue
                        else:
                            matched_lora.append(lora_name)
                    for lycoris in model.lycoris:
                        lycoris_name, ext = os.path.splitext(lycoris.model)
                        if engine_lycoris_dict.get(lycoris_name, None) != lycoris.weight:
                            mismatched = True
                            continue
                        else:
                            matched_lycoris.append(lycoris_name)
                    for inversion in model.inversion:
                        inversion_name, ext = os.path.splitext(inversion.model)
                        if inversion_name not in engine_inversion:
                            mismatched = True
                            continue
                        else:
                            matched_inversion.append(inversion_name)
                    if (
                        len(matched_lora) == len(engine_lora_dict.keys())
                        and len(matched_lycoris) == len(engine_lycoris_dict.keys())
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
                    "lycoris": engine_lycoris,
                    "inversion": engine_inversion,
                    "used": engine_used,
                    "used_by": list(set(engine_used_by)),
                    "size": engine_size,
                    "bytes": os.path.getsize(engine),
                }
            )
        return engines

    @handlers.path("^/api/tensorrt/(?P<model_name>[^\/]+)/(?P<engine_type>[a-z]+)/(?P<engine_key>[a-zA-Z0-9]+)$")
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
        engine_dir = os.path.join(self.engine_root, "tensorrt", model_name, engine_type, engine_key)
        if not os.path.exists(engine_dir):
            raise NotFoundError(f"Couldn't find {engine_type} TensorRT engine for {model_name} with key {engine_key}")
        shutil.rmtree(engine_dir)

    @handlers.path("^/api/models/(?P<model_name>[^\/]+)/status$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured("DiffusionModel", "read")
    def get_model_status(self, request: Request, response: Response, model_name: str) -> Dict[str, Any]:
        """
        Gets status for a particular model
        """
        model = (
            self.database.query(self.orm.DiffusionModel)
            .filter(self.orm.DiffusionModel.name == model_name)
            .one_or_none()
        )
        if not model:
            raise NotFoundError(f"No model named {model_name}")

        main_model_status = DiffusionPipelineManager.get_status(
            self.engine_root,
            model.model,
            model.size,
            [(lora.model, lora.weight) for lora in model.lora],
            [(lycoris.model, lycoris.weight) for lycoris in model.lycoris],
            [inversion.model for inversion in model.inversion],
        )

        if model.inpainter:
            inpainter_model = model.inpainter[0].model
            inpainter_model_status = DiffusionPipelineManager.get_status(
                self.engine_root,
                model.inpainter[0].model,
                model.size,
            )
        else:
            model_name, ext = os.path.splitext(model.model)
            inpainter_model = f"{model_name}-inpainting{ext}"
            inpainter_model_status = DiffusionPipelineManager.get_status(
                self.engine_root,
                inpainter_model,
                model.size,
            )

        if model.refiner:
            refiner_model = model.refiner[0].model
            refiner_model_status = DiffusionPipelineManager.get_status(
                self.engine_root,
                refiner_model,
                model.size,
            )
        else:
            refiner_model = None
            refiner_model_status = None

        return {
            "model": model.model,
            "refiner": refiner_model,
            "inpainter": inpainter_model,
            "tensorrt": {
                "base": main_model_status,
                "inpainter": inpainter_model_status,
                "refiner": refiner_model_status,
            },
        }

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

        step = DiffusionStep(prompt="a green field, blue sky, outside", width=plan.size, height=plan.size)
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
        return self.invoke(
            request.token.user.id,
            plan,
            save=False,
            disable_intermediate_decoding=True,
            metadata={"tensorrt_build": build_metadata},
        ).format()

    @handlers.path("^/api/models/(?P<model_name>[^\/]+)$")
    @handlers.methods("PATCH")
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

        for existing_lora in model.lora:
            self.database.delete(existing_lora)

        for existing_lycoris in model.lycoris:
            self.database.delete(existing_lycoris)

        for existing_inversion in model.inversion:
            self.database.delete(existing_inversion)

        for existing_scheduler in model.scheduler:
            self.database.delete(existing_scheduler)

        for existing_refiner in model.refiner:
            self.database.delete(existing_refiner)

        for existing_inpainter in model.inpainter:
            self.database.delete(existing_inpainter)

        for existing_config in model.config:
            self.database.delete(existing_config)

        for existing_vae in model.vae:
            self.database.delete(existing_vae)
        
        for existing_vae in model.refiner_vae:
            self.database.delete(existing_vae)
        
        for existing_vae in model.inpainter_vae:
            self.database.delete(existing_vae)

        self.database.commit()
        
        model.name = request.parsed.get("name", model.name)
        model.model = request.parsed.get("checkpoint", model.model)
        model.size = request.parsed.get("size", model.size)
        model.prompt = request.parsed.get("prompt", model.prompt)
        model.negative_prompt = request.parsed.get("negative_prompt", model.negative_prompt)
        
        self.database.commit()

        refiner = request.parsed.get("refiner", None)
        if refiner:
            self.database.add(
                self.orm.DiffusionModelRefiner(
                    diffusion_model_name=model_name, model=refiner, size=request.parsed.get("refiner_size", None)
                )
            )

        inpainter = request.parsed.get("inpainter", None)
        if inpainter:
            self.database.add(
                self.orm.DiffusionModelInpainter(
                    diffusion_model_name=model_name, model=inpainter, size=request.parsed.get("inpainter_size", None)
                )
            )

        scheduler = request.parsed.get("scheduler", None)
        if scheduler:
            self.database.add(
                self.orm.DiffusionModelScheduler(
                    diffusion_model_name=model_name,
                    name=scheduler,
                )
            )

        vae = request.parsed.get("vae", None)
        if vae:
            self.database.add(
                self.orm.DiffusionModelVAE(
                    diffusion_model_name=model_name,
                    name=vae,
                )
            )
        
        refiner_vae = request.parsed.get("refiner_vae", None)
        if refiner_vae:
            self.database.add(
                self.orm.DiffusionModelRefinerVAE(
                    diffusion_model_name=model_name,
                    name=refiner_vae,
                )
            )
        
        inpainter_vae = request.parsed.get("inpainter_vae", None)
        if inpainter_vae:
            self.database.add(
                self.orm.DiffusionModelInpainterVAE(
                    diffusion_model_name=model_name,
                    name=inpainter_vae,
                )
            )

        for lora in request.parsed.get("lora", []):
            new_lora = self.orm.DiffusionModelLora(
                diffusion_model_name=model.name, model=lora["model"], weight=lora["weight"]
            )
            self.database.add(new_lora)

        for lycoris in request.parsed.get("lycoris", []):
            new_lycoris = self.orm.DiffusionModelLycoris(
                diffusion_model_name=model.name, model=lycoris["model"], weight=lycoris["weight"]
            )
            self.database.add(new_lycoris)

        for inversion in request.parsed.get("inversion", []):
            new_inversion = self.orm.DiffusionModelInversion(
                diffusion_model_name=model.name,
                model=inversion,
            )
            self.database.add(new_inversion)

        for field_name in self.MODEL_DEFAULT_FIELDS:
            field_value = request.parsed.get(field_name, None)
            if field_value is not None:
                new_config = self.orm.DiffusionModelDefaultConfiguration(
                    diffusion_model_name=model.name, configuration_key=field_name, configuration_value=field_value
                )
                self.database.add(new_config)

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
        for lycoris in model.lycoris:
            self.database.delete(lycoris)
        for inversion in model.inversion:
            self.database.delete(inversion)
        for refiner in model.refiner:
            self.database.delete(refiner)
        for inpainter in model.inpainter:
            self.database.delete(inpainter)
        for scheduler in model.scheduler:
            self.database.delete(scheduler)
        for vae in model.vae:
            self.database.delete(vae)
        for vae in model.refiner_vae:
            self.database.delete(vae)
        for vae in model.inpainter_vae:
            self.database.delete(vae)
        for config in model.config:
            self.database.delete(config)

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
            refiner = request.parsed.get("refiner", None)
            if refiner:
                new_refiner = self.orm.DiffusionModelRefiner(
                    diffusion_model_name=new_model.name, model=refiner, size=request.parsed.get("refiner_size", None)
                )
                self.database.add(new_refiner)
                self.database.commit()
            inpainter = request.parsed.get("inpainter", None)
            if inpainter:
                new_inpainter = self.orm.DiffusionModelInpainter(
                    diffusion_model_name=new_model.name,
                    model=inpainter,
                    size=request.parsed.get("inpainter_size", None),
                )
                self.database.add(new_inpainter)
                self.database.commit()
            scheduler = request.parsed.get("scheduler", None)
            if scheduler:
                new_scheduler = self.orm.DiffusionModelScheduler(diffusion_model_name=new_model.name, name=scheduler)
                self.database.add(new_scheduler)
                self.database.commit()
            vae = request.parsed.get("vae", None)
            if vae:
                new_vae = self.orm.DiffusionModelVAE(diffusion_model_name=new_model.name, name=vae)
                self.database.add(new_vae)
                self.database.commit()
            refiner_vae = request.parsed.get("refiner_vae", None)
            if refiner_vae:
                new_refiner_vae = self.orm.DiffusionModelRefinerVAE(diffusion_model_name=new_model.name, name=refiner_vae)
                self.database.add(new_refiner_vae)
                self.database.commit()
            inpainter_vae = request.parsed.get("inpainter_vae", None)
            if inpainter_vae:
                new_inpainter_vae = self.orm.DiffusionModelInpainterVAE(diffusion_model_name=new_model.name, name=inpainter_vae)
                self.database.add(new_inpainter_vae)
                self.database.commit()
            for lora in request.parsed.get("lora", []):
                new_lora = self.orm.DiffusionModelLora(
                    diffusion_model_name=new_model.name, model=lora["model"], weight=lora["weight"]
                )
                self.database.add(new_lora)
                self.database.commit()
            for lycoris in request.parsed.get("lycoris", []):
                new_lycoris = self.orm.DiffusionModelLycoris(
                    diffusion_model_name=new_model.name, model=lycoris["model"], weight=lycoris["weight"]
                )
                self.database.add(new_lycoris)
                self.database.commit()
            for inversion in request.parsed.get("inversion", []):
                new_inversion = self.orm.DiffusionModelInversion(diffusion_model_name=new_model.name, model=inversion)
                self.database.add(new_inversion)
                self.database.commit()
            for field_name in self.MODEL_DEFAULT_FIELDS:
                field_value = request.parsed.get(field_name, None)
                if field_value is not None:
                    new_config = self.orm.DiffusionModelDefaultConfiguration(
                        diffusion_model_name=new_model.name,
                        configuration_key=field_name,
                        configuration_value=field_value,
                    )
                    self.database.add(new_config)
                    self.database.commit()
            return new_model
        except KeyError as ex:
            raise BadRequestError(f"Missing required parameter {ex}")

    @handlers.path("^/api/model-options$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured("DiffusionModel", "read")
    def get_all_models(self, request: Request, response: Response) -> List[Dict[str, Any]]:
        """
        Gets all checkpoints and model names for the picker.
        """
        checkpoints_dir = self.configuration.get(
            "enfugue.engine.checkpoint", os.path.join(self.engine_root, "checkpoint")
        )
        if not os.path.exists(checkpoints_dir):
            checkpoints = []
        else:
            checkpoints = os.listdir(checkpoints_dir)
        
        checkpoints_dir = self.configuration.get(
            "enfugue.engine.checkpoint", os.path.join(self.engine_root, "checkpoint")
        )
        checkpoints  = list(find_files_in_directory(checkpoints_dir))
        checkpoints.sort(key=lambda item: os.path.getmtime(os.path.join(checkpoints_dir, item)))
        checkpoints = [
            os.path.basename(checkpoint)
            for checkpoint in checkpoints
        ]
        for checkpoint in self.DEFAULT_CHECKPOINTS:
            if checkpoint not in checkpoints:
                checkpoints.append(checkpoint)
        
        model_names = self.database.query(self.orm.DiffusionModel.name).all()
        return [
            {"type": "checkpoint", "name": checkpoint} for checkpoint in checkpoints
        ] + [
            {"type": "model", "name": model[0]} for model in model_names
        ]
