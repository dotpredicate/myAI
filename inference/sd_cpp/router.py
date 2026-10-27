from dataclasses import asdict
from typing import Any, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference.engine import ImageProvider
from inference.registry import image_registry
from log_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/sd-cpp", tags=["sd-cpp"])


class AddImageModelRequest(BaseModel):
    name: str
    recipe: str
    files: dict[str, str] = Field(default_factory=dict)


def _provider() -> ImageProvider:
    try:
        return image_registry.get("stable-diffusion.cpp")
    except KeyError as e:
        raise RuntimeError("stable-diffusion.cpp image provider is not registered") from e


def _default_model(models: list[dict[str, Any]]) -> str:
    for model in models:
        if model.get("available"):
            return str(model.get("id", ""))
    return str(models[0].get("id", "")) if models else ""


@router.get("/models")
async def list_models() -> JSONResponse:
    try:
        provider = _provider()
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"error": str(e)})
    try:
        models = await provider.list_models()
    except RuntimeError as e:
        logger.error("Failed to load image models: %s", e)
        return JSONResponse(status_code=503, content={"error": str(e)})
    return JSONResponse(content={"models": models, "default": _default_model(models) if models else ""})


@router.get("/recipes")
async def list_recipes() -> JSONResponse:
    try:
        provider = _provider()
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"error": str(e)})
    try:
        return JSONResponse(content={"recipes": await provider.list_recipes()})
    except NotImplementedError:
        return JSONResponse(status_code=501, content={"error": "provider does not support recipes"})
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"error": str(e)})


@router.get("/files")
async def list_files() -> JSONResponse:
    try:
        provider = _provider()
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"error": str(e)})
    try:
        return JSONResponse(content={"files": await provider.list_files()})
    except NotImplementedError:
        return JSONResponse(status_code=501, content={"error": "this image provider has no file catalog"})
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"error": str(e)})


@router.post("/models")
async def add_model(req: AddImageModelRequest) -> JSONResponse:
    try:
        provider = _provider()
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"error": str(e)})
    try:
        model = await provider.add_model(req.name, req.recipe, req.files)
    except NotImplementedError:
        return JSONResponse(status_code=501, content={"error": "this image provider does not support adding models"})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.exception("Failed to add image model")
        return JSONResponse(status_code=502, content={"error": str(e)})
    return JSONResponse(content=model)


@router.delete("/models/{model_id}")
async def delete_model(model_id: str) -> JSONResponse:
    try:
        provider = _provider()
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"error": str(e)})
    try:
        deleted = await provider.remove_model(model_id)
    except NotImplementedError:
        return JSONResponse(status_code=501, content={"error": "this image provider does not support removing models"})
    if not deleted:
        return JSONResponse(status_code=404, content={"error": f"unknown image model '{model_id}'"})
    return JSONResponse(content={"deleted": True})


@router.get("/inference-params")
async def inference_params(model: Optional[str] = None) -> JSONResponse:
    try:
        provider = _provider()
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"error": str(e)})
    try:
        if model is None:
            models = await provider.list_models()
            model = next((str(item["id"]) for item in models if item.get("available")), "")
        if not model:
            return JSONResponse(status_code=404, content={"error": "no image model available"})
        params = await provider.get_inference_params(model)
    except RuntimeError as e:
        logger.error("Failed to load image parameters: %s", e)
        return JSONResponse(status_code=503, content={"error": str(e)})
    if not params:
        return JSONResponse(status_code=404, content={"error": f"unknown image model '{model}'"})
    return JSONResponse(content=[asdict(param) for param in params])
