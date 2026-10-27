from typing import Any, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference.engine import ImageGenerationRequest
from inference.registry import image_registry
from log_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


class GenerateImageRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    params: dict[str, Any] = Field(default_factory=dict)


@router.post("/api/images/generate")
async def generate_image(req: GenerateImageRequest) -> JSONResponse:
    if not req.prompt.strip():
        return JSONResponse(status_code=400, content={"error": "prompt required"})
    try:
        provider = image_registry.get_default()
    except RuntimeError:
        logger.error("No image provider registered")
        return JSONResponse(status_code=503, content={"error": "no image provider registered"})
    try:
        model_id = req.model
        if not model_id:
            models = await provider.list_models()
            model_id = next((str(item["id"]) for item in models if item["available"]), "")
        known = bool(model_id) and bool(await provider.get_inference_params(model_id))
    except RuntimeError as e:
        logger.error("Failed to load image models: %s", e)
        return JSONResponse(status_code=503, content={"error": str(e)})
    if not model_id or not known:
        return JSONResponse(status_code=400, content={"error": f"unknown image model '{model_id}'"})
    try:
        b64 = await provider.generate_image(ImageGenerationRequest(
            model=model_id,
            prompt=req.prompt,
            params=req.params,
        ))
    except FileNotFoundError:
        logger.error("Image provider backend not found")
        return JSONResponse(status_code=503, content={"error": "image provider backend not found"})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error("Image generation failed: %s", e)
        return JSONResponse(status_code=502, content={"error": str(e)})
    return JSONResponse(content={"image": b64})
