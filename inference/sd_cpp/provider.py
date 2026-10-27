import asyncio
import os
from pathlib import Path
from typing import Any

import httpx

import inference.managed_processes as mp
from inference.engine import ImageGenerationRequest, ImageProvider, InferenceParam
from inference.managed_processes import get_free_port
from log_config import get_logger
from . import image_models
from .recipes import (
    _files_by_id,
    _generation_params,
    _missing_slots,
    _recipe_slots,
    _recipes_by_id,
    _slot_categories,
    _slot_label,
    build_generation_request,
)

logger = get_logger(__name__)

MODEL_DIRS = [
    Path(part).expanduser()
    for part in os.getenv("MYAI_SD_MODEL_DIRS", "~/.cache/myai/models").split(":")
    if part
]


async def _wait_for_job(client: httpx.AsyncClient, base: str, job: dict[str, Any]) -> dict[str, Any]:
    poll_url = job.get("poll_url")
    if not isinstance(poll_url, str):
        raise RuntimeError("sd-server returned an invalid job")
    while job.get("status") in ("queued", "generating"):
        await asyncio.sleep(0.5)
        response = await client.get(f"{base}{poll_url}")
        if not response.is_success:
            raise RuntimeError(f"sd-server returned {response.status_code}: {response.text[:500]}")
        job = response.json()
    return job


def parse_job_result(job: object) -> str:
    if not isinstance(job, dict):
        raise RuntimeError("sd-server returned an invalid job")
    if job.get("status") != "completed":
        raise RuntimeError(f"sd-server job {job.get('status')}: {job.get('error')}")
    result = job.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("images"), list) or not result["images"]:
        raise RuntimeError("sd-server returned no image")
    first = result["images"][0]
    if not isinstance(first, dict) or not isinstance(first.get("b64_json"), str):
        raise RuntimeError("sd-server returned no image")
    return first["b64_json"]


class SdCppImageProvider(ImageProvider):

    def __init__(self) -> None:
        self._port: str | None = None
        self._loaded: tuple[str, tuple[tuple[str, str], ...]] | None = None
        self._recipes: dict[str, dict[str, Any]] | None = None
        self._lock = asyncio.Lock()

    async def _ensure_server(self) -> str:
        if self._port is not None:
            managed = mp.get_managed_process(self._port)
            if managed is not None and managed.process is not None and managed.process.poll() is None:
                return self._port
            self._port = None
            self._loaded = None

        port = str(get_free_port())
        cmd = ["sd-server", "--listen-ip", "127.0.0.1", "--listen-port", port]
        for model_dir in MODEL_DIRS:
            if model_dir.is_dir():
                cmd += ["--model-dir", str(model_dir)]

        async def _check() -> bool:
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.get(f"http://127.0.0.1:{port}/sdcpp/v1/recipes")
                return response.status_code == 200

        await mp.ensure_server_started(port, "stable-diffusion.cpp", "server", cmd, readiness_check=_check)
        self._port = port
        self._loaded = None
        return port

    async def _catalog(self) -> dict[str, dict[str, Any]]:
        if self._recipes is not None:
            return self._recipes
        port = await self._ensure_server()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"http://127.0.0.1:{port}/sdcpp/v1/recipes")
            response.raise_for_status()
            payload = response.json()
        recipes = _recipes_by_id(payload.get("recipes") if isinstance(payload, dict) else None)
        if not recipes:
            raise RuntimeError("sd-server returned no recipes")
        self._recipes = recipes
        return recipes

    async def _refresh_catalog(self) -> None:
        port = await self._ensure_server()
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"http://127.0.0.1:{port}/sdcpp/v1/models/refresh")
            response.raise_for_status()
        self._recipes = None

    async def _model_files(self) -> dict[str, dict[str, Any]]:
        port = await self._ensure_server()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"http://127.0.0.1:{port}/sdcpp/v1/model-files")
            response.raise_for_status()
            payload = response.json()
        return _files_by_id(payload.get("files") if isinstance(payload, dict) else None)

    async def list_models(self) -> list[dict[str, Any]]:
        records = await image_models.list_image_models()
        if not records:
            return []
        await self._refresh_catalog()
        catalog = await self._catalog()
        files = await self._model_files()
        models: list[dict[str, Any]] = []
        for record in records:
            recipe = catalog.get(record.recipe)
            missing = ["unknown recipe: " + record.recipe] if recipe is None else _missing_slots(recipe, record.files, files)
            models.append({
                "id": str(record.id),
                "name": record.name,
                "recipe": record.recipe,
                "available": not missing,
                "missing": missing,
            })
        return models

    async def get_inference_params(self, model_id: str) -> list[InferenceParam]:
        record = await image_models.get_image_model_by_ref(model_id)
        if record is None:
            return []
        catalog = await self._catalog()
        recipe = catalog.get(record.recipe)
        if recipe is None:
            return []
        defaults = recipe.get("defaults")
        return _generation_params(defaults if isinstance(defaults, dict) else {})

    async def list_recipes(self) -> list[dict[str, Any]]:
        await self._refresh_catalog()
        catalog = await self._catalog()
        recipes: list[dict[str, Any]] = []
        for recipe in catalog.values():
            recipes.append({
                "id": recipe.get("id"),
                "name": recipe.get("name"),
                "description": recipe.get("description"),
                "model_url": recipe.get("model_url"),
                "slots": [
                    {
                        "id": slot["id"],
                        "label": _slot_label(slot),
                        "required": bool(slot.get("required")),
                        "categories": _slot_categories(slot),
                    }
                    for slot in _recipe_slots(recipe)
                ],
                "required_any": recipe.get("required_any", []),
            })
        return sorted(recipes, key=lambda item: str(item.get("name") or item.get("id")))

    async def list_files(self) -> list[dict[str, Any]]:
        await self._refresh_catalog()
        return list((await self._model_files()).values())

    async def add_model(self, name: str, recipe_id: str, files: dict[str, str]) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("model name is required")
        if await image_models.get_image_model_by_name(name) is not None:
            raise ValueError(f"image model '{name}' already exists")

        await self._refresh_catalog()
        catalog = await self._catalog()
        recipe = catalog.get(recipe_id)
        if recipe is None:
            raise ValueError(f"unknown recipe: {recipe_id}")
        slots = {str(slot["id"]): slot for slot in _recipe_slots(recipe)}
        unknown = sorted(set(files) - set(slots))
        if unknown:
            raise ValueError("unknown slots: " + ", ".join(unknown))
        selected = {slot: file_id for slot, file_id in files.items() if file_id}
        available = await self._model_files()
        missing = _missing_slots(recipe, selected, available)
        if missing:
            raise ValueError("invalid model files: " + ", ".join(missing))

        record = await image_models.create_image_model(name, recipe_id, selected)
        return {
            "id": str(record.id),
            "name": record.name,
            "recipe": record.recipe,
            "available": True,
            "missing": [],
        }

    async def remove_model(self, model_id: str) -> bool:
        record = await image_models.get_image_model_by_ref(model_id)
        if record is None:
            return False
        deleted = await image_models.delete_image_model(record.id)
        if deleted and self._loaded is not None:
            self._loaded = None
        return deleted

    async def _load_model(self, record: image_models.StoredImageModel) -> None:
        selection = {slot: file_id for slot, file_id in record.files.items() if file_id}
        key = (record.recipe, tuple(sorted(selection.items())))
        if self._loaded == key:
            return
        port = await self._ensure_server()
        async with httpx.AsyncClient(timeout=900.0) as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/sdcpp/v1/models/load",
                json={"recipe": record.recipe, "models": selection},
            )
        if response.status_code != 200:
            message = response.text[:500]
            if response.status_code in (400, 422):
                raise ValueError(f"cannot load model '{record.name}': {message}")
            raise RuntimeError(f"sd-server returned {response.status_code}: {message}")
        self._loaded = key

    async def generate_image(self, request: ImageGenerationRequest) -> str:
        record = await image_models.get_image_model_by_ref(request.model)
        if record is None:
            raise ValueError(f"unknown image model: {request.model}")
        catalog = await self._catalog()
        recipe = catalog.get(record.recipe)
        if recipe is None:
            raise ValueError(f"unknown recipe: {record.recipe}")

        body = build_generation_request(request.params)
        body["prompt"] = request.prompt

        async with self._lock:
            await self._load_model(record)
            port = await self._ensure_server()
            base = f"http://127.0.0.1:{port}"
            async with mp.track_server_activity(port):
                async with httpx.AsyncClient(timeout=None) as client:
                    response = await client.post(f"{base}/sdcpp/v1/img_gen", json=body)
                    if not response.is_success:
                        raise RuntimeError(f"sd-server returned {response.status_code}: {response.text[:500]}")
                    job = await _wait_for_job(client, base, response.json())
        return parse_job_result(job)
