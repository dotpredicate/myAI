from typing import Any

from inference.engine import InferenceParam, InferenceParamType

# Recipe `defaults` keys mapped to POST /sdcpp/v1/img_gen. This mirrors
# apply_recipe_defaults() and the reverse of make_sample_params_json() /
# make_img_gen_defaults_json() in stable-diffusion.cpp (commit 95ff762):
# https://github.com/leejet/stable-diffusion.cpp/blob/95ff76230889898e1bf40058ba272a3699b50dc6/examples/server/model_catalog.cpp#L166
# https://github.com/leejet/stable-diffusion.cpp/blob/95ff76230889898e1bf40058ba272a3699b50dc6/examples/server/routes_sdcpp.cpp#L102
# https://github.com/leejet/stable-diffusion.cpp/blob/95ff76230889898e1bf40058ba272a3699b50dc6/examples/server/routes_sdcpp.cpp#L142
# https://github.com/leejet/stable-diffusion.cpp/blob/95ff76230889898e1bf40058ba272a3699b50dc6/examples/common/common.cpp#L2105
GEN_PATHS: dict[str, tuple[str, ...]] = {
    "width": ("width",),
    "height": ("height",),
    "seed": ("seed",),
    "strength": ("strength",),
    "moe_boundary": ("moe_boundary",),
    "video_frames": ("video_frames",),
    "fps": ("fps",),

    "steps": ("sample_params", "sample_steps"),
    "sampling_method": ("sample_params", "sample_method"),
    "scheduler": ("sample_params", "scheduler"),
    "flow_shift": ("sample_params", "flow_shift"),
    "shifted_timestep": ("sample_params", "shifted_timestep"),
    "custom_sigmas": ("sample_params", "custom_sigmas"),

    "cfg_scale": ("sample_params", "guidance", "txt_cfg"),
    "img_cfg_scale": ("sample_params", "guidance", "img_cfg"),
    "distilled_guidance": ("sample_params", "guidance", "distilled_guidance"),

    "high_noise_steps": ("high_noise_sample_params", "sample_steps"),
    "high_noise_sampling_method": ("high_noise_sample_params", "sample_method"),
    "high_noise_scheduler": ("high_noise_sample_params", "scheduler"),
    "high_noise_flow_shift": ("high_noise_sample_params", "flow_shift"),
    "high_noise_cfg_scale": ("high_noise_sample_params", "guidance", "txt_cfg"),
}


def _generation_params(defaults: dict[str, Any]) -> list[InferenceParam]:
    params: list[InferenceParam] = []
    for name in GEN_PATHS:
        value = defaults.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            param_type = InferenceParamType.INT
        elif isinstance(value, float):
            param_type = InferenceParamType.FLOAT
        elif isinstance(value, str):
            param_type = InferenceParamType.STR
        else:
            continue
        params.append(InferenceParam(name=name, type=param_type, default=value, description=name))

    if "seed" not in defaults:
        params.append(InferenceParam(name="seed", type=InferenceParamType.INT, default=-1, description="Seed (-1 is random)"))
    params.append(InferenceParam(name="negative_prompt", type=InferenceParamType.STR, default="", description="Negative prompt"))
    return params


def _recipes_by_id(raw: object) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        return {}
    return {recipe["id"]: recipe for recipe in raw if isinstance(recipe, dict) and isinstance(recipe.get("id"), str)}


def _files_by_id(raw: object) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        return {}
    return {item["id"]: item for item in raw if isinstance(item, dict) and isinstance(item.get("id"), str)}


def _recipe_slots(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    slots = recipe.get("slots")
    if not isinstance(slots, list):
        return []
    return [slot for slot in slots if isinstance(slot, dict) and isinstance(slot.get("id"), str)]


def _slot_label(slot: dict[str, Any]) -> str:
    label = slot.get("label")
    return label if isinstance(label, str) and label else str(slot["id"])


def _slot_categories(slot: dict[str, Any]) -> list[str]:
    categories = slot.get("categories")
    if isinstance(categories, list):
        return [category for category in categories if isinstance(category, str)]
    category = slot.get("category")
    return [category] if isinstance(category, str) else []


def _file_matches_slot(slot: dict[str, Any], file: dict[str, Any]) -> bool:
    categories = _slot_categories(slot)
    return not categories or file.get("category") in categories


def _missing_selection(recipe: dict[str, Any], selected: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for slot in _recipe_slots(recipe):
        if slot.get("required") and not selected.get(str(slot["id"])):
            missing.append(_slot_label(slot))
    for group in recipe.get("required_any", []):
        if isinstance(group, list) and group and not any(selected.get(str(slot)) for slot in group):
            missing.append("one of: " + ", ".join(str(slot) for slot in group))
    return missing


def _missing_slots(recipe: dict[str, Any], files: dict[str, str], catalog: dict[str, dict[str, Any]]) -> list[str]:
    selected = {slot: value for slot, value in files.items() if value}
    missing = _missing_selection(recipe, selected)
    for slot in _recipe_slots(recipe):
        slot_id = str(slot["id"])
        file_id = selected.get(slot_id)
        if not file_id:
            continue
        file = catalog.get(file_id)
        if file is None:
            missing.append(f"{_slot_label(slot)} (file not in model directory)")
        elif not _file_matches_slot(slot, file):
            missing.append(f"{_slot_label(slot)} (wrong category)")
    return missing


def _set_path(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = target
    for key in path[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[path[-1]] = value


def build_generation_request(params: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {"output_format": "png", "batch_count": 1}

    negative_prompt = params.get("negative_prompt")
    if negative_prompt:
        body["negative_prompt"] = str(negative_prompt)

    for name, path in GEN_PATHS.items():
        value = params.get(name)
        if value is not None:
            _set_path(body, path, value)
    return body
