"""Pure helpers for the Anima inference web GUI: build the CLI argv for a single embedded-prompt
generation, and discover the sampler/scheduler choices from the inference script (without importing
it, so the GUI stays dependency-free)."""

import json
import os
import re
from typing import Any, Dict, List, Tuple


INFERENCE_SCRIPT_RELATIVE_PATH = os.path.join("sd-scripts", "anima_minimal_inference.py")

# Used only as a fallback if the choices cannot be parsed out of the inference script source.
FALLBACK_SAMPLER_CHOICES = ("euler", "er_sde", "euler_ancestral")
FALLBACK_SCHEDULER_CHOICES = ("default", "beta57", "simple", "diffusers", "diffusers_dynamic")

DEFAULT_SAVE_PATH = "./anima_out"
DEFAULT_INFER_STEPS = 40
DEFAULT_GUIDANCE_SCALE = 4.5

# Standard ~1-megapixel aspect-ratio buckets (base 1024, all divisible by 32 as the inference script
# requires). NOTE: this is NOT a verified "official Anima" list - no such list was found in the repo;
# 1216x832 / 832x1216 match the project's own example commands. Replace with the real list if there is one.
ANIMA_RESOLUTION_PRESETS = [
    {"label": "1024 x 1024 (1:1)", "width": 1024, "height": 1024},
    {"label": "1216 x 832 (landscape 3:2)", "width": 1216, "height": 832},
    {"label": "832 x 1216 (portrait 2:3)", "width": 832, "height": 1216},
    {"label": "1152 x 896 (landscape 9:7)", "width": 1152, "height": 896},
    {"label": "896 x 1152 (portrait 7:9)", "width": 896, "height": 1152},
    {"label": "1344 x 768 (landscape 16:9)", "width": 1344, "height": 768},
    {"label": "768 x 1344 (portrait 9:16)", "width": 768, "height": 1344},
    {"label": "1536 x 640 (wide 12:5)", "width": 1536, "height": 640},
    {"label": "640 x 1536 (tall 5:12)", "width": 640, "height": 1536},
    # 1536-basis buckets: the 1024-basis ratios above scaled 1.5x, snapped to multiples of 64.
    {"label": "1536 x 1536 (1:1) [1536]", "width": 1536, "height": 1536},
    {"label": "1856 x 1280 (landscape 3:2) [1536]", "width": 1856, "height": 1280},
    {"label": "1280 x 1856 (portrait 2:3) [1536]", "width": 1280, "height": 1856},
    {"label": "1728 x 1344 (landscape 9:7) [1536]", "width": 1728, "height": 1344},
    {"label": "1344 x 1728 (portrait 7:9) [1536]", "width": 1344, "height": 1728},
    {"label": "2048 x 1152 (landscape 16:9) [1536]", "width": 2048, "height": 1152},
    {"label": "1152 x 2048 (portrait 9:16) [1536]", "width": 1152, "height": 2048},
    {"label": "2304 x 960 (wide 12:5) [1536]", "width": 2304, "height": 960},
    {"label": "960 x 2304 (tall 5:12) [1536]", "width": 960, "height": 2304},
]


def build_inference_command(generation_request: Dict[str, Any]) -> List[str]:
    """Build the argv list for one single-prompt (embedded) generation.

    Required in the request: positive_prompt, and dit_path unless an enabled dit_test_folder supplies the
    DiT(s). save_path falls back to DEFAULT_SAVE_PATH.
    vae_path / text_encoder_path are omitted when blank (e.g. an all-in-one --dit checkpoint). loras is
    a list of {"path", "strength"} dicts; blank-path rows are skipped. Only the fields the GUI exposes
    are emitted; everything else uses the inference script's own defaults (one image per run).
    """
    dit_path = str(generation_request.get("dit_path", "")).strip()
    positive_prompt = str(generation_request.get("positive_prompt", ""))
    negative_prompt = str(generation_request.get("negative_prompt", ""))
    save_path = str(generation_request.get("save_path", "")).strip() or DEFAULT_SAVE_PATH
    mode = str(generation_request.get("mode", "single")).strip() or "single"

    # A DiT test sweep supplies the DiT(s) from a folder, so --dit may be omitted in that case.
    dit_test_folder = str(generation_request.get("dit_test_folder", "")).strip()
    dit_test_folder_enabled = bool(generation_request.get("dit_test_folder_enabled", True))
    dit_test_sweep_active = bool(dit_test_folder and dit_test_folder_enabled)

    if not dit_path and not dit_test_sweep_active:
        raise ValueError("dit_path is required unless a DiT test folder is set")

    command_argv: List[str] = ["uv", "run", INFERENCE_SCRIPT_RELATIVE_PATH]
    if dit_path:
        command_argv += ["--dit", dit_path]

    vae_path = str(generation_request.get("vae_path", "")).strip()
    if vae_path:
        command_argv += ["--vae", vae_path]

    text_encoder_path = str(generation_request.get("text_encoder_path", "")).strip()
    if text_encoder_path:
        command_argv += ["--text_encoder", text_encoder_path]

    # Only enabled LoRAs are merged (--lora_list), but ALL rows (path, multiplier, enabled) are recorded
    # in the settings sidecar (--record_lora_rows_json) so a disabled row survives a save/reload.
    lora_list_tokens: List[str] = []
    recorded_lora_rows: List[Dict[str, Any]] = []
    for lora_entry in generation_request.get("loras", []) or []:
        lora_path = str(lora_entry.get("path", "")).strip()
        if not lora_path:
            continue
        lora_strength = str(lora_entry.get("strength", "")).strip() or "1.0"
        is_enabled = bool(lora_entry.get("enabled", True))  # default enabled
        recorded_lora_rows.append({"path": lora_path, "multiplier": lora_strength, "enabled": is_enabled})
        if is_enabled:
            lora_list_tokens += [lora_path, lora_strength]
    if lora_list_tokens:
        command_argv += ["--lora_list"] + lora_list_tokens
    if recorded_lora_rows:
        command_argv += ["--record_lora_rows_json", json.dumps(recorded_lora_rows)]

    # LoRA test sweep: run the whole generation once per top-level .safetensors in this folder, on top
    # of the fixed LoRAs above. Multiplier defaults to 1.0.
    lora_test_folder = str(generation_request.get("lora_test_folder", "")).strip()
    lora_test_folder_enabled = bool(generation_request.get("lora_test_folder_enabled", True))
    if lora_test_folder and lora_test_folder_enabled:
        lora_test_multiplier = str(generation_request.get("lora_test_multiplier", "")).strip() or "1.0"
        command_argv += ["--lora_test_folder", lora_test_folder, lora_test_multiplier]

    # DiT test sweep: run the whole generation once per top-level .safetensors DiT in this folder (plus
    # --dit if distinct). If a LoRA test folder is also set, the two sweeps nest (each DiT x each LoRA).
    if dit_test_sweep_active:
        command_argv += ["--dit_test_folder", dit_test_folder]

    # Number of images per run, seed-incremented in a single model load (the GUI 'quantity' maps here).
    images_per_prompt = coerce_to_int_with_default(generation_request.get("images_per_prompt"), 1)
    if images_per_prompt > 1:
        command_argv += ["--images_per_prompt", str(images_per_prompt)]

    # Prompt source depends on the mode. In the file/image modes the positive/negative fields are the
    # pre-prompt (--pre_prompt / --pre_prompt_neg); in single mode they are the actual prompt.
    if mode == "from_image":
        source_image_folder = str(generation_request.get("source_image_folder", "")).strip()
        if not source_image_folder:
            raise ValueError("source_image_folder is required for from_image mode")
        command_argv += ["--from_image_embed", source_image_folder]
        if positive_prompt.strip():
            command_argv += ["--pre_prompt", positive_prompt]
        if negative_prompt.strip():
            command_argv += ["--pre_prompt_neg", negative_prompt]
    elif mode == "from_prompt_list":
        prompt_list_path = str(generation_request.get("prompt_list_path", "")).strip()
        if not prompt_list_path:
            raise ValueError("prompt_list_path is required for from_prompt_list mode")
        command_argv += ["--from_file", prompt_list_path]
        if positive_prompt.strip():
            command_argv += ["--pre_prompt", positive_prompt]
        if negative_prompt.strip():
            command_argv += ["--pre_prompt_neg", negative_prompt]
    else:  # single embedded prompt
        if not positive_prompt.strip():
            raise ValueError("positive_prompt is required")
        command_argv += ["--prompt", positive_prompt]
        if negative_prompt.strip():
            command_argv += ["--negative_prompt", negative_prompt]

    sampler = str(generation_request.get("sampler", "")).strip()
    if sampler:
        command_argv += ["--sampler", sampler]

    scheduler = str(generation_request.get("scheduler", "")).strip()
    if scheduler:
        command_argv += ["--scheduler", scheduler]

    # base/max shift only affect the diffusers_dynamic scheduler; emit them only for it to keep other
    # commands clean.
    if scheduler == "diffusers_dynamic":
        command_argv += [
            "--dynamic_base_shift",
            str(coerce_to_float_with_default(generation_request.get("dynamic_base_shift"), 0.5)),
        ]
        command_argv += [
            "--dynamic_max_shift",
            str(coerce_to_float_with_default(generation_request.get("dynamic_max_shift"), 0.9)),
        ]

    # Image size is passed as HEIGHT WIDTH (the script's --image_size order). Both or neither.
    image_height = str(generation_request.get("image_height", "")).strip()
    image_width = str(generation_request.get("image_width", "")).strip()
    if image_height and image_width:
        try:
            parsed_height, parsed_width = int(image_height), int(image_width)
        except ValueError:
            raise ValueError("image_height and image_width must be integers")
        command_argv += ["--image_size", str(parsed_height), str(parsed_width)]
    elif image_height or image_width:
        raise ValueError("provide both image_height and image_width, or neither")

    flow_shift = str(generation_request.get("flow_shift", "")).strip()
    if flow_shift:
        try:
            float(flow_shift)
        except ValueError:
            raise ValueError("flow_shift must be a number")
        command_argv += ["--flow_shift", flow_shift]

    seed = str(generation_request.get("seed", "")).strip()
    if seed:
        try:
            int(seed)
        except ValueError:
            raise ValueError("seed must be an integer (use -1 for random)")
        command_argv += ["--seed", seed]

    # Steps and CFG always emit, coerced to int/float, falling back to defaults when unparseable.
    infer_steps = coerce_to_int_with_default(generation_request.get("infer_steps"), DEFAULT_INFER_STEPS)
    guidance_scale = coerce_to_float_with_default(generation_request.get("guidance_scale"), DEFAULT_GUIDANCE_SCALE)
    command_argv += ["--infer_steps", str(infer_steps), "--guidance_scale", str(guidance_scale)]

    # Serialize model-file disk reads across concurrent generations: when a lock file path is given,
    # each spawned process holds an exclusive lock on it while loading models from disk, so concurrent
    # runs load one at a time (and their GPU denoise phases naturally stagger). Omitted when blank.
    model_load_disk_lock_file = str(generation_request.get("model_load_disk_lock_file", "")).strip()
    if model_load_disk_lock_file:
        command_argv += ["--model_load_disk_lock_file", model_load_disk_lock_file]

    # Serialize GPU compute across concurrent generations: when a lock file path is given, each spawned
    # process holds an exclusive lock on it while doing GPU work, so only one uses the GPU at a time. The
    # scope selects how much GPU work is guarded (see the inference script's --gpu_lock_scope).
    gpu_compute_lock_file = str(generation_request.get("gpu_compute_lock_file", "")).strip()
    if gpu_compute_lock_file:
        command_argv += ["--gpu_compute_lock_file", gpu_compute_lock_file]
        gpu_lock_scope = str(generation_request.get("gpu_lock_scope", "")).strip()
        if gpu_lock_scope:
            command_argv += ["--gpu_lock_scope", gpu_lock_scope]

    # Anima-Safe PAG: emitted only when the GUI's PAG checkbox is on. All settings are passed through so
    # a render reproduces exactly what the form shows; head_indices is omitted when blank (= all heads).
    if generation_request.get("pag_enabled"):
        command_argv += ["--pag"]
        command_argv += ["--pag_scale", str(coerce_to_float_with_default(generation_request.get("pag_scale"), 4.0))]
        pag_block_indices = str(generation_request.get("pag_block_indices", "")).strip()
        if pag_block_indices:
            command_argv += ["--pag_block_indices", pag_block_indices]
        command_argv += [
            "--pag_perturbation_strength",
            str(coerce_to_float_with_default(generation_request.get("pag_perturbation_strength"), 0.75)),
        ]
        pag_head_indices = str(generation_request.get("pag_head_indices", "")).strip()
        if pag_head_indices:
            command_argv += ["--pag_head_indices", pag_head_indices]
        command_argv += ["--pag_start_percent", str(coerce_to_float_with_default(generation_request.get("pag_start_percent"), 0.0))]
        command_argv += ["--pag_end_percent", str(coerce_to_float_with_default(generation_request.get("pag_end_percent"), 0.7))]
        command_argv += ["--pag_rescale", str(coerce_to_float_with_default(generation_request.get("pag_rescale"), 0.2))]
        pag_rescale_mode = str(generation_request.get("pag_rescale_mode", "")).strip()
        if pag_rescale_mode:
            command_argv += ["--pag_rescale_mode", pag_rescale_mode]

        if generation_request.get("flow_matched_pag_enabled"):
            command_argv += ["--flow_matched_pag"]
            command_argv += [
                "--flow_matched_pag_strength",
                str(coerce_to_float_with_default(generation_request.get("flow_matched_pag_strength"), 1.0)),
            ]
            command_argv += [
                "--flow_matched_pag_curve_exponent",
                str(coerce_to_float_with_default(generation_request.get("flow_matched_pag_curve_exponent"), 1.0)),
            ]

    command_argv += ["--save_path", save_path, "--output_type", "images"]
    return command_argv


def coerce_to_int_with_default(raw_value: Any, default_value: int) -> int:
    """Parse raw_value to an int (truncating a float, ignoring whitespace); default_value if unparseable."""
    try:
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
        return int(float(raw_value))
    except (ValueError, TypeError):
        return default_value


def coerce_to_float_with_default(raw_value: Any, default_value: float) -> float:
    """Parse raw_value to a float (ignoring whitespace); default_value if unparseable."""
    try:
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
        return float(raw_value)
    except (ValueError, TypeError):
        return default_value


def coerce_quantity_to_positive_int(raw_quantity: Any) -> int:
    """Normalize a quantity value to a positive integer.

    Truncates a float toward zero (3.7 -> 3), ignores surrounding whitespace, and falls back to 1 for
    any non-numeric string / empty / None, or for results below 1.
    """
    try:
        if isinstance(raw_quantity, str):
            raw_quantity = raw_quantity.strip()
        number = float(raw_quantity)
    except (ValueError, TypeError):
        return 1
    quantity = int(number)  # truncate toward zero
    return quantity if quantity >= 1 else 1


def coerce_concurrency_to_positive_int(raw_concurrency: Any) -> int:
    """Normalize a 'max concurrent generations' value to a positive integer (>= 1).

    Same rules as coerce_quantity_to_positive_int: truncate a float toward zero, ignore whitespace,
    and fall back to 1 for any non-numeric / empty / None / below-1 value.
    """
    return coerce_quantity_to_positive_int(raw_concurrency)


def _parse_choice_tuple_from_source(source_text: str, choice_variable_name: str) -> Tuple[str, ...]:
    """Extract the quoted strings from a `NAME = ("a", "b", ...)` assignment in Python source text."""
    match = re.search(choice_variable_name + r"\s*=\s*\(([^)]*)\)", source_text)
    if not match:
        return ()
    return tuple(re.findall(r"[\"']([^\"']+)[\"']", match.group(1)))


def load_sampler_and_scheduler_choices(repo_root: str) -> Dict[str, List[str]]:
    """Read the sampler/scheduler choices from the inference script source (falling back to constants).

    Parses the source text rather than importing the module, so the GUI needs no torch/venv."""
    script_path = os.path.join(repo_root, INFERENCE_SCRIPT_RELATIVE_PATH)
    samplers: Tuple[str, ...] = ()
    schedulers: Tuple[str, ...] = ()
    try:
        with open(script_path, "r", encoding="utf-8") as script_file:
            source_text = script_file.read()
        samplers = _parse_choice_tuple_from_source(source_text, "SAMPLER_OPTION_CHOICES")
        schedulers = _parse_choice_tuple_from_source(source_text, "SCHEDULER_OPTION_CHOICES")
    except OSError:
        pass

    return {
        "samplers": list(samplers) if samplers else list(FALLBACK_SAMPLER_CHOICES),
        "schedulers": list(schedulers) if schedulers else list(FALLBACK_SCHEDULER_CHOICES),
    }
