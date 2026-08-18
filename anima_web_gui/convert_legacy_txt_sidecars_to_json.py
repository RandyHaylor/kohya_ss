"""One-time migration: create '<name>.json' settings sidecars from the legacy '<name>.txt' sidecars
the inference script used to write, so older generated images gain the JSON the GUI's Load button reads.

The legacy .txt format is one 'key: value' line per setting, with repeated 'lora: <path> <mult>' lines
and optional 'source_image:' / 'test_lora:' lines. This converts that into the same structured schema
build_generation_settings_dict now emits (loras as {path, multiplier, enabled}).

Run (defaults to the repo's anima_out):
    python3 convert_legacy_txt_sidecars_to_json.py [folder]
Existing .json files are never overwritten.
"""

import json
import os
import sys

SCALAR_SETTING_KEYS_IN_OUTPUT_ORDER = [
    "prompt",
    "negative_prompt",
    "width",
    "height",
    "steps",
    "guidance_scale",
    "flow_shift",
    "seed",
    "sampler",
    "scheduler",
    "dit",
    "vae",
    "text_encoder",
]
TRAILING_SCALAR_SETTING_KEYS_IN_OUTPUT_ORDER = ["source_image", "test_lora"]
INTEGER_SETTING_KEYS = {"width", "height", "steps", "seed"}
FLOAT_SETTING_KEYS = {"guidance_scale", "flow_shift"}
ALL_SCALAR_SETTING_KEYS = set(SCALAR_SETTING_KEYS_IN_OUTPUT_ORDER) | set(TRAILING_SCALAR_SETTING_KEYS_IN_OUTPUT_ORDER)


def coerce_scalar_setting_value(key, raw_value):
    """Coerce a legacy string value to int/float for numeric keys; keep the raw string if it does not
    parse (best-effort migration should never drop data)."""
    if key in INTEGER_SETTING_KEYS:
        try:
            return int(raw_value)
        except ValueError:
            return raw_value
    if key in FLOAT_SETTING_KEYS:
        try:
            return float(raw_value)
        except ValueError:
            return raw_value
    return raw_value


def coerce_lora_multiplier(raw_multiplier):
    try:
        return float(raw_multiplier)
    except ValueError:
        return raw_multiplier


def parse_legacy_txt_sidecar_to_settings_dict(txt_content):
    """Parse the legacy 'key: value' settings .txt into the structured settings dict. A line that does
    not begin with a known 'key: ' is treated as a continuation of the previous scalar value (so a
    prompt that contained a newline is preserved)."""
    parsed_scalars = {}
    parsed_loras = []
    last_scalar_key = None
    for raw_line in txt_content.splitlines():
        key, separator, value = raw_line.partition(": ")
        if separator and key == "lora":
            lora_path, _, multiplier_text = value.rpartition(" ")
            if lora_path:
                parsed_loras.append(
                    {"path": lora_path, "multiplier": coerce_lora_multiplier(multiplier_text), "enabled": True}
                )
            last_scalar_key = None
            continue
        if separator and key in ALL_SCALAR_SETTING_KEYS:
            parsed_scalars[key] = coerce_scalar_setting_value(key, value)
            last_scalar_key = key
            continue
        if last_scalar_key is not None and isinstance(parsed_scalars.get(last_scalar_key), str):
            parsed_scalars[last_scalar_key] = parsed_scalars[last_scalar_key] + "\n" + raw_line

    settings = {key: parsed_scalars[key] for key in SCALAR_SETTING_KEYS_IN_OUTPUT_ORDER if key in parsed_scalars}
    if parsed_loras:
        settings["loras"] = parsed_loras
    for key in TRAILING_SCALAR_SETTING_KEYS_IN_OUTPUT_ORDER:
        if key in parsed_scalars:
            settings[key] = parsed_scalars[key]
    return settings


def convert_legacy_txt_sidecars_in_folder_tree(root_folder):
    """Walk root_folder (recursively) and write '<name>.json' next to each legacy '<name>.txt' that has
    no .json yet and parses to at least one setting. Returns a counts dict. Never overwrites a .json."""
    created_count = 0
    skipped_existing_json_count = 0
    skipped_unrecognized_count = 0
    for directory_path, _subdirectories, file_names in os.walk(root_folder):
        for file_name in file_names:
            if not file_name.endswith(".txt"):
                continue
            base_name = file_name[: -len(".txt")]
            txt_path = os.path.join(directory_path, file_name)
            json_path = os.path.join(directory_path, base_name + ".json")
            if os.path.isfile(json_path):
                skipped_existing_json_count += 1
                continue
            with open(txt_path, "r", encoding="utf-8") as txt_file:
                settings = parse_legacy_txt_sidecar_to_settings_dict(txt_file.read())
            if not settings:
                skipped_unrecognized_count += 1
                continue
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(settings, json_file, indent=2, ensure_ascii=False)
                json_file.write("\n")
            created_count += 1
    return {
        "created": created_count,
        "skipped_existing_json": skipped_existing_json_count,
        "skipped_unrecognized": skipped_unrecognized_count,
    }


def resolve_default_anima_output_folder():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "anima_out")


def main():
    target_folder = sys.argv[1] if len(sys.argv) > 1 else resolve_default_anima_output_folder()
    if not os.path.isdir(target_folder):
        print(f"Not a folder: {target_folder}")
        sys.exit(1)
    counts = convert_legacy_txt_sidecars_in_folder_tree(target_folder)
    print(
        f"Converted {counts['created']} .txt sidecar(s) to .json in {target_folder} "
        f"(skipped {counts['skipped_existing_json']} already having .json, "
        f"{counts['skipped_unrecognized']} unrecognized)."
    )


if __name__ == "__main__":
    main()
