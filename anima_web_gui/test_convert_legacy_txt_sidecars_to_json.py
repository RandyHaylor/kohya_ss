"""Stdlib-only tests for the legacy .txt -> .json sidecar migration.

Run: cd anima_web_gui then  python3 -m pytest test_convert_legacy_txt_sidecars_to_json.py -q
"""

import json
import os

from convert_legacy_txt_sidecars_to_json import (
    convert_legacy_txt_sidecars_in_folder_tree,
    parse_legacy_txt_sidecar_to_settings_dict,
)

LEGACY_TXT_SIDECAR_SAMPLE = (
    "prompt: masterpiece, a fox\n"
    "negative_prompt: low quality\n"
    "width: 832\n"
    "height: 1216\n"
    "steps: 50\n"
    "guidance_scale: 3.5\n"
    "flow_shift: 5.0\n"
    "seed: 42\n"
    "sampler: er_sde\n"
    "scheduler: beta57\n"
    "dit: /models/dit.safetensors\n"
    "vae: /models/vae.safetensors\n"
    "text_encoder: /models/te.safetensors\n"
    "lora: /loras/a.safetensors 0.8\n"
    "lora: /loras/b.safetensors 1.0\n"
    "source_image: /refs/ref.png\n"
    "test_lora: /loras/test.safetensors 1.0\n"
)


def test_parse_maps_scalars_with_numeric_types():
    settings = parse_legacy_txt_sidecar_to_settings_dict(LEGACY_TXT_SIDECAR_SAMPLE)
    assert settings["prompt"] == "masterpiece, a fox"
    assert settings["negative_prompt"] == "low quality"
    assert settings["width"] == 832 and settings["height"] == 1216
    assert settings["steps"] == 50
    assert settings["guidance_scale"] == 3.5
    assert settings["flow_shift"] == 5.0
    assert settings["seed"] == 42
    assert settings["sampler"] == "er_sde" and settings["scheduler"] == "beta57"
    assert settings["dit"].endswith("dit.safetensors")


def test_parse_collects_loras_as_structured_enabled_rows():
    settings = parse_legacy_txt_sidecar_to_settings_dict(LEGACY_TXT_SIDECAR_SAMPLE)
    assert settings["loras"] == [
        {"path": "/loras/a.safetensors", "multiplier": 0.8, "enabled": True},
        {"path": "/loras/b.safetensors", "multiplier": 1.0, "enabled": True},
    ]


def test_parse_keeps_source_image_and_test_lora():
    settings = parse_legacy_txt_sidecar_to_settings_dict(LEGACY_TXT_SIDECAR_SAMPLE)
    assert settings["source_image"] == "/refs/ref.png"
    assert settings["test_lora"] == "/loras/test.safetensors 1.0"


def test_parse_preserves_a_prompt_that_spans_multiple_lines():
    txt = "prompt: line one\nline two\nnegative_prompt: neg\n"
    settings = parse_legacy_txt_sidecar_to_settings_dict(txt)
    assert settings["prompt"] == "line one\nline two"
    assert settings["negative_prompt"] == "neg"


def test_folder_conversion_creates_json_and_skips_existing(tmp_path):
    directory = str(tmp_path)
    with open(os.path.join(directory, "image_a.txt"), "w") as sidecar_a:
        sidecar_a.write(LEGACY_TXT_SIDECAR_SAMPLE)
    # image_b already has a .json -> must be left untouched
    with open(os.path.join(directory, "image_b.txt"), "w") as sidecar_b:
        sidecar_b.write("prompt: b\n")
    with open(os.path.join(directory, "image_b.json"), "w") as existing_json:
        existing_json.write('{"prompt": "keep me"}')

    counts = convert_legacy_txt_sidecars_in_folder_tree(directory)

    assert counts["created"] == 1
    assert counts["skipped_existing_json"] == 1
    with open(os.path.join(directory, "image_a.json")) as created_json:
        assert json.load(created_json)["width"] == 832
    with open(os.path.join(directory, "image_b.json")) as untouched_json:
        assert json.load(untouched_json)["prompt"] == "keep me"


def test_folder_conversion_recurses_into_subfolders(tmp_path):
    subfolder = tmp_path / "batch1"
    subfolder.mkdir()
    with open(str(subfolder / "nested.txt"), "w") as nested_sidecar:
        nested_sidecar.write(LEGACY_TXT_SIDECAR_SAMPLE)

    counts = convert_legacy_txt_sidecars_in_folder_tree(str(tmp_path))

    assert counts["created"] == 1
    assert os.path.isfile(str(subfolder / "nested.json"))
