import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_command_builder import (
    ANIMA_RESOLUTION_PRESETS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_INFER_STEPS,
    DEFAULT_SAVE_PATH,
    build_inference_command,
    coerce_quantity_to_positive_int,
    coerce_to_float_with_default,
    coerce_to_int_with_default,
    load_sampler_and_scheduler_choices,
    _parse_choice_tuple_from_source,
)


def test_coerce_to_int_with_default():
    assert coerce_to_int_with_default("30", 50) == 30
    assert coerce_to_int_with_default("  30 ", 50) == 30
    assert coerce_to_int_with_default("30.9", 50) == 30  # truncates
    assert coerce_to_int_with_default("abc", 50) == 50
    assert coerce_to_int_with_default("", 50) == 50
    assert coerce_to_int_with_default(None, 50) == 50


def test_coerce_to_float_with_default():
    assert coerce_to_float_with_default("4.5", 3.5) == 4.5
    assert coerce_to_float_with_default("  4.5 ", 3.5) == 4.5
    assert coerce_to_float_with_default("7", 3.5) == 7.0
    assert coerce_to_float_with_default("abc", 3.5) == 3.5
    assert coerce_to_float_with_default("", 3.5) == 3.5


def test_steps_and_cfg_always_emitted_with_defaults_or_parsed_values():
    argv_default = build_inference_command({"dit_path": "/m/dit.safetensors", "positive_prompt": "p"})
    assert argv_default[argv_default.index("--infer_steps") + 1] == str(DEFAULT_INFER_STEPS)
    assert argv_default[argv_default.index("--guidance_scale") + 1] == str(DEFAULT_GUIDANCE_SCALE)

    argv_custom = build_inference_command(
        {"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "infer_steps": "30", "guidance_scale": "7"}
    )
    assert argv_custom[argv_custom.index("--infer_steps") + 1] == "30"
    assert argv_custom[argv_custom.index("--guidance_scale") + 1] == "7.0"

    argv_bad = build_inference_command(
        {"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "infer_steps": "xx", "guidance_scale": "yy"}
    )
    assert argv_bad[argv_bad.index("--infer_steps") + 1] == str(DEFAULT_INFER_STEPS)
    assert argv_bad[argv_bad.index("--guidance_scale") + 1] == str(DEFAULT_GUIDANCE_SCALE)


def test_coerce_quantity_to_positive_int():
    assert coerce_quantity_to_positive_int(1) == 1
    assert coerce_quantity_to_positive_int("3") == 3
    assert coerce_quantity_to_positive_int("  4  ") == 4  # whitespace ignored
    assert coerce_quantity_to_positive_int("3.7") == 3  # float truncates
    assert coerce_quantity_to_positive_int(2.9) == 2
    assert coerce_quantity_to_positive_int("abc") == 1  # non-numeric string -> 1
    assert coerce_quantity_to_positive_int("") == 1
    assert coerce_quantity_to_positive_int("0") == 1  # below 1 -> 1
    assert coerce_quantity_to_positive_int("-5") == 1
    assert coerce_quantity_to_positive_int(None) == 1


def test_build_minimal_command_uses_defaults():
    argv = build_inference_command({"dit_path": "/models/dit.safetensors", "positive_prompt": "a cat"})
    assert argv[:5] == ["uv", "run", "sd-scripts/anima_minimal_inference.py", "--dit", "/models/dit.safetensors"]
    assert "--prompt" in argv and argv[argv.index("--prompt") + 1] == "a cat"
    assert argv[argv.index("--save_path") + 1] == DEFAULT_SAVE_PATH
    assert argv[argv.index("--output_type") + 1] == "images"
    # optional flags omitted when not provided
    assert "--vae" not in argv
    assert "--text_encoder" not in argv
    assert "--negative_prompt" not in argv
    assert "--lora_list" not in argv


def test_build_full_command_includes_all_provided_fields():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "vae_path": "/m/vae.safetensors",
            "text_encoder_path": "/m/te.safetensors",
            "positive_prompt": "a fox",
            "negative_prompt": "blurry",
            "sampler": "euler_ancestral",
            "scheduler": "simple",
            "save_path": "/out/dir",
            "loras": [{"path": "/l/a.safetensors", "strength": "0.8"}, {"path": "/l/b.safetensors", "strength": "1.0"}],
        }
    )
    assert argv[argv.index("--vae") + 1] == "/m/vae.safetensors"
    assert argv[argv.index("--text_encoder") + 1] == "/m/te.safetensors"
    assert argv[argv.index("--negative_prompt") + 1] == "blurry"
    assert argv[argv.index("--sampler") + 1] == "euler_ancestral"
    assert argv[argv.index("--scheduler") + 1] == "simple"
    assert argv[argv.index("--save_path") + 1] == "/out/dir"
    lora_index = argv.index("--lora_list")
    assert argv[lora_index + 1 : lora_index + 5] == ["/l/a.safetensors", "0.8", "/l/b.safetensors", "1.0"]


def test_lora_rows_with_blank_path_are_skipped_and_strength_defaults():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "loras": [{"path": "", "strength": "2"}, {"path": "/l/keep.safetensors", "strength": ""}],
        }
    )
    lora_index = argv.index("--lora_list")
    assert argv[lora_index + 1 : lora_index + 3] == ["/l/keep.safetensors", "1.0"]


def test_image_size_flow_shift_and_seed_are_included_when_provided():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "image_height": "1216",
            "image_width": "832",
            "flow_shift": "6.0",
            "seed": "42",
        }
    )
    image_size_index = argv.index("--image_size")
    assert argv[image_size_index + 1 : image_size_index + 3] == ["1216", "832"]  # HEIGHT WIDTH
    assert argv[argv.index("--flow_shift") + 1] == "6.0"
    assert argv[argv.index("--seed") + 1] == "42"


def test_image_size_and_flow_shift_and_seed_omitted_when_blank():
    argv = build_inference_command({"dit_path": "/m/dit.safetensors", "positive_prompt": "p"})
    assert "--image_size" not in argv
    assert "--flow_shift" not in argv
    assert "--seed" not in argv


def test_seed_minus_one_allowed():
    argv = build_inference_command({"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "seed": "-1"})
    assert argv[argv.index("--seed") + 1] == "-1"


def test_partial_image_size_raises():
    try:
        build_inference_command({"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "image_height": "1024"})
    except ValueError:
        return
    raise AssertionError("expected ValueError when only one of height/width is given")


def test_bad_flow_shift_and_seed_raise():
    for bad in ({"flow_shift": "abc"}, {"seed": "3.5"}):
        request = {"dit_path": "/m/dit.safetensors", "positive_prompt": "p"}
        request.update(bad)
        try:
            build_inference_command(request)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_resolution_presets_are_divisible_by_32():
    assert len(ANIMA_RESOLUTION_PRESETS) > 0
    for preset in ANIMA_RESOLUTION_PRESETS:
        assert preset["width"] % 32 == 0, preset
        assert preset["height"] % 32 == 0, preset
        assert preset["label"]


def test_disabled_lora_rows_are_excluded():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "loras": [
                {"path": "/l/on.safetensors", "strength": "1.0", "enabled": True},
                {"path": "/l/off.safetensors", "strength": "1.0", "enabled": False},
                {"path": "/l/default.safetensors", "strength": "0.5"},  # missing enabled -> enabled
            ],
        }
    )
    lora_index = argv.index("--lora_list")
    lora_tokens = argv[lora_index + 1 :]
    lora_tokens = lora_tokens[: lora_tokens.index("--prompt")] if "--prompt" in lora_tokens else lora_tokens
    assert lora_tokens == ["/l/on.safetensors", "1.0", "/l/default.safetensors", "0.5"]
    assert "/l/off.safetensors" not in argv


def test_all_loras_disabled_yields_no_lora_flag():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "loras": [{"path": "/l/off.safetensors", "strength": "1.0", "enabled": False}],
        }
    )
    assert "--lora_list" not in argv


def test_from_image_mode_uses_from_image_embed_and_pre_prompt():
    argv = build_inference_command(
        {
            "mode": "from_image",
            "dit_path": "/m/dit.safetensors",
            "source_image_folder": "/src/pngs",
            "positive_prompt": "masterpiece",
            "negative_prompt": "blurry",
        }
    )
    assert argv[argv.index("--from_image_embed") + 1] == "/src/pngs"
    assert argv[argv.index("--pre_prompt") + 1] == "masterpiece"
    assert argv[argv.index("--pre_prompt_neg") + 1] == "blurry"
    assert "--prompt" not in argv
    assert "--negative_prompt" not in argv


def test_from_prompt_list_mode_uses_from_file_and_pre_prompt():
    argv = build_inference_command(
        {
            "mode": "from_prompt_list",
            "dit_path": "/m/dit.safetensors",
            "prompt_list_path": "/tmp/prompts.txt",
            "positive_prompt": "masterpiece",
        }
    )
    assert argv[argv.index("--from_file") + 1] == "/tmp/prompts.txt"
    assert argv[argv.index("--pre_prompt") + 1] == "masterpiece"
    assert "--prompt" not in argv


def test_from_image_requires_source_folder():
    try:
        build_inference_command({"mode": "from_image", "dit_path": "/m/dit.safetensors"})
    except ValueError:
        return
    raise AssertionError("expected ValueError when source_image_folder missing")


def test_from_prompt_list_requires_path():
    try:
        build_inference_command({"mode": "from_prompt_list", "dit_path": "/m/dit.safetensors"})
    except ValueError:
        return
    raise AssertionError("expected ValueError when prompt_list_path missing")


def test_single_mode_still_requires_positive_prompt():
    try:
        build_inference_command({"mode": "single", "dit_path": "/m/dit.safetensors"})
    except ValueError:
        return
    raise AssertionError("expected ValueError when positive_prompt missing in single mode")


def test_multiline_prompt_is_a_single_argv_element():
    multiline_prompt = "line one,\nline two,\nline three"
    argv = build_inference_command({"dit_path": "/m/dit.safetensors", "positive_prompt": multiline_prompt})
    assert argv[argv.index("--prompt") + 1] == multiline_prompt


def test_missing_dit_or_prompt_raises():
    for bad_request in ({"positive_prompt": "p"}, {"dit_path": "/m/dit.safetensors"}, {"dit_path": "/m/dit.safetensors", "positive_prompt": "   "}):
        try:
            build_inference_command(bad_request)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad_request}")


def test_parse_choice_tuple_from_source():
    source = 'SAMPLER_OPTION_CHOICES = ("euler", "er_sde", "euler_ancestral")\nSCHEDULER_OPTION_CHOICES = ("default", "beta57", "simple")\n'
    assert _parse_choice_tuple_from_source(source, "SAMPLER_OPTION_CHOICES") == ("euler", "er_sde", "euler_ancestral")
    assert _parse_choice_tuple_from_source(source, "SCHEDULER_OPTION_CHOICES") == ("default", "beta57", "simple")
    assert _parse_choice_tuple_from_source(source, "MISSING") == ()


def test_load_choices_reads_from_real_inference_script():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    choices = load_sampler_and_scheduler_choices(repo_root)
    assert "euler_ancestral" in choices["samplers"]
    assert "simple" in choices["schedulers"]


def test_load_choices_falls_back_when_script_missing(tmp_path):
    choices = load_sampler_and_scheduler_choices(str(tmp_path))
    assert choices["samplers"] == ["euler", "er_sde", "euler_ancestral"]
    assert choices["schedulers"] == ["default", "beta57", "simple"]


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func) and "tmp_path" not in func.__code__.co_varnames[: func.__code__.co_argcount]:
            func()
    print("ok")
