import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_command_builder import (
    ANIMA_RESOLUTION_PRESETS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_INFER_STEPS,
    DEFAULT_SAVE_PATH,
    build_inference_command,
    coerce_concurrency_to_positive_int,
    coerce_quantity_to_positive_int,
    coerce_to_float_with_default,
    coerce_to_int_with_default,
    load_sampler_and_scheduler_choices,
    _parse_choice_tuple_from_source,
)


def collect_flag_values(argv, flag):
    """Return the argv tokens after `flag`, up to (not including) the next '--' option token."""
    values = []
    for token in argv[argv.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


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


def test_coerce_concurrency_to_positive_int():
    assert coerce_concurrency_to_positive_int(1) == 1
    assert coerce_concurrency_to_positive_int("4") == 4
    assert coerce_concurrency_to_positive_int("  3  ") == 3
    assert coerce_concurrency_to_positive_int("2.9") == 2  # float truncates
    assert coerce_concurrency_to_positive_int("abc") == 1  # non-numeric -> 1
    assert coerce_concurrency_to_positive_int("") == 1
    assert coerce_concurrency_to_positive_int("0") == 1  # below 1 -> 1
    assert coerce_concurrency_to_positive_int("-3") == 1
    assert coerce_concurrency_to_positive_int(None) == 1


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


def test_dynamic_shift_flags_emitted_for_dynamic_scheduler():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "scheduler": "diffusers_dynamic",
            "dynamic_base_shift": "0.4",
            "dynamic_max_shift": "1.1",
        }
    )
    assert argv[argv.index("--dynamic_base_shift") + 1] == "0.4"
    assert argv[argv.index("--dynamic_max_shift") + 1] == "1.1"


def test_dynamic_shift_flags_absent_for_non_dynamic_scheduler():
    argv = build_inference_command(
        {"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "scheduler": "diffusers", "dynamic_base_shift": "0.4"}
    )
    assert "--dynamic_base_shift" not in argv
    assert "--dynamic_max_shift" not in argv


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


def test_1536_basis_presets_are_appended_after_the_1024_basis_ones():
    expected_1536_basis_dimensions_in_order = [
        (1536, 1536),
        (1856, 1280),
        (1280, 1856),
        (1728, 1344),
        (1344, 1728),
        (2048, 1152),
        (1152, 2048),
        (2304, 960),
        (960, 2304),
    ]
    trailing_presets = ANIMA_RESOLUTION_PRESETS[-len(expected_1536_basis_dimensions_in_order):]
    actual_trailing_dimensions = [(preset["width"], preset["height"]) for preset in trailing_presets]
    assert actual_trailing_dimensions == expected_1536_basis_dimensions_in_order

    # The 1024-basis presets remain first (1024x1024 leads the list, unchanged).
    assert (ANIMA_RESOLUTION_PRESETS[0]["width"], ANIMA_RESOLUTION_PRESETS[0]["height"]) == (1024, 1024)


def test_images_per_prompt_emitted_when_gt_one():
    argv = build_inference_command({"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "images_per_prompt": 4})
    assert argv[argv.index("--images_per_prompt") + 1] == "4"


def test_images_per_prompt_omitted_when_one_or_absent():
    argv_one = build_inference_command({"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "images_per_prompt": 1})
    argv_absent = build_inference_command({"dit_path": "/m/dit.safetensors", "positive_prompt": "p"})
    assert "--images_per_prompt" not in argv_one
    assert "--images_per_prompt" not in argv_absent


def test_model_load_disk_lock_file_emitted_when_provided():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "model_load_disk_lock_file": "/tmp/model_load_disk.lock",
        }
    )
    assert argv[argv.index("--model_load_disk_lock_file") + 1] == "/tmp/model_load_disk.lock"


def test_model_load_disk_lock_file_omitted_when_blank_or_absent():
    argv_blank = build_inference_command(
        {"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "model_load_disk_lock_file": "  "}
    )
    argv_absent = build_inference_command({"dit_path": "/m/dit.safetensors", "positive_prompt": "p"})
    assert "--model_load_disk_lock_file" not in argv_blank
    assert "--model_load_disk_lock_file" not in argv_absent


def test_gpu_compute_lock_file_and_scope_emitted_when_provided():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "gpu_compute_lock_file": "/tmp/gpu_compute.lock",
            "gpu_lock_scope": "all",
        }
    )
    assert argv[argv.index("--gpu_compute_lock_file") + 1] == "/tmp/gpu_compute.lock"
    assert argv[argv.index("--gpu_lock_scope") + 1] == "all"


def test_gpu_lock_scope_omitted_when_no_lock_file():
    argv = build_inference_command(
        {"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "gpu_lock_scope": "all"}
    )
    assert "--gpu_compute_lock_file" not in argv
    assert "--gpu_lock_scope" not in argv  # scope is meaningless without the lock file, so not emitted


def test_gpu_compute_lock_file_omitted_when_blank_or_absent():
    argv_blank = build_inference_command(
        {"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "gpu_compute_lock_file": "  "}
    )
    argv_absent = build_inference_command({"dit_path": "/m/dit.safetensors", "positive_prompt": "p"})
    assert "--gpu_compute_lock_file" not in argv_blank
    assert "--gpu_compute_lock_file" not in argv_absent


def test_pag_flags_emitted_when_enabled():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "pag_enabled": True,
            "pag_scale": "4.0",
            "pag_block_indices": "18-20",
            "pag_perturbation_strength": "0.75",
            "pag_head_indices": "0,1",
            "pag_start_percent": "0.0",
            "pag_end_percent": "0.7",
            "pag_rescale": "0.2",
            "pag_rescale_mode": "full",
        }
    )
    assert "--pag" in argv
    assert argv[argv.index("--pag_scale") + 1] == "4.0"
    assert argv[argv.index("--pag_block_indices") + 1] == "18-20"
    assert argv[argv.index("--pag_perturbation_strength") + 1] == "0.75"
    assert argv[argv.index("--pag_head_indices") + 1] == "0,1"
    assert argv[argv.index("--pag_start_percent") + 1] == "0.0"
    assert argv[argv.index("--pag_end_percent") + 1] == "0.7"
    assert argv[argv.index("--pag_rescale") + 1] == "0.2"
    assert argv[argv.index("--pag_rescale_mode") + 1] == "full"


def test_pag_flags_absent_when_disabled():
    argv = build_inference_command({"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "pag_enabled": False})
    assert "--pag" not in argv
    assert "--pag_scale" not in argv


def test_pag_head_indices_omitted_when_blank():
    argv = build_inference_command(
        {"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "pag_enabled": True, "pag_head_indices": "  "}
    )
    assert "--pag" in argv
    assert "--pag_head_indices" not in argv  # blank = all heads, flag omitted


def test_flow_matched_pag_flags_emitted_when_enabled():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "pag_enabled": True,
            "flow_matched_pag_enabled": True,
            "flow_matched_pag_strength": "0.5",
            "flow_matched_pag_curve_exponent": "2.0",
        }
    )
    assert "--flow_matched_pag" in argv
    assert argv[argv.index("--flow_matched_pag_strength") + 1] == "0.5"
    assert argv[argv.index("--flow_matched_pag_curve_exponent") + 1] == "2.0"


def test_flow_matched_pag_flags_absent_when_pag_disabled():
    # FlowMatched PAG scales PAG, so it only emits inside the pag_enabled block.
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "pag_enabled": False,
            "flow_matched_pag_enabled": True,
        }
    )
    assert "--flow_matched_pag" not in argv


def test_flow_matched_pag_flags_absent_when_disabled():
    argv = build_inference_command(
        {"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "pag_enabled": True, "flow_matched_pag_enabled": False}
    )
    assert "--flow_matched_pag" not in argv


def test_lora_test_folder_emitted_with_default_multiplier():
    argv = build_inference_command(
        {"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "lora_test_folder": "/loras/_totest"}
    )
    index = argv.index("--lora_test_folder")
    assert argv[index + 1 : index + 3] == ["/loras/_totest", "1.0"]


def test_lora_test_folder_uses_provided_multiplier():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "lora_test_folder": "/loras/_totest",
            "lora_test_multiplier": "0.8",
        }
    )
    index = argv.index("--lora_test_folder")
    assert argv[index + 1 : index + 3] == ["/loras/_totest", "0.8"]


def test_lora_test_folder_omitted_when_blank():
    argv = build_inference_command({"dit_path": "/m/dit.safetensors", "positive_prompt": "p"})
    assert "--lora_test_folder" not in argv


def test_dit_test_folder_emitted_when_set():
    argv = build_inference_command(
        {"dit_path": "/m/dit.safetensors", "positive_prompt": "p", "dit_test_folder": "/dits/_totest"}
    )
    index = argv.index("--dit_test_folder")
    assert argv[index + 1] == "/dits/_totest"


def test_dit_test_folder_omitted_when_blank():
    argv = build_inference_command({"dit_path": "/m/dit.safetensors", "positive_prompt": "p"})
    assert "--dit_test_folder" not in argv


def test_dit_path_may_be_empty_when_dit_test_folder_set():
    argv = build_inference_command(
        {"dit_path": "", "positive_prompt": "p", "dit_test_folder": "/dits/_totest"}
    )
    assert "--dit" not in argv
    index = argv.index("--dit_test_folder")
    assert argv[index + 1] == "/dits/_totest"


def test_dit_path_required_when_no_dit_test_folder():
    try:
        build_inference_command({"dit_path": "", "positive_prompt": "p"})
    except ValueError:
        return
    raise AssertionError("expected ValueError when dit_path is empty and no DiT test folder is set")


def test_dit_path_required_when_dit_test_folder_disabled():
    try:
        build_inference_command(
            {
                "dit_path": "",
                "positive_prompt": "p",
                "dit_test_folder": "/dits/_totest",
                "dit_test_folder_enabled": False,
            }
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError when dit_path is empty and the DiT test folder is disabled")


def test_dit_test_folder_omitted_when_disabled():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "dit_test_folder": "/dits/_totest",
            "dit_test_folder_enabled": False,
        }
    )
    assert "--dit_test_folder" not in argv


def test_lora_test_folder_omitted_when_disabled():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "lora_test_folder": "/loras/_totest",
            "lora_test_folder_enabled": False,
        }
    )
    assert "--lora_test_folder" not in argv


def test_dit_test_folder_nested_with_lora_test_folder():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "dit_test_folder": "/dits/_totest",
            "lora_test_folder": "/loras/_totest",
        }
    )
    dit_index = argv.index("--dit_test_folder")
    assert argv[dit_index + 1] == "/dits/_totest"
    lora_index = argv.index("--lora_test_folder")
    assert argv[lora_index + 1 : lora_index + 3] == ["/loras/_totest", "1.0"]


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
    lora_tokens = collect_flag_values(argv, "--lora_list")
    assert lora_tokens == ["/l/on.safetensors", "1.0", "/l/default.safetensors", "0.5"]
    assert "/l/off.safetensors" not in lora_tokens  # disabled row is not merged


def test_all_lora_rows_including_disabled_are_recorded_for_the_sidecar():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "loras": [
                {"path": "/l/on.safetensors", "strength": "1.0", "enabled": True},
                {"path": "/l/off.safetensors", "strength": "0.7", "enabled": False},
            ],
        }
    )
    recorded_rows = json.loads(argv[argv.index("--record_lora_rows_json") + 1])
    assert recorded_rows == [
        {"path": "/l/on.safetensors", "multiplier": "1.0", "enabled": True},
        {"path": "/l/off.safetensors", "multiplier": "0.7", "enabled": False},
    ]


def test_all_loras_disabled_yields_no_lora_flag_but_still_records_rows():
    argv = build_inference_command(
        {
            "dit_path": "/m/dit.safetensors",
            "positive_prompt": "p",
            "loras": [{"path": "/l/off.safetensors", "strength": "1.0", "enabled": False}],
        }
    )
    assert "--lora_list" not in argv
    recorded_rows = json.loads(argv[argv.index("--record_lora_rows_json") + 1])
    assert recorded_rows == [{"path": "/l/off.safetensors", "multiplier": "1.0", "enabled": False}]


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
    assert "diffusers" in choices["schedulers"]
    assert "diffusers_dynamic" in choices["schedulers"]


def test_load_choices_falls_back_when_script_missing(tmp_path):
    choices = load_sampler_and_scheduler_choices(str(tmp_path))
    assert choices["samplers"] == ["euler", "er_sde", "euler_ancestral"]
    assert choices["schedulers"] == ["default", "beta57", "simple", "diffusers", "diffusers_dynamic"]


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func) and "tmp_path" not in func.__code__.co_varnames[: func.__code__.co_argcount]:
            func()
    print("ok")
