# CLAUDE.md — working notes for this workspace

Guidance for agents working in `kohya_ss` (this repo) and its `sd-scripts` submodule. Everything here
was verified during the work that produced it; re-verify with tools before relying on any detail, and
correct this file when you find it stale.

## Repos, forks, and git workflow

- **Two git repos:**
  - Parent: `kohya_ss` (this dir). Working branch: **`master`**. `origin` = `bmaltais/kohya_ss` (upstream,
    not ours); **`randyhaylor` = `RandyHaylor/kohya_ss` (our fork)** — push here.
  - Submodule: `sd-scripts/` (separate repo). Working branch: **`main`**. `origin` = `kohya-ss/sd-scripts`
    (upstream); **`randyhaylor` = `RandyHaylor/sd-scripts` (our fork)** — push here.
- **We work directly on `master`/`main` — no feature branches.** Commits are the unit of separation.
- **Push to the `randyhaylor` remotes**, never to `origin` (upstream, no write access assumed).
- **Submodule pin:** the parent records a specific `sd-scripts` commit SHA (a lockfile-style pin), not
  "latest on the fork". After committing in `sd-scripts` and pushing, you must **bump** the pin in the
  parent: from the parent, `git add sd-scripts` then commit + push. Otherwise the parent still points at
  the old sd-scripts commit. `.gitmodules` already points `sd-scripts` at the fork with `branch = main`.
- Typical change spanning both: (1) commit+push in `sd-scripts`; (2) in parent `git add sd-scripts` +
  any parent files, commit ("Bump sd-scripts: …"), push.
- **Commit messages:** no Claude/AI attribution or session lines (user's standing rule). Describe the change.

## The model & the inference script

- **Anima** is a Qwen-Image–based **rectified-flow (CONST)** DiT. The DiT outputs velocity `v`; the
  denoised estimate is `x0 = x - sigma*v`. Flow sigmas are in `[0,1]` descending to 0.
- Main script: **`sd-scripts/anima_minimal_inference.py`** (run with `uv run sd-scripts/anima_minimal_inference.py …`).
- **Model inputs:** split files via `--dit` / `--vae` / `--text_encoder`. `--dit` also accepts an
  **all-in-one civitai/ComfyUI checkpoint** (DiT+VAE+text-encoder baked in under `model.diffusion_model.`
  / `first_stage_model.` / `cond_stage_model.`); it's auto-detected and extracted once to a sibling
  folder named for the model (reused after), making `--vae`/`--text_encoder` optional.
- **Generation modes** (see `dispatch_generation`): single `--prompt`; `--from_file <prompts.txt>`
  (one prompt per line, per-line `--w/--h/--s/--l/--g/--fs/--d/--n` overrides); `--from_folder <dir>`
  (caption .txt files); `--from_image_embed <folder>` (pull prompts from PNG metadata — A1111
  `parameters`, ComfyUI `prompt` graph, and EXIF UserComment; keywords `prompts_only`,
  `ignore_negative_prompt`, `prompt_only_and_all_settings`).
- **`--lora_test_folder <dir> [mult]`**: A/B sweep — runs the whole configured generation once per
  top-level `.safetensors` on top of the fixed LoRAs (reloads models per test LoRA). A `<loraname>.txt`
  sidecar injects trigger text after `--pre_prompt`.
- **`--prompt_count` / `--prompt_count_skip_first`**: usable-counted limit/pagination (skipped/unusable
  prompts don't count). **`--images_per_prompt N`**: N seed-incremented images per prompt in ONE model
  load — honored in **single `--prompt`** and **`--from_image_embed`** (NOT from_file/from_folder yet).
- **Samplers** (`--sampler`): `euler`, `er_sde`, `euler_ancestral`. **Schedulers** (`--scheduler`):
  `default`, `beta57`, `simple`. Defaults are **`er_sde` / `beta57`**. Sampler/scheduler ports live in
  `library/anima_er_sde_sampling.py` and are **faithful ports of ComfyUI** — verify against the ComfyUI
  source (fetch it) before touching the math; don't hand-roll.
- **Known-correct behaviors (don't "fix"):** `default` and `simple` schedulers are (near-)identical for a
  flow model — that matches ComfyUI (both reduce to shift-of-even-timesteps). `beta57` is the one that
  meaningfully differs.
- **Settings sidecar `<name>.txt`** is written **before** generation in all modes (readable while
  rendering). Output PNGs also embed an A1111-style `parameters` chunk (path-free) via
  `build_png_generation_metadata_text`.

## LoRA formats (a real gotcha)

- `load_dit_model` → `select_dit_lora_state_dict` auto-detects: keeps kohya `lora_unet_` keys, else
  converts ComfyUI/PEFT `diffusion_model.<module>.lora_A/lora_B/.alpha` → the `lora_down/lora_up/.alpha`
  form the merge (`library/lora_utils.py` `weight_hook_func`) matches on bare model keys (`.`→`_`).
- Text-encoder side still only handles kohya `lora_te_`. ComfyUI LoRAs with TE keys aren't converted yet
  (the anima turbo LoRAs have DiT keys only).
- Debugging "LoRA has no effect": inspect the LoRA's key prefixes with `safetensors.safe_open`; if all
  keys were filtered out, it merged nothing.

## Tests (required for code changes — write/adjust them first)

- sd-scripts: **`sd-scripts/tests/test_anima_from_image_embed.py`** (pure-helper coverage for the inference
  features). Run: `cd sd-scripts` then `uv run python -m pytest tests/test_anima_from_image_embed.py -q`.
- GUI: **`anima_web_gui/test_inference_command_builder.py`**. Run: `cd anima_web_gui` then
  `python3 -m pytest test_inference_command_builder.py -q` (stdlib only, no `uv`).
- Prefer extracting a **pure function** for new logic so it's unit-testable without a GPU. GPU renders
  are the final proof for model-loading/sampler changes — do an A/B and compare (md5/pixels), don't
  assume.

## Local web GUI

- `anima_web_gui/` — **stdlib-only** HTTP server that builds the single-`--prompt`/from_image/from_file
  command and runs generations one at a time via a queue. Launch (own window, to watch output):
  `python3 /media/aikenyon/NVME_2/kohya_ss/anima_web_gui/anima_inference_gui_server.py` → http://127.0.0.1:7861.
- `inference_command_builder.py` = pure argv builder (unit-tested); `anima_inference_gui_server.py` =
  server + embedded HTML/JS. Sampler/scheduler choices are parsed out of the inference script source
  (no import) so they can't drift. GUI **quantity → `--images_per_prompt`** (one subprocess, N images).
- Verify GUI/DOM changes with the **playwright-browser-emulation** skill (screenshots + `eval`); the
  sandbox may block inline `eval` — put the CLI call in a small script if so.

## Environment / tooling caveats

- Always run Python via **`uv run`** in `sd-scripts` (deps/venv). The GUI server is plain `python3`.
- The Bash tool blocks command **chaining/expansion** (`;`, `&&`, `$()`, pipes into some commands, `sed`).
  Write a small `/tmp/*.sh` script and run that instead.
- **Do not commit** `anima_out/` (generated images) or `dataset/…` (~hundreds of MB). `anima_web_gui/`
  has a `.gitignore` for `generation.log` and `queued_prompt_lists/`.
- **`inference-command.txt`, `prompts.txt`** at the parent root are the user's **personal notes/scratch**,
  not docs — confirm before editing them.

## User working preferences (applied here)

- Verbose, purpose-driven names clear out of context. Self-documenting code; comments say what IS, not
  what changed.
- For a fix: state the minimal vs the proper fix, then do the proper one. Verify claims with tools; say
  plainly what you did/didn't run. Finding a mismatch is not fixing it — prove the fix.
