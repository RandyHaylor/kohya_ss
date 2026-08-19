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
- **`--model_load_disk_lock_file <path>`**: serialize model-file disk reads ACROSS processes. While a
  process reads the DiT/text-encoder/VAE/LoRA weights from disk it holds an exclusive `flock` on this
  file; other inference processes sharing the same path wait until it finishes loading. Denoising runs
  outside the lock, so concurrent runs stagger disk-load vs GPU work. No flag = no locking (unchanged).
  The lock is held across a whole **model-loading phase** — one process keeps it while loading ALL the
  files it needs (VAE + DiT + text encoder), released only before GPU work — so it is NOT dropped between
  files (dropping between files caused thrashing: a competing process grabbed it mid-load and requeued
  the first). Each mode wraps its contiguous load region in `serialize_model_loading_phase(args)`, and the
  per-file `serialize_model_file_disk_reads` locks inside `load_dit_model` / `load_text_encoder` /
  `load_qwen_image_vae_serialized` nest under it. Both go through `hold_exclusive_cross_process_file_lock`,
  which is **re-entrant per path within a thread** (thread-local depth map) so the phase lock and the
  nested per-file locks share one held `flock` without self-deadlocking (a second `flock` on a new fd of
  the same file would otherwise block against the first, even in one process). Streaming modes load VAE+TE
  under the phase lock up front; their DiT loads lazily in `generate()` under its own held per-file lock.
  POSIX-only (no-op if `fcntl` is unavailable). The web GUI passes a fixed lock path to every spawned gen.
- **`--gpu_compute_lock_file <path>` / `--gpu_lock_scope {denoise,all}`**: serialize GPU compute ACROSS
  processes so only one uses the GPU at a time. Combined with the disk lock, concurrent runs become a
  pipeline (one loads from disk while another uses the GPU). `denoise` (default) locks only the denoise
  loop (the heavy sustained cost); `all` also locks text-encode + VAE-decode (one process on the GPU at
  any instant). Implemented via `serialize_gpu_compute(args, gpu_phase)` wrapping the three GPU spans;
  every guarded span is disk-read free, so the GPU lock can't deadlock against the disk lock (a process
  never waits on the disk lock while holding the GPU lock). Uses the shared
  `hold_exclusive_cross_process_file_lock` context manager (same `flock` core as the disk lock).
  In `process_batch_prompts` (single `--prompt` incl. `--images_per_prompt N`, and `--from_file`) the GPU
  lock is held ACROSS the whole batch — the text-encode precompute loop and every image's generate/decode
  as one block — so a multi-image batch keeps the GPU for all N images (no release between images) rather
  than re-acquiring per image. The per-image locks nest under it via the re-entrant lock.
- **`--prompt_count` / `--prompt_count_skip_first`**: usable-counted limit/pagination (skipped/unusable
  prompts don't count). **`--images_per_prompt N`**: N seed-incremented images per prompt in ONE model
  load — honored in **single `--prompt`** and **`--from_image_embed`** (NOT from_file/from_folder yet).
- **Anima-Safe PAG** (`--pag` + `--pag_scale/--pag_block_indices/--pag_perturbation_strength/`
  `--pag_head_indices/--pag_start_percent/--pag_end_percent/--pag_rescale/--pag_rescale_mode`): a **faithful
  port** of https://github.com/iljung1106/comfyui-anima-safe-pag. On active steps it runs one extra
  conditional forward with selected self-attention blocks blended toward the value path (`lerp(attn, v,
  perturbation_strength)`, optional per-head) and steers the CFG result away from that weak prediction
  (`guidance = (cond − pag) * scale`, std-rescaled). Perturbation hook lives in `library/anima_models.py`
  (`apply_soft_pag_attention_perturbation`, `Attention.pag_perturbation`, `Anima.enable/disable_soft_pag_perturbation`);
  the guidance/active-range/index helpers are pure in `anima_minimal_inference.py`
  (`rescale_pag_guidance`, `pag_is_active_for_sigma`, `parse_pag_index_spec`). Defaults match the repo
  (scale 4.0, block 18, strength 0.75, start 0.0, end 0.7, rescale 0.20, mode full). Costs an extra DiT
  forward per active step (on top of CFG's two). GPU A/B render is the final proof — the pure helpers are
  unit-tested but the perturbed forward is not. GUI: **Anima-Safe PAG** checkbox (default off) + fields.
  PAG state (incl. `enabled`) is recorded under a **`pag`** key in the JSON settings sidecar
  (`build_generation_settings_dict`); images made before PAG have no `pag` key. In the GUI, a generated
  image's **Load** control is split into three buttons: **Settings** (gen params incl. PAG, NOT model
  paths, NOT LoRAs), **Models** (dit/vae/text_encoder only), and **LoRAs** (the LoRA stack — paths,
  strengths, enabled). Loading Settings from an image with no `pag` key fills the PAG defaults and leaves
  the checkbox unchecked.
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
  command and runs queued generations up to **`max_concurrent_generations`** at once (default 1; startup
  flag `--max_concurrent_generations` and a live UI control — the `/concurrency` GET/POST endpoint). Each
  concurrent generation is its own subprocess loading its own full model copy (N× VRAM, GPU time-slices),
  so raise it only as far as VRAM allows. A single dispatcher thread gates on a Condition; "Stop Current"
  terminates **all** in-flight processes. The server passes every spawned generation a shared
  `--model_load_disk_lock_file` (`model_load_disk.lock`) and `--gpu_compute_lock_file` (`gpu_compute.lock`),
  so concurrent runs read model files from disk one at a time and use the GPU one at a time (a load-vs-GPU
  pipeline). Three topbar checkboxes control it: **lock disk** / **lock GPU** (both default on) and
  **strict** (default off → GPU `all`-compute scope vs denoise-only), translated to flags server-side by
  `apply_resource_serialization_locks_from_request`. Launch (own window, to watch output):
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
