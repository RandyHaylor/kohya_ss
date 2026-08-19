"""Tiny local web GUI for single-prompt Anima generations.

Scope: build the embedded --prompt command line from a form (positive/negative prompt, sampler and
scheduler dropdowns, model paths, and an add/remove LoRA list) and run generations from a queue, up to
max_concurrent_generations at once (default 1; each concurrent generation loads its own full copy of the
model, so raise it only as far as VRAM allows). Buttons: Queue Gen, Stop Current, Stop All. Each spawned generation's console output
is streamed to this server's stdout (launch it in your own window to watch it) and appended to a log
file next to this script. Stdlib only - no torch/venv needed to run the server; it just spawns
`uv run sd-scripts/anima_minimal_inference.py ...` from the repo root.
"""

import argparse
import collections
import json
import os
import subprocess
import tempfile
import threading
import queue as queue_module
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from inference_command_builder import (
    ANIMA_RESOLUTION_PRESETS,
    DEFAULT_SAVE_PATH,
    build_inference_command,
    coerce_concurrency_to_positive_int,
    coerce_quantity_to_positive_int,
    load_sampler_and_scheduler_choices,
)
from generated_image_gallery import (
    is_path_within_allowed_directories,
    list_generated_png_files_in_directories,
    resolve_save_directory_absolute_path,
    resolve_settings_sidecar_path_for_image,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATION_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generation.log")
QUEUED_PROMPT_LISTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queued_prompt_lists")
# Shared by every spawned generation so their model-file disk reads serialize across processes (one loads
# from disk at a time; see --model_load_disk_lock_file in the inference script).
MODEL_LOAD_DISK_LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_load_disk.lock")
# Shared GPU lock so concurrent generations use the GPU one at a time (see --gpu_compute_lock_file).
GPU_COMPUTE_LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_compute.lock")
GPU_LOCK_SCOPE_STRICT = "all"  # lock text-encode + denoise + VAE-decode
GPU_LOCK_SCOPE_DENOISE_ONLY = "denoise"  # lock only the denoise loop

DEFAULT_MAX_CONCURRENT_GENERATIONS = 1

generation_request_queue: "queue_module.Queue[dict]" = queue_module.Queue()
server_state_lock = threading.Lock()
# A slot frees (a generation finishes) -> notify the dispatcher so it can start the next queued one.
# Reuses server_state_lock so the running-set, the counter, and the limit are all read/written under one lock.
generation_slot_available_condition = threading.Condition(server_state_lock)
max_concurrent_generations = DEFAULT_MAX_CONCURRENT_GENERATIONS
active_generation_count = 0  # generations currently running (dispatched, not yet finished)
next_running_generation_id = 0  # monotonic id handed to each dispatched generation
running_generations_by_id = {}  # id -> {"process": Popen, "label": str} for every in-flight generation
recent_log_lines = collections.deque(maxlen=500)
generation_log_file = open(GENERATION_LOG_PATH, "a", encoding="utf-8")
observed_save_directories = []  # ordered, de-duplicated absolute save dirs seen this session (for the gallery)


def append_log_line(text: str) -> None:
    """Record one line to the in-memory tail (for the UI), this server's stdout, and the log file."""
    line = text.rstrip("\n")
    with server_state_lock:
        recent_log_lines.append(line)
    print(line, flush=True)
    generation_log_file.write(line + "\n")
    generation_log_file.flush()


def register_observed_save_directory(save_path: str) -> None:
    """Record (once) the absolute save directory for a generation so the gallery knows where to look
    for its output PNGs and is allowed to serve them."""
    absolute_directory = resolve_save_directory_absolute_path(save_path or DEFAULT_SAVE_PATH, REPO_ROOT)
    with server_state_lock:
        if absolute_directory not in observed_save_directories:
            observed_save_directories.append(absolute_directory)


def list_gallery_images_newest_first() -> list:
    """Return the generated PNGs across every observed save directory, newest-first, as
    {'path', 'name', 'mtime'} dicts for the gallery panel."""
    with server_state_lock:
        directories = list(observed_save_directories)
    listing = list_generated_png_files_in_directories(directories)
    listing.reverse()  # helper returns oldest-first; the gallery shows newest first
    return [
        {"path": entry["absolute_path"], "name": entry["file_name"], "mtime": entry["modified_time"]}
        for entry in listing
    ]


def materialize_pasted_prompt_list(generation_request: dict) -> None:
    """For from_prompt_list mode: if the user pasted a prompt list (and gave no file path), write it to
    a .txt file and set prompt_list_path to it, so the CLI's --from_file has a real file to read."""
    if generation_request.get("mode") != "from_prompt_list":
        return
    existing_path = str(generation_request.get("prompt_list_path", "")).strip()
    pasted_text = str(generation_request.get("prompt_list_text", "")).strip()
    if existing_path or not pasted_text:
        return
    os.makedirs(QUEUED_PROMPT_LISTS_DIR, exist_ok=True)
    file_descriptor, list_path = tempfile.mkstemp(suffix=".txt", prefix="prompts_", dir=QUEUED_PROMPT_LISTS_DIR)
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as list_file:
        list_file.write(pasted_text + "\n")
    generation_request["prompt_list_path"] = list_path
    append_log_line(f"Wrote pasted prompt list to: {list_path}")


def apply_resource_serialization_locks_from_request(generation_request: dict) -> None:
    """Translate the GUI's serialization checkboxes into the inference script's lock flags, using this
    server's shared lock file paths so all spawned generations contend on the same locks.

    - serialize_disk_loads (default on): pass --model_load_disk_lock_file so only one process reads model
      files from disk at a time.
    - serialize_gpu_compute (default on): pass --gpu_compute_lock_file so only one process uses the GPU
      at a time. gpu_lock_strict (default off) selects the 'all compute' scope (text-encode + denoise +
      VAE-decode) vs denoise-only.
    """
    if generation_request.get("serialize_disk_loads", True):
        generation_request["model_load_disk_lock_file"] = MODEL_LOAD_DISK_LOCK_PATH
    if generation_request.get("serialize_gpu_compute", True):
        generation_request["gpu_compute_lock_file"] = GPU_COMPUTE_LOCK_PATH
        generation_request["gpu_lock_scope"] = (
            GPU_LOCK_SCOPE_STRICT if generation_request.get("gpu_lock_strict", False) else GPU_LOCK_SCOPE_DENOISE_ONLY
        )


def make_request_label(generation_request: dict) -> str:
    prompt_preview = str(generation_request.get("positive_prompt", "")).strip().replace("\n", " ")
    return (prompt_preview[:60] + "...") if len(prompt_preview) > 60 else (prompt_preview or "(no prompt)")


def set_max_concurrent_generations(raw_value) -> int:
    """Set how many generations may run at once (coerced to >= 1) and wake the dispatcher so it can
    fill any newly-available slots. Returns the value that was set."""
    global max_concurrent_generations
    new_limit = coerce_concurrency_to_positive_int(raw_value)
    with generation_slot_available_condition:
        max_concurrent_generations = new_limit
        generation_slot_available_condition.notify_all()
    return new_limit


def get_max_concurrent_generations() -> int:
    with server_state_lock:
        return max_concurrent_generations


def _register_running_generation(process, label: str) -> int:
    """Record a freshly-spawned generation under a new id (for status display and stop)."""
    global next_running_generation_id
    with server_state_lock:
        generation_id = next_running_generation_id
        next_running_generation_id += 1
        running_generations_by_id[generation_id] = {"process": process, "label": label}
    return generation_id


def _deregister_running_generation(generation_id: int) -> None:
    with server_state_lock:
        running_generations_by_id.pop(generation_id, None)


def _release_generation_slot() -> None:
    """Free the slot the dispatcher reserved for one generation and wake the dispatcher. Called exactly
    once per dispatched generation, by its worker thread, whether it succeeded or failed."""
    global active_generation_count
    with generation_slot_available_condition:
        active_generation_count -= 1
        generation_slot_available_condition.notify_all()


def run_one_generation(generation_request: dict) -> None:
    """Spawn the inference CLI for one request and stream its output until it returns. Registers the
    process in the running set so status/stop can see it, and deregisters it when done."""
    command_argv = build_inference_command(generation_request)
    label = make_request_label(generation_request)
    append_log_line(f"=== START generation: {label} ===")
    append_log_line("command: " + " ".join(command_argv))

    process = subprocess.Popen(
        command_argv,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    generation_id = _register_running_generation(process, label)
    try:
        for output_line in process.stdout:
            append_log_line(output_line)
        process.wait()
        append_log_line(f"=== END generation (exit code {process.returncode}): {label} ===")
    finally:
        _deregister_running_generation(generation_id)


def _run_generation_in_worker_thread(generation_request: dict) -> None:
    """Body of each concurrent worker thread: run one generation, then always free its slot and mark
    the queue item done. Owns the slot counter so it is released exactly once per generation."""
    try:
        run_one_generation(generation_request)
    except Exception as error:  # keep the server alive across a single bad request
        append_log_line(f"ERROR running generation: {error}")
    finally:
        _release_generation_slot()
        generation_request_queue.task_done()


def generation_dispatcher_loop() -> None:
    """Single dispatcher: wait for a free slot (up to max_concurrent_generations), pull one queued
    request, reserve a slot, and hand it to a worker thread. Only this thread increments the active
    count, so the concurrency limit can never be exceeded even while it changes at runtime."""
    global active_generation_count
    while True:
        with generation_slot_available_condition:
            while active_generation_count >= max_concurrent_generations:
                generation_slot_available_condition.wait()
        generation_request = generation_request_queue.get()  # blocks until something is queued
        with generation_slot_available_condition:
            active_generation_count += 1
        worker_thread = threading.Thread(
            target=_run_generation_in_worker_thread,
            args=(generation_request,),
            name="generation-worker",
            daemon=True,
        )
        worker_thread.start()


def terminate_all_running_generations() -> int:
    """Signal every in-flight generation to stop (SIGTERM). Returns how many were running."""
    with server_state_lock:
        processes = [entry["process"] for entry in running_generations_by_id.values()]
    if not processes:
        return 0
    append_log_line(f"Stop requested; terminating {len(processes)} running generation(s).")
    for process in processes:
        process.terminate()
    return len(processes)


def clear_pending_queue() -> int:
    """Drop all queued (not-yet-started) requests. Returns how many were dropped."""
    dropped = 0
    while True:
        try:
            generation_request_queue.get_nowait()
            generation_request_queue.task_done()
            dropped += 1
        except queue_module.Empty:
            break
    return dropped


def build_status_snapshot() -> dict:
    with server_state_lock:
        running_labels = [entry["label"] for entry in running_generations_by_id.values()]
        return {
            "running": len(running_labels) > 0,
            "running_count": len(running_labels),
            "running_labels": running_labels,
            "max_concurrent": max_concurrent_generations,
            "queued": generation_request_queue.qsize(),
            "log_tail": list(recent_log_lines)[-40:],
        }


class AnimaInferenceGuiRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status_code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length == 0:
            return {}
        return json.loads(self.rfile.read(content_length).decode("utf-8"))

    def log_message(self, *args) -> None:
        pass  # keep the console focused on generation output, not HTTP access lines

    def _send_png_file(self, absolute_path: str) -> None:
        with open(absolute_path, "rb") as png_file:
            body = png_file.read()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")  # a path may be overwritten by a new gen
        self.end_headers()
        self.wfile.write(body)

    def _serve_generated_image(self, query_string: str) -> None:
        requested_absolute_path = (parse_qs(query_string).get("path") or [""])[0]
        with server_state_lock:
            allowed_directories = list(observed_save_directories)
        if (
            not requested_absolute_path
            or not is_path_within_allowed_directories(requested_absolute_path, allowed_directories)
            or not os.path.isfile(requested_absolute_path)
        ):
            self._send_json({"error": "not found"}, status_code=404)
            return
        self._send_png_file(requested_absolute_path)

    def _serve_image_settings_sidecar(self, query_string: str) -> None:
        """Return the JSON settings sidecar for a generated image so the GUI can reload its settings.
        404s when the image path is outside the known save dirs or has no sidecar."""
        requested_image_path = (parse_qs(query_string).get("path") or [""])[0]
        with server_state_lock:
            allowed_directories = list(observed_save_directories)
        if not requested_image_path or not is_path_within_allowed_directories(requested_image_path, allowed_directories):
            self._send_json({"error": "not found"}, status_code=404)
            return
        sidecar_path = resolve_settings_sidecar_path_for_image(requested_image_path)
        if not os.path.isfile(sidecar_path):
            self._send_json({"error": "no settings sidecar for this image"}, status_code=404)
            return
        try:
            with open(sidecar_path, "r", encoding="utf-8") as sidecar_file:
                settings = json.load(sidecar_file)
        except (OSError, json.JSONDecodeError) as error:
            self._send_json({"error": f"could not read settings sidecar: {error}"}, status_code=500)
            return
        self._send_json({"settings": settings})

    def do_GET(self) -> None:
        parsed_request = urlparse(self.path)
        route = parsed_request.path
        if route == "/" or route.startswith("/index"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif route == "/choices":
            choices = load_sampler_and_scheduler_choices(REPO_ROOT)
            choices["resolution_presets"] = ANIMA_RESOLUTION_PRESETS
            self._send_json(choices)
        elif route == "/status":
            self._send_json(build_status_snapshot())
        elif route == "/concurrency":
            self._send_json({"max_concurrent": get_max_concurrent_generations()})
        elif route == "/generated_images":
            self._send_json({"images": list_gallery_images_newest_first()})
        elif route == "/image":
            self._serve_generated_image(parsed_request.query)
        elif route == "/image_settings":
            self._serve_image_settings_sidecar(parsed_request.query)
        else:
            self._send_json({"error": "not found"}, status_code=404)

    def do_POST(self) -> None:
        if self.path == "/queue":
            try:
                generation_request = self._read_json_body()
                quantity = coerce_quantity_to_positive_int(generation_request.pop("quantity", 1))
                # quantity -> images_per_prompt: ONE subprocess renders N seed-incremented images with a
                # single model load, instead of enqueuing N reload-the-model copies.
                generation_request["images_per_prompt"] = quantity
                apply_resource_serialization_locks_from_request(generation_request)  # disk/GPU serialization per the GUI checkboxes
                materialize_pasted_prompt_list(generation_request)  # paste -> temp .txt for --from_file
                build_inference_command(generation_request)  # validate before queueing
                register_observed_save_directory(generation_request.get("save_path", ""))
            except (ValueError, json.JSONDecodeError) as error:
                self._send_json({"error": str(error)}, status_code=400)
                return
            generation_request_queue.put(generation_request)
            append_log_line(f"Queued generation (images_per_prompt={quantity}): {make_request_label(generation_request)}")
            self._send_json({"queued": generation_request_queue.qsize(), "images_per_prompt": quantity})
        elif self.path == "/clear_queue":
            dropped = clear_pending_queue()  # drops pending only; does NOT touch the running generation
            append_log_line(f"Clear queue requested; dropped {dropped} pending (running generation left alone).")
            self._send_json({"dropped": dropped})
        elif self.path == "/stop_current":
            stopped_count = terminate_all_running_generations()
            self._send_json({"stopped_count": stopped_count})
        elif self.path == "/stop_all":
            dropped = clear_pending_queue()
            stopped_count = terminate_all_running_generations()
            append_log_line(f"Stop all requested; dropped {dropped} queued, terminated {stopped_count} running.")
            self._send_json({"dropped": dropped, "stopped_count": stopped_count})
        elif self.path == "/concurrency":
            try:
                request_body = self._read_json_body()
            except json.JSONDecodeError as error:
                self._send_json({"error": str(error)}, status_code=400)
                return
            new_limit = set_max_concurrent_generations(request_body.get("max_concurrent", 1))
            append_log_line(f"Max concurrent generations set to {new_limit}.")
            self._send_json({"max_concurrent": new_limit})
        else:
            self._send_json({"error": "not found"}, status_code=404)


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Anima Inference GUI</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 12px; max-width: 1000px; font-size: 13px; }
  h2 { margin: 2px 0 6px; font-size: 15px; }
  label { display: block; font-weight: 600; font-size: 11px; margin: 0 0 2px; color: #333; }
  textarea, input, select { width: 100%; box-sizing: border-box; padding: 4px 6px; font-size: 13px; }
  input[type=checkbox] { width: auto; }  /* checkboxes size to content, not the 100% above */
  textarea { resize: vertical; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
  .span2 { grid-column: 1 / -1; }
  .field { display: flex; flex-direction: column; }
  .fourCol { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px 12px; }
  .miniField { display: flex; flex-direction: column; min-width: 0; }
  .pagGroup { grid-column: 1 / -1; border: 1px solid #bbb; border-radius: 4px; padding: 2px 8px 8px; margin: 0; min-width: 0; }
  .pagGroup > legend { font-weight: 600; font-size: 11px; color: #333; padding: 0 4px; }
  .pagAdvancedGroup { border: 1px solid #ccc; border-radius: 4px; padding: 2px 8px 8px; margin-top: 8px; min-width: 0; }
  .pagAdvancedGroup > legend { font-size: 11px; color: #555; padding: 0 4px; }
  .row { display: flex; gap: 6px; align-items: center; }
  .row input { flex: 1; }
  .row .strength { flex: 0 0 70px; }
  .row .increment { flex: 0 0 80px; }
  .row select { flex: 1 1 0; min-width: 0; }
  .row input[type=checkbox] { flex: 0 0 auto; width: auto; }
  .cycleLabel { flex: 0 0 auto; font-weight: normal; display: flex; align-items: center; gap: 3px; white-space: nowrap; font-size: 12px; }
  .topbar { position: sticky; top: 0; background: #fff; padding: 6px 0; margin-bottom: 8px; display: flex; gap: 8px; align-items: center; border-bottom: 1px solid #ddd; z-index: 10; }
  button { padding: 6px 12px; cursor: pointer; }
  #status { margin-top: 12px; padding: 8px; background: #f4f4f4; border: 1px solid #ccc; }
  #logTail { white-space: pre-wrap; font-family: monospace; font-size: 12px; max-height: 280px; overflow:auto; background:#111; color:#ddd; padding:8px; }
  .copyPasteFieldWrapper { position: relative; box-sizing: border-box; }
  .copyPasteControl { position: absolute; top: 3px; right: 2px; display: flex; flex-direction: column; gap: 1px; z-index: 3; }
  .numberStepControl { position: absolute; top: 3px; right: 16px; display: flex; flex-direction: column; gap: 1px; z-index: 3; }
  .copyPasteButton { width: 12px; height: 12px; padding: 0; font-size: 9px; line-height: 12px; text-align: center; cursor: pointer; border: 1px solid #aaa; border-radius: 2px; background: rgba(255,255,255,0.85); color: #333; }
  .copyPasteButton:hover { background: #fff; }
  #galleryPanel { margin-top: 12px; border: 1px solid #ccc; }
  #galleryHeader { cursor: pointer; padding: 6px 8px; background: #eee; font-weight: 600; user-select: none; }
  #galleryThumbs { display: flex; flex-wrap: wrap; gap: 6px; padding: 8px; max-height: 340px; overflow: auto; }
  .thumbnailWrapper { position: relative; display: inline-block; line-height: 0; }
  .thumbnailWrapper img { height: 110px; width: auto; cursor: pointer; border: 1px solid #ccc; background: #fafafa; }
  .imageLoadGroup { position: absolute; bottom: 3px; right: 3px; margin: 0; padding: 0 3px 2px; border: 1px solid #999; border-radius: 3px; background: rgba(255,255,255,0.9); display: flex; flex-direction: column; gap: 1px; }
  .imageLoadGroup > legend { font-size: 9px; color: #333; padding: 0 2px; }
  .imageLoadButton { font-size: 10px; line-height: 1; padding: 2px 5px; cursor: pointer; }
  .refreshImageButton { position: absolute; bottom: 78px; right: 3px; font-size: 10px; line-height: 1; padding: 2px 5px; cursor: pointer; background: rgba(255,255,255,0.9); border: 1px solid #999; border-radius: 3px; }
  #galleryEmpty { color: #777; font-style: italic; }
  #lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.88); align-items: center; justify-content: center; z-index: 1000; cursor: zoom-out; }
  #lightbox img { max-width: 96vw; max-height: 96vh; object-fit: contain; }
  .lightboxNavButton { position: absolute; bottom: 16px; font-size: 26px; line-height: 1; padding: 6px 18px; cursor: pointer; background: rgba(255,255,255,0.85); border: none; border-radius: 6px; }
  #lightboxPrevButton { left: 16px; }
  #lightboxNextButton { right: 16px; }
  #lightboxLoadGroup { position: absolute; bottom: 64px; right: 16px; margin: 0; padding: 2px 8px 6px; border: 1px solid #999; border-radius: 6px; background: rgba(255,255,255,0.9); display: flex; gap: 6px; }
  #lightboxLoadGroup > legend { font-size: 12px; color: #333; padding: 0 4px; }
  .lightboxLoadButton { font-size: 15px; padding: 6px 14px; cursor: pointer; }
</style>
</head>
<body>
<h2>Anima Inference GUI - queued, up to N concurrent generations</h2>

<div class="topbar">
  <button type="button" onclick="queueGeneration()">Queue Gen</button>
  <input id="quantity" type="text" value="1" title="how many to queue" data-number-step="1" style="width:74px;flex:0 0 74px" onchange="normalizeQuantityField()">
  <button type="button" onclick="postAction('/clear_queue')" title="drop pending queued gens; does not stop the running one">Clear Queue</button>
  <button type="button" onclick="postAction('/stop_current')">Stop Current</button>
  <button type="button" onclick="postAction('/stop_all')">Stop All</button>
  <label class="cycleLabel" title="how many generations run at once; each loads its own full model copy, so raise only as far as VRAM allows">concurrent<input id="maxConcurrent" type="text" value="1" data-number-step="1" style="width:68px;flex:0 0 68px" onchange="applyMaxConcurrentChange()"></label>
  <label class="cycleLabel" title="serialize model-file disk reads across concurrent generations (one process reads models from disk at a time)"><input type="checkbox" id="serializeDiskLoads" checked> lock disk</label>
  <label class="cycleLabel" title="serialize GPU use across concurrent generations (one process on the GPU at a time)"><input type="checkbox" id="serializeGpuCompute" checked> lock GPU</label>
  <label class="cycleLabel" title="strict GPU lock: also serialize text-encode and VAE-decode, not just the denoise loop"><input type="checkbox" id="gpuLockStrict"> strict</label>
  <select id="modeSelect" title="generation mode" style="flex:0 0 auto;width:auto" onchange="applyModeVisibility()">
    <option value="single">Single prompt</option>
    <option value="from_image">From image folder</option>
    <option value="from_prompt_list">From prompt list</option>
  </select>
</div>

<div class="grid">
  <div class="field span2 mode-from_image" style="display:none"><label>Source PNG folder (--from_image_embed)</label><input id="sourceImageFolder" type="text" placeholder="/path/to/folder/of/pngs"></div>
  <div class="field span2 mode-from_prompt_list" style="display:none"><label>Prompt list file (--from_file), OR paste below</label><input id="promptListPath" type="text" placeholder="/path/to/prompts.txt (leave blank if pasting)"></div>
  <div class="field span2 mode-from_prompt_list" style="display:none"><label>Paste prompt list (one prompt per line)</label><textarea id="promptListText" rows="4"></textarea></div>

  <div class="field span2"><label id="positivePromptLabel">Positive prompt</label><textarea id="positivePrompt" rows="2">masterpiece,best quality,score_7, the cutest bunny</textarea></div>
  <div class="field span2"><label id="negativePromptLabel">Negative prompt</label><textarea id="negativePrompt" rows="2">worst quality, low quality, blurry</textarea></div>

  <div class="field"><label>Sampler (cycle advances each Queue Gen)</label><div class="row"><select id="sampler"></select><label class="cycleLabel"><input type="checkbox" id="samplerCycle"> cycle</label></div></div>
  <div class="field"><label>Scheduler</label><div class="row"><select id="scheduler"></select><label class="cycleLabel"><input type="checkbox" id="schedulerCycle"> cycle</label></div></div>

  <div class="field"><label>Resolution preset (fills H/W)</label><div class="row"><select id="resolutionPreset"></select><label class="cycleLabel"><input type="checkbox" id="resolutionCycle"> cycle</label></div></div>
  <div class="field"><label>Width / Height (passed to --image_size as H W)</label><div class="row"><input id="imageWidth" type="text" value="832" placeholder="W" data-number-step-snap="16"><input id="imageHeight" type="text" value="1216" placeholder="H" data-number-step-snap="16"></div></div>

  <div class="field span2"><div class="fourCol">
    <div class="miniField"><label>Steps + inc/gen</label><div class="row"><input id="inferSteps" type="text" value="50" data-number-step="1" onchange="normalizeIntField('inferSteps', 50)"><input id="inferStepsIncrement" class="increment" type="text" value="0" data-number-step="1" title="+/- steps after each Queue Gen"></div></div>
    <div class="miniField"><label>CFG + inc/gen</label><div class="row"><input id="guidanceScale" type="text" value="3.5" data-number-step="0.1" onchange="normalizeFloatField('guidanceScale', 3.5)"><input id="guidanceScaleIncrement" class="increment" type="text" value="0" data-number-step="0.1" title="+/- CFG after each Queue Gen"></div></div>
    <div class="miniField"><label>Flow shift + inc/gen</label><div class="row"><input id="flowShift" type="text" value="5.0" data-number-step="0.1"><input id="flowShiftIncrement" class="increment" type="text" value="0" data-number-step="0.1" title="+/- flow shift after each Queue Gen"></div></div>
    <div class="miniField"><label>Seed + inc/gen (-1=random)</label><div class="row"><input id="seed" type="text" value="42" data-number-step="1"><input id="seedIncrement" class="increment" type="text" value="0" data-number-step="1" title="+/- seed after each Queue Gen"></div></div>
  </div></div>

  <fieldset class="pagGroup">
    <legend><label style="display:inline-flex; align-items:center; gap:4px; cursor:pointer;"><input type="checkbox" id="pagEnabled" style="width:auto; margin:0;"> Anima-Safe PAG (perturbed attention guidance)</label></legend>
    <div class="miniField" style="margin-bottom:4px"><label>Recommended preset (fills the fields below)</label><select id="pagPreset" title="Pick a recommended PAG setting combination to fill the fields below; 'custom' leaves them as-is. Choosing a preset does not toggle the enable checkbox." onchange="applyPagPreset()"></select></div>
    <div class="fourCol">
      <div class="miniField"><label>PAG scale</label><input id="pagScale" type="text" value="4.0" data-number-step="0.1" title="PAG correction strength: how hard to steer the result away from the perturbed (weak) prediction. Higher = stronger effect; 0 disables the effect."></div>
      <div class="miniField"><label>Block indices</label><input id="pagBlockIndices" type="text" value="18" title="Which DiT transformer block(s) to perturb: a single index, comma list, or range (e.g. 18, 18,20, or 18-22)."></div>
      <div class="miniField"><label>Perturb strength</label><input id="pagPerturbationStrength" type="text" value="0.75" data-number-step="0.1" title="How far each perturbed block's self-attention output is blended toward its value/identity path (0..1). Higher = weaker 'guide' prediction, stronger guidance."></div>
      <div class="miniField"><label>Start %</label><input id="pagStartPercent" type="text" value="0.0" data-number-step="0.1" data-number-min="0" data-number-max="1" onchange="clampNumericFieldToBounds(this)" title="Sampling-progress fraction (0..1) at which PAG turns ON. 0 = from the first step."></div>
      <div class="miniField"><label>End %</label><input id="pagEndPercent" type="text" value="0.7" data-number-step="0.1" data-number-min="0" data-number-max="1" onchange="clampNumericFieldToBounds(this)" title="Sampling-progress fraction (0..1) at which PAG turns OFF. Should be >= start %."></div>
    </div>
    <fieldset class="pagAdvancedGroup">
      <legend>advanced</legend>
      <div class="fourCol">
        <div class="miniField"><label>Head indices (blank=all)</label><input id="pagHeadIndices" type="text" value="" title="Optional attention-head filter for the perturbation: index, comma list, or range. Blank = perturb all heads."></div>
        <div class="miniField"><label>Rescale</label><input id="pagRescale" type="text" value="0.2" data-number-step="0.1" data-number-min="0" data-number-max="1" onchange="clampNumericFieldToBounds(this)" title="Std-based guidance rescale (0..1) for contrast control; 0 = no rescaling, higher pulls guided-result contrast back toward the conditional prediction."></div>
        <div class="miniField"><label>Rescale mode</label><select id="pagRescaleMode" title="Which statistic the rescale normalizes against: 'full' = the CFG result, 'partial' = the conditional prediction."><option value="full">full</option><option value="partial">partial</option></select></div>
      </div>
    </fieldset>
  </fieldset>

  <div class="field span2"><label>DiT path (--dit; all-in-one checkpoint OK)</label><input id="ditPath" type="text" value="/media/aikenyon/WDRed16TB/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors"></div>
  <div class="field"><label>VAE (--vae, optional)</label><input id="vaePath" type="text" value="/media/aikenyon/WDRed16TB/models/anima/split_files/vae/qwen_image_vae.safetensors"></div>
  <div class="field"><label>Text encoder (--text_encoder, optional)</label><input id="textEncoderPath" type="text" value="/media/aikenyon/WDRed16TB/models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors"></div>
  <div class="field span2"><label>Save path (--save_path)</label><input id="savePath" type="text" value="./anima_out"></div>

  <div class="field span2"><label>LoRAs</label><div id="loraList"></div><button type="button" onclick="addLoraRow()" style="margin-top:4px;align-self:flex-start;">Add LoRA</button></div>

  <div class="field span2"><label>LoRA test folder (--lora_test_folder): runs the whole gen once per .safetensors in the folder, on top of the LoRAs above (blank = off)</label>
    <div class="row">
      <input id="loraTestFolder" type="text" placeholder="/path/to/folder/of/loras (blank = off)">
      <input id="loraTestMultiplier" class="increment" type="text" value="1" data-number-step="0.1" title="multiplier for each test LoRA">
    </div>
  </div>
</div>

<div id="status">
  <div id="statusLine">status: ...</div>
  <div id="logTail"></div>
</div>

<div id="galleryPanel">
  <div id="galleryHeader" onclick="toggleGalleryExpanded()"><span id="galleryCaret">&#9656;</span> Generated images (<span id="galleryCount">0</span>)</div>
  <div id="galleryThumbs" style="display:none"><span id="galleryEmpty">No images yet.</span></div>
</div>

<div id="lightbox" onclick="closeLightbox()">
  <button type="button" id="lightboxPrevButton" class="lightboxNavButton" title="previous image" onclick="event.stopPropagation(); showLightboxImageRelativeToCurrent(-1)">&lt;</button>
  <img id="lightboxImage" alt="generated image">
  <fieldset id="lightboxLoadGroup" onclick="event.stopPropagation()">
    <legend>Load</legend>
    <button type="button" class="lightboxLoadButton" title="load this image's generation settings (not model paths)" onclick="loadImageGenerationSettings(currentLightboxImagePath)">Settings</button>
    <button type="button" class="lightboxLoadButton" title="load this image's model paths (DiT/VAE/text encoder)" onclick="loadImageModelPaths(currentLightboxImagePath)">Models</button>
    <button type="button" class="lightboxLoadButton" title="load this image's LoRA stack (paths, strengths, enabled)" onclick="loadImageLoras(currentLightboxImagePath)">LoRAs</button>
  </fieldset>
  <button type="button" id="lightboxNextButton" class="lightboxNavButton" title="next image" onclick="event.stopPropagation(); showLightboxImageRelativeToCurrent(1)">&gt;</button>
</div>

<script>
function addLoraRow(path, strength, enabled) {
  const container = document.getElementById('loraList');
  const row = document.createElement('div');
  row.className = 'row';
  const enabledCheckbox = document.createElement('input');
  enabledCheckbox.type = 'checkbox';
  enabledCheckbox.className = 'loraEnabled';
  enabledCheckbox.checked = (enabled === undefined) ? true : Boolean(enabled);  // enabled by default
  enabledCheckbox.title = 'enable/disable this LoRA';
  enabledCheckbox.style.flex = '0 0 auto';
  enabledCheckbox.style.width = 'auto';  // override the global input{width:100%} so it doesn't eat the row
  const pathInput = document.createElement('input');
  pathInput.type = 'text';
  pathInput.placeholder = 'LoRA path';
  pathInput.value = path || '';
  pathInput.style.flex = '1 1 auto';  // width-based basis so the path field fills the row (like other fields)
  pathInput.style.minWidth = '0';     // allow it to shrink for the fixed strength/checkbox/x, not collapse
  const strengthInput = document.createElement('input');
  strengthInput.type = 'text';
  strengthInput.className = 'strength';
  strengthInput.placeholder = 'strength';
  strengthInput.value = strength || '1.0';
  strengthInput.dataset.numberStep = '0.1';  // float field: up/down adjust by 0.1
  const removeButton = document.createElement('button');
  removeButton.type = 'button';
  removeButton.textContent = 'x';
  removeButton.onclick = function() { container.removeChild(row); };
  row.appendChild(pathInput);
  row.appendChild(strengthInput);
  row.appendChild(enabledCheckbox);
  row.appendChild(removeButton);
  container.appendChild(row);
  attachCopyPasteButtonsToField(pathInput);
  attachCopyPasteButtonsToField(strengthInput);
}

function copyFieldValueToClipboard(field) {
  const text = field.value;
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).catch(function() { legacyCopyFieldSelection(field); });
  } else {
    legacyCopyFieldSelection(field);
  }
}

function legacyCopyFieldSelection(field) {
  field.focus();
  field.select();
  try { document.execCommand('copy'); } catch (e) { /* clipboard unavailable */ }
}

function pasteClipboardIntoField(field) {
  if (navigator.clipboard && navigator.clipboard.readText) {
    navigator.clipboard.readText().then(function(text) {
      field.value = text;
      field.dispatchEvent(new Event('change'));
    }).catch(function() { alert('Paste not permitted by the browser; click the field and press Ctrl+V.'); });
  } else {
    alert('Paste not supported by the browser; click the field and press Ctrl+V.');
  }
}

// Overlay a stacked tiny copy(c)/paste(p) control at the right edge of a text field, preserving the
// field's flex sizing by moving it onto the wrapper. Idempotent per field.
function attachCopyPasteButtonsToField(field) {
  if (field.dataset.copyPasteEnhanced === 'true') { return; }
  field.dataset.copyPasteEnhanced = 'true';

  const computedStyle = window.getComputedStyle(field);
  const wrapper = document.createElement('span');
  wrapper.className = 'copyPasteFieldWrapper';
  wrapper.style.flexGrow = computedStyle.flexGrow;
  wrapper.style.flexShrink = computedStyle.flexShrink;
  wrapper.style.flexBasis = computedStyle.flexBasis;
  wrapper.style.minWidth = computedStyle.minWidth;
  wrapper.style.display = (computedStyle.display === 'block') ? 'block' : 'inline-block';
  wrapper.style.width = (field.style.width === '100%' || computedStyle.width) ? '100%' : 'auto';

  field.parentNode.insertBefore(wrapper, field);
  wrapper.appendChild(field);
  field.style.width = '100%';
  field.style.boxSizing = 'border-box';
  // Numeric fields also get an up/down column left of copy/paste, so they need extra right padding.
  const isNumericField = Boolean(field.dataset.numberStep || field.dataset.numberStepSnap);
  field.style.paddingRight = isNumericField ? '31px' : '17px';  // keep text clear of the overlaid buttons

  if (isNumericField) {
    attachUpDownStepButtonsToWrapper(wrapper, field);
  }

  const control = document.createElement('span');
  control.className = 'copyPasteControl';
  const copyButton = document.createElement('button');
  copyButton.type = 'button';
  copyButton.className = 'copyPasteButton';
  copyButton.textContent = 'c';
  copyButton.title = 'copy';
  copyButton.onclick = function() { copyFieldValueToClipboard(field); };
  const pasteButton = document.createElement('button');
  pasteButton.type = 'button';
  pasteButton.className = 'copyPasteButton';
  pasteButton.textContent = 'p';
  pasteButton.title = 'paste';
  pasteButton.onclick = function() { pasteClipboardIntoField(field); };
  control.appendChild(copyButton);
  control.appendChild(pasteButton);
  wrapper.appendChild(control);
}

// Overlay a stacked up(^)/down(v) control just left of the copy/paste control, for numeric fields only.
// The buttons step the field by its data-number-step (integer by 1, float by 0.1, etc.).
function attachUpDownStepButtonsToWrapper(wrapper, field) {
  const control = document.createElement('span');
  control.className = 'numberStepControl';
  const upButton = document.createElement('button');
  upButton.type = 'button';
  upButton.className = 'copyPasteButton';
  upButton.textContent = '^';
  upButton.title = 'increase';
  upButton.onclick = function() { stepNumericField(field, 1); };
  const downButton = document.createElement('button');
  downButton.type = 'button';
  downButton.className = 'copyPasteButton';
  downButton.textContent = 'v';
  downButton.title = 'decrease';
  downButton.onclick = function() { stepNumericField(field, -1); };
  control.appendChild(upButton);
  control.appendChild(downButton);
  wrapper.appendChild(control);
}

function attachCopyPasteButtonsToAllTextFields() {
  document.querySelectorAll('input[type=text], textarea').forEach(attachCopyPasteButtonsToField);
}

function collectLoras() {
  const loras = [];
  document.querySelectorAll('#loraList .row').forEach(function(row) {
    const enabledCheckbox = row.querySelector('.loraEnabled');
    const textInputs = row.querySelectorAll('input[type=text]');
    loras.push({ path: textInputs[0].value, strength: textInputs[1].value, enabled: enabledCheckbox.checked });
  });
  return loras;
}

function parseQuantityValue(rawValue) {
  const trimmed = (rawValue || '').trim();
  const parsed = Number(trimmed);
  if (!isFinite(parsed)) { return 1; }  // non-numeric string -> 1
  const truncated = Math.trunc(parsed);
  return truncated < 1 ? 1 : truncated;
}

function normalizeQuantityField() {
  const field = document.getElementById('quantity');
  field.value = parseQuantityValue(field.value);
}

function parseIntWithDefault(rawValue, defaultValue) {
  const parsed = Number((rawValue || '').trim());
  return isFinite(parsed) ? Math.trunc(parsed) : defaultValue;
}

function parseFloatWithDefault(rawValue, defaultValue) {
  const parsed = Number((rawValue || '').trim());
  return isFinite(parsed) ? parsed : defaultValue;
}

// Increase/decrease a numeric field (direction is +1 or -1), dispatching 'change' so any field
// normalizer/handler still runs. A field with data-number-step-snap=N snaps to the next multiple of N
// in the click direction (so an off-grid value like 513 lands on 528 up / 512 down, never below N).
// Otherwise it steps by data-number-step: an integer step stays integer; a fractional step (0.1) steps
// as a float, rounded to avoid float noise.
// Clamp a numeric field to its optional data-number-min / data-number-max bounds (blank/non-numeric is
// left untouched). No-op when neither bound is set.
function clampNumericFieldToBounds(field) {
  const value = Number((field.value || '').trim());
  if (!isFinite(value)) { return; }
  const minValue = Number(field.dataset.numberMin);
  const maxValue = Number(field.dataset.numberMax);
  let clamped = value;
  if (isFinite(minValue)) { clamped = Math.max(minValue, clamped); }
  if (isFinite(maxValue)) { clamped = Math.min(maxValue, clamped); }
  field.value = clamped;
}

function stepNumericField(field, direction) {
  const snapModulus = Number(field.dataset.numberStepSnap);
  if (isFinite(snapModulus) && snapModulus > 0) {
    const current = parseIntWithDefault(field.value, snapModulus);
    const snapped = (direction > 0)
      ? Math.floor(current / snapModulus) * snapModulus + snapModulus
      : Math.ceil(current / snapModulus) * snapModulus - snapModulus;
    field.value = Math.max(snapModulus, snapped);  // never drop below one step (no zero/negative size)
  } else {
    const stepValue = Number(field.dataset.numberStep);
    if (!isFinite(stepValue) || stepValue === 0) { return; }
    if (Number.isInteger(stepValue)) {
      field.value = parseIntWithDefault(field.value, 0) + direction * stepValue;
    } else {
      const next = parseFloatWithDefault(field.value, 0) + direction * stepValue;
      field.value = Math.round(next * 10000) / 10000;
    }
  }
  clampNumericFieldToBounds(field);  // respect data-number-min/max (e.g. PAG start/end % in [0,1])
  field.dispatchEvent(new Event('change'));
}

function normalizeIntField(elementId, defaultValue) {
  const field = document.getElementById(elementId);
  field.value = parseIntWithDefault(field.value, defaultValue);
}

function normalizeFloatField(elementId, defaultValue) {
  const field = document.getElementById(elementId);
  field.value = parseFloatWithDefault(field.value, defaultValue);
}

function applyIntIncrementForNextGen(fieldId, incrementId, fallback) {
  const increment = parseIntWithDefault(document.getElementById(incrementId).value, 0);
  if (increment === 0) { return; }  // 0 / unparseable increment leaves the field untouched
  const field = document.getElementById(fieldId);
  field.value = parseIntWithDefault(field.value, fallback) + increment;
}

function applyFloatIncrementForNextGen(fieldId, incrementId, fallback) {
  const increment = parseFloatWithDefault(document.getElementById(incrementId).value, 0);
  if (increment === 0) { return; }
  const field = document.getElementById(fieldId);
  const next = parseFloatWithDefault(field.value, fallback) + increment;
  field.value = Math.round(next * 10000) / 10000;  // avoid float noise like 3.6000000001
}

function cycleSelectForNextGen(selectId) {
  const select = document.getElementById(selectId);
  if (select.options.length > 0) {
    select.selectedIndex = (select.selectedIndex + 1) % select.options.length;
  }
}

function cycleResolutionForNextGen() {
  const select = document.getElementById('resolutionPreset');
  const optionCount = select.options.length;
  if (optionCount <= 1) { return; }
  let nextIndex = select.selectedIndex + 1;
  if (nextIndex >= optionCount) { nextIndex = 1; }  // loop back to first preset, skipping index 0 (custom)
  if (nextIndex < 1) { nextIndex = 1; }
  select.selectedIndex = nextIndex;
  const selectedOption = select.options[nextIndex];
  if (selectedOption.dataset.width) {
    document.getElementById('imageWidth').value = selectedOption.dataset.width;
    document.getElementById('imageHeight').value = selectedOption.dataset.height;
  }
}

function applyPostQueueIncrementsAndCycles() {
  applyIntIncrementForNextGen('inferSteps', 'inferStepsIncrement', 50);
  applyFloatIncrementForNextGen('guidanceScale', 'guidanceScaleIncrement', 3.5);
  applyFloatIncrementForNextGen('flowShift', 'flowShiftIncrement', 5.0);
  applyIntIncrementForNextGen('seed', 'seedIncrement', 42);
  if (document.getElementById('samplerCycle').checked) { cycleSelectForNextGen('sampler'); }
  if (document.getElementById('schedulerCycle').checked) { cycleSelectForNextGen('scheduler'); }
  if (document.getElementById('resolutionCycle').checked) { cycleResolutionForNextGen(); }
}

function applyModeVisibility() {
  const mode = document.getElementById('modeSelect').value;
  document.querySelectorAll('.mode-from_image').forEach(function(el) { el.style.display = (mode === 'from_image') ? '' : 'none'; });
  document.querySelectorAll('.mode-from_prompt_list').forEach(function(el) { el.style.display = (mode === 'from_prompt_list') ? '' : 'none'; });
  const isFileOrImageMode = (mode === 'from_image' || mode === 'from_prompt_list');
  document.getElementById('positivePromptLabel').textContent = isFileOrImageMode ? 'Pre-prompt (--pre_prompt)' : 'Positive prompt';
  document.getElementById('negativePromptLabel').textContent = isFileOrImageMode ? 'Pre-prompt negative (--pre_prompt_neg)' : 'Negative prompt';
}

function buildRequest() {
  return {
    quantity: parseQuantityValue(document.getElementById('quantity').value),
    mode: document.getElementById('modeSelect').value,
    source_image_folder: document.getElementById('sourceImageFolder').value,
    prompt_list_path: document.getElementById('promptListPath').value,
    prompt_list_text: document.getElementById('promptListText').value,
    positive_prompt: document.getElementById('positivePrompt').value,
    negative_prompt: document.getElementById('negativePrompt').value,
    sampler: document.getElementById('sampler').value,
    scheduler: document.getElementById('scheduler').value,
    image_height: document.getElementById('imageHeight').value,
    image_width: document.getElementById('imageWidth').value,
    infer_steps: parseIntWithDefault(document.getElementById('inferSteps').value, 50),
    guidance_scale: parseFloatWithDefault(document.getElementById('guidanceScale').value, 3.5),
    flow_shift: document.getElementById('flowShift').value,
    seed: document.getElementById('seed').value,
    dit_path: document.getElementById('ditPath').value,
    vae_path: document.getElementById('vaePath').value,
    text_encoder_path: document.getElementById('textEncoderPath').value,
    save_path: document.getElementById('savePath').value,
    loras: collectLoras(),
    lora_test_folder: document.getElementById('loraTestFolder').value,
    lora_test_multiplier: document.getElementById('loraTestMultiplier').value,
    serialize_disk_loads: document.getElementById('serializeDiskLoads').checked,
    serialize_gpu_compute: document.getElementById('serializeGpuCompute').checked,
    gpu_lock_strict: document.getElementById('gpuLockStrict').checked,
    pag_enabled: document.getElementById('pagEnabled').checked,
    pag_scale: document.getElementById('pagScale').value,
    pag_block_indices: document.getElementById('pagBlockIndices').value,
    pag_perturbation_strength: document.getElementById('pagPerturbationStrength').value,
    pag_head_indices: document.getElementById('pagHeadIndices').value,
    pag_start_percent: document.getElementById('pagStartPercent').value,
    pag_end_percent: document.getElementById('pagEndPercent').value,
    pag_rescale: document.getElementById('pagRescale').value,
    pag_rescale_mode: document.getElementById('pagRescaleMode').value
  };
}

function queueGeneration() {
  normalizeQuantityField();
  fetch('/queue', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(buildRequest()) })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.error) { alert('Error: ' + data.error); return; }
      applyPostQueueIncrementsAndCycles();  // bump fields for the NEXT gen only after a successful queue
    })
    .catch(function(e) { alert('Request failed: ' + e); });
}

function postAction(path) {
  fetch(path, { method: 'POST' }).catch(function(e) { alert('Request failed: ' + e); });
}

function refreshStatus() {
  fetch('/status').then(function(r) { return r.json(); }).then(function(s) {
    const runningLabels = s.running_labels || [];
    const runningSuffix = runningLabels.length ? ' [' + runningLabels.join('] [') + ']' : '';
    document.getElementById('statusLine').textContent =
      'running: ' + (s.running_count || 0) + '/' + (s.max_concurrent || 1) + runningSuffix + '  |  queued: ' + s.queued;
    document.getElementById('logTail').textContent = (s.log_tail || []).join('\\n');
  }).catch(function() {});
}

function applyMaxConcurrentChange() {
  const field = document.getElementById('maxConcurrent');
  const value = parseQuantityValue(field.value);
  field.value = value;
  fetch('/concurrency', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ max_concurrent: value }) })
    .then(function(r) { return r.json(); })
    .then(function(data) { if (data.max_concurrent) { field.value = data.max_concurrent; } })
    .catch(function(e) { alert('Failed to set concurrency: ' + e); });
}

let galleryExpanded = false;
const renderedGalleryImagePaths = new Set();
let orderedGalleryImagePathsNewestFirst = [];  // display order (index 0 = newest); drives lightbox cycling
let currentLightboxImagePath = null;
let galleryImageCacheBustCounter = 0;  // bumped per manual refresh to force a fresh fetch of one thumbnail

// Re-fetch a thumbnail's image with a unique cache-busting query param, so a thumbnail that only
// partially loaded (image fetched mid-write) is replaced by the complete file.
function reloadThumbnailImageBustingCache(thumbnail, imagePath) {
  galleryImageCacheBustCounter += 1;
  thumbnail.src = imageUrlForPath(imagePath) + '&cache_bust=' + galleryImageCacheBustCounter;
}

function toggleGalleryExpanded() {
  galleryExpanded = !galleryExpanded;
  document.getElementById('galleryThumbs').style.display = galleryExpanded ? 'flex' : 'none';
  document.getElementById('galleryCaret').innerHTML = galleryExpanded ? '&#9662;' : '&#9656;';
}

function imageUrlForPath(imagePath) {
  return '/image?path=' + encodeURIComponent(imagePath);
}

function openLightbox(imagePath) {
  currentLightboxImagePath = imagePath;
  document.getElementById('lightboxImage').src = imageUrlForPath(imagePath);
  document.getElementById('lightbox').style.display = 'flex';
}

function closeLightbox() {
  document.getElementById('lightbox').style.display = 'none';
  document.getElementById('lightboxImage').src = '';
  currentLightboxImagePath = null;
}

// While the expanded (lightbox) view is open: Left = previous, Right/Space = next, Escape = close.
function handleLightboxKeydown(event) {
  if (currentLightboxImagePath === null) { return; }  // only act while the expanded view is open
  if (event.key === 'ArrowLeft') {
    showLightboxImageRelativeToCurrent(-1);
    event.preventDefault();
  } else if (event.key === 'ArrowRight' || event.key === ' ' || event.key === 'Spacebar') {
    showLightboxImageRelativeToCurrent(1);
    event.preventDefault();
  } else if (event.key === 'Escape') {
    closeLightbox();
    event.preventDefault();
  }
}
document.addEventListener('keydown', handleLightboxKeydown);

// Move the lightbox by offset positions in the current display order (wrapping around). Resolves the
// current image by path each time so images arriving while open do not desync navigation.
function showLightboxImageRelativeToCurrent(offset) {
  const imageCount = orderedGalleryImagePathsNewestFirst.length;
  if (imageCount === 0) { return; }
  const currentIndex = orderedGalleryImagePathsNewestFirst.indexOf(currentLightboxImagePath);
  const startIndex = currentIndex === -1 ? 0 : currentIndex;
  const nextIndex = ((startIndex + offset) % imageCount + imageCount) % imageCount;
  openLightbox(orderedGalleryImagePathsNewestFirst[nextIndex]);
}

function setFieldValueIfPresent(elementId, value) {
  if (value === undefined || value === null) { return; }
  document.getElementById(elementId).value = value;
}

// Repo defaults for the PAG form fields, used when a loaded image predates PAG (has no 'pag' settings).
var PAG_FORM_FIELD_DEFAULTS = {
  scale: '4.0', block_indices: '18', perturbation_strength: '0.75', head_indices: '',
  start_percent: '0.0', end_percent: '0.7', rescale: '0.2', rescale_mode: 'full'
};

// Recommended PAG setting combinations for the preset dropdown. Only "Repo Default" is verified (the
// node's shipped defaults; Anima has 28 blocks 0-27, block 18 is the balanced choice). The other presets
// are research-based estimates (general PAG guidance + the repo's block advice), NOT tested on Anima
// renders. Selecting a preset fills the fields below; it does not toggle the enable checkbox.
var PAG_PRESETS = [
  { label: '- custom (edit fields below) -', values: null },
  { label: 'Repo Default (balanced detail)', values: { scale: '4.0', block_indices: '18', perturbation_strength: '0.75', head_indices: '', start_percent: '0.0', end_percent: '0.7', rescale: '0.2', rescale_mode: 'full' } },
  { label: 'Subtle cleanup (light touch) [est.]', values: { scale: '2.5', block_indices: '20', perturbation_strength: '0.6', head_indices: '', start_percent: '0.0', end_percent: '0.6', rescale: '0.15', rescale_mode: 'full' } },
  { label: 'Strong structure guidance [est.]', values: { scale: '6.0', block_indices: '16-20', perturbation_strength: '0.85', head_indices: '', start_percent: '0.0', end_percent: '0.8', rescale: '0.3', rescale_mode: 'full' } },
  { label: 'Early composition (shape first) [est.]', values: { scale: '4.0', block_indices: '18', perturbation_strength: '0.75', head_indices: '', start_percent: '0.0', end_percent: '0.4', rescale: '0.2', rescale_mode: 'full' } },
  { label: 'Late detail refinement (sharpen) [est.]', values: { scale: '3.0', block_indices: '20', perturbation_strength: '0.65', head_indices: '', start_percent: '0.5', end_percent: '0.9', rescale: '0.15', rescale_mode: 'full' } },
  { label: 'Contrast-safe aggressive (partial rescale) [est.]', values: { scale: '5.0', block_indices: '18-20', perturbation_strength: '0.8', head_indices: '', start_percent: '0.0', end_percent: '0.7', rescale: '0.4', rescale_mode: 'partial' } }
];

function populatePagPresetOptions() {
  const select = document.getElementById('pagPreset');
  PAG_PRESETS.forEach(function(preset) {
    const option = document.createElement('option');
    option.textContent = preset.label;
    select.appendChild(option);
  });
}

// Fill the PAG fields from the selected preset (index maps 1:1 to PAG_PRESETS). 'custom' (index 0) is a
// no-op. Does not change the enable checkbox, mirroring how the resolution preset only fills H/W.
function applyPagPreset() {
  const preset = PAG_PRESETS[document.getElementById('pagPreset').selectedIndex];
  if (!preset || !preset.values) { return; }
  const values = preset.values;
  document.getElementById('pagScale').value = values.scale;
  document.getElementById('pagBlockIndices').value = values.block_indices;
  document.getElementById('pagPerturbationStrength').value = values.perturbation_strength;
  document.getElementById('pagHeadIndices').value = values.head_indices;
  document.getElementById('pagStartPercent').value = values.start_percent;
  document.getElementById('pagEndPercent').value = values.end_percent;
  document.getElementById('pagRescale').value = values.rescale;
  document.getElementById('pagRescaleMode').value = values.rescale_mode;
}

function applyPagSettingsToForm(settings) {
  document.getElementById('pagPreset').selectedIndex = 0;  // loaded values are 'custom', not a named preset
  const pag = settings.pag;
  if (pag) {
    document.getElementById('pagEnabled').checked = (pag.enabled === true);
    setFieldValueIfPresent('pagScale', pag.scale);
    setFieldValueIfPresent('pagBlockIndices', pag.block_indices);
    setFieldValueIfPresent('pagPerturbationStrength', pag.perturbation_strength);
    document.getElementById('pagHeadIndices').value = (pag.head_indices == null) ? '' : pag.head_indices;
    setFieldValueIfPresent('pagStartPercent', pag.start_percent);
    setFieldValueIfPresent('pagEndPercent', pag.end_percent);
    setFieldValueIfPresent('pagRescale', pag.rescale);
    setFieldValueIfPresent('pagRescaleMode', pag.rescale_mode);
  } else {
    // Image predates PAG: load PAG defaults but leave the box unchecked (its PAG state is unknown).
    document.getElementById('pagEnabled').checked = false;
    document.getElementById('pagScale').value = PAG_FORM_FIELD_DEFAULTS.scale;
    document.getElementById('pagBlockIndices').value = PAG_FORM_FIELD_DEFAULTS.block_indices;
    document.getElementById('pagPerturbationStrength').value = PAG_FORM_FIELD_DEFAULTS.perturbation_strength;
    document.getElementById('pagHeadIndices').value = PAG_FORM_FIELD_DEFAULTS.head_indices;
    document.getElementById('pagStartPercent').value = PAG_FORM_FIELD_DEFAULTS.start_percent;
    document.getElementById('pagEndPercent').value = PAG_FORM_FIELD_DEFAULTS.end_percent;
    document.getElementById('pagRescale').value = PAG_FORM_FIELD_DEFAULTS.rescale;
    document.getElementById('pagRescaleMode').value = PAG_FORM_FIELD_DEFAULTS.rescale_mode;
  }
}

// Load only the model paths (DiT/VAE/text encoder) from an image's settings — kept separate from the
// generation settings so you can swap the prompt/params without changing models, and vice versa.
function applyModelPathsFromSettings(settings) {
  setFieldValueIfPresent('ditPath', settings.dit);
  setFieldValueIfPresent('vaePath', settings.vae);
  setFieldValueIfPresent('textEncoderPath', settings.text_encoder);
}

// Load the LoRA rows (path, strength, enabled) from an image's settings — separate from generation
// settings and model paths, so the LoRA stack (and its strengths) can be loaded on its own.
function applyLorasFromSettings(settings) {
  document.getElementById('loraList').innerHTML = '';
  (settings.loras || []).forEach(function(lora) {
    addLoraRow(lora.path, String(lora.multiplier), lora.enabled !== false);  // preserve disabled rows
  });
}

// Load the generation settings (prompt, size, steps, sampler, PAG, ...) but NOT the model paths or LoRAs.
function applyGenerationSettingsExcludingModelsAndLoras(settings) {
  // These settings describe a single embedded-prompt render, so reproduce them in single-prompt mode
  // with the exact height/width (preset selector back to custom).
  document.getElementById('modeSelect').value = 'single';
  applyModeVisibility();
  document.getElementById('resolutionPreset').value = '';

  setFieldValueIfPresent('positivePrompt', settings.prompt);
  setFieldValueIfPresent('negativePrompt', settings.negative_prompt);
  setFieldValueIfPresent('imageWidth', settings.width);
  setFieldValueIfPresent('imageHeight', settings.height);
  setFieldValueIfPresent('inferSteps', settings.steps);
  setFieldValueIfPresent('guidanceScale', settings.guidance_scale);
  setFieldValueIfPresent('flowShift', settings.flow_shift);
  setFieldValueIfPresent('seed', settings.seed);
  setFieldValueIfPresent('sampler', settings.sampler);
  setFieldValueIfPresent('scheduler', settings.scheduler);

  applyPagSettingsToForm(settings);
}

function fetchImageSettingsThen(imagePath, applySettingsCallback) {
  if (!imagePath) { return; }
  fetch('/image_settings?path=' + encodeURIComponent(imagePath))
    .then(function(r) { return r.json().then(function(data) { return { ok: r.ok, data: data }; }); })
    .then(function(response) {
      if (!response.ok || !response.data.settings) {
        alert((response.data && response.data.error) || 'No settings found for this image.');
        return;
      }
      applySettingsCallback(response.data.settings);
    })
    .catch(function(e) { alert('Failed to load settings: ' + e); });
}

function loadImageGenerationSettings(imagePath) { fetchImageSettingsThen(imagePath, applyGenerationSettingsExcludingModelsAndLoras); }
function loadImageModelPaths(imagePath) { fetchImageSettingsThen(imagePath, applyModelPathsFromSettings); }
function loadImageLoras(imagePath) { fetchImageSettingsThen(imagePath, applyLorasFromSettings); }

// Build the small bordered "Load" group (Settings / Models buttons) overlaid on a gallery thumbnail.
function createImageLoadGroup(imagePath) {
  const group = document.createElement('fieldset');
  group.className = 'imageLoadGroup';
  const legend = document.createElement('legend');
  legend.textContent = 'Load';
  const settingsButton = document.createElement('button');
  settingsButton.type = 'button';
  settingsButton.className = 'imageLoadButton';
  settingsButton.textContent = 'Settings';
  settingsButton.title = "load this image's generation settings (not model paths)";
  settingsButton.onclick = function(event) { event.stopPropagation(); loadImageGenerationSettings(imagePath); };
  const modelsButton = document.createElement('button');
  modelsButton.type = 'button';
  modelsButton.className = 'imageLoadButton';
  modelsButton.textContent = 'Models';
  modelsButton.title = "load this image's model paths (DiT/VAE/text encoder)";
  modelsButton.onclick = function(event) { event.stopPropagation(); loadImageModelPaths(imagePath); };
  const lorasButton = document.createElement('button');
  lorasButton.type = 'button';
  lorasButton.className = 'imageLoadButton';
  lorasButton.textContent = 'LoRAs';
  lorasButton.title = "load this image's LoRA stack (paths, strengths, enabled)";
  lorasButton.onclick = function(event) { event.stopPropagation(); loadImageLoras(imagePath); };
  group.appendChild(legend);
  group.appendChild(settingsButton);
  group.appendChild(modelsButton);
  group.appendChild(lorasButton);
  return group;
}

function refreshGeneratedImages() {
  fetch('/generated_images').then(function(r) { return r.json(); }).then(function(data) {
    const images = data.images || [];  // newest-first from the server
    orderedGalleryImagePathsNewestFirst = images.map(function(image) { return image.path; });
    document.getElementById('galleryCount').textContent = images.length;
    const thumbsContainer = document.getElementById('galleryThumbs');
    const emptyPlaceholder = document.getElementById('galleryEmpty');
    if (emptyPlaceholder && images.length > 0) { emptyPlaceholder.remove(); }
    // Iterate oldest-first and prepend, so the newest image ends up at the front and older
    // already-rendered thumbnails keep their place.
    for (let i = images.length - 1; i >= 0; i--) {
      const image = images[i];
      if (renderedGalleryImagePaths.has(image.path)) { continue; }
      renderedGalleryImagePaths.add(image.path);
      const thumbnailWrapper = document.createElement('div');
      thumbnailWrapper.className = 'thumbnailWrapper';
      const thumbnail = document.createElement('img');
      thumbnail.src = imageUrlForPath(image.path);
      thumbnail.title = image.name;
      thumbnail.onclick = function() { openLightbox(image.path); };
      const refreshImageButton = document.createElement('button');
      refreshImageButton.type = 'button';
      refreshImageButton.className = 'refreshImageButton';
      refreshImageButton.textContent = 'Refresh';
      refreshImageButton.title = 're-fetch this image (fixes a thumbnail that only partially loaded)';
      refreshImageButton.onclick = function(event) { event.stopPropagation(); reloadThumbnailImageBustingCache(thumbnail, image.path); };
      thumbnailWrapper.appendChild(thumbnail);
      thumbnailWrapper.appendChild(refreshImageButton);
      thumbnailWrapper.appendChild(createImageLoadGroup(image.path));
      thumbsContainer.insertBefore(thumbnailWrapper, thumbsContainer.firstChild);
    }
  }).catch(function() {});
}

fetch('/choices').then(function(r) { return r.json(); }).then(function(choices) {
  const samplerSelect = document.getElementById('sampler');
  const schedulerSelect = document.getElementById('scheduler');
  (choices.samplers || []).forEach(function(name) {
    const option = document.createElement('option'); option.value = name; option.textContent = name; samplerSelect.appendChild(option);
  });
  (choices.schedulers || []).forEach(function(name) {
    const option = document.createElement('option'); option.value = name; option.textContent = name; schedulerSelect.appendChild(option);
  });
  if ((choices.samplers || []).indexOf('er_sde') >= 0) { samplerSelect.value = 'er_sde'; }
  if ((choices.schedulers || []).indexOf('beta57') >= 0) { schedulerSelect.value = 'beta57'; }

  const resolutionSelect = document.getElementById('resolutionPreset');
  const customOption = document.createElement('option');
  customOption.value = ''; customOption.textContent = '- custom (edit height/width below) -';
  resolutionSelect.appendChild(customOption);
  (choices.resolution_presets || []).forEach(function(preset) {
    const option = document.createElement('option');
    option.value = preset.width + 'x' + preset.height;
    option.textContent = preset.label;
    option.dataset.width = preset.width;
    option.dataset.height = preset.height;
    resolutionSelect.appendChild(option);
  });
  resolutionSelect.onchange = function() {
    const selected = resolutionSelect.options[resolutionSelect.selectedIndex];
    if (selected && selected.dataset.width) {
      document.getElementById('imageWidth').value = selected.dataset.width;
      document.getElementById('imageHeight').value = selected.dataset.height;
    }
  };
  // Default the preset selector to match the default height/width fields (832x1216 portrait).
  resolutionSelect.value = '832x1216';
});

addLoraRow('/media/aikenyon/WDRed16TB/models/loras/div2k_anima/div2k_anima_v1-step00000360.safetensors', '1');
applyModeVisibility();
populatePagPresetOptions();
attachCopyPasteButtonsToAllTextFields();

fetch('/concurrency').then(function(r) { return r.json(); }).then(function(data) {
  if (data.max_concurrent) { document.getElementById('maxConcurrent').value = data.max_concurrent; }
}).catch(function() {});

setInterval(refreshStatus, 1500);
setInterval(refreshGeneratedImages, 2500);
refreshStatus();
refreshGeneratedImages();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web GUI for single-prompt Anima generations")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind (default localhost only)")
    parser.add_argument("--port", type=int, default=7861, help="Port to serve on (default 7861)")
    parser.add_argument(
        "--max_concurrent_generations",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_GENERATIONS,
        help="How many generation subprocesses may run at once (default 1). Adjustable live in the UI. "
        "Each concurrent generation loads its own full copy of the model, so raise this only as far as VRAM allows.",
    )
    args = parser.parse_args()

    set_max_concurrent_generations(args.max_concurrent_generations)
    register_observed_save_directory(DEFAULT_SAVE_PATH)  # show the default output folder from the start

    dispatcher_thread = threading.Thread(target=generation_dispatcher_loop, name="generation-dispatcher", daemon=True)
    dispatcher_thread.start()

    httpd = ThreadingHTTPServer((args.host, args.port), AnimaInferenceGuiRequestHandler)
    append_log_line(f"Anima inference GUI serving at http://{args.host}:{args.port} (repo root: {REPO_ROOT})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        append_log_line("Shutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
