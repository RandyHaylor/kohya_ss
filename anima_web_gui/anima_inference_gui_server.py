"""Tiny local web GUI for single-prompt Anima generations.

v1 scope: build the embedded --prompt command line from a form (positive/negative prompt, sampler and
scheduler dropdowns, model paths, and an add/remove LoRA list) and run generations strictly one at a
time via a queue. Buttons: Queue Gen, Stop Current, Stop All. The spawned generation's console output
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

from inference_command_builder import (
    ANIMA_RESOLUTION_PRESETS,
    build_inference_command,
    coerce_quantity_to_positive_int,
    load_sampler_and_scheduler_choices,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATION_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generation.log")
QUEUED_PROMPT_LISTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queued_prompt_lists")

generation_request_queue: "queue_module.Queue[dict]" = queue_module.Queue()
server_state_lock = threading.Lock()
currently_running_process = None  # subprocess.Popen while a generation is running, else None
currently_running_label = None
recent_log_lines = collections.deque(maxlen=500)
generation_log_file = open(GENERATION_LOG_PATH, "a", encoding="utf-8")


def append_log_line(text: str) -> None:
    """Record one line to the in-memory tail (for the UI), this server's stdout, and the log file."""
    line = text.rstrip("\n")
    with server_state_lock:
        recent_log_lines.append(line)
    print(line, flush=True)
    generation_log_file.write(line + "\n")
    generation_log_file.flush()


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


def make_request_label(generation_request: dict) -> str:
    prompt_preview = str(generation_request.get("positive_prompt", "")).strip().replace("\n", " ")
    return (prompt_preview[:60] + "...") if len(prompt_preview) > 60 else (prompt_preview or "(no prompt)")


def run_one_generation(generation_request: dict) -> None:
    """Spawn the inference CLI for one request and stream its output until it returns."""
    global currently_running_process, currently_running_label

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
    with server_state_lock:
        currently_running_process = process
        currently_running_label = label

    for output_line in process.stdout:
        append_log_line(output_line)
    process.wait()

    with server_state_lock:
        currently_running_process = None
        currently_running_label = None
    append_log_line(f"=== END generation (exit code {process.returncode}): {label} ===")


def generation_worker_loop() -> None:
    """Single worker: pull one request at a time and run it to completion before the next."""
    while True:
        generation_request = generation_request_queue.get()
        try:
            run_one_generation(generation_request)
        except Exception as error:  # keep the worker alive across a single bad request
            append_log_line(f"ERROR running generation: {error}")
        finally:
            generation_request_queue.task_done()


def terminate_current_generation() -> bool:
    """Signal the currently running generation to stop (SIGTERM). Returns True if one was running."""
    with server_state_lock:
        process = currently_running_process
    if process is None:
        return False
    append_log_line("Stop current requested; terminating running generation.")
    process.terminate()
    return True


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
        return {
            "running": currently_running_process is not None,
            "running_label": currently_running_label,
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

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/choices":
            choices = load_sampler_and_scheduler_choices(REPO_ROOT)
            choices["resolution_presets"] = ANIMA_RESOLUTION_PRESETS
            self._send_json(choices)
        elif self.path == "/status":
            self._send_json(build_status_snapshot())
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
                materialize_pasted_prompt_list(generation_request)  # paste -> temp .txt for --from_file
                build_inference_command(generation_request)  # validate before queueing
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
            was_running = terminate_current_generation()
            self._send_json({"stopped_current": was_running})
        elif self.path == "/stop_all":
            dropped = clear_pending_queue()
            was_running = terminate_current_generation()
            append_log_line(f"Stop all requested; dropped {dropped} queued, terminated running={was_running}.")
            self._send_json({"dropped": dropped, "stopped_current": was_running})
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
  textarea { resize: vertical; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
  .span2 { grid-column: 1 / -1; }
  .field { display: flex; flex-direction: column; }
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
</style>
</head>
<body>
<h2>Anima Inference GUI - single prompt, queued one at a time</h2>

<div class="topbar">
  <button type="button" onclick="queueGeneration()">Queue Gen</button>
  <input id="quantity" type="text" value="1" title="how many to queue" style="width:50px;flex:0 0 50px" onchange="normalizeQuantityField()">
  <button type="button" onclick="postAction('/clear_queue')" title="drop pending queued gens; does not stop the running one">Clear Queue</button>
  <button type="button" onclick="postAction('/stop_current')">Stop Current</button>
  <button type="button" onclick="postAction('/stop_all')">Stop All</button>
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
  <div class="field"><label>Height / Width (--image_size H W)</label><div class="row"><input id="imageHeight" type="text" value="1216" placeholder="H"><input id="imageWidth" type="text" value="832" placeholder="W"></div></div>

  <div class="field"><label>Steps + increment/gen</label><div class="row"><input id="inferSteps" type="text" value="50" onchange="normalizeIntField('inferSteps', 50)"><input id="inferStepsIncrement" class="increment" type="text" value="0" title="+/- steps after each Queue Gen"></div></div>
  <div class="field"><label>CFG + increment/gen</label><div class="row"><input id="guidanceScale" type="text" value="3.5" onchange="normalizeFloatField('guidanceScale', 3.5)"><input id="guidanceScaleIncrement" class="increment" type="text" value="0" title="+/- CFG after each Queue Gen"></div></div>

  <div class="field"><label>Flow shift + increment/gen (blank = default)</label><div class="row"><input id="flowShift" type="text" value="5.0"><input id="flowShiftIncrement" class="increment" type="text" value="0" title="+/- flow shift after each Queue Gen"></div></div>
  <div class="field"><label>Seed + increment/gen (-1/blank = random)</label><div class="row"><input id="seed" type="text" value="42"><input id="seedIncrement" class="increment" type="text" value="0" title="+/- seed after each Queue Gen"></div></div>

  <div class="field span2"><label>DiT path (--dit; all-in-one checkpoint OK)</label><input id="ditPath" type="text" value="/media/aikenyon/WDRed16TB/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors"></div>
  <div class="field"><label>VAE (--vae, optional)</label><input id="vaePath" type="text" value="/media/aikenyon/WDRed16TB/models/anima/split_files/vae/qwen_image_vae.safetensors"></div>
  <div class="field"><label>Text encoder (--text_encoder, optional)</label><input id="textEncoderPath" type="text" value="/media/aikenyon/WDRed16TB/models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors"></div>
  <div class="field span2"><label>Save path (--save_path)</label><input id="savePath" type="text" value="./anima_out"></div>

  <div class="field span2"><label>LoRAs</label><div id="loraList"></div><button type="button" onclick="addLoraRow()" style="margin-top:4px;align-self:flex-start;">Add LoRA</button></div>

  <div class="field span2"><label>LoRA test folder (--lora_test_folder): runs the whole gen once per .safetensors in the folder, on top of the LoRAs above (blank = off)</label>
    <div class="row">
      <input id="loraTestFolder" type="text" placeholder="/path/to/folder/of/loras (blank = off)">
      <input id="loraTestMultiplier" class="increment" type="text" value="1" title="multiplier for each test LoRA">
    </div>
  </div>
</div>

<div id="status">
  <div id="statusLine">status: ...</div>
  <div id="logTail"></div>
</div>

<script>
function addLoraRow(path, strength) {
  const container = document.getElementById('loraList');
  const row = document.createElement('div');
  row.className = 'row';
  const enabledCheckbox = document.createElement('input');
  enabledCheckbox.type = 'checkbox';
  enabledCheckbox.className = 'loraEnabled';
  enabledCheckbox.checked = true;  // enabled by default
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
  const removeButton = document.createElement('button');
  removeButton.type = 'button';
  removeButton.textContent = 'x';
  removeButton.onclick = function() { container.removeChild(row); };
  row.appendChild(pathInput);
  row.appendChild(strengthInput);
  row.appendChild(enabledCheckbox);
  row.appendChild(removeButton);
  container.appendChild(row);
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
    lora_test_multiplier: document.getElementById('loraTestMultiplier').value
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
    document.getElementById('statusLine').textContent =
      'running: ' + s.running + (s.running_label ? ' [' + s.running_label + ']' : '') + '  |  queued: ' + s.queued;
    document.getElementById('logTail').textContent = (s.log_tail || []).join('\\n');
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

setInterval(refreshStatus, 1500);
refreshStatus();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web GUI for single-prompt Anima generations")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind (default localhost only)")
    parser.add_argument("--port", type=int, default=7861, help="Port to serve on (default 7861)")
    args = parser.parse_args()

    worker_thread = threading.Thread(target=generation_worker_loop, name="generation-worker", daemon=True)
    worker_thread.start()

    httpd = ThreadingHTTPServer((args.host, args.port), AnimaInferenceGuiRequestHandler)
    append_log_line(f"Anima inference GUI serving at http://{args.host}:{args.port} (repo root: {REPO_ROOT})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        append_log_line("Shutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
