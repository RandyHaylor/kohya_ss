"""Tests for the GUI server's concurrency state (limit setter/getter, slot accounting, status snapshot).

Importing the server module is side-effecting (it opens generation.log next to this file and creates the
queue), but it does NOT start any threads until main() runs, so these tests exercise the pure state
functions without spawning subprocesses or a GPU."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anima_inference_gui_server as server


def reset_concurrency_state():
    server.set_max_concurrent_generations(1)
    with server.server_state_lock:
        server.active_generation_count = 0
        server.running_generations_by_id.clear()


def test_set_max_concurrent_generations_coerces_and_returns():
    reset_concurrency_state()
    assert server.set_max_concurrent_generations(4) == 4
    assert server.get_max_concurrent_generations() == 4
    assert server.set_max_concurrent_generations("3") == 3
    assert server.get_max_concurrent_generations() == 3
    assert server.set_max_concurrent_generations("0") == 1  # below 1 -> 1
    assert server.set_max_concurrent_generations("abc") == 1  # non-numeric -> 1
    reset_concurrency_state()


def test_status_snapshot_reports_running_and_limit():
    reset_concurrency_state()
    server.set_max_concurrent_generations(2)
    snapshot = server.build_status_snapshot()
    assert snapshot["max_concurrent"] == 2
    assert snapshot["running_count"] == 0
    assert snapshot["running_labels"] == []
    assert snapshot["running"] is False

    generation_id = server._register_running_generation(process=None, label="a cat")
    snapshot = server.build_status_snapshot()
    assert snapshot["running_count"] == 1
    assert snapshot["running_labels"] == ["a cat"]
    assert snapshot["running"] is True

    server._deregister_running_generation(generation_id)
    assert server.build_status_snapshot()["running_count"] == 0
    reset_concurrency_state()


def test_release_generation_slot_decrements_active_count():
    reset_concurrency_state()
    with server.server_state_lock:
        server.active_generation_count = 2
    server._release_generation_slot()
    with server.server_state_lock:
        assert server.active_generation_count == 1
    reset_concurrency_state()


def test_serialization_locks_default_to_both_on_denoise_scope():
    request = {}
    server.apply_resource_serialization_locks_from_request(request)
    assert request["model_load_disk_lock_file"] == server.MODEL_LOAD_DISK_LOCK_PATH
    assert request["gpu_compute_lock_file"] == server.GPU_COMPUTE_LOCK_PATH
    assert request["gpu_lock_scope"] == server.GPU_LOCK_SCOPE_DENOISE_ONLY


def test_serialization_locks_can_be_individually_disabled():
    disk_off = {"serialize_disk_loads": False}
    server.apply_resource_serialization_locks_from_request(disk_off)
    assert "model_load_disk_lock_file" not in disk_off
    assert disk_off["gpu_compute_lock_file"] == server.GPU_COMPUTE_LOCK_PATH

    gpu_off = {"serialize_gpu_compute": False}
    server.apply_resource_serialization_locks_from_request(gpu_off)
    assert "gpu_compute_lock_file" not in gpu_off
    assert "gpu_lock_scope" not in gpu_off
    assert gpu_off["model_load_disk_lock_file"] == server.MODEL_LOAD_DISK_LOCK_PATH


def test_serialization_locks_strict_selects_all_compute_scope():
    request = {"gpu_lock_strict": True}
    server.apply_resource_serialization_locks_from_request(request)
    assert request["gpu_lock_scope"] == server.GPU_LOCK_SCOPE_STRICT


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
    print("ok")
