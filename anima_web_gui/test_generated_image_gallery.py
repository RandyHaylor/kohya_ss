"""Stdlib-only unit tests for the generated-image gallery helpers.

Run: cd anima_web_gui then  python3 -m pytest test_generated_image_gallery.py -q
"""

import os

from generated_image_gallery import (
    is_path_within_allowed_directories,
    list_generated_png_files_in_directories,
    resolve_save_directory_absolute_path,
)


def write_png_file_with_modified_time(directory, file_name, modified_time_epoch_seconds):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "wb") as png_file:
        png_file.write(b"\x89PNG\r\n\x1a\n")  # PNG magic bytes; contents irrelevant to these tests
    os.utime(file_path, (modified_time_epoch_seconds, modified_time_epoch_seconds))
    return file_path


def test_relative_save_path_resolves_against_base_directory():
    resolved = resolve_save_directory_absolute_path("./anima_out", "/repo/root")
    assert resolved == "/repo/root/anima_out"


def test_absolute_save_path_is_returned_normalized_unchanged():
    resolved = resolve_save_directory_absolute_path("/tmp/out/../out", "/repo/root")
    assert resolved == "/tmp/out"


def test_lists_only_png_files_oldest_first(tmp_path):
    directory = str(tmp_path)
    write_png_file_with_modified_time(directory, "newer.png", 2000)
    write_png_file_with_modified_time(directory, "older.png", 1000)
    with open(os.path.join(directory, "notes.txt"), "w") as non_image_file:
        non_image_file.write("ignore me")

    listing = list_generated_png_files_in_directories([directory])

    assert [entry["file_name"] for entry in listing] == ["older.png", "newer.png"]


def test_missing_directory_is_skipped_and_duplicates_collapsed(tmp_path):
    real_directory = str(tmp_path)
    write_png_file_with_modified_time(real_directory, "image.png", 1500)

    listing = list_generated_png_files_in_directories(
        [real_directory, real_directory, os.path.join(real_directory, "does_not_exist")]
    )

    assert [entry["file_name"] for entry in listing] == ["image.png"]


def test_file_inside_allowed_directory_is_permitted(tmp_path):
    directory = str(tmp_path)
    file_path = write_png_file_with_modified_time(directory, "image.png", 1500)
    assert is_path_within_allowed_directories(file_path, [directory]) is True


def test_file_outside_allowed_directories_is_rejected(tmp_path):
    allowed_directory = str(tmp_path / "allowed")
    other_directory = str(tmp_path / "other")
    os.makedirs(allowed_directory)
    os.makedirs(other_directory)
    outside_file_path = write_png_file_with_modified_time(other_directory, "secret.png", 1500)
    assert is_path_within_allowed_directories(outside_file_path, [allowed_directory]) is False


def test_directory_traversal_escape_is_rejected(tmp_path):
    allowed_directory = str(tmp_path / "allowed")
    os.makedirs(allowed_directory)
    traversal_path = os.path.join(allowed_directory, "..", "etc_passwd_lookalike.png")
    assert is_path_within_allowed_directories(traversal_path, [allowed_directory]) is False
