"""Pure helpers for the GUI's generated-image gallery.

These resolve where a generation writes its PNGs, list the PNGs sitting in those directories, and
guard which absolute paths the server is allowed to serve. Kept free of server/HTTP state and side
effects so they are unit-testable without spawning the server or a GPU.
"""

import os


def resolve_save_directory_absolute_path(save_path: str, base_directory: str) -> str:
    """Resolve a request's --save_path (often relative, e.g. './anima_out') to an absolute path.

    Relative paths are interpreted against base_directory (the repo root the inference CLI runs from,
    since the server spawns it with cwd=repo_root), matching where the images actually land.
    """
    if os.path.isabs(save_path):
        return os.path.normpath(save_path)
    return os.path.normpath(os.path.join(base_directory, save_path))


def list_generated_png_files_in_directories(directories) -> list:
    """Return the .png files across the given directories, oldest-first by modification time.

    Each entry is {'absolute_path', 'file_name', 'modified_time'}. Directories that do not exist are
    skipped, and a file reachable via more than one directory is listed once.
    """
    png_files = []
    already_listed_absolute_paths = set()
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for file_name in os.listdir(directory):
            if not file_name.lower().endswith(".png"):
                continue
            absolute_path = os.path.normpath(os.path.join(directory, file_name))
            if absolute_path in already_listed_absolute_paths:
                continue
            if not os.path.isfile(absolute_path):
                continue
            already_listed_absolute_paths.add(absolute_path)
            png_files.append(
                {
                    "absolute_path": absolute_path,
                    "file_name": file_name,
                    "modified_time": os.path.getmtime(absolute_path),
                }
            )
    png_files.sort(key=lambda entry: entry["modified_time"])
    return png_files


def is_path_within_allowed_directories(candidate_absolute_path: str, allowed_directories) -> bool:
    """True only if candidate_absolute_path (symlinks resolved) sits strictly inside one of the
    allowed_directories (symlinks resolved). Guards the image-serving route against path traversal
    outside the known save directories."""
    real_candidate_path = os.path.realpath(candidate_absolute_path)
    for directory in allowed_directories:
        real_allowed_directory = os.path.realpath(directory)
        if real_candidate_path == real_allowed_directory:
            continue  # the directory itself is not a servable file
        try:
            if os.path.commonpath([real_candidate_path, real_allowed_directory]) == real_allowed_directory:
                return True
        except ValueError:
            continue  # different drives / mixed absolute-relative -> not within
    return False
