import re
import pathlib
from typing import Final

ALLOWED_EXTENSIONS: Final[frozenset] = frozenset({".txt", ".md"})
MAX_FILENAME_LENGTH: Final[int] = 200
MAX_CONTENT_BYTES: Final[int] = 10 * 1024 * 1024

_UNSAFE_CHARS: Final = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_PATH_TRAVERSAL: Final = re.compile(r"\.\.")


def sanitize_filename(name: str) -> str:
    name = _UNSAFE_CHARS.sub("_", name)
    name = name.strip(". ")
    if not name:
        return "untitled"
    return name[:MAX_FILENAME_LENGTH]


def validate_path(raw: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(raw).resolve()
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File extension '{path.suffix}' is not permitted.")
    safe_name = sanitize_filename(path.name)
    resolved = (path.parent / safe_name).resolve()
    if _PATH_TRAVERSAL.search(str(resolved)):
        raise ValueError("Path traversal detected.")
    return resolved


def validate_content(content: str) -> None:
    if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
        raise ValueError("Content exceeds the 10 MB limit.")