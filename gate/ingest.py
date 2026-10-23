"""Save an uploaded file to disk and describe it (modality, size)."""
import os
import time
from pathlib import Path

UPLOAD_DIR = Path("uploads")

EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".bmp", ".webp"},
    "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
    "pdf": {".pdf"},
}


def detect_modality(filename):
    ext = Path(filename).suffix.lower()
    for modality, exts in EXTENSIONS.items():
        if ext in exts:
            return modality
    return "unknown"


def ingest(filename, data: bytes):
    """Persist bytes under uploads/ and return a metadata dict."""
    UPLOAD_DIR.mkdir(exist_ok=True)
    safe = Path(filename).name
    path = UPLOAD_DIR / f"{int(time.time() * 1000)}_{safe}"
    path.write_bytes(data)
    return {
        "filename": safe,
        "path": str(path),
        "modality": detect_modality(safe),
        "size_kb": round(len(data) / 1024, 2),
    }


def ingest_path(path):
    """Ingest a file already on disk (e.g. from samples/)."""
    p = Path(path)
    return ingest(p.name, p.read_bytes())
