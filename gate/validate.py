"""One validator per modality. Each returns {passed: bool, reason: str, metrics: dict}."""
import os

import cv2
import librosa
import numpy as np
import pymupdf

CONFIG = {
    "max_file_mb": 20,
    "image_min_blur_var": 100.0,   # Laplacian variance below this -> blurry
    "image_min_side_px": 256,
    "audio_min_rms_dbfs": -45.0,   # overall loudness below this -> near-silent
    "audio_silence_dbfs": -50.0,   # a frame quieter than this counts as silent
    "audio_max_silence_ratio": 0.8,
    "audio_max_minutes": 10,
    "pdf_min_chars_per_page": 50,
    "pdf_max_pages": 50,
}


def _result(failures, metrics):
    return {"passed": not failures, "reason": "; ".join(failures) or "ok", "metrics": metrics}


def validate_image(path):
    img = cv2.imread(path)
    if img is None:
        return _result(["unreadable image"], {})
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    m = {"width": w, "height": h, "blur_var": round(blur, 1)}
    fails = []
    if blur < CONFIG["image_min_blur_var"]:
        fails.append("blurry")
    if min(w, h) < CONFIG["image_min_side_px"]:
        fails.append("too small")
    return _result(fails, m)


def validate_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000, mono=True)
    except Exception as e:
        return _result([f"unreadable audio ({type(e).__name__})"], {})
    duration = len(y) / sr
    if len(y) == 0:
        return _result(["silent/near-silent"], {"duration_s": 0})
    eps = 1e-10
    rms_db = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + eps)
    frame_db = 20 * np.log10(librosa.feature.rms(y=y)[0] + eps)
    silence = float(np.mean(frame_db < CONFIG["audio_silence_dbfs"]))
    m = {"duration_s": round(duration, 2), "rms_dbfs": round(float(rms_db), 1),
         "silence_ratio": round(silence, 3)}
    fails = []
    if rms_db < CONFIG["audio_min_rms_dbfs"] or silence > CONFIG["audio_max_silence_ratio"]:
        fails.append("silent/near-silent")
    if duration > CONFIG["audio_max_minutes"] * 60:
        fails.append("too long")
    return _result(fails, m)


def pdf_text(path):
    with pymupdf.open(path) as doc:
        return [page.get_text() for page in doc]


def check_pdf_text(pages):
    """Shared by validate_pdf and the OCR re-check in route.py."""
    n = len(pages)
    chars = sum(len(t.strip()) for t in pages) / max(n, 1)
    m = {"pages": n, "chars_per_page": round(chars, 1)}
    fails = []
    if chars < CONFIG["pdf_min_chars_per_page"]:
        fails.append("scanned/no text layer")
    if n > CONFIG["pdf_max_pages"]:
        fails.append("too large")
    return _result(fails, m)


def validate_pdf(path):
    try:
        return check_pdf_text(pdf_text(path))
    except Exception as e:
        return _result([f"unreadable pdf ({type(e).__name__})"], {})


VALIDATORS = {"image": validate_image, "audio": validate_audio, "pdf": validate_pdf}


def validate(path, modality):
    size_mb = os.path.getsize(path) / 1024 / 1024
    if size_mb > CONFIG["max_file_mb"]:
        return _result(["too large"], {"size_mb": round(size_mb, 2)})
    if modality not in VALIDATORS:
        return _result(["unsupported"], {})
    return VALIDATORS[modality](path)
