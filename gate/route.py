"""Passed -> call the right model. Failed -> try a fix (sharpen / normalize / OCR), re-check, log both."""
import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import soundfile as sf
import librosa
from dotenv import load_dotenv

from gate import log
from gate.validate import pdf_text, validate

load_dotenv()

MODELS = {
    "image": os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
    "audio": os.getenv("OPENAI_AUDIO_MODEL", "whisper-1"),
    "pdf": os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"),
}
# USD per 1M tokens (input, output) and per audio minute. Rough estimates for the dashboard.
TOKEN_PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}
AUDIO_PRICE_PER_MIN = 0.006


def _client():
    from openai import OpenAI
    return OpenAI(timeout=60)


def _token_cost(model, usage):
    p_in, p_out = TOKEN_PRICES.get(model, (0.15, 0.60))
    return (usage.prompt_tokens * p_in + usage.completion_tokens * p_out) / 1e6


def _call_image(path):
    model = MODELS["image"]
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    r = _client().chat.completions.create(model=model, max_tokens=200, messages=[{
        "role": "user", "content": [
            {"type": "text", "text": "Describe this image in 2-3 sentences."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}])
    return model, r.choices[0].message.content, _token_cost(model, r.usage)


def _call_audio(path):
    model = MODELS["audio"]
    with open(path, "rb") as f:
        r = _client().audio.transcriptions.create(model=model, file=f)
    minutes = librosa.get_duration(path=path) / 60
    return model, r.text, minutes * AUDIO_PRICE_PER_MIN


def _call_pdf(path):
    model = MODELS["pdf"]
    text = "\n".join(pdf_text(path))[:12000]
    r = _client().chat.completions.create(model=model, max_tokens=250, messages=[
        {"role": "system", "content": "Summarize the document in 3 bullet points."},
        {"role": "user", "content": text}])
    return model, r.choices[0].message.content, _token_cost(model, r.usage)


CALLERS = {"image": _call_image, "audio": _call_audio, "pdf": _call_pdf}


def call_model(path, modality):
    """Never raises: returns (model_used, output, cost)."""
    if not os.getenv("OPENAI_API_KEY"):
        return "skipped (no OPENAI_API_KEY)", "Model call skipped: set OPENAI_API_KEY in .env.", 0.0
    try:
        return CALLERS[modality](path)
    except Exception as e:
        return f"error: {MODELS[modality]}", f"Model call failed: {type(e).__name__}: {e}", 0.0


# ---------- fixes: each returns a new file path, or None if the failure isn't fixable ----------

def _fixed_path(path, ext=None):
    p = Path(path)
    return str(p.with_name(f"fixed_{p.stem}{ext or p.suffix}"))


def fix_image(path, reason):
    if "blurry" not in reason or "too small" in reason:
        return None
    img = cv2.imread(path)
    out = img
    for _ in range(2):  # two passes of unsharp masking
        soft = cv2.GaussianBlur(out, (0, 0), 3)
        out = cv2.addWeighted(out, 3.0, soft, -2.0, 0)
    dst = _fixed_path(path)
    cv2.imwrite(dst, out)
    return dst


def fix_audio(path, reason):
    if "silent" not in reason:
        return None
    y, sr = librosa.load(path, sr=None, mono=True)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak < 1e-3:  # nothing but noise floor: boosting it would only amplify hiss
        return None
    dst = _fixed_path(path, ".wav")
    sf.write(dst, y * (0.9 / peak), sr)
    return dst


def fix_pdf(path, reason):
    if "scanned" not in reason:
        return None
    out = pymupdf.open()
    with pymupdf.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            with pymupdf.open("pdf", pix.pdfocr_tobytes()) as ocr_page:  # needs tesseract
                out.insert_pdf(ocr_page)
    dst = _fixed_path(path)
    out.save(dst)
    return dst


FIXERS = {"image": fix_image, "audio": fix_audio, "pdf": fix_pdf}


def _gate_one(path, filename, modality, size_kb):
    t0 = time.perf_counter()
    result = validate(path, modality)
    model_used, output, cost = None, None, None
    if result["passed"]:
        model_used, output, cost = call_model(path, modality)
    latency = round((time.perf_counter() - t0) * 1000, 1)
    event_id = log.insert_event(filename, modality, size_kb, result["passed"], result["reason"],
                                result["metrics"], model_used, latency, cost)
    return {"id": event_id, "filename": filename, "path": path, **result,
            "model_used": model_used, "output": output, "latency_ms": latency, "est_cost_usd": cost}


def process(meta):
    """Run the gate on an ingested file. Returns a list of attempts: [original] or [original, fixed]."""
    first = _gate_one(meta["path"], meta["filename"], meta["modality"], meta["size_kb"])
    attempts = [first]
    if first["passed"] or meta["modality"] not in FIXERS:
        return attempts
    try:
        fixed = FIXERS[meta["modality"]](meta["path"], first["reason"])
    except Exception as e:
        first["fix_error"] = f"{type(e).__name__}: {e}"
        fixed = None
    if fixed:
        size_kb = round(os.path.getsize(fixed) / 1024, 2)
        attempts.append(_gate_one(fixed, f"{meta['filename']} (fixed)", meta["modality"], size_kb))
    return attempts
