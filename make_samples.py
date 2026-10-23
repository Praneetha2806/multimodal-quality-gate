"""Generate demo files in samples/good and samples/bad. Run once: python make_samples.py"""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import soundfile as sf

GOOD, BAD = Path("samples/good"), Path("samples/bad")
TEXT = ("Quarterly report. Revenue grew 12 percent year over year, driven by strong demand "
        "in the enterprise segment. Operating costs were flat. The team shipped three new "
        "features and reduced support tickets by a fifth. Next quarter we plan to expand "
        "into two new regions and hire ten engineers.")


def image_samples():
    img = np.full((512, 512, 3), 255, np.uint8)
    for i in range(0, 512, 32):
        cv2.line(img, (i, 0), (i, 511), (40, 40, 40), 1)
    cv2.rectangle(img, (60, 60), (240, 220), (0, 90, 200), -1)
    cv2.circle(img, (360, 320), 90, (20, 160, 60), -1)
    cv2.putText(img, "QUALITY GATE", (40, 470), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    cv2.imwrite(str(GOOD / "sharp_chart.png"), img)
    cv2.imwrite(str(BAD / "blurry_chart.png"), cv2.GaussianBlur(img, (0, 0), 2.5))
    cv2.imwrite(str(BAD / "tiny_thumbnail.png"), cv2.resize(img, (128, 128)))


def audio_samples():
    speech = GOOD / "speech.wav"
    if shutil.which("espeak-ng"):
        subprocess.run(["espeak-ng", "-s", "150", "-w", str(speech), TEXT[:160]], check=True)
        y, sr = sf.read(speech)
    else:  # fallback: harmonic tone so the validator still has something to measure
        sr = 16000
        t = np.linspace(0, 5, 5 * sr, endpoint=False)
        y = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t)) / 2
        sf.write(speech, y, sr)
    sf.write(BAD / "quiet_speech.wav", y * 0.004, sr)  # fixable by normalization
    sf.write(BAD / "silence.wav", np.random.default_rng(0).normal(0, 1e-5, 5 * sr), sr)


def pdf_samples():
    doc = pymupdf.open()
    for n in range(2):
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 760), f"Page {n + 1}\n\n{TEXT}", fontsize=13)
    doc.save(GOOD / "text_report.pdf")
    # "Scanned" copy: each page rendered to an image, no text layer.
    scan = pymupdf.open()
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        p = scan.new_page(width=page.rect.width, height=page.rect.height)
        p.insert_image(p.rect, stream=pix.tobytes("png"))
    scan.save(BAD / "scanned_report.pdf", deflate=True)


if __name__ == "__main__":
    GOOD.mkdir(parents=True, exist_ok=True)
    BAD.mkdir(parents=True, exist_ok=True)
    image_samples()
    audio_samples()
    pdf_samples()
    (BAD / "notes.xyz").write_text("unsupported file type")
    print(sorted(str(p) for p in Path("samples").rglob("*.*")))
