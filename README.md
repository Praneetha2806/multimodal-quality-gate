# Multi-Modal Quality Gate

Checks images, audio and PDFs **before** they reach an AI model. Bad inputs are caught with a reason,
fixed when possible (sharpen / normalize / OCR) and re-checked; good inputs are routed to an OpenAI model.
Every attempt is logged to DuckDB (`gate.duckdb`, table `events`) and shown on a dashboard.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY (optional: without it, model calls are skipped)
python make_samples.py        # builds samples/good and samples/bad
streamlit run app.py
```
System deps: `ffmpeg` (mp3/m4a decoding), `tesseract-ocr` (scanned-PDF fix), `espeak-ng` (optional, speech sample).

## Rules (`gate/validate.py` → `CONFIG`)
| Modality | Fails when | Auto-fix |
|---|---|---|
| any | > 20 MB → too large; unknown extension → unsupported | – |
| image | Laplacian variance < 100 → blurry; short side < 256 px → too small | unsharp mask (blurry only) |
| audio | RMS < -45 dBFS or > 80% silent frames → silent/near-silent; > 10 min → too long | peak-normalize (if there's signal) |
| pdf | < 50 chars/page → scanned/no text layer; > 50 pages → too large | tesseract OCR |

## Demo script
1. Upload tab → pick `samples/bad/blurry_chart.png` → FAILED blurry → fixed version PASSES.
2. Same with `scanned_report.pdf` (OCR) and `quiet_speech.wav` (normalize).
3. `tiny_thumbnail.png`, `silence.wav`, `notes.xyz` → caught, no fix possible.
4. Dashboard tab → pass rate, failures by reason, latency, cost → export failures as CSV/JSON.
