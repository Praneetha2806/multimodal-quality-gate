import json
from pathlib import Path

import pandas as pd
import streamlit as st

from gate import log
from gate.ingest import ingest, ingest_path
from gate.route import process
from gate.validate import pdf_text

st.set_page_config(page_title="Multi-Modal Quality Gate", layout="wide")
st.title("Multi-Modal Quality Gate")
st.caption("Checks images, audio and PDFs before they reach a model. Bad inputs are caught, fixed if possible, and logged.")

upload_tab, dash_tab = st.tabs(["Upload", "Dashboard"])


def show_preview(path, modality):
    if modality == "image":
        st.image(path, use_container_width=True)
    elif modality == "audio":
        st.audio(path)
    elif modality == "pdf":
        text = "\n".join(pdf_text(path)).strip()
        st.text_area("Extracted text", text[:600] or "(no text layer)", height=150, disabled=True,
                     key=f"txt_{path}")


def show_attempt(a, modality):
    title = "Fixed version" if a["filename"].endswith("(fixed)") else "Original"
    st.subheader(title)
    if a["passed"]:
        st.success(f"PASSED → model: {a['model_used']}")
    else:
        st.error(f"FAILED: {a['reason']}")
    show_preview(a["path"], modality)
    st.json(a["metrics"])
    if a.get("output"):
        st.markdown("**Model output**")
        st.write(a["output"])
    if a.get("fix_error"):
        st.warning(f"Fix attempt failed: {a['fix_error']}")
    st.caption(f"latency {a['latency_ms']} ms · est. cost ${a['est_cost_usd'] or 0:.5f} · event #{a['id']}")


with upload_tab:
    left, right = st.columns(2)
    with left:
        f = st.file_uploader("Upload an image, audio file, or PDF")
        run_upload = st.button("Run gate", type="primary")
    with right:
        samples = sorted(str(p) for p in Path("samples").rglob("*.*"))
        sample = st.selectbox("…or pick a demo sample", samples)
        run_sample = st.button("Run gate on sample")

    meta = None
    if run_upload and f is None:
        st.warning("Choose a file first.")
    elif run_upload:
        meta = ingest(f.name, f.getvalue())
    elif run_sample and sample:
        meta = ingest_path(sample)

    if meta:
        attempts = process(meta)
        if meta["modality"] == "unknown":
            st.error(f"FAILED: {attempts[0]['reason']} ({meta['filename']})")
        else:
            cols = st.columns(len(attempts))
            for col, a in zip(cols, attempts):
                with col:
                    show_attempt(a, meta["modality"])
            if len(attempts) == 1 and not attempts[0]["passed"]:
                st.info("No automatic fix available for this failure.")

with dash_tab:
    df = log.query_events()
    df["passed"] = df["passed"].fillna(False).astype(bool)
    if df.empty:
        st.info("No events yet. Run some files through the gate.")
    else:
        total = len(df)
        pass_rate = df["passed"].mean() * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Files checked", total)
        c2.metric("Pass rate", f"{pass_rate:.0f}%")
        c3.metric("Avg latency", f"{df['latency_ms'].mean():.0f} ms")
        c4.metric("Est. model cost", f"${df['est_cost_usd'].fillna(0).sum():.4f}")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Failures by reason**")
            reasons = df.loc[~df["passed"], "reason"].str.split("; ").explode()
            if reasons.empty:
                st.write("No failures.")
            else:
                st.bar_chart(reasons.value_counts(), horizontal=True)
        with g2:
            st.markdown("**Pass / fail by modality**")
            by_mod = (df.assign(result=df["passed"].map({True: "passed", False: "failed"}))
                        .pivot_table(index="modality", columns="result", values="id",
                                     aggfunc="count", fill_value=0))
            st.bar_chart(by_mod)

        st.markdown("**Latency per event (ms)**")
        st.line_chart(df.sort_values("created_at").set_index("created_at")["latency_ms"])

        st.markdown("**Recent events**")
        st.dataframe(df, use_container_width=True, hide_index=True)

        failures = df[~df["passed"]].copy()
        failures["created_at"] = failures["created_at"].astype(str)
        st.markdown(f"**Export failure cases** ({len(failures)})")
        e1, e2, _ = st.columns([1, 1, 4])
        e1.download_button("Download CSV", failures.to_csv(index=False), "failures.csv", "text/csv")
        records = failures.assign(metrics=failures["metrics_json"].map(json.loads)) \
                          .drop(columns="metrics_json").to_dict(orient="records")
        e2.download_button("Download JSON", json.dumps(records, indent=2, default=str),
                           "failures.json", "application/json")
