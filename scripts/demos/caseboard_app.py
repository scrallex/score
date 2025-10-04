#!/usr/bin/env python3
"""Streamlit Caseboard: compare baseline vs reality filter with slider replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import streamlit as st

DEFAULT_THRESHOLDS = {
    "r_min": 2,
    "lambda_max": 0.55,
    "sigma_min": 0.28,
    "semantic_threshold": 0.25,
    "structural_threshold": 0.46,
}


def load_detail(detail_path: Path):
    records = []
    with detail_path.open() as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def recompute_sentence(sentence: Dict[str, object], thresholds: Dict[str, float]):
    metrics = sentence.get("metrics", {})
    decisions = {
        "repeat_ok": metrics.get("repetitions", 0) >= thresholds["r_min"],
        "hazard_ok": metrics.get("lambda", metrics.get("hazard", 1.0)) <= thresholds["lambda_max"],
        "semantic_ok": metrics.get("semantic", 0.0) >= thresholds["sigma_min"],
        "structural_ok": metrics.get("patternability", 0.0) >= thresholds["structural_threshold"],
    }
    admit = all(decisions.values())
    sentence = dict(sentence)
    sentence["decisions"] = {**sentence.get("decisions", {}), **decisions, "admit": admit}
    if not admit and sentence.get("repair_span"):
        sentence["action"] = "repair"
    else:
        sentence["action"] = "emit" if admit else "decline"
    return sentence


def assemble_answer(sentences, fallback="No supporting evidence."):
    out = []
    for sentence in sentences:
        if sentence["decisions"].get("admit"):
            out.append(sentence["sentence"])
        elif sentence.get("action") == "repair" and sentence.get("repair_span"):
            out.append(sentence["repair_span"])
        else:
            out.append(fallback)
    return " ".join(out)


def main():
    st.set_page_config(page_title="Reality Filter Caseboard", layout="wide")
    detail_path = st.sidebar.text_input("Evaluation detail path", "results/eval/docs_demo/eval_detail.jsonl")
    detail_file = Path(detail_path)
    if not detail_file.exists():
        st.warning(f"Detail file not found: {detail_file}")
        st.stop()

    records = load_detail(detail_file)
    claim_ids = [record.get("id", f"claim_{idx}") for idx, record in enumerate(records)]
    selected_id = st.sidebar.selectbox("Claim", claim_ids)
    record = next((rec for rec in records if rec.get("id") == selected_id), records[0])

    st.sidebar.write("Thresholds")
    thresholds = {}
    thresholds["r_min"] = st.sidebar.slider("Repeat min", 0, 5, DEFAULT_THRESHOLDS["r_min"])
    thresholds["lambda_max"] = st.sidebar.slider("Lambda max", 0.0, 1.0, DEFAULT_THRESHOLDS["lambda_max"], step=0.01)
    thresholds["sigma_min"] = st.sidebar.slider("Semantic min", 0.0, 1.0, DEFAULT_THRESHOLDS["sigma_min"], step=0.01)
    thresholds["semantic_threshold"] = thresholds["sigma_min"]
    thresholds["structural_threshold"] = st.sidebar.slider("Patternability min", 0.0, 1.0, DEFAULT_THRESHOLDS["structural_threshold"], step=0.01)

    recomputed = [recompute_sentence(sentence, thresholds) for sentence in record.get("sentences", [])]
    final_answer = assemble_answer(recomputed)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Baseline LLM")
        st.write(record.get("raw_answer", ""))
    with col2:
        st.subheader("Decision Log")
        for sentence in recomputed:
            badges = " ".join(
                [
                    "✅" if sentence["decisions"].get(key) else "❌"
                    for key in ["repeat_ok", "hazard_ok", "semantic_ok", "structural_ok"]
                ]
            )
            st.markdown(f"**{sentence['action'].upper()}** {badges}<br/>{sentence['sentence']}", unsafe_allow_html=True)
            twins = sentence.get("twins", [])
            if twins:
                st.caption(", ".join(twin.get("string", "") for twin in twins))
    with col3:
        st.subheader("Reality Filter Answer")
        st.write(final_answer)

    st.markdown("---")
    st.markdown(f"**Expected:** {record.get('expected')} · **Original Pred:** {record.get('predicted')} · **Question:** {record.get('question')}")


if __name__ == "__main__":
    main()
