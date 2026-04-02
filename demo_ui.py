import os

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("DEMO_API_URL", "http://localhost:8000")


def call_health(api_base_url: str):
    r = requests.get(f"{api_base_url}/health", timeout=8)
    r.raise_for_status()
    return r.json()


def call_predict(api_base_url: str, payload: dict):
    r = requests.post(f"{api_base_url}/v1/predict", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def render_level(title: str, items: list[dict]):
    st.subheader(title)
    for item in items:
        st.write(f"- {item['label']} ({item['prob']:.1%})")
        st.progress(float(item["prob"]))


def main():
    st.set_page_config(page_title="HARNN Demo", page_icon="NLP", layout="wide")
    st.title("HARNN Demo - Client Calling API")

    with st.expander("API config", expanded=True):
        api_base_url = st.text_input("API base URL", value=DEFAULT_API_URL).rstrip("/")
        if st.button("Check API health"):
            try:
                st.json(call_health(api_base_url))
            except Exception as e:
                st.error(f"Health check failed: {e}")

    col1, col2 = st.columns([2, 1])
    with col2:
        threshold = st.slider("Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        top_k = st.slider("Top-k fallback", min_value=1, max_value=5, value=3, step=1)

    with col1:
        input_text = st.text_area("Input text", value="Nhap doan van tieng Viet de du doan.", height=240)
        run_btn = st.button("Predict", type="primary")

    if run_btn:
        if not input_text.strip():
            st.warning("Please input text before predicting.")
            return

        payload = {
            "text": input_text,
            "threshold": threshold,
            "top_k": top_k,
            "return_tokens": True,
        }
        try:
            result = call_predict(api_base_url, payload)
        except Exception as e:
            st.error(f"Predict request failed: {e}")
            return

        labels = result.get("labels") or {
            "l1": result.get("l1", []),
            "l2": result.get("l2", []),
            "l3": result.get("l3", []),
        }
        st.caption(f"Tokens (first 10): {result.get('tokens') or []}")

        meta = result.get("meta") or {}
        if meta:
            st.write(
                f"Device: {meta.get('device')} | Tokenizer: {meta.get('tokenizer')} | "
                f"Latency: {meta.get('latency_ms')} ms"
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            render_level("L1", labels.get("l1", []))
        with c2:
            render_level("L2", labels.get("l2", []))
        with c3:
            render_level("L3", labels.get("l3", []))


if __name__ == "__main__":
    main()
