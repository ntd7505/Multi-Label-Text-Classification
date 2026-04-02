import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent
UI_ENTRY = PROJECT_DIR / "demo_ui.py"
CHECKPOINT = PROJECT_DIR / "output" / "models" / "checkpoints" / "best_model.pt"
VOCAB_FILE = PROJECT_DIR / "data" / "process_data" / "vocab.json"
LABEL_FILE = PROJECT_DIR / "data" / "process_data" / "label_map.json"
STOPWORDS_FILE = PROJECT_DIR / "data" / "dictionary" / "vietnamese-stopwords.txt"

MAX_LEN = 512
DEFAULT_API_PORT = 8000
DEFAULT_UI_PORT = 8501


class HARNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes_per_level, dropout=0.5):
        super().__init__()
        self.num_levels = len(num_classes_per_level)
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bigru = nn.GRU(embed_dim, hidden_size, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.attention = nn.ModuleList([nn.Linear(hidden_size * 2, 1) for _ in range(self.num_levels)])
        self.ham = nn.LSTMCell(hidden_size * 2, hidden_size)
        self.classifiers = nn.ModuleList([nn.Linear(hidden_size * 3, n) for n in num_classes_per_level])

    def forward(self, x):
        doc = self.drop(self.bigru(self.drop(self.emb(x)))[0])
        h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        c = torch.zeros(x.size(0), self.hidden_size, device=x.device)

        preds = []
        for lv in range(self.num_levels):
            context = (torch.softmax(self.attention[lv](doc), dim=1) * doc).sum(dim=1)
            h, c = self.ham(context, (h, c))
            feat = self.drop(torch.cat([context, h], dim=-1))
            preds.append(self.classifiers[lv](feat))
        return preds


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    top_k: int = Field(3, ge=1, le=10)
    return_tokens: bool = True


class InferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Extra calibration to reduce overconfident predictions.
        self.global_temp_multiplier = 1.15
        self.confidence_shrink = 0.88
        self._load_assets()

    def _normalize_state_dict(self, state_dict):
        sd = dict(state_dict)
        if "embedding.weight" in sd and "emb.weight" not in sd:
            sd["emb.weight"] = sd["embedding.weight"]
        sd.pop("embedding.weight", None)
        return sd

    def _load_assets(self):
        for p in [CHECKPOINT, VOCAB_FILE, LABEL_FILE, STOPWORDS_FILE]:
            if not p.exists():
                raise FileNotFoundError(f"Missing required file: {p}")

        with open(VOCAB_FILE, encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(LABEL_FILE, encoding="utf-8") as f:
            label_map = json.load(f)
        with open(STOPWORDS_FILE, encoding="utf-8") as f:
            self.stopwords = {line.strip() for line in f if line.strip()}

        self.idx_to_label = {
            level: {v: k for k, v in label_map[level].items()}
            for level in ["l1", "l2", "l3"]
        }
        num_classes = [len(label_map["l1"]), len(label_map["l2"]), len(label_map["l3"])]

        self.model = HARNN(
            vocab_size=len(self.vocab),
            embed_dim=100,
            hidden_size=256,
            num_classes_per_level=num_classes,
        ).to(self.device)

        ckpt = torch.load(CHECKPOINT, map_location=self.device)
        state_dict = self._normalize_state_dict(ckpt["model_state"])
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        default_temps = {"l1": 1.6, "l2": 1.6, "l3": 1.6}
        loaded_temps = ckpt.get("temperatures", default_temps)
        self.temperatures = {
            level: float(max(0.5, min(10.0, loaded_temps.get(level, default_temps[level]))))
            for level in ["l1", "l2", "l3"]
        }

        self.model_epoch = ckpt.get("epoch", None)
        self.val_f1_l1 = ckpt.get("val_f1_l1", None)

        try:
            from underthesea import word_tokenize

            self._word_tokenize = lambda t: word_tokenize(t, format="text").split()
            self.tokenizer_name = "underthesea"
        except Exception:
            self._word_tokenize = lambda t: t.split()
            self.tokenizer_name = "whitespace"

    def clean_text(self, text: str) -> str:
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", " ", text)
        return re.sub(r"\s+", " ", text).strip().lower()

    def tokenize(self, text: str) -> list[str]:
        cleaned = self.clean_text(text)
        tokens = self._word_tokenize(cleaned)
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]

    def text_to_tensor(self, text: str):
        tokens = self.tokenize(text)
        ids = [self.vocab.get(t, 1) for t in tokens][:MAX_LEN]
        ids += [0] * (MAX_LEN - len(ids))
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        return x, tokens

    def _safe_temperature(self, level: str) -> float:
        base = float(self.temperatures.get(level, 1.0))
        t = base * float(self.global_temp_multiplier)
        return min(max(t, 0.5), 10.0)

    def _calibrate_probs(self, logits: torch.Tensor, level: str) -> np.ndarray:
        probs = torch.sigmoid(logits / self._safe_temperature(level)).detach().cpu().numpy()
        alpha = min(max(float(self.confidence_shrink), 0.0), 1.0)
        return (0.5 + (probs - 0.5) * alpha).clip(1e-6, 1 - 1e-6)

    @torch.no_grad()
    def predict(self, req: PredictRequest):
        start = time.perf_counter()
        x, tokens = self.text_to_tensor(req.text)
        logits = self.model(x)

        result = {"tokens": tokens[:10] if req.return_tokens else None}
        labels = {}

        for i, level in enumerate(["l1", "l2", "l3"]):
            probs = self._calibrate_probs(logits[i][0], level)
            idx_to_label = self.idx_to_label[level]

            selected = [
                {"label": idx_to_label[j], "prob": float(probs[j])}
                for j in range(len(probs))
                if probs[j] >= req.threshold
            ]
            if not selected:
                top_indices = np.argsort(probs)[::-1][: req.top_k]
                selected = [{"label": idx_to_label[j], "prob": float(probs[j])} for j in top_indices]

            selected.sort(key=lambda x: x["prob"], reverse=True)
            labels[level] = selected
            result[level] = selected

        result["labels"] = labels
        result["meta"] = {
            "threshold": req.threshold,
            "top_k": req.top_k,
            "device": str(self.device),
            "tokenizer": self.tokenizer_name,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "model_epoch": self.model_epoch,
            "val_f1_l1": self.val_f1_l1,
            "temperatures": {level: self._safe_temperature(level) for level in ["l1", "l2", "l3"]},
            "global_temp_multiplier": self.global_temp_multiplier,
            "confidence_shrink": self.confidence_shrink,
        }
        return result


app = FastAPI(title="HARNN Demo API", version="1.0.0")
try:
    ENGINE = InferenceEngine()
except Exception as e:
    ENGINE = None
    INIT_ERROR = str(e)
else:
    INIT_ERROR = None


@app.get("/health")
def health():
    if ENGINE is None:
        return {"status": "error", "detail": INIT_ERROR}
    return {"status": "ok", "device": str(ENGINE.device), "tokenizer": ENGINE.tokenizer_name}


@app.post("/v1/predict")
def predict(req: PredictRequest):
    if ENGINE is None:
        raise HTTPException(status_code=500, detail=f"Engine init failed: {INIT_ERROR}")
    try:
        return ENGINE.predict(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict failed: {e}")


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


def render_ui(default_api_url: str):
    st.set_page_config(page_title="HARNN Demo", page_icon="NLP", layout="wide")
    st.title("HARNN Demo - Client Calling API")

    with st.expander("API config", expanded=True):
        api_base_url = st.text_input("API base URL", value=default_api_url).rstrip("/")
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


def run_api(port: int):
    import uvicorn

    uvicorn.run("demo:app", host="0.0.0.0", port=port, reload=False)


def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def find_free_port(start_port: int, max_tries: int = 20) -> int:
    for p in range(start_port, start_port + max_tries):
        if is_port_free(p):
            return p
    raise RuntimeError(f"No free port found from {start_port} to {start_port + max_tries - 1}")


def can_use_existing_api(port: int) -> bool:
    try:
        r = requests.get(f"http://localhost:{port}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def run_all(api_port: int, ui_port: int):
    python_exe = sys.executable
    file_path = str(Path(__file__).resolve())

    api_proc = None
    if can_use_existing_api(api_port):
        print(f"API already running at http://localhost:{api_port} (reuse)")
    else:
        if not is_port_free(api_port):
            new_api_port = find_free_port(api_port + 1)
            print(f"API port {api_port} is busy -> switch to {new_api_port}")
            api_port = new_api_port

        api_cmd = [python_exe, file_path, "api", "--api-port", str(api_port)]
        api_proc = subprocess.Popen(api_cmd, cwd=PROJECT_DIR)

        time.sleep(2)
        if api_proc.poll() is not None:
            raise RuntimeError("API failed to start. Check logs.")

    if not is_port_free(ui_port):
        new_ui_port = find_free_port(ui_port + 1)
        print(f"UI port {ui_port} is busy -> switch to {new_ui_port}")
        ui_port = new_ui_port

    print(f"API: http://localhost:{api_port}")
    print(f"UI : http://localhost:{ui_port}")
    print("Press Ctrl+C to stop.")

    ui_cmd = [
        python_exe,
        "-m",
        "streamlit",
        "run",
        str(UI_ENTRY),
        "--server.headless",
        "true",
        "--server.port",
        str(ui_port),
    ]

    ui_env = os.environ.copy()
    ui_env["DEMO_API_URL"] = f"http://localhost:{api_port}"

    try:
        subprocess.run(ui_cmd, cwd=PROJECT_DIR, check=False, env=ui_env)
    finally:
        if api_proc is not None and api_proc.poll() is None:
            api_proc.terminate()
            try:
                api_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_proc.kill()


def run_ui(api_url: str, ui_port: int):
    python_exe = sys.executable
    ui_cmd = [
        python_exe,
        "-m",
        "streamlit",
        "run",
        str(UI_ENTRY),
        "--server.headless",
        "true",
        "--server.port",
        str(ui_port),
    ]
    ui_env = os.environ.copy()
    ui_env["DEMO_API_URL"] = api_url
    subprocess.run(ui_cmd, cwd=PROJECT_DIR, check=False, env=ui_env)


def parse_args():
    parser = argparse.ArgumentParser(description="Single-file demo: API + UI")
    sub = parser.add_subparsers(dest="command")

    p_api = sub.add_parser("api", help="Run only API")
    p_api.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)

    p_ui = sub.add_parser("ui", help="Run only Streamlit UI")
    p_ui.add_argument("--api-url", type=str, default=f"http://localhost:{DEFAULT_API_PORT}")
    p_ui.add_argument("--ui-port", type=int, default=DEFAULT_UI_PORT)

    p_all = sub.add_parser("all", help="Run API + UI together")
    p_all.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    p_all.add_argument("--ui-port", type=int, default=DEFAULT_UI_PORT)

    return parser.parse_args()


def main():
    # When started by `streamlit run demo.py`, render UI directly instead of launcher mode.
    if os.getenv("DEMO_STREAMLIT_MODE") == "1" or os.getenv("STREAMLIT_SERVER_PORT") is not None:
        render_ui(os.getenv("DEMO_API_URL", f"http://localhost:{DEFAULT_API_PORT}"))
        return

    args = parse_args()
    if args.command == "api":
        run_api(args.api_port)
    elif args.command == "ui":
        run_ui(args.api_url, args.ui_port)
    else:
        api_port = getattr(args, "api_port", DEFAULT_API_PORT)
        ui_port = getattr(args, "ui_port", DEFAULT_UI_PORT)
        run_all(api_port, ui_port)


if __name__ == "__main__":
    main()
