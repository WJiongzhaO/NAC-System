"""
本地演示前端服务。

用法:
    python run_web.py
然后打开 http://127.0.0.1:7860
"""

from __future__ import annotations

import cgi
import json
import mimetypes
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eval.latency_timer import LatencyTimer
from src.llm.deepseek_client import DeepSeekClient
from src.pipeline.baseline_pipeline import BaselinePipeline
from src.pipeline.streaming_pipeline import StreamingPipeline
from src.safety.safety_gate import SafetyGate
from src.tts.piper_tts import PiperTTS


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
OUTPUT_DIR = ROOT / "output" / "web"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
TTS_MODEL = "pretrained_models/piper/zh_CN-huayan-medium.onnx"


class TextRunner:
    def __init__(self):
        self.safety = SafetyGate()
        self._llm: DeepSeekClient | None = None
        self._tts: PiperTTS | None = None

    @property
    def llm(self) -> DeepSeekClient:
        if self._llm is None:
            self._llm = DeepSeekClient()
        return self._llm

    @property
    def tts(self) -> PiperTTS:
        if self._tts is None:
            self._tts = PiperTTS(model_path=TTS_MODEL)
        return self._tts

    def run(self, text: str) -> dict:
        timer = LatencyTimer()
        timer.mark("start")
        safety_result = self.safety.check(text)
        timer.mark("safety_end")

        if safety_result["blocked"]:
            answer = safety_result.get("response_text") or self.safety.get_safe_response(
                safety_result.get("category", "unsafe")
            )
            llm_ms = 0
        else:
            llm_start = time.perf_counter()
            llm_result = self.llm.generate(text)
            answer = llm_result["text"]
            llm_ms = (time.perf_counter() - llm_start) * 1000

        audio_path = OUTPUT_DIR / f"text_answer_{int(time.time() * 1000)}.wav"
        tts_result = self.tts.synthesize(answer, str(audio_path))
        timer.mark("end")

        return {
            "mode": "text",
            "asr_text": text,
            "llm_text": answer,
            "safety": safety_result,
            "latency": {
                "safety_ms": timer.elapsed_ms("start", "safety_end"),
                "llm_ms": llm_ms,
                "tts_ms": tts_result["latency_ms"],
                "total_ms": timer.elapsed_ms("start", "end"),
            },
            "audio_url": output_url(audio_path),
        }


TEXT_RUNNER: TextRunner | None = None


def output_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        rel = p.relative_to(ROOT / "output")
    except ValueError:
        return None
    return "/outputs/" + quote(str(rel).replace("\\", "/"))


def json_bytes(payload: dict, status: int = 200) -> tuple[int, bytes]:
    return status, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


class WebHandler(BaseHTTPRequestHandler):
    server_version = "NACDemo/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_file(WEB_DIR / "index.html")
            return
        if parsed.path.startswith("/assets/"):
            self.serve_file(WEB_DIR / parsed.path.removeprefix("/assets/"))
            return
        if parsed.path.startswith("/outputs/"):
            rel = unquote(parsed.path.removeprefix("/outputs/"))
            self.serve_file(ROOT / "output" / rel)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path == "/api/query":
            status, body = self.handle_query()
            self.respond_json(status, body)
            return
        if self.path == "/api/safety":
            status, body = self.handle_safety()
            self.respond_json(status, body)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_safety(self) -> tuple[int, bytes]:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        gate = SafetyGate()
        return json_bytes({"safety": gate.check(payload.get("text", ""))})

    def handle_query(self) -> tuple[int, bytes]:
        global TEXT_RUNNER

        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )

            mode = form.getfirst("mode", "streaming")
            text = (form.getfirst("text", "") or "").strip()
            audio_item = form["audio"] if "audio" in form else None

            if text:
                if TEXT_RUNNER is None:
                    TEXT_RUNNER = TextRunner()
                return json_bytes(TEXT_RUNNER.run(text))

            if audio_item is None or not getattr(audio_item, "filename", ""):
                return json_bytes({"error": "请上传音频文件，或输入文本问题。"}, 400)

            filename = Path(audio_item.filename).name or "question.wav"
            upload_path = UPLOAD_DIR / f"{int(time.time() * 1000)}_{filename}"
            with upload_path.open("wb") as f:
                f.write(audio_item.file.read())

            if mode == "baseline":
                pipeline = BaselinePipeline(tts_model_path=TTS_MODEL)
                record = pipeline.run(str(upload_path), output_dir=str(OUTPUT_DIR))
                response = {
                    "mode": "baseline",
                    "asr_text": record.get("asr_text", ""),
                    "llm_text": record.get("llm_text", ""),
                    "safety": {
                        "blocked": record.get("safety_blocked", False),
                        "category": record.get("safety_category", "allowed"),
                        "reason": record.get("safety_reason"),
                    },
                    "latency": {
                        "asr_ms": record.get("asr_latency_ms"),
                        "llm_ms": record.get("llm_total_ms"),
                        "tts_ms": record.get("tts_latency_ms"),
                        "total_ms": record.get("total_latency_ms"),
                    },
                    "audio_url": output_url(OUTPUT_DIR / f"{upload_path.stem}_output.wav"),
                }
            else:
                pipeline = StreamingPipeline(tts_model_path=TTS_MODEL)
                result = pipeline.run(str(upload_path), output_dir=str(OUTPUT_DIR))
                response = {
                    "mode": "streaming",
                    "asr_text": result.get("asr_text", ""),
                    "llm_text": result.get("llm_text", ""),
                    "safety": {
                        "blocked": result.get("safety_blocked", False),
                        "category": result.get("safety_category", "allowed"),
                        "reason": result.get("safety_reason"),
                    },
                    "latency": {
                        "asr_ms": result.get("asr_latency_ms"),
                        "llm_ttft_ms": result.get("llm_ttft_ms"),
                        "llm_total_ms": result.get("llm_total_ms"),
                        "tts_first_ms": result.get("tts_first_ms"),
                        "first_playable_ms": result.get("first_playable_ms"),
                        "total_ms": result.get("total_ms"),
                    },
                    "audio_url": output_url(result.get("first_output")),
                    "audio_urls": [output_url(p) for p in result.get("outputs", [])],
                }
            return json_bytes(response)
        except Exception as exc:
            return json_bytes({"error": str(exc)}, 500)

    def serve_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def respond_json(self, status: int, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args):
        print(f"[Web] {self.address_string()} - {fmt % args}")


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 7860), WebHandler)
    print("NAC-System Web Demo: http://127.0.0.1:7860")
    print("按 Ctrl+C 停止服务。")
    server.serve_forever()


if __name__ == "__main__":
    main()
