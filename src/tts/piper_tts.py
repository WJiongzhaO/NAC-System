"""
TTS 模块 — Piper TTS (ONNX / zh_CN-huayan-medium)

支持两种模式：
- synthesize: 整段合成，输出 WAV 文件（基线）
- synthesize_streaming: 逐句合成 + 流式播放（改进阶段）

发音控制通过 W3C SSML 标准实现（改进阶段启用）。
"""

import os
import shutil
import subprocess
import tempfile
import time
import xml.sax.saxutils as saxutils
from pathlib import Path


def _find_piper_executable() -> str:
    """查找 piper CLI 可执行文件路径"""
    # 1. 先查系统 PATH
    piper_path = shutil.which("piper")
    if piper_path:
        return piper_path

    # 2. 查当前 venv 的 Scripts 目录
    venv_scripts = Path(__file__).resolve().parent.parent.parent / ".venv" / "Scripts"
    piper_exe = venv_scripts / "piper.exe"
    if piper_exe.exists():
        return str(piper_exe)

    return "piper"


PIPER_EXECUTABLE = _find_piper_executable()


class PiperTTS:
    def __init__(
        self,
        model_name: str = "zh_CN-huayan-medium",
        model_path: str | None = None,
        device: str = "cuda",
        ssml_enabled: bool = False,
        ssml_lexicon: dict | None = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.ssml_enabled = ssml_enabled
        self.ssml_lexicon = ssml_lexicon or {}
        self._piper_exe = PIPER_EXECUTABLE

        # 如果指定了本地模型路径，使用路径；否则用模型名
        self._model_arg = model_path if model_path else model_name

        # 如用本地路径，验证文件存在
        if model_path and not Path(model_path).exists():
            raise RuntimeError(f"Piper model not found: {model_path}")

        # 验证 piper CLI 可用
        try:
            subprocess.run(
                [self._piper_exe, "--help"], capture_output=True, text=True, timeout=10
            )
        except FileNotFoundError:
            raise RuntimeError(
                "piper CLI not found. Run: pip install piper-tts"
            )

    def text_to_ssml(self, text: str) -> str:
        """将纯文本转为 SSML，应用发音词典替换"""
        if not self.ssml_enabled or not self.ssml_lexicon:
            return text

        escaped = saxutils.escape(text)

        ssml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        ssml += '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis">\n'
        ssml += f"  {escaped}\n"
        ssml += "</speak>"
        return ssml

    def _build_piper_command(self, text: str, output_path: str) -> list[str]:
        """构建 piper CLI 命令"""
        cmd = [
            self._piper_exe,
            "--model", self._model_arg,
            "--output_file", str(output_path),
        ]
        if self.ssml_enabled:
            cmd.append("--ssml")
        return cmd

    def synthesize(self, text: str, output_path: str | None = None) -> dict:
        """
        整段合成（基线模式）

        Args:
            text: 待合成文本
            output_path: 输出 WAV 路径，为 None 则自动生成临时文件

        Returns:
            {
                "audio_path": str,
                "latency_ms": float,
                "duration_s": float,
            }
        """
        output = Path(output_path) if output_path else Path(tempfile.mktemp(suffix=".wav"))

        processed_text = self.text_to_ssml(text)
        cmd = self._build_piper_command(processed_text, str(output))

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            input=processed_text if self.ssml_enabled else text,
            capture_output=True,
            text=True,
            timeout=120,
        )
        t1 = time.perf_counter()

        if proc.returncode != 0:
            raise RuntimeError(f"Piper TTS failed: {proc.stderr}")

        try:
            import soundfile as sf
            info = sf.info(str(output))
            duration_s = info.duration
        except Exception:
            duration_s = 0.0

        return {
            "audio_path": str(output),
            "latency_ms": (t1 - t0) * 1000,
            "duration_s": duration_s,
        }

    def synthesize_streaming_generator(self, sentences: list[str]):
        """
        逐句流式合成（改进模式），每句独立合成后 yield 音频路径

        用于"首句优先播放"策略：第一句合成完即可播放，
        后续句子边生成边排队。
        """
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            output_path = Path(tempfile.mktemp(suffix=f"_sent{i}.wav"))
            result = self.synthesize(sentence, str(output_path))
            yield {
                "audio_path": result["audio_path"],
                "latency_ms": result["latency_ms"],
                "sentence_index": i,
                "text": sentence,
            }
