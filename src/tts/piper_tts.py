"""
TTS 模块 — Piper TTS (ONNX / zh_CN-huayan-medium)

支持两种模式：
- synthesize: 整段合成，输出 WAV 文件（基线）
- synthesize_streaming: 逐句合成 + 流式播放（改进阶段）

发音控制通过 W3C SSML 标准实现（改进阶段启用）。
"""

import os
import tempfile
import time
import xml.sax.saxutils as saxutils
from pathlib import Path

from piper import PiperVoice, SynthesisConfig


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
        self._model_path = self._resolve_model_path(model_path, model_name)
        self._voice: PiperVoice | None = None

        if not self._model_path.exists():
            raise RuntimeError(f"Piper model not found: {self._model_path}")

    def _resolve_model_path(self, model_path: str | None, model_name: str) -> Path:
        """解析 Piper 语音模型路径。"""
        if model_path:
            return Path(model_path)

        candidate_paths = [
            Path("pretrained_models/piper") / f"{model_name}.onnx",
            Path("pretrained_models/piper") / model_name,
            Path(model_name),
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                return candidate

        return candidate_paths[0]

    def _resolve_use_cuda(self) -> bool:
        """仅在显式要求且运行时可用时启用 CUDA。"""
        if self.device.lower() != "cuda":
            return False

        try:
            import onnxruntime

            return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        except Exception:
            return False

    @property
    def voice(self) -> PiperVoice:
        """懒加载并缓存 PiperVoice，避免重复加载模型。"""
        if self._voice is None:
            self._voice = PiperVoice.load(
                self._model_path,
                use_cuda=self._resolve_use_cuda(),
            )
        return self._voice

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

        t0 = time.perf_counter()
        wav_file = None
        try:
            import wave

            wav_file = wave.open(str(output), "wb")
            self.voice.synthesize_wav(text, wav_file)
        finally:
            if wav_file is not None:
                wav_file.close()
        t1 = time.perf_counter()

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
