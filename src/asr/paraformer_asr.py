"""
ASR 模块 — paraformer-zh (FunASR 流式主力)

支持两种模式：
- transcribe: 整段转写（基线）
- transcribe_streaming: VAD 分句 + 逐句输出（改进阶段）
"""

import time

import librosa
import numpy as np
import soundfile as sf
from funasr import AutoModel

# Paraformer / FSMN-VAD 训练采样率
TARGET_SAMPLE_RATE = 16000


class ParaformerASR:
    def __init__(
        self,
        model_id: str = "paraformer-zh",
        device: str = "cuda",
        vad_model: str = "fsmn-vad",
        vad_max_segment_time: int = 30000,
    ):
        self.model_id = model_id
        self.device = device
        print(f"[ASR] 加载模型 {model_id} (device={device}) ...")
        self.model = AutoModel(
            model=model_id,
            vad_model=vad_model,
            vad_kwargs={"max_single_segment_time": vad_max_segment_time},
            device=device,
        )
        print("[ASR] 模型加载完成")

    @staticmethod
    def _load_audio_mono_16k(audio_path: str) -> np.ndarray:
        """加载单声道 float32 音频并重采样到 16 kHz（Paraformer 要求）。"""
        audio_data, sample_rate = sf.read(audio_path, dtype="float32")
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        if sample_rate != TARGET_SAMPLE_RATE:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=TARGET_SAMPLE_RATE,
            )
        return audio_data.astype(np.float32)

    def transcribe(self, audio_path: str, hotword: str = "") -> dict:
        """
        整段转写（基线模式）

        使用 soundfile 加载并重采样到 16 kHz，绕过 ffmpeg 依赖。
        """
        audio_data = self._load_audio_mono_16k(audio_path)

        t0 = time.perf_counter()
        result = self.model.generate(
            input=[audio_data],
            hotword=hotword if hotword else None,
            language="zh",
            use_itn=True,
        )
        t1 = time.perf_counter()

        if not result or len(result) == 0:
            return {"text": "", "segments": [], "latency_ms": (t1 - t0) * 1000}

        full_text = result[0].get("text", "")
        segments = result[0].get("sentences", [])

        return {
            "text": full_text,
            "segments": [
                {"text": seg.get("text", ""), "start_ms": seg.get("start", 0), "end_ms": seg.get("end", 0)}
                for seg in segments
            ],
            "latency_ms": (t1 - t0) * 1000,
        }

    def transcribe_streaming_generator(self, audio_path: str, hotword: str = ""):
        """
        流式分句转写（改进模式），逐句 yield

        利用内置 FSMN-VAD 自动分句，每识别完一句即产出。
        """
        result = self.transcribe(audio_path, hotword=hotword)
        for seg in result["segments"]:
            yield {
                "text": seg["text"],
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
            }
