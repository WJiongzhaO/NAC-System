"""
ASR 模块 — paraformer-zh (FunASR)

支持三种模式：
- transcribe: 整段转写（基线，paraformer-zh + VAD）
- transcribe_streaming / transcribe_streaming_generator:
  在线 chunk 流式（paraformer-zh-streaming + chunk_size + cache）
"""

import time
from typing import Any, Iterator

import librosa
import numpy as np
import soundfile as sf
from funasr import AutoModel

# Paraformer / FSMN-VAD 训练采样率
TARGET_SAMPLE_RATE = 16000

# FunASR 流式默认：600ms 步进 + 300ms 前瞻（单位 60ms 帧）
DEFAULT_STREAMING_CHUNK_SIZE = [0, 10, 5]
# chunk_size[1] * 960 samples @ 16kHz = 600ms
STREAMING_FRAME_SAMPLES = 960


class ParaformerASR:
    def __init__(
        self,
        model_id: str = "paraformer-zh",
        streaming_model_id: str = "paraformer-zh-streaming",
        device: str = "cuda",
        vad_model: str = "fsmn-vad",
        vad_max_segment_time: int = 30000,
        streaming_chunk_size: list[int] | None = None,
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
        load_offline_model: bool = True,
    ):
        self.model_id = model_id
        self.streaming_model_id = streaming_model_id
        self.device = device
        self.vad_model = vad_model
        self.vad_max_segment_time = vad_max_segment_time
        self.streaming_chunk_size = streaming_chunk_size or list(DEFAULT_STREAMING_CHUNK_SIZE)
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back
        self.load_offline_model = load_offline_model

        self._model: AutoModel | None = None
        self._streaming_model: AutoModel | None = None

        if load_offline_model:
            self._ensure_offline_model()

    def _ensure_offline_model(self) -> AutoModel:
        if self._model is None:
            print(f"[ASR] 加载离线模型 {self.model_id} (device={self.device}) ...")
            self._model = AutoModel(
                model=self.model_id,
                vad_model=self.vad_model,
                vad_kwargs={"max_single_segment_time": self.vad_max_segment_time},
                device=self.device,
                disable_update=True,
            )
            print("[ASR] 离线模型加载完成")
        return self._model

    @property
    def model(self) -> AutoModel:
        return self._ensure_offline_model()

    @property
    def streaming_model(self) -> AutoModel:
        """懒加载流式模型，避免基线模式占用双倍显存。"""
        if self._streaming_model is None:
            print(
                f"[ASR] 加载流式模型 {self.streaming_model_id} (device={self.device}) ..."
            )
            self._streaming_model = AutoModel(
                model=self.streaming_model_id,
                device=self.device,
                disable_update=True,
            )
            print("[ASR] 流式模型加载完成")
        return self._streaming_model

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

    @staticmethod
    def _extract_text(infer_result: Any) -> str:
        """从 FunASR generate 返回值中解析文本。"""
        if not infer_result:
            return ""
        if isinstance(infer_result, list):
            if not infer_result:
                return ""
            first = infer_result[0]
            if isinstance(first, dict):
                return first.get("text", "") or ""
            return str(first)
        if isinstance(infer_result, dict):
            return infer_result.get("text", "") or ""
        return str(infer_result)

    @staticmethod
    def _merge_streaming_text(full_text: str, chunk_text: str) -> tuple[str, str]:
        """
        合并流式 chunk 文本。

        FunASR 流式每次 generate 通常只返回当前块的增量；少数情况下会返回
        截至当前的累计全文（以已有全文为前缀）。返回 (累计全文, 本块增量)。
        """
        chunk_text = chunk_text.strip()
        if not chunk_text:
            return full_text, ""

        if full_text and chunk_text.startswith(full_text):
            return chunk_text, chunk_text[len(full_text) :]

        if full_text and full_text.endswith(chunk_text):
            return full_text, ""

        return full_text + chunk_text, chunk_text

    def _streaming_chunk_stride(self) -> int:
        """每个流式 chunk 的采样点数（由 chunk_size[1] 决定）。"""
        return max(self.streaming_chunk_size[1] * STREAMING_FRAME_SAMPLES, STREAMING_FRAME_SAMPLES)

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

    def transcribe_streaming_generator(
        self, audio_path: str, hotword: str = ""
    ) -> Iterator[dict]:
        """
        在线 chunk 流式转写（FunASR 官方 API）。

        使用 paraformer-zh-streaming + chunk_size + cache 逐块解码，
        每块 yield 累计文本与增量 delta_text。
        """
        audio_data = self._load_audio_mono_16k(audio_path)
        chunk_stride = self._streaming_chunk_stride()
        chunk_size = self.streaming_chunk_size

        cache: dict = {}
        total_chunks = max(1, int((len(audio_data) - 1) / chunk_stride + 1))
        full_text = ""
        started_at = time.perf_counter()

        generate_kwargs: dict[str, Any] = {
            "chunk_size": chunk_size,
            "encoder_chunk_look_back": self.encoder_chunk_look_back,
            "decoder_chunk_look_back": self.decoder_chunk_look_back,
        }
        # 流式 online 模型不支持与离线相同的热词参数，避免干扰识别
        _ = hotword

        for chunk_index in range(total_chunks):
            start = chunk_index * chunk_stride
            end = min(start + chunk_stride, len(audio_data))
            speech_chunk = audio_data[start:end]
            if len(speech_chunk) == 0:
                continue

            is_final = chunk_index == total_chunks - 1
            infer_result = self.streaming_model.generate(
                input=speech_chunk,
                cache=cache,
                is_final=is_final,
                **generate_kwargs,
            )

            chunk_text = self._extract_text(infer_result)
            full_text, delta_text = self._merge_streaming_text(full_text, chunk_text)
            if not delta_text and not is_final:
                continue

            elapsed_ms = (time.perf_counter() - started_at) * 1000
            yield {
                "text": full_text,
                "delta_text": delta_text,
                "chunk_index": chunk_index + 1,
                "chunk_total": total_chunks,
                "is_final": is_final,
                "latency_ms": elapsed_ms,
            }

    def transcribe_streaming(self, audio_path: str, hotword: str = "") -> dict:
        """
        在线 chunk 流式转写的同步封装。

        返回最终完整文本，同时附带首个有效 chunk 和总耗时，供上层做延迟统计。
        """
        started_at = time.perf_counter()
        final_text = ""
        first_chunk_ms = None
        partial_count = 0
        final_segments: list[dict] = []

        for partial in self.transcribe_streaming_generator(audio_path, hotword=hotword):
            partial_count += 1
            final_text = partial["text"]
            if first_chunk_ms is None and len(partial.get("text", "")) >= 2:
                first_chunk_ms = partial["latency_ms"]
            final_segments.append(
                {
                    "text": partial["text"],
                    "start_ms": 0,
                    "end_ms": int(partial["latency_ms"]),
                }
            )

        total_ms = (time.perf_counter() - started_at) * 1000
        return {
            "text": final_text,
            "segments": final_segments,
            "latency_ms": total_ms,
            "first_chunk_ms": first_chunk_ms or total_ms,
            "chunk_count": partial_count,
        }
