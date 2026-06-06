"""
TTS 引擎 — Edge-TTS 适配 (微软神经 TTS，在线)
================================================

为什么引入：Piper 的中文走 espeak-ng 前端，G2P(字→音)阶段经常丢声调，
导致"字字一声、多音字读错"，这是 Piper 中文的架构天花板，调参无法解决。
Edge-TTS 是微软商用级神经 TTS，中文原生建模、声调/多音字/韵律均为工业水准，
且零模型下载、零 GPU、几十种中文音色。

设计目标
--------
- 与 PiperTTS **同接口**：synthesize() / synthesize_streaming_generator()，
  返回结构一致 {audio_path, latency_ms, duration_s}，管线零改动即可替换。
- 输出统一为 WAV（与 Piper 一致，下游拼接/播放不受影响）。
- 复用方向 A 文本前端规整(术语/数字/单位)。
- 在线失败可由上层工厂自动降级到 Piper（见 tts_engine.py）。

依赖：edge-tts、soundfile（libsndfile 自带 MP3 解码）。均为 pip 包，无 GPU。

音色（造船专业沉稳推荐）：
  zh-CN-YunjianNeural  云健 — 沉稳，新闻/体育播报男声（默认）
  zh-CN-YunyangNeural  云扬 — 专业新闻播报男声
  zh-CN-YunxiNeural    云希 — 活力男声
  zh-CN-XiaoxiaoNeural 晓晓 — 自然女声
"""

from __future__ import annotations

import asyncio
import io
import os
import ssl
import tempfile
import time
from pathlib import Path

import edge_tts
import soundfile as sf

try:
    from src.tts.text_frontend import TextFrontend
except Exception:
    try:
        from text_frontend import TextFrontend
    except Exception:
        TextFrontend = None


# ============================================================
# 沙箱/代理环境的 TLS 兼容（仅当显式开启时生效；生产默认正常校验）
# ============================================================
def _maybe_patch_ssl_for_proxy():
    """
    某些企业网关对 HTTPS 做 TLS 拦截(self-signed CA)，aiohttp 经 HTTPS 代理
    隧道时无法用常规 connector 传入 CA。仅当环境变量 EDGE_TTS_INSECURE=1 时，
    放宽 asyncio start_tls 的校验。生产环境正常联网时**不要**设置该变量。
    """
    if os.environ.get("EDGE_TTS_INSECURE") != "1":
        return
    import asyncio as _a
    loop_cls = _a.SelectorEventLoop
    if getattr(loop_cls, "_edge_tts_patched", False):
        return
    orig = loop_cls.start_tls

    async def patched(self, transport, protocol, sslcontext, *a, **k):
        return await orig(self, transport, protocol,
                          ssl._create_unverified_context(), *a, **k)

    loop_cls.start_tls = patched
    loop_cls._edge_tts_patched = True


class EdgeTTS:
    """Edge-TTS 适配类，接口对齐 PiperTTS。"""

    def __init__(
        self,
        voice: str = "zh-CN-YunjianNeural",   # 沉稳专业男声
        rate: str = "+0%",                      # 语速，如 "-10%" 更沉稳
        volume: str = "+0%",
        pitch: str = "+0Hz",                    # 音调
        sample_rate: int = 22050,               # 与 Piper 对齐
        text_normalize: bool = True,
        tn_lexicon_path: str | None = None,
        proxy: str | None = None,
        # 兼容 PiperTTS 的构造签名(忽略不适用的参数)，便于工厂统一调用
        **_ignored,
    ):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.sample_rate = sample_rate
        self.proxy = proxy or os.environ.get("HTTPS_PROXY") or None

        self.text_normalize = bool(text_normalize and TextFrontend is not None)
        self._frontend = (
            TextFrontend(extra_lexicon_path=tn_lexicon_path)
            if self.text_normalize else None
        )
        _maybe_patch_ssl_for_proxy()

    # —— 内部：异步合成为 MP3 字节 ——
    async def _synth_bytes(self, text: str) -> bytes:
        kwargs = dict(text=text, voice=self.voice,
                      rate=self.rate, volume=self.volume, pitch=self.pitch)
        if self.proxy:
            kwargs["proxy"] = self.proxy
        communicate = edge_tts.Communicate(**kwargs)
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        return bytes(buf)

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # 已在事件循环中(极少见)，用新线程跑
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            return ex.submit(lambda: asyncio.run(coro)).result()

    def synthesize(self, text: str, output_path: str | None = None) -> dict:
        """整段合成，输出 WAV。接口与 PiperTTS.synthesize 完全一致。"""
        output = Path(output_path) if output_path else Path(tempfile.mktemp(suffix=".wav"))

        if self._frontend is not None:
            text = self._frontend.normalize(text)

        t0 = time.perf_counter()
        mp3_bytes = self._run_async(self._synth_bytes(text))
        # MP3 -> WAV(libsndfile 支持 MP3 解码)，重采样到目标采样率
        data, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # 转单声道
        if sr != self.sample_rate:
            data = _resample_linear(data, sr, self.sample_rate)
        sf.write(str(output), data, self.sample_rate, subtype="PCM_16")
        t1 = time.perf_counter()

        try:
            duration_s = sf.info(str(output)).duration
        except Exception:
            duration_s = len(data) / self.sample_rate

        return {
            "audio_path": str(output),
            "latency_ms": (t1 - t0) * 1000,
            "duration_s": duration_s,
        }

    def synthesize_streaming_generator(self, sentences: list[str]):
        """逐句合成，接口对齐 PiperTTS。"""
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            out = Path(tempfile.mktemp(suffix=f"_sent{i}.wav"))
            r = self.synthesize(sentence, str(out))
            yield {
                "audio_path": r["audio_path"],
                "latency_ms": r["latency_ms"],
                "sentence_index": i,
                "text": sentence,
            }


def _resample_linear(x, src_sr: int, dst_sr: int):
    """轻量线性重采样，避免引入 librosa。"""
    import numpy as np
    if src_sr == dst_sr:
        return x
    n_dst = int(round(len(x) * dst_sr / src_sr))
    if n_dst <= 1:
        return x
    xp = np.linspace(0, 1, num=len(x), endpoint=False)
    xq = np.linspace(0, 1, num=n_dst, endpoint=False)
    return np.interp(xq, xp, x).astype("float32")


# ============================================================
# 独立运行：python src/tts/edge_tts_engine.py [文本]
# ============================================================
if __name__ == "__main__":
    import sys
    txt = sys.argv[1] if len(sys.argv) > 1 else \
        "根据 CCS 规范，No.3 货舱钢板厚度不应小于 5.5mm，预热温度需 ≤300℃。"
    out = "output_edge_demo.wav"
    tts = EdgeTTS(voice="zh-CN-YunjianNeural", rate="-5%")
    print(f"音色: {tts.voice}")
    print(f"原文: {txt}")
    if tts._frontend:
        print(f"规整: {tts._frontend.normalize(txt)}")
    r = tts.synthesize(txt, out)
    print(f"输出: {r['audio_path']}  耗时 {r['latency_ms']:.0f} ms  时长 {r['duration_s']:.2f}s")
