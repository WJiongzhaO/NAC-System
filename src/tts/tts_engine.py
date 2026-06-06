"""
TTS 引擎工厂 + 自动降级 — 统一入口
=====================================

职责
----
1. 按配置选择主引擎：edge(Edge-TTS, 在线高质量) / piper(本地兜底)。
2. 自动降级：主引擎为 edge 但初始化/合成失败(断网、超时、服务异常)时，
   透明回退到 Piper，保证链路永不中断。
3. 对外暴露与 PiperTTS 完全一致的接口(synthesize /
   synthesize_streaming_generator)，现有 pipeline 零改动。

用法
----
    from src.tts.tts_engine import create_tts
    tts = create_tts(engine="edge", piper_model_path="...huayan-medium.onnx")
    tts.synthesize("文本", "out.wav")   # 与 PiperTTS 用法一致

设计：组合而非继承。EngineWithFallback 持有 primary + fallback 两个实现，
每次合成 try primary，异常则切 fallback 并记忆(避免反复重试拖慢)。
"""

from __future__ import annotations

import os
import sys
import time


def _make_piper(**kwargs):
    try:
        from src.tts.piper_tts import PiperTTS
    except Exception:
        from piper_tts import PiperTTS
    # 只透传 PiperTTS 认识的参数
    allowed = {"model_name", "model_path", "device", "ssml_enabled",
               "ssml_lexicon", "text_normalize", "tn_lexicon_path"}
    return PiperTTS(**{k: v for k, v in kwargs.items() if k in allowed})


def _make_edge(**kwargs):
    try:
        from src.tts.edge_tts_engine import EdgeTTS
    except Exception:
        from edge_tts_engine import EdgeTTS
    allowed = {"voice", "rate", "volume", "pitch", "sample_rate",
               "text_normalize", "tn_lexicon_path", "proxy"}
    return EdgeTTS(**{k: v for k, v in kwargs.items() if k in allowed})


class EngineWithFallback:
    """主引擎 + 兜底引擎。primary 失败时自动切 fallback。"""

    def __init__(self, primary, fallback, primary_name="edge", fallback_name="piper"):
        self._primary = primary
        self._fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self._primary_dead = False   # 记忆：primary 已失败则后续直接用 fallback

    @property
    def active_engine(self) -> str:
        return self.fallback_name if (self._primary_dead or self._primary is None) else self.primary_name

    def _active(self):
        if self._primary_dead or self._primary is None:
            return self._fallback
        return self._primary

    def synthesize(self, text: str, output_path: str | None = None) -> dict:
        eng = self._active()
        try:
            r = eng.synthesize(text, output_path)
            r["engine"] = self.active_engine
            return r
        except Exception as e:
            # primary 失败 → 降级
            if eng is self._primary and self._fallback is not None:
                print(f"[TTS] 主引擎 {self.primary_name} 失败，降级到 {self.fallback_name}: {e}",
                      file=sys.stderr)
                self._primary_dead = True
                r = self._fallback.synthesize(text, output_path)
                r["engine"] = self.fallback_name
                r["fallback_reason"] = str(e)
                return r
            raise

    def synthesize_streaming_generator(self, sentences: list[str]):
        eng = self._active()
        try:
            yield from eng.synthesize_streaming_generator(sentences)
        except Exception as e:
            if eng is self._primary and self._fallback is not None:
                print(f"[TTS] 主引擎 {self.primary_name} 流式失败，降级到 {self.fallback_name}: {e}",
                      file=sys.stderr)
                self._primary_dead = True
                yield from self._fallback.synthesize_streaming_generator(sentences)
            else:
                raise

    # 透传 voice 属性(供 web 预热 .voice 时不报错)
    @property
    def voice(self):
        eng = self._active()
        return getattr(eng, "voice", None)


def create_tts(engine: str = "edge",
               # edge 参数
               voice: str = "zh-CN-YunjianNeural",
               rate: str = "+0%",
               volume: str = "+0%",
               pitch: str = "+0Hz",
               sample_rate: int = 22050,
               proxy: str | None = None,
               # piper 参数
               piper_model_name: str = "zh_CN-huayan-medium",
               piper_model_path: str | None = None,
               piper_device: str = "cpu",
               # 公共
               text_normalize: bool = True,
               tn_lexicon_path: str | None = None,
               enable_fallback: bool = True):
    """
    创建 TTS 引擎。

    engine="edge"  : 主 Edge-TTS，可选 Piper 兜底(enable_fallback)
    engine="piper" : 仅 Piper(纯离线)

    返回对象接口与 PiperTTS 一致。
    """
    common = dict(text_normalize=text_normalize, tn_lexicon_path=tn_lexicon_path)

    if engine == "piper":
        return _make_piper(model_name=piper_model_name, model_path=piper_model_path,
                           device=piper_device, **common)

    # engine == "edge"
    edge = None
    try:
        edge = _make_edge(voice=voice, rate=rate, volume=volume, pitch=pitch,
                          sample_rate=sample_rate, proxy=proxy, **common)
    except Exception as e:
        print(f"[TTS] Edge-TTS 初始化失败: {e}", file=sys.stderr)

    fallback = None
    if enable_fallback:
        try:
            fallback = _make_piper(model_name=piper_model_name,
                                   model_path=piper_model_path,
                                   device=piper_device, **common)
        except Exception as e:
            print(f"[TTS] Piper 兜底初始化失败(将无兜底): {e}", file=sys.stderr)

    if edge is None and fallback is None:
        raise RuntimeError("Edge-TTS 与 Piper 兜底均不可用")
    if edge is None:
        return fallback  # 只剩兜底
    return EngineWithFallback(edge, fallback,
                              primary_name="edge", fallback_name="piper")


if __name__ == "__main__":
    # 独立冒烟测试
    txt = "根据 CCS 规范，No.3 货舱钢板厚度不应小于 5.5mm。"
    print("=== engine=edge (+piper fallback) ===")
    tts = create_tts(engine="edge", rate="-5%",
                     piper_model_path="pretrained_models/piper/zh_CN-huayan-medium.onnx")
    r = tts.synthesize(txt, "output_engine_smoke.wav")
    print("  active:", getattr(tts, "active_engine", "piper"),
          "| engine_used:", r.get("engine"),
          "| ms:", round(r["latency_ms"]), "| dur:", round(r["duration_s"], 2))
