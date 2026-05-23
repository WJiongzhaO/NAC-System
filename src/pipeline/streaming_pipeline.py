"""
流式改进管线 — ASR → Safety → LLM(stream) → TTS(逐句)

核心优化：
- VAD 分句后逐句送入 LLM + TTS
- 首句 LLM token 到达 + TTS 首 chunk 合成 = 首段可播放
- 后续句子流水线并行
"""

import re
import time
from pathlib import Path

from src.asr.paraformer_asr import ParaformerASR
from src.llm.deepseek_client import DeepSeekClient
from src.tts.piper_tts import PiperTTS
from src.safety.safety_gate import SafetyGate
from src.eval.latency_timer import LatencyTimer


class StreamingPipeline:
    def __init__(
        self,
        asr_model_id: str = "paraformer-zh",
        asr_device: str = "cpu",
        llm_api_base: str = "https://api.deepseek.com",
        llm_api_key: str | None = None,
        llm_model: str = "deepseek-v4-flash",
        tts_model_name: str = "zh_CN-huayan-medium",
        tts_model_path: str | None = None,
        hotword: str = "",
    ):
        self.hotword = hotword
        self._asr_kwargs = {"model_id": asr_model_id, "device": asr_device}
        self._llm_kwargs = {"api_base": llm_api_base, "api_key": llm_api_key, "model": llm_model}
        self._tts_kwargs = {"model_name": tts_model_name, "model_path": tts_model_path}
        self._asr: ParaformerASR | None = None
        self._llm: DeepSeekClient | None = None
        self._tts: PiperTTS | None = None
        self._safety: SafetyGate | None = None

    @property
    def asr(self):
        if self._asr is None: self._asr = ParaformerASR(**self._asr_kwargs)
        return self._asr
    @property
    def llm(self):
        if self._llm is None: self._llm = DeepSeekClient(**self._llm_kwargs)
        return self._llm
    @property
    def tts(self):
        if self._tts is None: self._tts = PiperTTS(**self._tts_kwargs)
        return self._tts
    @property
    def safety(self):
        if self._safety is None: self._safety = SafetyGate()
        return self._safety

    def _split_sentences(self, text: str) -> list[str]:
        """按中文标点分句"""
        parts = re.split(r"(?<=[。！？])", text)
        return [s.strip() for s in parts if s.strip() and not re.match(r'^[，、的]+$', s.strip())]

    def run(self, audio_path: str, output_dir: str = "output") -> dict:
        """
        流式级联：VAD分句 → LLM流式 → TTS逐句 → 首句优先播放

        Returns dict with latency breakdown and output paths.
        """
        sample_id = Path(audio_path).stem
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        t = LatencyTimer()

        # ---- Stage 1: ASR + VAD ----
        print(f"\n[Stream] [{sample_id}] ASR+VAD 转写...")
        t.mark("asr_start")
        asr_result = self.asr.transcribe(audio_path, hotword=self.hotword)
        t.mark("asr_end")
        asr_text = asr_result["text"]
        print(f"  识别: {asr_text[:60]}...")
        print(f"  ASR延迟: {asr_result['latency_ms']:.0f} ms")

        # ---- Stage 2: Safety ----
        safety_check = self.safety.check(asr_text)
        if safety_check["blocked"]:
            print(f"  [拦截] {safety_check['reason']}")
            llm_text = self.safety.get_safe_response()
            sentences = self._split_sentences(llm_text)
        else:
            # ---- Stage 3+4: LLM流式 + TTS逐句 ----
            print(f"[Stream] [{sample_id}] LLM流式生成 + TTS逐句合成...")
            llm_full = []
            sentences = []

            buf = ""
            t.mark("llm_start")
            for chunk in self.llm.generate_stream(asr_text):
                delta = chunk["delta"]
                if chunk["ttft_ms"] is not None:
                    t.mark("llm_ttft")

                llm_full.append(delta)
                buf += delta

                # 每攒够一句就送入 TTS
                while True:
                    m = re.match(r"^(.*?[。！？])", buf)
                    if not m:
                        break
                    sent = m.group(1)
                    sentences.append(sent)
                    buf = buf[len(sent):]

            # 剩余文本
            if buf.strip():
                sentences.append(buf.strip())

            llm_text = "".join(llm_full)

        t.mark("llm_end")

        # ---- TTS 逐句合成 ----
        print(f"  分句数: {len(sentences)}")
        first_tts_latency = None
        first_tts_path = None
        all_outputs = []

        for i, sent in enumerate(sentences):
            if not sent.strip():
                continue
            out = output / f"{sample_id}_sent{i}.wav"
            tts_result = self.tts.synthesize(sent, str(out))
            if i == 0:
                first_tts_latency = tts_result["latency_ms"]
                first_tts_path = str(out)
                t.mark("tts_first_packet")
            all_outputs.append(str(out))

        t.mark("pipeline_end")

        # ---- 延迟汇总 ----
        asr_lat = asr_result["latency_ms"]
        llm_ttft = t.elapsed_ms("llm_start", "llm_ttft") if "llm_ttft" in t._marks else None
        llm_total = t.elapsed_ms("llm_start", "llm_end")
        tts_first = first_tts_latency or 0
        total = t.elapsed_ms("asr_start", "pipeline_end")

        # 首段可播放 = VAD结束 + LLM首token + TTS首包
        first_playable = llm_ttft + tts_first if llm_ttft else total

        print(f"\n  ══════ 流式管线延迟 ══════")
        print(f"  ASR:              {asr_lat:>8.0f} ms")
        print(f"  LLM TTFT:         {llm_ttft:>8.0f} ms" if llm_ttft else f"  LLM总耗时:         {llm_total:>8.0f} ms")
        print(f"  TTS首包:          {tts_first:>8.0f} ms")
        print(f"  ─────────────────────────")
        print(f"  首段可播放 ★:     {first_playable:>8.0f} ms ← 用户感知延迟")
        print(f"  端到端总计:       {total:>8.0f} ms")

        return {
            "asr_text": asr_text,
            "llm_text": llm_text,
            "sentences": len(sentences),
            "asr_latency_ms": asr_lat,
            "llm_ttft_ms": llm_ttft,
            "llm_total_ms": llm_total,
            "tts_first_ms": tts_first,
            "first_playable_ms": first_playable,
            "total_ms": total,
            "outputs": all_outputs,
            "first_output": first_tts_path,
        }
