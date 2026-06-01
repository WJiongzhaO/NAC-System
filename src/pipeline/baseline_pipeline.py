"""
基线串行管线 — ASR → Safety → LLM → TTS

完整的端到端延迟测量管线。
"""

import time
from pathlib import Path
from typing import Optional

from src.asr.paraformer_asr import ParaformerASR
from src.asr.hotwords import load_hotwords
from src.llm.deepseek_client import DeepSeekClient
from src.tts.piper_tts import PiperTTS
from src.safety.safety_gate import SafetyGate
from src.eval.latency_timer import LatencyTimer, LatencyRecord


class BaselinePipeline:
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
        print("=" * 60)
        print("[Pipeline] 初始化基线串行管线")
        print("=" * 60)

        self.hotword = hotword or load_hotwords()

        # 按需初始化各模块（延迟加载以节省显存）
        self._asr: ParaformerASR | None = None
        self._llm: DeepSeekClient | None = None
        self._tts: PiperTTS | None = None
        self._safety: SafetyGate | None = None

        self._asr_kwargs = {"model_id": asr_model_id, "device": asr_device}
        self._llm_kwargs = {
            "api_base": llm_api_base,
            "api_key": llm_api_key,
            "model": llm_model,
        }
        self._tts_kwargs = {"model_name": tts_model_name, "model_path": tts_model_path}

    @property
    def asr(self) -> ParaformerASR:
        if self._asr is None:
            self._asr = ParaformerASR(**self._asr_kwargs)
        return self._asr

    @property
    def llm(self) -> DeepSeekClient:
        if self._llm is None:
            self._llm = DeepSeekClient(**self._llm_kwargs)
        return self._llm

    @property
    def tts(self) -> PiperTTS:
        if self._tts is None:
            self._tts = PiperTTS(**self._tts_kwargs)
        return self._tts

    @property
    def safety(self) -> SafetyGate:
        if self._safety is None:
            self._safety = SafetyGate()
        return self._safety

    def run(self, audio_path: str, output_dir: str = "output") -> LatencyRecord:
        """
        运行单条音频的完整基线链路

        Args:
            audio_path: 输入 WAV 文件路径
            output_dir: 输出目录

        Returns:
            LatencyRecord 包含各环节延迟数据
        """
        sample_id = Path(audio_path).stem
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        timer = LatencyTimer()
        record: LatencyRecord = {"sample_id": sample_id}

        # ──── Stage 1: ASR ────
        print(f"\n[Pipeline] [{sample_id}] Stage 1/4: ASR 转写...")
        timer.mark("asr_start")
        asr_result = self.asr.transcribe(audio_path, hotword=self.hotword)
        timer.mark("asr_end")
        asr_text = asr_result["text"]
        record["asr_latency_ms"] = asr_result["latency_ms"]
        record["asr_text"] = asr_text
        print(f"  ASR 文本: {asr_text[:80]}...")
        print(f"  ASR 延迟: {asr_result['latency_ms']:.1f} ms")

        if not asr_text.strip():
            print("[Pipeline] ASR 未识别到文本，跳过后续步骤")
            return record

        # ──── Stage 2: Safety Gate ────
        print(f"[Pipeline] [{sample_id}] Stage 2/4: 安全门控检查...")
        timer.mark("safety_start")
        safety_result = self.safety.check(asr_text)
        timer.mark("safety_end")

        if safety_result["blocked"]:
            print(f"  [拦截] {safety_result['reason']}")
            llm_text = safety_result.get("response_text") or self.safety.get_safe_response(
                safety_result.get("category", "unsafe")
            )
            record["llm_text"] = llm_text
            record["safety_blocked"] = True
            record["safety_category"] = safety_result.get("category")
            record["safety_reason"] = safety_result.get("reason")
            record["llm_total_ms"] = 0
        else:
            print("  [通过] 未检测到敏感内容")
            record["safety_blocked"] = False

            # ──── Stage 3: LLM ────
            print(f"[Pipeline] [{sample_id}] Stage 3/4: LLM 生成回答...")
            timer.mark("llm_start")
            llm_result = self.llm.generate(asr_text)
            timer.mark("llm_end")
            llm_text = llm_result["text"]
            record["llm_total_ms"] = llm_result["total_ms"]
            record["llm_text"] = llm_text
            print(f"  LLM 回答: {llm_text[:80]}...")
            print(f"  LLM 延迟: {llm_result['total_ms']:.1f} ms")

        # ──── Stage 4: TTS ────
        print(f"[Pipeline] [{sample_id}] Stage 4/4: TTS 合成语音...")
        tts_output = output / f"{sample_id}_output.wav"
        timer.mark("tts_start")
        tts_result = self.tts.synthesize(llm_text, str(tts_output))
        timer.mark("tts_end")
        record["tts_latency_ms"] = tts_result["latency_ms"]
        record["audio_duration_s"] = tts_result["duration_s"]
        print(f"  TTS 输出: {tts_output}")
        print(f"  TTS 延迟: {tts_result['latency_ms']:.1f} ms")
        print(f"  音频时长: {tts_result['duration_s']:.1f} s")

        # ──── 汇总 ────
        timer.mark("pipeline_end")
        total = timer.elapsed_ms("asr_start", "pipeline_end")
        record["total_latency_ms"] = total
        print(f"\n[Pipeline] [{sample_id}] 端到端总延迟: {total:.0f} ms")

        return record
