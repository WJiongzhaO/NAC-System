"""
延迟测量工具 — 精确计时埋点

按作业要求定义测量点：
- T_vad_end: VAD 尾点检测完成 → 下游可开始处理
- T_asr: 音频输入完成 → ASR 转写文本输出
- T_llm_ttft: LLM prompt 发送 → 首个 token 返回
- T_tts_first_packet: 文本输入 → 第一段 PCM 输出
- T_total: 用户停止说话 → 首段音频开始播放
"""

import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict


class LatencyRecord(TypedDict, total=False):
    sample_id: str
    asr_latency_ms: float
    llm_total_ms: float
    llm_ttft_ms: float | None
    tts_latency_ms: float
    first_playable_ms: float
    total_latency_ms: float
    audio_duration_s: float
    asr_text: str
    llm_text: str


class LatencyTimer:
    def __init__(self):
        self._marks: dict[str, float] = {}

    def mark(self, name: str):
        self._marks[name] = time.perf_counter()

    def elapsed_ms(self, start_mark: str, end_mark: str) -> float:
        return (self._marks[end_mark] - self._marks[start_mark]) * 1000

    def now(self) -> float:
        return time.perf_counter()


class EvalRecorder:
    """评测记录器，汇总多条音频的延迟数据"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[LatencyRecord] = []

    def add(self, record: LatencyRecord):
        self.records.append(record)

    def summary(self) -> dict:
        """输出延迟汇总统计"""
        if not self.records:
            return {}

        metrics = {}
        for key in ["asr_latency_ms", "llm_total_ms", "tts_latency_ms", "first_playable_ms", "total_latency_ms"]:
            values = [r[key] for r in self.records if key in r and r[key] is not None]
            if values:
                metrics[key] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        return metrics

    def save_report(self, filepath: str = "output/latency_report.json"):
        """保存延迟报告为 JSON"""
        report = {
            "summary": self.summary(),
            "details": self.records,
        }
        path = Path(filepath)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[评测] 报告已保存至 {path}")
        return str(path)

    def print_summary(self):
        """打印延迟汇总表"""
        s = self.summary()
        print("\n" + "=" * 70)
        print("评测结果汇总")
        print("=" * 70)
        print(f"{'指标':<20} {'均值(ms)':<12} {'中位数(ms)':<12} {'标准差(ms)':<12} {'样本数':<8}")
        print("-" * 60)
        labels = {
            "asr_latency_ms": "ASR 延迟",
            "llm_total_ms": "LLM 总耗时",
            "tts_latency_ms": "TTS 延迟",
            "first_playable_ms": "首段可播放",
            "total_latency_ms": "端到端总延迟",
        }
        for key, label in labels.items():
            if key in s:
                m = s[key]
                print(f"{label:<20} {m['mean']:<12.1f} {m['median']:<12.1f} {m['stdev']:<12.1f} {m['count']:<8}")
