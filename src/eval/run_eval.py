"""
批量评测脚本 — 基线 & 流式管线

用法:
    python run_eval.py                          # 基线模式
    python run_eval.py --mode streaming          # 流式模式
    python run_eval.py --test-dir data/test_audio --output-dir output
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.baseline_pipeline import BaselinePipeline
from src.pipeline.streaming_pipeline import StreamingPipeline
from src.eval.latency_timer import EvalRecorder

TTS_MODEL = "pretrained_models/piper/zh_CN-huayan-medium.onnx"


def main():
    parser = argparse.ArgumentParser(description="NAC-System 批量评测")
    parser.add_argument("--test-dir", default="data/test_audio")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--hotword", default="")
    parser.add_argument("--mode", choices=["baseline", "streaming"], default="baseline")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"[错误] 评测目录不存在: {test_dir}")
        sys.exit(1)

    audio_files = sorted(test_dir.glob("*.wav"))
    if not audio_files:
        print(f"[错误] 目录 {test_dir} 中没有 .wav 文件")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f" 评测模式: {args.mode}")
    print(f" 测试音频: {len(audio_files)} 条")
    print(f" 文件列表: {[f.name for f in audio_files]}")
    print(f"{'=' * 60}")

    recorder = EvalRecorder(output_dir=args.output_dir)

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n─ [{i}/{len(audio_files)}] {audio_file.name}")

        try:
            if args.mode == "streaming":
                pipeline = StreamingPipeline(
                    tts_model_path=TTS_MODEL, hotword=args.hotword
                )
                result = pipeline.run(str(audio_file), output_dir=args.output_dir)
                record = {
                    "sample_id": audio_file.stem,
                    "asr_latency_ms": result["asr_latency_ms"],
                    "llm_total_ms": result["llm_total_ms"],
                    "llm_ttft_ms": result["llm_ttft_ms"],
                    "tts_latency_ms": result["tts_first_ms"],
                    "total_latency_ms": result["total_ms"],
                    "asr_text": result["asr_text"],
                    "llm_text": result["llm_text"],
                }
            else:
                pipeline = BaselinePipeline(
                    tts_model_path=TTS_MODEL, hotword=args.hotword
                )
                record = pipeline.run(str(audio_file), output_dir=args.output_dir)

            recorder.add(record)
        except Exception as e:
            print(f"  [错误] {e}")
            import traceback
            traceback.print_exc()

    recorder.print_summary()
    recorder.save_report(f"{args.output_dir}/latency_report.json")


if __name__ == "__main__":
    main()
