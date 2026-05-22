"""
评测脚本 — 批量运行基线评测集

用法:
    python -m src.eval.run_eval --test-dir data/test_audio --output-dir output
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.baseline_pipeline import BaselinePipeline
from src.eval.latency_timer import EvalRecorder


def main():
    parser = argparse.ArgumentParser(description="NAC-System 基线评测")
    parser.add_argument("--test-dir", default="data/test_audio", help="评测音频目录")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--hotword", default="", help="热词列表，逗号分隔")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"[错误] 评测目录不存在: {test_dir}")
        print("请先创建 data/test_audio/ 目录并放入测试音频 (.wav)")
        sys.exit(1)

    audio_files = sorted(test_dir.glob("*.wav"))
    if not audio_files:
        print(f"[错误] 目录 {test_dir} 中没有 .wav 文件")
        sys.exit(1)

    print(f"\n找到 {len(audio_files)} 条测试音频")
    print(f"文件列表: {[f.name for f in audio_files]}\n")

    pipeline = BaselinePipeline(hotword=args.hotword)
    recorder = EvalRecorder(output_dir=args.output_dir)

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'=' * 60}")
        print(f" [{i}/{len(audio_files)}] 处理: {audio_file.name}")
        print(f"{'=' * 60}")

        try:
            record = pipeline.run(str(audio_file), output_dir=args.output_dir)
            recorder.add(record)
        except Exception as e:
            print(f"[错误] {audio_file.name} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    recorder.print_summary()
    recorder.save_report(f"{args.output_dir}/latency_report.json")


if __name__ == "__main__":
    main()
