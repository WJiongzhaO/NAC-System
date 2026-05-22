"""
基线管线 — 单条音频端到端语音回答
用法: python run_baseline.py [audio_path]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.baseline_pipeline import BaselinePipeline

audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_audio/test.wav"

pipeline = BaselinePipeline(
    tts_model_path="pretrained_models/piper/zh_CN-huayan-medium.onnx",
)

record = pipeline.run(audio_path, output_dir="output")
print(f"\n>>> 语音回答已保存: output/{Path(audio_path).stem}_output.wav")
