"""
流式改进管线 — 首句优先播放
用法: python run_streaming.py [audio_path]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.streaming_pipeline import StreamingPipeline

audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_audio/test.wav"

pipeline = StreamingPipeline(
    tts_model_path="pretrained_models/piper/zh_CN-huayan-medium.onnx",
)

result = pipeline.run(audio_path, output_dir="output")
print(f"\n>>> 首句语音: {result['first_output']}")
print(f">>> 首段可播放延迟: {result['first_playable_ms']:.0f} ms")
