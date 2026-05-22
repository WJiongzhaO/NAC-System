# NAC-System — 级联式造船安全低延迟语音问答系统

A.2 选题：ASR → LLM → TTS 级联语音问答基线复现与流式改进。

## 技术栈

| 模块 | 选型 | 说明 |
|------|------|------|
| ASR | paraformer-zh (FunASR) | 中文离线流式语音识别 |
| LLM | DeepSeek-V4-Flash (API) | 云端大模型文本生成 |
| TTS | Piper TTS (ONNX) | 轻量本地语音合成，zh_CN-huayan-medium |

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv && source .venv/Scripts/activate  # Windows

# 2. 安装依赖
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 4. 下载 Piper 语音包
# 从 https://hf-mirror.com/rhasspy/piper-voices 下载 zh_CN-huayan-medium 的
# .onnx 和 .onnx.json 文件，放入 pretrained_models/piper/

# 5. 运行基线
python run_baseline.py data/test_audio/test.wav
```

## 项目结构

```
src/
├── asr/paraformer_asr.py     # ASR 封装
├── llm/deepseek_client.py    # LLM API 客户端
├── tts/piper_tts.py          # TTS 封装
├── safety/safety_gate.py     # 安全门控
├── pipeline/
│   ├── baseline_pipeline.py  # 串行基线
│   └── streaming_pipeline.py # 流式改进
└── eval/
    ├── latency_timer.py      # 延迟测量
    └── run_eval.py           # 批量评测

data/
├── test_audio/               # 评测音频
├── hotwords.txt              # 造船领域热词
└── ssml_lexicon.json         # TTS 发音词典
```

## 优化阶段

| 阶段 | 内容 | 首段延迟 |
|------|------|----------|
| 基线 | 串行 ASR→LLM→TTS | ~9s |
| 流式 | VAD分句+流式LLM+TTS | ~1.0s |
| 质量 | hotword热词+SSML发音 | ~1.0s |
| 安全 | 门控钩子+安全提示 | ~1.0s |
