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

## Web 演示与安全模块

```bash
# 启动本地前端
python run_web.py
# 浏览器打开 http://127.0.0.1:7860
```

前端支持两种输入：

- 上传 WAV 音频，选择“串行基线”或“流式低延迟”运行完整 ASR → Safety → LLM → TTS 链路。
- 点击“开始录音/停止录音”现场录制语音问题，前端会转换为 16k 单声道 WAV 后提交给 ASR。
- 直接输入文本，用于快速验证安全门控和 TTS 固定安全提示。

安全模块位于 `src/safety/safety_gate.py`，在 ASR 和 LLM 之间执行：

- 有害内容拦截：命中爆炸物、恶意代码、越狱提示等关键词时短路，不调用 LLM。
- 领域越界拦截：问题不属于造船、船舶制造或船舶安全领域时短路。
- 固定提示播放：拦截后直接进入 TTS，生成安全提示语音。

ASR 热词位于 `data/hotwords.txt`。默认管线会自动加载该文件，用于提升“肋板、肘板、船艏、轴系校中”等造船术语识别质量。

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
