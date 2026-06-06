# TTS 引擎升级交付包 — 方向 C：Edge-TTS 主引擎 + Piper 自动兜底

适配项目：NAC-System（造船领域 ASR→安全门控→LLM→TTS 语音问答）
改动边界：仅 TTS 合成部分，不触碰 ASR / LLM / 安全门控。

---

## 一、为什么要换引擎（问题根因）

此前反馈"声调全是一声、很多字听不出来"，**不是参数问题，是模型架构天花板**：

- Piper 中文走 **espeak-ng 前端做 G2P（字→音）**，这一步会**丢失声调信息**。
- 声调是汉语区分字义的核心，espeak 路线注定"字字一声、多音字读错"。
- 这是架构级限制，**调 length_scale / noise_scale 等任何参数都救不了**。

> 结论：要"提高发音下限"，必须换成**原生对中文声调建模**的引擎。

**方向 C：引入 Edge-TTS（微软神经 TTS）作为主引擎**
- 中文原生建模，声调 / 多音字 / 韵律均为工业水准；
- 零模型下载、零 GPU、几十种中文音色；
- 合成更快（实测约 1~2s/句，Piper 约 6~14s/句）；
- **保留 Piper 作为离线兜底**：Edge 初始化/合成失败（断网/超时/网关异常）时
  **透明降级到 Piper，链路永不中断**。

---

## 二、交付物清单

| 文件 | 作用 | 类型 |
|------|------|------|
| `src/tts/edge_tts_engine.py` | Edge-TTS 适配类，**接口与 PiperTTS 完全一致** | 新增 |
| `src/tts/tts_engine.py` | 引擎工厂 `create_tts()` + `EngineWithFallback`（主+兜底） | 新增 |
| `config.yaml` | TTS 段新增 `engine` / `enable_fallback` / `voice` 等配置项 | 修改 |
| `output_compare/A_edge_yunjian.wav` | Edge 云健（沉稳男声，默认）试听 | 对照音频 |
| `output_compare/A_edge_yunyang.wav` | Edge 云扬（专业播报男声）试听 | 对照音频 |
| `output_compare/B_piper_huayan.wav` | 旧 Piper 同文本试听（声调对照） | 对照音频 |

> 方向 A（文本前端规整）/ 方向 B（线程调优）成果**全部复用**：Edge 合成前
> 同样调用 `TextFrontend.normalize()`，术语/数字/单位规整一致生效。

**新增依赖**（均为 pip 包，无 GPU）：
```
edge-tts        # 微软神经 TTS 客户端
soundfile       # MP3→WAV 解码 + 重采样（libsndfile 自带 MP3，无需 ffmpeg）
```
`numpy` 项目已有。Piper 相关依赖保持不变（兜底仍需要）。

---

## 三、单模块独立运行测试

```bash
# 1) Edge-TTS 适配类单测（整段合成，输出 WAV）
python src/tts/edge_tts_engine.py "根据 CCS 规范，No.3 货舱钢板厚度不应小于 5.5mm。"

# 2) 引擎工厂冒烟（edge 主 + piper 兜底）
python src/tts/tts_engine.py
#   输出: active: edge | engine_used: edge | ms: ~1000

# 3) 验证自动降级（强制 edge 失败 → 自动切 Piper）
python -c "
from src.tts.tts_engine import create_tts
tts = create_tts(engine='edge', proxy='http://127.0.0.1:9',
                 piper_model_path='pretrained_models/piper/zh_CN-huayan-medium.onnx')
r = tts.synthesize('测试降级', 'fallback.wav')
print('active:', tts.active_engine, '| engine_used:', r['engine'])
#   输出: active: piper | engine_used: piper
"
```

> 沙箱/企业 TLS 拦截网关下联网测试，临时加 `EDGE_TTS_INSECURE=1` 环境变量
> 放宽校验；**生产正常联网环境请勿设置该变量**（默认正常校验证书）。

---

## 四、嵌入现有项目（一行替换，接口零改动）

工厂返回对象与 `PiperTTS` **同接口**（`synthesize` /
`synthesize_streaming_generator` / `.voice`），现有 pipeline 逻辑无需任何改动，
只需把"直接 new PiperTTS"换成"调 create_tts()"。共 3 处调用点：

### 1) `run_web.py:54`
```python
# 旧：
self._tts = PiperTTS(model_path=TTS_MODEL)
# 新：
from src.tts.tts_engine import create_tts
self._tts = create_tts(engine="edge", piper_model_path=TTS_MODEL)
```

### 2) `src/pipeline/baseline_pipeline.py:66`
### 3) `src/pipeline/streaming_pipeline.py:79`
```python
# 旧：
self._tts = PiperTTS(**self._tts_kwargs)
# 新：
from src.tts.tts_engine import create_tts
self._tts = create_tts(engine="edge", piper_model_path=self._tts_kwargs.get("model_path"))
```

> 建议把 `engine` / `voice` / `rate` 等从 `config.yaml` 的 `tts:` 段读出后透传给
> `create_tts()`，实现配置驱动；不接配置也能用（参数有合理默认）。

---

## 五、参数配置项（config.yaml → tts:）

```yaml
tts:
  engine: "edge"                     # edge=Edge主+Piper兜底; piper=纯离线
  enable_fallback: true              # engine=edge 时是否启用 Piper 兜底
  # —— Edge-TTS 参数 ——
  voice: "zh-CN-YunjianNeural"       # 云健(沉稳默认); 云扬=zh-CN-YunyangNeural
  rate: "-5%"                        # 语速，负值更沉稳
  volume: "+0%"
  pitch: "+0Hz"
  proxy: ""                          # 可选 HTTPS 代理；留空走系统 HTTPS_PROXY
  # —— Piper 参数（engine=piper 或 edge 兜底时生效）——
  model_name: "zh_CN-huayan-medium"
  device: "cpu"
```

沉稳专业推荐音色：
| voice | 说明 |
|-------|------|
| `zh-CN-YunjianNeural` | 云健 — 沉稳，新闻/体育播报男声（**默认**） |
| `zh-CN-YunyangNeural` | 云扬 — 专业新闻播报男声 |
| `zh-CN-YunxiNeural` | 云希 — 活力男声 |
| `zh-CN-XiaoxiaoNeural` | 晓晓 — 自然女声 |

---

## 六、实测结论（本沙箱，同一段造船文本）

| 引擎/音色 | 合成耗时 | 音频时长 | 声调正确性 |
|-----------|----------|----------|------------|
| Edge 云健 | ~1~2 s | ~12 s | ✅ 原生声调，多音字/韵律自然 |
| Edge 云扬 | ~1.1 s | ~11 s | ✅ 同上 |
| Piper 华研 | ~14 s | ~12 s | ❌ 声调坍缩为一声（espeak 架构限制） |

对照音频见 `output_compare/`。Edge 在**音质（声调）与速度上均显著优于 Piper**，
且故障时自动回落 Piper 保证可用性。

> 降级是**透明且带记忆**的：一旦 Edge 失败，本次会话后续合成直接走 Piper，
> 不再反复重试拖慢响应；返回结果中 `engine` 字段标明实际使用的引擎，
> 降级时附 `fallback_reason` 便于排查。
