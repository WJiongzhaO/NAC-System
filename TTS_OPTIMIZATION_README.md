# TTS 优化模块交付包 — 方向 A（文本前端规整）+ 方向 B（推理加速/量化）

适配项目：NAC-System（造船领域 ASR→LLM→TTS 语音问答）。
改动边界：仅 TTS 合成部分，不触碰 ASR / LLM / 安全门控。

---

## 一、交付物清单

| 文件 | 作用 | 类型 |
|------|------|------|
| `src/tts/text_frontend.py` | 方向 A：文本前端规整（术语/数字/单位/符号→可朗读中文） | 新增 |
| `src/tts/tts_accel.py` | 方向 B：线程调优 + INT8 量化脚本 + 基准工具 | 新增 |
| `data/tn_lexicon.json` | 造船术语读音扩展词典（可运营增改） | 新增 |
| `tests/test_tts_modules.py` | 单模块独立测试（轻量档无需模型 / 完整档实测） | 新增 |
| `src/tts/piper_tts.py` | 集成两模块（向后兼容，旧调用零改动） | 修改 |
| `config.yaml` | 新增 TTS 配置项（TN / 量化 / 线程） | 修改 |

**依赖**：均为项目已有依赖（`onnxruntime`/`onnx` 由 `piper-tts` 传递引入），
方向 A 纯标准库（仅 `re`/`json`），**零新增第三方包**。

---

## 二、单模块独立运行测试

```bash
# 轻量档：只测 TN 逻辑 + 加速模块纯函数，无需 piper / 模型（秒级）
python tests/test_tts_modules.py

# 完整档：额外加载模型实测 fp32 vs int8 的耗时/RTF + 集成
python tests/test_tts_modules.py --full --model pretrained_models/piper/zh_CN-huayan-medium.onnx

# 各模块也可单独自测
python src/tts/text_frontend.py                 # TN 规整示例
python src/tts/tts_accel.py env                  # 打印线程环境
python src/tts/tts_accel.py quantize <src.onnx>  # 生成 *.int8.onnx
python src/tts/tts_accel.py bench <model.onnx>   # 基准测试 RTF
```

---

## 三、嵌入现有项目（已完成，开箱即用）

### 方向 A：文本前端规整
- `PiperTTS.__init__` 新增 `text_normalize=True` / `tn_lexicon_path`，
  在 `synthesize()` 合成前自动对文本做规整。
- 旧调用 `PiperTTS(model_path=...)` 无需改动即可享受 TN（默认开启）；
  若想关闭：`PiperTTS(..., text_normalize=False)`。
- 也可独立调用：`from src.tts.text_frontend import normalize`。

### 方向 B：推理加速
- `piper_tts.py` 顶部在导入 piper 前调用 `apply_runtime_env()`，
  对多核服务器把 ONNX Runtime intra 线程**封顶为 8**，避免线程抖动。
- INT8 量化通过 `model_path` 直接加载 `*.int8.onnx`，与现有加载逻辑零耦合。
  config 中 `use_quantized_model` 由调用方（run_web.py/run_eval.py）决定是否
  改用量化模型路径。

---

## 四、参数配置项（config.yaml → tts:）

```yaml
text_normalize: true                 # 方向A 总开关
tn_lexicon_path: "data/tn_lexicon.json"   # 术语扩展词典
use_quantized_model: false           # 方向B：是否用 int8 模型（默认关）
quantized_model_path: "pretrained_models/piper/zh_CN-huayan-medium.int8.onnx"
intra_op_threads: 0                  # 0=自动(min(核数,8))
inter_op_threads: 1                  # TTS 单流推理建议 1
```

---

## 五、实测结论（本沙箱 CPU，huayan-medium，n=5）

### 方向 A：文本前端规整 —— 效果显著、强展示性
| 原文 | 规整前（Piper 直念） | 规整后 |
|------|----------|--------|
| `CCS 规范` | 念错/跳读字母 | 中国船级社规范 |
| `5.5mm` | 念成 "五点五 mm" | 五点五毫米 |
| `No.3 货舱` | 念 "No 三" | 三号货舱 |
| `≤300℃` | 跳读符号 | 小于等于三百摄氏度 |
| `3200kW` | 念 "三千二百 kW" | 三千二百千瓦 |

单元测试 11/11 通过；附 `output_demo_TN_off.wav` vs `output_demo_TN_on.wav` 对照试听。

### 方向 B：推理加速 —— 重要硬件依赖结论
- **线程封顶**：在 384 核服务器上，默认按核数起线程会严重拖慢小模型；
  封顶到 8 是更稳的默认值。
- **INT8 量化**：模型体积稳定压缩 **3.38x**（63MB→18.7MB）；
  但**本机 CPU 缺 INT8 GEMM 加速指令（无 AVX-512 VNNI），量化后反而变慢约 3.5x**。
  → 故**默认关闭** `use_quantized_model`，并在代码/文档中明确标注：
  量化的提速收益**仅在支持 INT8 指令的边缘/ARM 设备（Piper 目标场景）成立**，
  需在目标硬件 `bench` 实测确认后再启用。体积压缩则在任何硬件都成立。

> 诚实提示：本交付包对 INT8 不做"提速"承诺，只承诺"体积压缩 + 提供实测工具"；
> 真正稳定的速度收益来自线程封顶。
