"""
单模块独立运行测试 — TTS 优化方向 A (文本前端规整) + B (推理加速/量化)
==========================================================================

特点：不依赖 ASR / LLM / 安全门控，不需要跑整条链路。
分两档运行：
  · 轻量档 (默认)：只测 TN 规整逻辑 + 量化模块的纯函数，无需 piper / 模型。
  · 完整档 (--full)：额外加载 Piper 模型，实测 fp32 vs int8 的合成耗时/RTF，
    需已安装 piper-tts 且模型存在。

用法：
  python tests/test_tts_modules.py            # 轻量档
  python tests/test_tts_modules.py --full --model pretrained_models/piper/zh_CN-huayan-medium.onnx
"""

import argparse
import sys
from pathlib import Path

# 兼容两种运行方式：项目根目录 / 直接进 tests 目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "tts"))

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ============================================================
# 测试 A：文本前端规整
# ============================================================
def test_text_frontend():
    print("\n[A] 文本前端规整 (TN) 单元测试")
    from src.tts.text_frontend import TextFrontend, normalize

    fe = TextFrontend()
    cases = [
        ("CCS 规范", "中国船级社规范", "英文缩写映射"),
        ("钢板厚度 5.5mm。", "钢板厚度五点五毫米。", "小数+单位"),
        ("裕度约为 20%。", "裕度约为百分之二十。", "百分比"),
        ("No.3 货舱", "三号货舱", "序号"),
        ("温度 ≤300℃", "温度小于等于三百摄氏度", "符号+单位"),
        ("分为 Ⅰ 级", "分为一级", "罗马数字"),
        ("主机功率 3200kW", "主机功率三千二百千瓦", "千瓦"),
        ("船速 21kn", "船速二十一节", "节"),
    ]
    for raw, expect, desc in cases:
        got = fe.normalize(raw)
        check(f"{desc}: '{raw}' -> '{got}'", got == expect,
              f"期望 '{expect}'")

    # 边界
    check("空字符串不报错", normalize("") == "")
    check("无可规整内容原样返回",
          normalize("船体结构很重要。") == "船体结构很重要。")

    # 外部词典扩展
    lex = ROOT / "data" / "tn_lexicon.json"
    if lex.exists():
        fe2 = TextFrontend(extra_lexicon_path=str(lex))
        got = fe2.normalize("EEDI 是重要指标。")
        check(f"外部词典 EEDI -> '{got}'", "船舶能效设计指数" in got)


# ============================================================
# 测试 B：推理加速/量化 纯函数
# ============================================================
def test_accel_pure():
    print("\n[B] 推理加速/量化 模块纯函数测试")
    from src.tts.tts_accel import apply_runtime_env, best_providers

    env = apply_runtime_env()
    check("apply_runtime_env 返回线程配置",
          "OMP_NUM_THREADS" in env and "ORT_INTRA_OP_NUM_THREADS" in env,
          str(env))
    providers = best_providers()
    check("best_providers 至少含 CPU EP",
          "CPUExecutionProvider" in providers, str(providers))

    # build_session_options 需要 onnxruntime
    try:
        from src.tts.tts_accel import build_session_options
        so = build_session_options(intra_threads=4, inter_threads=1)
        check("build_session_options 生成 SessionOptions",
              so.intra_op_num_threads == 4)
    except Exception as e:
        print(f"  [SKIP] build_session_options (onnxruntime 未安装): {e}")


# ============================================================
# 测试 B(完整)：fp32 vs int8 实测
# ============================================================
def test_accel_full(model_path):
    print("\n[B-full] fp32 vs INT8 量化 实测 (需 piper + 模型)")
    from src.tts.tts_accel import quantize_model, _benchmark

    src = Path(model_path)
    if not src.exists():
        print(f"  [SKIP] 模型不存在: {src}")
        return

    print("\n  >>> fp32 基准:")
    fp32 = _benchmark(str(src), runs=5)

    int8_path = src.with_suffix(".int8.onnx")
    print("\n  >>> 量化为 INT8:")
    try:
        quantize_model(str(src), str(int8_path))
    except Exception as e:
        print(f"  [SKIP] 量化失败: {e}")
        return

    print("\n  >>> int8 基准:")
    int8 = _benchmark(str(int8_path), runs=5)

    if fp32 and int8:
        speedup = fp32["avg_ms"] / int8["avg_ms"] if int8["avg_ms"] else 0
        print(f"\n  ── 对比 (本机硬件实测) ──")
        print(f"  fp32: {fp32['avg_ms']:.1f} ms (RTF {fp32['rtf']:.3f})")
        print(f"  int8: {int8['avg_ms']:.1f} ms (RTF {int8['rtf']:.3f})")
        print(f"  耗时比 (fp32/int8): {speedup:.2f}x")
        if speedup >= 1.05:
            print(f"  => 本机 INT8 提速，可考虑启用 use_quantized_model")
        else:
            print(f"  => 本机 INT8 未提速 (CPU 缺 INT8 GEMM 加速)，建议保持 fp32")
        print(f"  注: 体积稳定压缩约 3x；提速与否取决于 CPU 是否支持 "
              f"AVX-512 VNNI / ARM i8mm")


# ============================================================
# 测试集成：PiperTTS 端到端 (TN 生效 + 量化模型可加载)
# ============================================================
def test_integration(model_path):
    print("\n[集成] PiperTTS 接入 TN + 量化模型")
    from src.tts.piper_tts import PiperTTS

    if not Path(model_path).exists():
        print(f"  [SKIP] 模型不存在: {model_path}")
        return

    tts = PiperTTS(model_path=model_path, device="cpu", text_normalize=True)
    out = tts.synthesize("CCS 规范要求钢板厚度不小于 5.5mm。")
    check("合成产出音频文件", Path(out["audio_path"]).exists(),
          str(out))
    check("合成耗时 > 0", out["latency_ms"] > 0)
    print(f"  合成: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="跑需要 piper+模型 的完整测试")
    parser.add_argument("--model",
                        default="pretrained_models/piper/zh_CN-huayan-medium.onnx")
    args = parser.parse_args()

    print("=" * 70)
    print("TTS 优化模块 单模块独立测试")
    print("=" * 70)

    test_text_frontend()
    test_accel_pure()

    if args.full:
        test_accel_full(args.model)
        test_integration(args.model)

    print("\n" + "=" * 70)
    print(f"结果: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    sys.exit(1 if FAIL else 0)
