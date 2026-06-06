"""
推理加速 / 量化模块 (Inference Acceleration & Quantization) — TTS 优化方向 B
============================================================================

目标：在不改变音质感知的前提下，降低 Piper(ONNX) 在 CPU 上的单句合成耗时，
直接改善"文本合成语音的时间"指标 (RTF, Real-Time Factor)。

两条独立可叠加的优化路径
------------------------
路径1  离线 INT8 动态量化 (quantize_dynamic)
   - 把 fp32 的 *.onnx 权重量化为 int8，模型体积↓约 60~70%，
     CPU 矩阵乘加速，medium 级模型音质近乎无损。
   - 产出一个新的 *.int8.onnx，PiperTTS 通过 model_path 直接加载，
     与现有加载逻辑零耦合。
   - 入口：quantize_model(src_onnx, dst_onnx)

路径2  ONNX Runtime 运行期调优 (线程 / 图优化)
   - 设置 intra/inter op 线程数、图优化级别 ORT_ENABLE_ALL。
   - 通过 build_session_options() 产出 SessionOptions；
     若安装的 piper 版本支持注入则注入，否则通过环境变量兜底
     (apply_runtime_env)，保证任何 piper 版本都能拿到线程收益。

设计原则
--------
- 仅依赖 onnxruntime / onnx (piper-tts 已传递依赖，无新增第三方包)。
- 可独立运行：python src/tts/tts_accel.py --help
- 可无缝嵌入：PiperTTS 仅需 (a) 传入量化后模型路径; (b) 启动前调用
  apply_runtime_env()。不侵入 piper 内部。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


# ============================================================
# 路径2-a：运行期环境变量调优 (最通用，任何 piper 版本生效)
# ============================================================
def apply_runtime_env(intra_threads: int | None = None,
                      inter_threads: int = 1,
                      max_cap: int = 8) -> dict:
    """
    在导入 onnxruntime / piper 之前调用，通过环境变量约束线程行为。

    关键经验：Piper 的 medium 模型很小，ONNX Runtime 默认会按"物理核数"
    起 intra 线程。在多核服务器(几十~几百核)上这会导致严重的线程同步开销，
    反而拖慢单句 TTS。因此对 intra 线程做上限封顶 (默认 8)。

    CPU 实时合成经验：intra 线程 = min(物理核, max_cap)；inter 线程 = 1
    (TTS 单流推理，过多 inter 线程只增加同步开销)。

    Returns: 实际设置的环境字典 (便于打印/记录)。
    """
    cores = os.cpu_count() or 4
    if intra_threads is None:
        intra_threads = min(cores, max_cap)   # 封顶，避免多核线程抖动
    intra_threads = max(1, intra_threads)

    env = {
        "OMP_NUM_THREADS": str(intra_threads),
        "OMP_WAIT_POLICY": "PASSIVE",          # 单流推理用 PASSIVE 降低空转占用
        "ORT_INTRA_OP_NUM_THREADS": str(intra_threads),
        "ORT_INTER_OP_NUM_THREADS": str(inter_threads),
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env


# ============================================================
# 路径2-b：构建 SessionOptions (供支持注入的调用方使用)
# ============================================================
def build_session_options(intra_threads: int | None = None,
                          inter_threads: int = 1):
    """
    返回调优后的 onnxruntime.SessionOptions。
    图优化级别拉满 + 线程绑定，CPU 上通常有 10~30% 提速。
    """
    import onnxruntime as ort

    if intra_threads is None:
        intra_threads = os.cpu_count() or 4

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = intra_threads
    so.inter_op_num_threads = inter_threads
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return so


def best_providers() -> list[str]:
    """返回可用的最优 EP 列表，CPU 优先保证可跑。"""
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        ordered = []
        for p in ("CUDAExecutionProvider", "CPUExecutionProvider"):
            if p in avail:
                ordered.append(p)
        return ordered or ["CPUExecutionProvider"]
    except Exception:
        return ["CPUExecutionProvider"]


# ============================================================
# 路径1：离线 INT8 动态量化
# ============================================================
def quantize_model(src_onnx: str, dst_onnx: str | None = None,
                   per_channel: bool = True) -> str:
    """
    对 Piper 的 fp32 ONNX 做 INT8 动态量化。

    Args:
        src_onnx:  原始 zh_CN-huayan-medium.onnx
        dst_onnx:  输出路径，默认 同目录 *.int8.onnx
        per_channel: 是否逐通道量化 (更高精度，体积略大)

    Returns: 量化后模型路径。

    注意：动态量化无需校准数据集，开箱即用；只量化权重，
    激活在推理时动态量化。模型体积稳定压缩约 3x。

    重要硬件依赖提示：
    INT8 是否"加速"取决于 CPU 是否具备 INT8 加速指令 (x86 AVX-512 VNNI、
    ARM dotprod/i8mm)。具备这些指令的边缘/移动端 (Piper 的目标场景) 通常
    可提速并显著省内存；但在缺乏 INT8 GEMM 加速的通用服务器 CPU 上，量化
    插入的 DynamicQuantize/Dequantize 算子开销可能反而使其变慢。因此本项目
    默认 use_quantized_model=false，请先用 `python src/tts/tts_accel.py bench`
    在目标硬件实测确认收益后再启用。
    本函数已自动复制 *.onnx.json 发音配置到新模型旁。
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    src = Path(src_onnx)
    if not src.exists():
        raise FileNotFoundError(f"源模型不存在: {src}")

    if dst_onnx is None:
        dst = src.with_suffix(".int8.onnx")
    else:
        dst = Path(dst_onnx)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"[量化] 输入: {src}  ({src.stat().st_size / 1e6:.1f} MB)")
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )
    print(f"[量化] 输出: {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")

    # 同步 Piper 配置 json (model.onnx.json)
    src_cfg = Path(str(src) + ".json")
    if src_cfg.exists():
        dst_cfg = Path(str(dst) + ".json")
        dst_cfg.write_bytes(src_cfg.read_bytes())
        print(f"[量化] 已复制发音配置: {dst_cfg}")
    else:
        print(f"[警告] 未找到发音配置 {src_cfg}，请手动放置 {dst}.json")

    ratio = src.stat().st_size / max(dst.stat().st_size, 1)
    print(f"[量化] 体积压缩比: {ratio:.2f}x")
    return str(dst)


# ============================================================
# 独立运行：量化 + 基准
# ============================================================
def _benchmark(model_path: str, text: str = "这是一句用于基准测试的造船领域语音合成文本。",
               runs: int = 5):
    """对给定模型跑 N 次合成，输出平均耗时 / RTF。需已安装 piper-tts 与模型。"""
    import time
    import tempfile
    import wave
    try:
        from piper import PiperVoice
    except Exception as e:
        print(f"[基准] 无法导入 piper，跳过: {e}")
        return None

    apply_runtime_env()
    voice = PiperVoice.load(model_path, use_cuda=False)

    # 预热一次 (排除首次加载/JIT)
    with wave.open(tempfile.mktemp(suffix=".wav"), "wb") as wf:
        voice.synthesize_wav(text, wf)

    import soundfile as sf
    times, durations = [], []
    for _ in range(runs):
        out = tempfile.mktemp(suffix=".wav")
        t0 = time.perf_counter()
        with wave.open(out, "wb") as wf:
            voice.synthesize_wav(text, wf)
        times.append(time.perf_counter() - t0)
        durations.append(sf.info(out).duration)

    avg_t = sum(times) / len(times)
    avg_d = sum(durations) / len(durations)
    rtf = avg_t / avg_d if avg_d else float("inf")
    print(f"\n  模型: {model_path}")
    print(f"  平均合成耗时: {avg_t * 1000:.1f} ms (n={runs})")
    print(f"  平均音频时长: {avg_d:.2f} s")
    print(f"  RTF: {rtf:.3f}  ({'实时' if rtf < 1 else '非实时'})")
    return {"avg_ms": avg_t * 1000, "rtf": rtf}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS 推理加速/量化工具 (优化方向 B)")
    sub = parser.add_subparsers(dest="cmd")

    pq = sub.add_parser("quantize", help="INT8 动态量化")
    pq.add_argument("src", help="原始 .onnx 路径")
    pq.add_argument("-o", "--out", default=None, help="输出 .onnx 路径")

    pb = sub.add_parser("bench", help="基准测试 (合成耗时/RTF)")
    pb.add_argument("model", help=".onnx 模型路径")
    pb.add_argument("-n", "--runs", type=int, default=5)

    pe = sub.add_parser("env", help="打印将设置的运行期线程环境")

    args = parser.parse_args()

    if args.cmd == "quantize":
        quantize_model(args.src, args.out)
    elif args.cmd == "bench":
        _benchmark(args.model, runs=args.runs)
    elif args.cmd == "env":
        print("运行期线程环境:")
        for k, v in apply_runtime_env().items():
            print(f"  {k}={v}")
        print(f"最优 Providers: {best_providers()}")
    else:
        parser.print_help()
