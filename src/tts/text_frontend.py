"""
文本前端规整模块 (Text Normalization, TN) — TTS 优化方向 A
================================================================

工业 TTS 前端必备环节：在送入声学模型合成前，把 LLM 原文里
"机器念不对/念得别扭"的内容规整为可正确发音的中文。

针对造船领域优化：术语英文缩写、计量单位、数字、特殊符号。

设计原则
--------
1. 零额外依赖：纯 Python 标准库 (re)，不引入任何第三方包。
2. 可独立运行：本文件可直接 `python text_frontend.py` 自测。
3. 可无缝嵌入：对外暴露单一入口 `normalize(text) -> str`，
   也可实例化 `TextFrontend` 复用编译后的正则，避免重复开销。
4. 词典外置可扩展：术语映射支持从 JSON 文件加载，便于运营增改。

处理流水线 (顺序敏感)
---------------------
  原始文本
    └─ 1. 全角→半角 + 空白规整
    └─ 2. 英文缩写/术语映射 (CCS→中国船级社规范 ...)
    └─ 3. 计量单位展开 (mm→毫米, ℃→摄氏度 ...)
    └─ 4. 特殊符号读法 (≤→小于等于, Φ→直径 ...)
    └─ 5. 数字读法 (含小数/百分比/序号/罗马数字)
    └─ 6. 残留无法朗读字符清理
  规整后文本 (可直接喂给 Piper)
"""

from __future__ import annotations

import json
import re
from pathlib import Path


# ============================================================
# 1. 造船领域术语 / 英文缩写映射表 (内置默认，可被外部 JSON 覆盖/扩展)
#    key 必须为大写，匹配时大小写不敏感。
# ============================================================
DEFAULT_TERM_MAP: dict[str, str] = {
    # —— 船级社 / 规范机构 ——
    "CCS": "中国船级社",
    "IMO": "国际海事组织",
    "SOLAS": "国际海上人命安全公约",
    "MARPOL": "国际防止船舶造成污染公约",
    "IACS": "国际船级社协会",
    "ABS": "美国船级社",
    "DNV": "挪威船级社",
    "LR": "英国劳氏船级社",
    "BV": "法国船级社",
    # —— 船型 / 缩写 ——
    "LNG": "液化天然气",
    "LPG": "液化石油气",
    "FPSO": "浮式生产储油卸油装置",
    "VLCC": "超大型油轮",
    "ULCC": "巨型油轮",
    "RORO": "滚装船",
    "TEU": "标准箱",
    # —— 工艺 / 检验 ——
    "NDT": "无损检测",
    "RT": "射线检测",
    "UT": "超声波检测",
    "MT": "磁粉检测",
    "PT": "渗透检测",
    "WPS": "焊接工艺规程",
    "PQR": "焊接工艺评定记录",
    "HAZ": "热影响区",
    # —— 文件 / 通用 ——
    "BL": "提单",
    "B/L": "提单",
    "GA": "总布置图",
    "MGO": "船用轻柴油",
    "HFO": "重燃油",
    "ETA": "预计到达时间",
}


# ============================================================
# 2. 计量单位映射 (跟在数字后面的单位才展开)
# ============================================================
UNIT_MAP: dict[str, str] = {
    "mm": "毫米",
    "cm": "厘米",
    "km": "千米",
    "m": "米",
    "kg": "千克",
    "g": "克",
    "t": "吨",
    "kN": "千牛",
    "MPa": "兆帕",
    "kPa": "千帕",
    "Pa": "帕",
    "kW": "千瓦",
    "MW": "兆瓦",
    "W": "瓦",
    "rpm": "转每分",
    "kn": "节",
    "Hz": "赫兹",
    "V": "伏",
    "A": "安培",
    "L": "升",
    "mL": "毫升",
}

# 单位按长度降序，保证 "mm" 先于 "m" 匹配
_UNIT_KEYS = sorted(UNIT_MAP.keys(), key=len, reverse=True)


# ============================================================
# 3. 特殊符号读法
# ============================================================
SYMBOL_MAP: dict[str, str] = {
    "≤": "小于等于",
    "≥": "大于等于",
    "≈": "约等于",
    "≠": "不等于",
    "±": "正负",
    "Φ": "直径",
    "φ": "直径",
    "°": "度",
    "℃": "摄氏度",
    "℉": "华氏度",
    "×": "乘以",
    "÷": "除以",
    "%": "百分之",   # 特殊：百分号需前置，见 _normalize_percent
    "&": "和",
    "@": "at",
}

# 罗马数字 (常见于规范分级、分隔等级)
ROMAN_MAP: dict[str, str] = {
    "Ⅰ": "一", "Ⅱ": "二", "Ⅲ": "三", "Ⅳ": "四", "Ⅴ": "五",
    "Ⅵ": "六", "Ⅶ": "七", "Ⅷ": "八", "Ⅸ": "九", "Ⅹ": "十",
}

CN_DIGITS = "零一二三四五六七八九"


def _int_to_cn(num_str: str) -> str:
    """整数串转中文读法 (按权位)。支持到亿级，超长退化为逐位读。"""
    n = int(num_str)
    if n == 0:
        return "零"
    units = ["", "十", "百", "千"]
    big_units = ["", "万", "亿"]
    if n >= 10 ** 12:  # 超大数逐位读，避免读法错乱
        return "".join(CN_DIGITS[int(c)] for c in num_str)

    s = str(n)
    groups = []
    while s:
        groups.append(s[-4:])
        s = s[:-4]

    result_parts = []
    for gi in range(len(groups) - 1, -1, -1):
        g = groups[gi].zfill(4) if gi != len(groups) - 1 else groups[gi]
        g_int = int(g)
        if g_int == 0:
            continue
        seg = ""
        gs = str(g_int).zfill(len(g)) if gi != len(groups) - 1 else str(g_int)
        length = len(gs)
        zero_flag = False
        for idx, ch in enumerate(gs):
            d = int(ch)
            pos = length - idx - 1
            if d == 0:
                zero_flag = True
            else:
                if zero_flag:
                    seg += "零"
                    zero_flag = False
                seg += CN_DIGITS[d] + units[pos]
        seg += big_units[gi]
        result_parts.append(seg)

    res = "".join(result_parts)
    # "一十" -> "十" (中文习惯)，仅句首
    res = re.sub(r"^一十", "十", res)
    return res


def _decimal_to_cn(num_str: str) -> str:
    """小数串转中文：整数部分按权位，小数部分逐位。"""
    if "." in num_str:
        int_part, dec_part = num_str.split(".", 1)
        int_cn = _int_to_cn(int_part) if int_part else "零"
        dec_cn = "".join(CN_DIGITS[int(c)] for c in dec_part)
        return f"{int_cn}点{dec_cn}"
    return _int_to_cn(num_str)


class TextFrontend:
    """文本前端规整器。编译期构建正则，调用 normalize() 复用。"""

    def __init__(self, term_map: dict[str, str] | None = None,
                 extra_lexicon_path: str | None = None,
                 enable_number: bool = True):
        # 合并术语表：默认 + 外部 JSON
        self.term_map: dict[str, str] = dict(DEFAULT_TERM_MAP)
        if term_map:
            self.term_map.update({k.upper(): v for k, v in term_map.items()})
        if extra_lexicon_path and Path(extra_lexicon_path).exists():
            try:
                with open(extra_lexicon_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.term_map.update({k.upper(): v for k, v in loaded.items()})
            except Exception:
                pass  # 词典加载失败不阻断主流程

        self.enable_number = enable_number

        # 术语按长度降序，保证 "B/L" 等长词优先；构建一个交替匹配正则
        keys = sorted(self.term_map.keys(), key=len, reverse=True)
        escaped = [re.escape(k) for k in keys]
        self._term_re = re.compile("|".join(escaped), re.IGNORECASE) if escaped else None

        # 数字+单位：捕获数字与紧随单位
        self._num_unit_re = re.compile(
            r"(\d+(?:\.\d+)?)\s*(" + "|".join(re.escape(u) for u in _UNIT_KEYS) + r")\b"
        )
        # 百分比： 20% / 20.5%
        self._percent_re = re.compile(r"(\d+(?:\.\d+)?)\s*%")
        # 序号： No.3 / NO.3 / #3
        self._serial_re = re.compile(r"(?:No\.?|NO\.?|#)\s*(\d+)", re.IGNORECASE)
        # 纯数字 (含小数)
        self._number_re = re.compile(r"\d+(?:\.\d+)?")

    # —— 各子步骤 ——
    @staticmethod
    def _to_halfwidth(text: str) -> str:
        out = []
        for ch in text:
            code = ord(ch)
            if code == 0x3000:
                code = 0x20
            elif 0xFF01 <= code <= 0xFF5E:
                code -= 0xFEE0
            out.append(chr(code))
        return "".join(out)

    def _normalize_terms(self, text: str) -> str:
        if not self._term_re:
            return text
        return self._term_re.sub(
            lambda m: self.term_map[m.group(0).upper()], text
        )

    def _normalize_units(self, text: str) -> str:
        def repl(m):
            num, unit = m.group(1), m.group(2)
            num_cn = _decimal_to_cn(num) if self.enable_number else num
            return f"{num_cn}{UNIT_MAP[unit]}"
        return self._num_unit_re.sub(repl, text)

    def _normalize_percent(self, text: str) -> str:
        def repl(m):
            num = m.group(1)
            num_cn = _decimal_to_cn(num) if self.enable_number else num
            return f"百分之{num_cn}"
        return self._percent_re.sub(repl, text)

    def _normalize_serial(self, text: str) -> str:
        def repl(m):
            num = m.group(1)
            num_cn = _int_to_cn(num) if self.enable_number else num
            return f"{num_cn}号"
        return self._serial_re.sub(repl, text)

    def _normalize_symbols(self, text: str) -> str:
        for sym, read in SYMBOL_MAP.items():
            if sym == "%":
                continue  # 已在 percent 阶段处理
            text = text.replace(sym, read)
        for r, cn in ROMAN_MAP.items():
            text = text.replace(r, cn)
        return text

    def _normalize_numbers(self, text: str) -> str:
        if not self.enable_number:
            return text
        return self._number_re.sub(lambda m: _decimal_to_cn(m.group(0)), text)

    @staticmethod
    def _cleanup(text: str) -> str:
        # 合并多空格；保留中文标点
        text = re.sub(r"[ \t]+", "", text)  # 中文 TTS 不需要词间空格
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    # —— 主入口 ——
    def normalize(self, text: str) -> str:
        """对外唯一入口：返回规整后的可朗读文本。"""
        if not text or not text.strip():
            return text
        text = self._to_halfwidth(text)
        text = self._normalize_terms(text)        # CCS -> 中国船级社
        text = self._normalize_serial(text)        # No.3 -> 三号
        text = self._normalize_percent(text)       # 20% -> 百分之二十
        text = self._normalize_units(text)         # 5.5mm -> 五点五毫米
        text = self._normalize_symbols(text)       # ≤ Φ ℃ 罗马数字
        text = self._normalize_numbers(text)       # 残留数字 -> 中文
        text = self._cleanup(text)
        return text


# 模块级单例 + 便捷函数，供 import 直接调用
_DEFAULT_FRONTEND: TextFrontend | None = None


def get_frontend(extra_lexicon_path: str | None = None) -> TextFrontend:
    global _DEFAULT_FRONTEND
    if _DEFAULT_FRONTEND is None:
        _DEFAULT_FRONTEND = TextFrontend(extra_lexicon_path=extra_lexicon_path)
    return _DEFAULT_FRONTEND


def normalize(text: str, extra_lexicon_path: str | None = None) -> str:
    """便捷函数：normalize("CCS 规范要求 5.5mm 钢板") -> '中国船级社规范要求五点五毫米钢板'"""
    return get_frontend(extra_lexicon_path).normalize(text)


# ============================================================
# 独立自测：python src/tts/text_frontend.py
# ============================================================
if __name__ == "__main__":
    cases = [
        "CCS 规范要求该处钢板厚度不小于 5.5mm。",
        "IMO 与 SOLAS 公约对 LNG 船的货舱有专门规定。",
        "焊接预热温度应保持在 ≤300℃，管子直径 Φ16。",
        "No.3 货舱的破损稳性裕度约为 20.5%，主机功率 3200kW。",
        "防火分隔等级分为 Ⅰ 级和 Ⅱ 级，转速 750rpm。",
        "船速 21kn，排水量 15000t，轴功率 ≥ 2MW。",
    ]
    fe = TextFrontend()
    print("=" * 70)
    print("文本前端规整 (TN) 自测")
    print("=" * 70)
    for c in cases:
        print(f"\n原文 : {c}")
        print(f"规整 : {fe.normalize(c)}")
    print("\n[OK] text_frontend 自测完成")
