"""
安全门控模块 — 输入安全过滤

在 ASR 和 LLM 之间插入，对有害、无关或恶意干扰输入短路处理。
安全门控命中时不调用大模型，直接播放固定安全提示。
"""

import re


class SafetyGate:
    def __init__(
        self,
        blocked_keywords: list[str] | None = None,
        domain_keywords: list[str] | None = None,
        enforce_domain: bool = True,
        safe_response_text: str = "抱歉，您的问题涉及不安全内容，我无法回答。请询问造船安全相关的问题。",
        out_of_domain_response_text: str = "抱歉，我只能回答造船、船舶制造和船舶安全相关的问题。请换一个造船领域的问题。",
    ):
        self.blocked_keywords = blocked_keywords or [
            "炸弹", "爆炸物", "武器", "枪支", "攻击", "黑客", "恶意代码",
            "木马", "病毒", "绕过", "越权", "泄露密钥", "口令破解",
            "忽略之前", "无视规则", "解除限制", "扮演", "越狱",
        ]
        self.domain_keywords = domain_keywords or [
            "船", "船舶", "船体", "船艏", "船尾", "舱", "甲板", "龙骨",
            "肋板", "肘板", "焊接", "涂装", "分段", "装配", "下水",
            "稳性", "破损", "柴油机", "轴系", "舵", "空泡", "坞修",
            "防火", "绝缘", "高强度钢", "造船", "安全规范",
        ]
        self.enforce_domain = enforce_domain
        self.safe_response_text = safe_response_text
        self.out_of_domain_response_text = out_of_domain_response_text
        self._patterns = [
            re.compile(re.escape(kw), re.IGNORECASE)
            for kw in self.blocked_keywords
        ]
        self._domain_patterns = [
            re.compile(re.escape(kw), re.IGNORECASE)
            for kw in self.domain_keywords
        ]

    @staticmethod
    def _normalize(text: str) -> str:
        """兼容 ASR 输出的字间空格与中英文大小写差异。"""
        return re.sub(r"\s+", "", text).lower()

    def check(self, text: str) -> dict:
        """
        检查文本是否包含敏感内容

        Returns:
            {
                "blocked": bool,
                "reason": str | None,     # 拦截原因
                "matched_word": str | None,
            }
        """
        normalized = self._normalize(text)
        if not normalized:
            return {"blocked": False, "reason": None, "matched_word": None}

        for i, pattern in enumerate(self._patterns):
            match = pattern.search(normalized)
            if match:
                return {
                    "blocked": True,
                    "category": "unsafe",
                    "reason": f"匹配到敏感关键词: {self.blocked_keywords[i]}",
                    "matched_word": self.blocked_keywords[i],
                    "response_text": self.safe_response_text,
                }

        if self.enforce_domain and not any(p.search(normalized) for p in self._domain_patterns):
            return {
                "blocked": True,
                "category": "out_of_domain",
                "reason": "问题不属于造船/船舶安全领域",
                "matched_word": None,
                "response_text": self.out_of_domain_response_text,
            }

        return {
            "blocked": False,
            "category": "allowed",
            "reason": None,
            "matched_word": None,
            "response_text": None,
        }

    def get_safe_response(self, category: str = "unsafe") -> str:
        """返回安全提示文本（用于 TTS 合成）"""
        if category == "out_of_domain":
            return self.out_of_domain_response_text
        return self.safe_response_text
