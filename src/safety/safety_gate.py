"""
安全门控模块 — 输入安全过滤

在 ASR 和 LLM 之间插入，对敏感/恶意输入短路处理。
使用关键词过滤 + 简单规则匹配。
"""

import re
from typing import Optional


class SafetyGate:
    def __init__(
        self,
        blocked_keywords: list[str] | None = None,
        safe_response_text: str = "抱歉，您的问题涉及不安全内容，我无法回答。请询问造船安全相关的问题。",
    ):
        self.blocked_keywords = blocked_keywords or [
            "炸弹", "武器", "攻击", "黑客", "恶意代码",
        ]
        self.safe_response_text = safe_response_text
        # 预编译正则，大小写不敏感
        self._patterns = [
            re.compile(re.escape(kw), re.IGNORECASE)
            for kw in self.blocked_keywords
        ]

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
        if not text.strip():
            return {"blocked": False, "reason": None, "matched_word": None}

        for i, pattern in enumerate(self._patterns):
            match = pattern.search(text)
            if match:
                return {
                    "blocked": True,
                    "reason": f"匹配到敏感关键词: {self.blocked_keywords[i]}",
                    "matched_word": self.blocked_keywords[i],
                }

        return {"blocked": False, "reason": None, "matched_word": None}

    def get_safe_response(self) -> str:
        """返回安全提示文本（用于 TTS 合成）"""
        return self.safe_response_text
