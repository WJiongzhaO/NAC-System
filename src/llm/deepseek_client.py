"""
LLM 模块 — DeepSeek-V4-Flash API 客户端

支持两种模式：
- generate: 完整生成（基线）
- generate_stream: 流式逐 token 输出（改进阶段）
"""

import os
import time
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from openai import OpenAI

# 自动加载项目根目录 .env 文件
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)


SYSTEM_PROMPT = """你是一个有帮助的助手，协助用户解答问题。请直接给出答案，不要输出任何无关内容。"""


class DeepSeekClient:
    def __init__(
        self,
        api_base: str = "https://api.deepseek.com",
        api_key: str | None = None,
        model: str = "deepseek-v4-flash",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ):
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("请设置环境变量 DEEPSEEK_API_KEY 或传入 api_key 参数")

        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def generate(self, user_text: str) -> dict:
        """
        完整生成（基线模式）

        Returns:
            {
                "text": str,           # 完整回答文本
                "ttft_ms": float,      # Time-To-First-Token (ms)
                "total_ms": float,     # 总耗时 (ms)
            }
        """
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False,
        )
        t1 = time.perf_counter()

        text = response.choices[0].message.content or ""

        return {
            "text": text,
            "ttft_ms": None,  # 非流式模式无 TTFT
            "total_ms": (t1 - t0) * 1000,
        }

    def generate_stream(self, user_text: str) -> Generator[dict, None, None]:
        """
        流式生成（改进模式），逐 token yield

        Yields:
            {"delta": str, "ttft_ms": float | None}
            首个 chunk 携带 ttft_ms 测量值
        """
        t0 = time.perf_counter()
        first_token = True
        ttft_ms = None

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if first_token and delta:
                ttft_ms = (time.perf_counter() - t0) * 1000
                first_token = False
            yield {"delta": delta, "ttft_ms": ttft_ms if first_token is False and ttft_ms else None}
