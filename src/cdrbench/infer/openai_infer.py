#!/usr/bin/env python3
"""
OpenAI-compatible inference backend shared by local vLLM and remote APIs.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base import BaseInfer

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)


class OpenAIInfer(BaseInfer):
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = 'EMPTY',
        concurrency: int = 8,
        max_tokens: int = 0,
        temperature: float = 0.0,
        num_runs: int = 1,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_thinking: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model=model,
            concurrency=concurrency,
            max_tokens=max_tokens,
            temperature=temperature,
            num_runs=num_runs,
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_thinking = enable_thinking
        self._extra_body: Dict[str, Any] = (
            extra_body
            if extra_body is not None
            else {'chat_template_kwargs': {'enable_thinking': enable_thinking}}
        )
        self._client = OpenAI(api_key=api_key, base_url=api_base)

    def _call_once(self, messages: List[dict]) -> str:
        last_exc: Exception = RuntimeError('no attempts made')
        delay = self.retry_delay

        for attempt in range(max(1, self.max_retries)):
            try:
                request_kwargs: Dict[str, Any] = {
                    'model': self.model,
                    'messages': messages,
                    'temperature': self.temperature,
                    'stream': False,
                }
                if self.max_tokens > 0:
                    request_kwargs['max_tokens'] = self.max_tokens
                if self._extra_body:
                    request_kwargs['extra_body'] = self._extra_body
                resp = self._client.chat.completions.create(**request_kwargs)
                return resp.choices[0].message.content or ''
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2

        raise last_exc

    def __repr__(self) -> str:
        return (
            f'OpenAIInfer(model={self.model!r}, '
            f'concurrency={self.concurrency}, '
            f'num_runs={self.num_runs}, '
            f'enable_thinking={self.enable_thinking})'
        )


def make_vllm_infer(
    model: str,
    api_base: str = 'http://127.0.0.1:8901/v1',
    concurrency: int = 128,
    max_tokens: int = 0,
    temperature: float = 0.0,
    num_runs: int = 1,
    enable_thinking: bool = False,
    max_retries: int = 2,
    retry_delay: float = 0.5,
) -> OpenAIInfer:
    return OpenAIInfer(
        model=model,
        api_base=api_base,
        api_key='EMPTY',
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        num_runs=num_runs,
        max_retries=max_retries,
        retry_delay=retry_delay,
        enable_thinking=enable_thinking,
    )


def make_api_infer(
    model: str,
    api_base: str = 'http://123.57.212.178:3333/v1',
    api_key: str = '',
    concurrency: int = 8,
    max_tokens: int = 0,
    temperature: float = 0.0,
    num_runs: int = 1,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> OpenAIInfer:
    resolved_key = api_key or os.getenv('DASHSCOPE_API_KEY', 'EMPTY')
    return OpenAIInfer(
        model=model,
        api_base=api_base,
        api_key=resolved_key,
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        num_runs=num_runs,
        max_retries=max_retries,
        retry_delay=retry_delay,
        enable_thinking=False,
    )
