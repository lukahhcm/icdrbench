from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


DEFAULT_BASE_URL = "http://123.57.212.178:3333/v1"
DEFAULT_MODEL = "gpt-4.1-2025-04-14"


def resolve_api_key(explicit: str | None = None) -> str:
    api_key = explicit or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("No API key found. Set OPENAI_API_KEY or DASHSCOPE_API_KEY.")
    return api_key


def resolve_base_url(explicit: str | None = None) -> str:
    return explicit or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL


def resolve_model(explicit: str | None = None) -> str:
    return explicit or os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_MODEL


def build_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    return OpenAI(api_key=resolve_api_key(api_key), base_url=resolve_base_url(base_url))


def chat_completion(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
) -> str:
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned empty content.")
    return content


def strip_code_fences(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL)
    return fenced.group(1) if fenced else text


def parse_json_response(text: str) -> Any:
    cleaned = strip_code_fences(text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise
