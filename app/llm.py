"""LLM backends: Anthropic (streaming) or a deterministic mock for running without an API key."""
import asyncio
import re
from typing import AsyncIterator

from . import config as C

Message = dict  # {"role": "user"|"assistant", "content": str}


class AnthropicLLM:
    def __init__(self):
        import anthropic
        self.client = anthropic.AsyncAnthropic()

    async def stream(self, system: str, messages: list[Message]) -> AsyncIterator[str]:
        async with self.client.messages.stream(
            model=C.LLM_MODEL,
            max_tokens=C.LLM_MAX_TOKENS,
            system=system,
            messages=messages,
            output_config={"effort": C.LLM_EFFORT},
        ) as stream:
            async for text in stream.text_stream:
                yield text


class MockLLM:
    """Echo-style assistant that visibly reacts to the QE block, so the two panels differ even offline."""

    async def stream(self, system: str, messages: list[Message]) -> AsyncIterator[str]:
        user = messages[-1]["content"]
        transcript, _, qe = user.partition("\n\n[ASR quality estimation]")
        if qe:
            low = re.findall(r'"([^"]+)" \((\d\.\d\d)\)', qe.split("Likely misrecognized:")[-1].split("\n")[0]) \
                if "Likely misrecognized:" in qe else []
            missing = re.findall(r"Possible missing words (.+?) \(", qe)
            parts = [f"Tôi hiểu bạn nói: “{transcript.strip()}”."]
            if low:
                parts.append("Tôi chưa chắc về " + ", ".join(f"“{w}”" for w, _ in low) + " — bạn xác nhận giúp nhé?")
            if missing:
                parts.append(f"Có vẻ một đoạn bị thiếu ({missing[0]}), bạn có thể nói lại phần đó không?")
            if not low and not missing:
                parts.append("Bản ghi âm rõ, tôi sẽ xử lý ngay. (mock LLM)")
            else:
                parts.append("(mock LLM)")
        else:
            parts = [f"Bạn nói: “{transcript.strip()}”. Tôi sẽ xử lý ngay. (mock LLM)"]
        for tok in " ".join(parts).split(" "):
            yield tok + " "
            await asyncio.sleep(0.03)


def build_llm():
    return AnthropicLLM() if C.LLM_BACKEND == "anthropic" else MockLLM()
