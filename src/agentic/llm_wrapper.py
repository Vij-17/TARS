"""LLM wrapper API and a Mock LLM for fast smoke tests.

The real integration should implement LLMInterface.generate(prompt, max_new_tokens)
to call your local DeepSeek model or remote LLM. The MockLLM returns deterministic
short replies so we can smoke-test the agent pipeline without loading large models.
"""
from typing import Optional


class LLMInterface:
    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate text from prompt. Real implementation should override this."""
        raise NotImplementedError()


class MockLLM(LLMInterface):
    def __init__(self, name: str = "mock-llm"):
        self.name = name

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        # Very simple heuristic: if asked to summarize, return short summary.
        lower = prompt.lower()
        if "summarize" in lower:
            return "- Short summary of the provided text.\n- Key point: none (mock)"
        if "plan" in lower:
            return "[\n  {\"type\": \"answer\", \"instruction\": \"Answer directly\"}\n]"
        # Default behaviour: echo a short answer
        head = prompt.strip().split('\n')[-1]
        return f"(mock answer) Responding to: {head[:120]}"


__all__ = ["LLMInterface", "MockLLM"]
