"""Lightweight prototype AgentController (SimpleAgentController) used for bootstrapping and tests.

This file is intentionally minimal and independent from the heavier agentic
modules present in `agent_controller.py` (the main project module). The
prototype controller is useful for quick experiments and smoke tests.
"""
from typing import Any, Dict, List, Optional
from .memory_store import MemoryStore
from .llm_wrapper import LLMInterface
import json
import logging

logger = logging.getLogger(__name__)


class SimpleAgentController:
    """Minimal plan->execute->store loop used in the prototype.

    Methods:
      - plan(user_input): asks the LLM for a JSON plan
      - execute_plan(plan, user_input): executes simple 'answer' steps
      - step(user_input): run full loop and store Q/A
    """

    def __init__(self, llm: LLMInterface, memory: Optional[MemoryStore] = None):
        self.llm = llm
        self.memory = memory or MemoryStore()

    def _build_plan_prompt(self, user_input: str, retrieved: List[str]) -> str:
        mem_text = "\n".join([m[0] for m in retrieved]) if retrieved else ""
        prompt = (
            f"You are an agent. Use the memory below to create a short plan in JSON.\n"
            f"Memory:\n{mem_text}\nUser: {user_input}\nPlan (JSON list of steps):"
        )
        return prompt

    def plan(self, user_input: str, k: int = 6) -> List[Dict[str, Any]]:
        retrieved = self.memory.retrieve(user_input, k=k) if self.memory else []
        prompt = self._build_plan_prompt(user_input, retrieved)
        raw = self.llm.generate(prompt, max_new_tokens=256)
        try:
            plan = json.loads(raw)
            if isinstance(plan, list):
                return plan
        except Exception:
            logger.debug("Plan parse failed, treating raw output as single-answer")
        return [{"type": "answer", "instruction": raw}]

    def execute_plan(self, plan: List[Dict[str, Any]], user_input: str) -> str:
        outputs = []
        for step in plan:
            stype = step.get("type", "answer")
            if stype == "answer":
                instr = step.get("instruction", "Answer directly")
                prompt = f"{instr}\nUser question:\n{user_input}\nAnswer:"
                out = self.llm.generate(prompt, max_new_tokens=256)
                outputs.append(out)
            else:
                outputs.append(f"[unsupported step type: {stype}]")
        return "\n".join(outputs)

    def step(self, user_input: str, store: bool = True) -> str:
        plan = self.plan(user_input)
        answer = self.execute_plan(plan, user_input)
        if store and self.memory is not None:
            try:
                self.memory.store_event(f"Q:{user_input}\nA:{answer}", metadata={"type": "qa"}, llm_generate_fn=self.llm.generate)
            except Exception as e:
                logger.debug(f"Failed to store event: {e}")
        return answer


__all__ = ["SimpleAgentController"]
