Agentic prototype for TARS
=========================

This directory contains a small prototype stack to run a lightweight agentic loop
and experiment with a continual-learning-like setup using a vector memory and
small adapter update skeletons.

Files added under `src/agentic`:

- `memory_store.py` - Chroma (with sentence-transformers) backed memory with a
  safe in-memory fallback.
- `llm_wrapper.py` - An LLM interface and a `MockLLM` used for smoke tests.
- `proto_agent.py` - A small `SimpleAgentController` implementing plan->execute->store.
- `agent_bootstrap.py` - CLI bootstrap that runs the prototype with the mock LLM.
- `lora_update.py` - Non-blocking template for LoRA/PEFT updates from an experience buffer.

How to run the quick smoke test (uses the MockLLM, safe to run anywhere):

```powershell
python -c "from src.agentic.agent_bootstrap import main; print('Run the bootstrap with: python src/agentic/agent_bootstrap.py')"
python src/agentic/agent_bootstrap.py
```

To integrate a real LLM, implement an adapter that conforms to the `LLMInterface`
in `src/agentic/llm_wrapper.py` and replace `MockLLM` in `agent_bootstrap.py`.
