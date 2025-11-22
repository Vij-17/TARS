"""Bootstrap runner for the prototype agent.

This script provides a small CLI that uses the MockLLM by default so it can be
run on any machine for quick experimentation. To use a real LLM, replace the
MockLLM with an implementation that calls your DeepSeek wrapper.
"""
from .llm_wrapper import MockLLM
from .proto_agent import SimpleAgentController
from .memory_store import MemoryStore


def main():
    llm = MockLLM()
    memory = MemoryStore()
    agent = SimpleAgentController(llm=llm, memory=memory)

    print("Prototype agent (mock LLM). Type 'exit' to quit.")
    while True:
        q = input("You: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        ans = agent.step(q)
        print("AI:", ans)


if __name__ == '__main__':
    main()
