"""TARS Demo - Comprehensive Examples"""

import torch
import sys
sys.path.append('..')

from src.tars_agent import TARS
from src.tools import create_tool_env


def demo_basic_qa():
    """Demo 1: Basic Question Answering"""
    print("\n" + "="*60)
    print("DEMO 1: Basic Question Answering")
    print("="*60)
    
    tars = TARS(
        vocab_size=50000,
        dim=512,
        humor_setting=75,
        honesty_setting=90
    ).cuda() if torch.cuda.is_available() else TARS().cpu()
    
    tools = create_tool_env()
    
    # Add some knowledge
    tools['rag_retrieve'].add_documents([
        "The capital of France is Paris.",
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
    ])
    
    questions = [
        "What is the capital of France?",
        "Tell me about Python",
        "What is machine learning?"
    ]
    
    for q in questions:
        print(f"\n❓ Question: {q}")
        answer = tars.quick_answer(q, tools, verbose=False)
        print(f"💬 TARS: {answer}")


def demo_personality_modes():
    """Demo 2: Different Personality Modes"""
    print("\n" + "="*60)
    print("DEMO 2: Personality Modes")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tools = create_tool_env()
    
    question = "What's 2+2?"
    
    # Mode 1: Professional
    print("\n--- Professional Mode (Humor: 10%, Honesty: 95%) ---")
    tars = TARS(humor_setting=10, honesty_setting=95).to(device)
    answer = tars.quick_answer(question, tools)
    print(f"💬 {answer}")
    
    # Mode 2: Balanced
    print("\n--- Balanced Mode (Humor: 60%, Honesty: 85%) ---")
    tars = TARS(humor_setting=60, honesty_setting=85).to(device)
    answer = tars.quick_answer(question, tools)
    print(f"💬 {answer}")
    
    # Mode 3: Witty
    print("\n--- Witty Mode (Humor: 90%, Honesty: 80%) ---")
    tars = TARS(humor_setting=90, honesty_setting=80).to(device)
    answer = tars.quick_answer(question, tools)
    print(f"💬 {answer}")


def demo_complex_mission():
    """Demo 3: Complex Multi-Step Mission"""
    print("\n" + "="*60)
    print("DEMO 3: Complex Mission")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tars = TARS(
        vocab_size=50000,
        dim=512,
        humor_setting=75,
        honesty_setting=90
    ).to(device)
    
    tars.introduce()
    
    tools = create_tool_env()
    
    # Add domain knowledge
    tools['rag_retrieve'].add_documents([
        "Artificial Intelligence includes machine learning and deep learning.",
        "Neural networks are inspired by biological neurons.",
        "Transformers revolutionized NLP in 2017.",
        "TARS uses pure reasoning without memorized facts.",
    ])
    
    mission = """
    Research artificial intelligence and explain:
    1. What it is
    2. Main approaches
    3. Recent breakthroughs
    Create a 3-sentence summary.
    """
    
    result = tars.execute_mission(
        goal=mission,
        tools=tools,
        max_iterations=20,
        enable_reflection=True,
        use_episodic_memory=True,
        verbose=True
    )
    
    print("\n📊 Mission Statistics:")
    print(f"   Success: {result['success']}")
    print(f"   Iterations: {result['iterations']}")
    print(f"   Steps: {len(result['execution_trace'])}")


def demo_calculation():
    """Demo 4: Calculation with Tools"""
    print("\n" + "="*60)
    print("DEMO 4: Calculations")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tars = TARS(humor_setting=70, honesty_setting=95).to(device)
    
    tools = create_tool_env()
    
    calculations = [
        "Calculate 157 * 23",
        "What is 2^10?",
        "Find the sum of 100, 200, and 300"
    ]
    
    for calc in calculations:
        print(f"\n🔢 {calc}")
        result = tars.execute_mission(
            goal=calc,
            tools=tools,
            max_iterations=5,
            enable_reflection=False,
            verbose=False
        )
        print(f"💬 TARS: {result['response']}")


def demo_episodic_memory():
    """Demo 5: Learning from Experience"""
    print("\n" + "="*60)
    print("DEMO 5: Episodic Memory (Learning from Experience)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tars = TARS(humor_setting=75, honesty_setting=90).to(device)
    
    tools = create_tool_env()
    
    # Mission 1
    print("\n--- First Mission ---")
    mission1 = "Find information about neural networks"
    result1 = tars.execute_mission(
        goal=mission1,
        tools=tools,
        max_iterations=10,
        use_episodic_memory=True,
        verbose=True
    )
    
    # Mission 2 (similar)
    print("\n--- Second Similar Mission (should benefit from memory) ---")
    mission2 = "Explain deep learning"
    result2 = tars.execute_mission(
        goal=mission2,
        tools=tools,
        max_iterations=10,
        use_episodic_memory=True,
        verbose=True
    )
    
    print(f"\n💾 Episodic Memory: {tars.episodic_memory.episode_count.item()} episodes stored")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║              TARS Comprehensive Demo                     ║
    ║       Thinking Autonomous Reasoning System               ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print("\nSelect a demo:")
    print("1. Basic Q&A")
    print("2. Personality Modes")
    print("3. Complex Mission")
    print("4. Calculations")
    print("5. Episodic Memory")
    print("6. Run All Demos")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        demo_basic_qa()
    elif choice == '2':
        demo_personality_modes()
    elif choice == '3':
        demo_complex_mission()
    elif choice == '4':
        demo_calculation()
    elif choice == '5':
        demo_episodic_memory()
    elif choice == '6':
        print("\n🚀 Running all demos...\n")
        demo_basic_qa()
        demo_personality_modes()
        demo_complex_mission()
        demo_calculation()
        demo_episodic_memory()
    else:
        print("Invalid choice")
        return
    
    print("\n" + "="*60)
    print("✨ Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()