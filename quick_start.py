"""TARS Quick Start - Get up and running in minutes"""

import torch
from src.tars_agent import TARS
from src.tools import create_tool_env


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                    T.A.R.S. v1.0                         ║
    ║          Thinking Autonomous Reasoning System            ║
    ║                                                          ║
    ║  "A pure reasoning engine with personality"              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Initialize TARS
    print("🚀 Initializing TARS...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    tars = TARS(
        vocab_size=50000,
        dim=512,
        reasoning_depth=6,
        max_iterations=20,
        humor_setting=75,      # 75% humor (witty but not annoying)
        honesty_setting=90     # 90% honest (direct but tactful)
    ).to(device)
    
    # Show intro
    tars.introduce()
    
    # Step 2: Setup tools
    print("\n🔧 Setting up tools...")
    tools = create_tool_env()
    
    # Add knowledge to RAG
    tools['rag_retrieve'].add_documents([
        "TARS is a Thinking Autonomous Reasoning System with personality.",
        "Pure reasoning engines store HOW to think, not facts.",
        "The Tolman-Eichenbaum Machine (TEM) enables relational reasoning.",
        "Working memory uses test-time learning via Titans neural memory.",
        "Episodic memory stores successful strategies from past missions.",
    ])
    
    print("   ✅ Tools ready!")
    
    # Step 3: Test different personality settings
    print("\n" + "="*60)
    print("🎭 PERSONALITY TEST")
    print("="*60)
    
    questions = [
        "What are you?",
        "What is a pure reasoning engine?",
        "Can you explain how you work?"
    ]
    
    # Test 1: Balanced personality
    print("\n--- Balanced TARS (Humor: 75%, Honesty: 90%) ---")
    for q in questions[:1]:
        answer = tars.quick_answer(q, tools, verbose=False)
        print(f"\n❓ {q}")
        print(f"💬 TARS: {answer}")
    
    # Test 2: Very serious
    print("\n--- Serious TARS (Humor: 20%, Honesty: 95%) ---")
    tars.set_humor(20)
    tars.set_honesty(95)
    for q in questions[1:2]:
        answer = tars.quick_answer(q, tools, verbose=False)
        print(f"\n❓ {q}")
        print(f"💬 TARS: {answer}")
    
    # Test 3: Maximum wit
    print("\n--- Witty TARS (Humor: 85%, Honesty: 85%) ---")
    tars.set_humor(85)
    tars.set_honesty(85)
    for q in questions[2:3]:
        answer = tars.quick_answer(q, tools, verbose=False)
        print(f"\n❓ {q}")
        print(f"💬 TARS: {answer}")
    
    # Step 4: Complex mission
    print("\n" + "="*60)
    print("🎯 COMPLEX MISSION")
    print("="*60)
    
    # Reset to balanced
    tars.set_humor(75)
    tars.set_honesty(90)
    
    mission = "Research how TARS works and explain it in simple terms"
    
    result = tars.execute_mission(
        goal=mission,
        tools=tools,
        max_iterations=15,
        enable_reflection=True,
        verbose=True
    )
    
    # Step 5: Show what's next
    print("\n" + "="*60)
    print("🎉 NEXT STEPS")
    print("="*60)
    print("""
    1. Adjust personality:
       tars.set_humor(50)      # 0-100
       tars.set_honesty(100)   # 0-100
    
    2. Try complex missions:
       result = tars.execute_mission(
           "Research X and create summary",
           tools=tools
       )
    
    3. Add your knowledge to RAG:
       tools['rag_retrieve'].add_documents([...])
    
    4. Train TARS:
       python training/train_reasoning.py
    
    5. See more examples:
       python examples/tars_demo.py
       python examples/personality_test.py
    
    6. Configure TARS:
       Edit configs/model_config.py
    """)
    
    print(f"\n💬 TARS: Ready for your missions.\n")


if __name__ == "__main__":
    main()