"""
TARS Setup Verification Script

Run this to verify your TARS installation is complete and working.
"""

import sys
import importlib


def test_import(module_name, friendly_name=None):
    """Test if a module can be imported"""
    friendly_name = friendly_name or module_name
    try:
        importlib.import_module(module_name)
        print(f"✅ {friendly_name}")
        return True
    except ImportError as e:
        print(f"❌ {friendly_name}: {e}")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║           TARS Setup Verification                        ║
╚══════════════════════════════════════════════════════════╝
""")
    
    all_passed = True
    
    # Test Python dependencies
    print("\n📦 Testing Python Dependencies:")
    print("-" * 60)
    deps = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('einops', 'Einops'),
        ('titans_pytorch', 'Titans'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('faiss', 'FAISS'),
        ('requests', 'Requests'),
        ('tqdm', 'tqdm'),
    ]
    
    for module, name in deps:
        if not test_import(module, name):
            all_passed = False
    
    # Test CUDA
    print("\n🎮 Testing CUDA:")
    print("-" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA Available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA Not Available (CPU mode)")
    except Exception as e:
        print(f"❌ CUDA Test Failed: {e}")
        all_passed = False
    
    # Test TARS modules
    print("\n🤖 Testing TARS Modules:")
    print("-" * 60)
    tars_modules = [
        ('src.tars_agent', 'TARS Agent'),
        ('src.personality', 'Personality Module'),
        ('src.core', 'Core Components'),
        ('src.reasoning', 'Reasoning Modules'),
        ('src.memory', 'Memory Systems'),
        ('src.tools', 'Tools'),
        ('src.agentic', 'Agentic Components'),
        ('configs', 'Configuration'),
    ]
    
    for module, name in tars_modules:
        if not test_import(module, name):
            all_passed = False
    
    # Test specific imports
    print("\n🔬 Testing Specific Imports:")
    print("-" * 60)
    try:
        from src.tars_agent import TARS
        print("✅ TARS class")
    except ImportError as e:
        print(f"❌ TARS class: {e}")
        all_passed = False
    
    try:
        from src.personality import TARSPersonality
        print("✅ TARSPersonality class")
    except ImportError as e:
        print(f"❌ TARSPersonality class: {e}")
        all_passed = False
    
    try:
        from src.tools import create_tool_env
        print("✅ create_tool_env function")
    except ImportError as e:
        print(f"❌ create_tool_env function: {e}")
        all_passed = False
    
    try:
        from src.reasoning import TolmanEichenbaumMachine
        print("✅ TolmanEichenbaumMachine class")
    except ImportError as e:
        print(f"❌ TolmanEichenbaumMachine class: {e}")
        all_passed = False
    
    try:
        from src.memory import WorkingMemory, EpisodicMemory
        print("✅ Memory classes")
    except ImportError as e:
        print(f"❌ Memory classes: {e}")
        all_passed = False
    
    # Test instantiation
    print("\n🏗️  Testing Instantiation:")
    print("-" * 60)
    try:
        from src.tars_agent import TARS
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tars = TARS(
            vocab_size=1000,  # Small for testing
            dim=128,          # Small for testing
            humor_setting=75,
            honesty_setting=90
        ).to(device)
        
        print(f"✅ TARS instantiation successful")
        print(f"   Device: {device}")
        print(f"   Parameters: {sum(p.numel() for p in tars.parameters())/1e6:.1f}M")
        print(f"   {tars.personality}")
        
    except Exception as e:
        print(f"❌ TARS instantiation failed: {e}")
        all_passed = False
    
    # Test tools
    print("\n🔧 Testing Tools:")
    print("-" * 60)
    try:
        from src.tools import create_tool_env
        tools = create_tool_env()
        
        required_tools = ['web_search', 'rag_retrieve', 'calculate', 'execute_code', 'reason']
        for tool_name in required_tools:
            if tool_name in tools:
                print(f"✅ {tool_name}")
            else:
                print(f"❌ {tool_name} missing")
                all_passed = False
        
    except Exception as e:
        print(f"❌ Tool creation failed: {e}")
        all_passed = False
    
    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("""
✨ Your TARS installation is complete and ready!

Next steps:
1. Run quick_start.py
2. Try examples/tars_demo.py
3. Adjust personality settings
4. Start training (optional)

Commands:
    python quick_start.py
    python examples/tars_demo.py
        """)
    else:
        print("⚠️  SOME TESTS FAILED")
        print("="*60)
        print("""
Please check:
1. All files copied to correct locations
2. All __init__.py files present
3. Dependencies installed: pip install -r requirements.txt
4. Correct directory structure

See INSTALLATION_GUIDE.md for details.
        """)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)