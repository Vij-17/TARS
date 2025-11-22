# TARS - Thinking Autonomous Reasoning System

*"A pure reasoning engine that knows HOW to think, not WHAT to think about"*

Inspired by intelligent systems that combine reasoning with personality - TARS is an agentic AI that:
- Stores reasoning methods (not facts) in its weights
- Retrieves all knowledge from external sources (web, RAG)
- Plans, acts, observes, and reflects autonomously
- Has configurable personality (humor, honesty levels)

## Features

### 🧠 Pure Reasoning Engine
- **TEM (Tolman-Eichenbaum Machine)**: Relational reasoning via graph neural networks
- **Recursive Reasoner**: Algorithmic and logical problem-solving
- **Meta-Learning**: Learns to learn from few examples

### 💾 Dual Memory Systems
- **Working Memory (Titans)**: Fast test-time adaptation
- **Episodic Memory**: Long-term strategy storage

### 🤖 Agentic Capabilities
- **Autonomous Planning**: Breaks down complex goals
- **Self-Reflection**: Learns from mistakes
- **Error Recovery**: Adapts when things go wrong
- **Multi-Step Execution**: Pursues long-term objectives

### 🔧 Tool Integration
- **Web Search**: Real-time information retrieval
- **RAG System**: Domain knowledge access
- **Calculator**: Precise computations
- **Code Execution**: Run and debug code

### 😄 TARS Personality
- **Humor Setting**: 0-100% (adjustable wit level)
- **Honesty Setting**: 0-100% (adjustable directness)
- **Helpful**: Always tries to solve your problem
- **Witty**: Delivers information with personality

## Installation

```bash
# Clone/create directory
mkdir tars
cd tars

# Install dependencies
pip install -r requirements.txt

# Quick start
python quick_start.py
```

## Quick Example

```python
from src.tars_agent import TARS
from src.tools import create_tool_env

# Initialize TARS
tars = TARS(
    vocab_size=50000,
    dim=512,
    humor_setting=75,      # 75% humor
    honesty_setting=90     # 90% honest
).cuda()

# Setup tools
tools = create_tool_env()

# Give it a goal
result = tars.execute_mission(
    goal="Research quantum computing and explain it simply",
    tools=tools,
    max_iterations=20
)

print(result['response'])
```

## Configuration

### Personality Settings

```python
# More serious, professional TARS
tars = TARS(humor_setting=20, honesty_setting=95)

# Witty, entertaining TARS
tars = TARS(humor_setting=80, honesty_setting=75)

# Brutally honest TARS
tars = TARS(humor_setting=50, honesty_setting=100)
```

## Project Structure

```
tars/
├── src/
│   ├── tars_agent.py          # Main TARS system
│   ├── personality.py         # Personality module
│   ├── core/                  # Core components
│   ├── reasoning/             # TEM + Recursive reasoner
│   ├── memory/                # Working + Episodic memory
│   ├── tools/                 # External tools
│   └── agentic/               # Planning, execution, reflection
├── configs/
├── training/
└── examples/
```

## Hardware Requirements

- **GPU**: RTX 4070 (8GB) or better
- **RAM**: 32GB+ recommended
- **Storage**: 10GB

## Model Architecture

- **Total Parameters**: ~430M
- **VRAM Usage**: ~5-6GB during training
- **Inference**: ~3-4GB

## Training

```bash
# Phase 1: Core reasoning (Week 1)
python training/train_reasoning.py

# Phase 2: Tool use (Week 2)
python training/train_tools.py

# Phase 3: Agentic (Week 3-4)
python training/train_agentic.py
```

## Examples

See `examples/` directory:
- `tars_demo.py` - Basic usage
- `personality_test.py` - Test different personality settings
- `complex_mission.py` - Multi-step autonomous task

## Why "TARS"?

Standing for **Thinking Autonomous Reasoning System**, TARS represents:
- **Autonomous**: Self-directed problem solving
- **Reasoning**: Pure logic, not memorized facts
- **Personality**: Helpful with character

## License

MIT

## Acknowledgments

- Titans paper for neural memory
- TEM paper for relational reasoning
- The inspiration for combining reasoning with personality
