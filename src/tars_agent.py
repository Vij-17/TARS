"""TARS - Thinking Autonomous Reasoning System"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import QueryUnderstanding
from .reasoning import TolmanEichenbaumMachine, RecursiveReasoningCore
from .memory import WorkingMemory, EpisodicMemory
from .tools import ToolUseController
from .agentic import AgentController
from .personality import TARSPersonality


class TARS(nn.Module):
    """
    TARS - Thinking Autonomous Reasoning System
    
    A pure reasoning engine with personality that:
    - Stores HOW to reason (not facts) in weights
    - Gets all knowledge from external sources
    - Plans and executes autonomously
    - Reflects and learns from experience
    - Responds with configurable wit and honesty
    
    Inspired by AI systems that combine intelligence with character.
    """
    
    def __init__(
        self,
        vocab_size=50000,
        dim=512,
        reasoning_depth=6,
        max_iterations=20,
        humor_setting=75,
        honesty_setting=90
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_iterations = max_iterations
        
        # Core identification
        self.name = "TARS"
        self.version = "1.0"
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, dim)
        
        # Core modules
        self.query_understanding = QueryUnderstanding(dim, depth=4)
        
        # Reasoning modules
        self.tem_core = TolmanEichenbaumMachine(
            input_dim=dim,
            num_factors=4,
            factor_dim=128,
            num_relation_types=10,
            num_gnn_layers=3,
            num_reasoning_steps=5
        )
        
        self.recursive_reasoner = RecursiveReasoningCore(dim, reasoning_depth)
        
        # Tool controller
        self.tool_controller = ToolUseController(dim, num_tools=5)
        
        # Memory systems
        self.working_memory = WorkingMemory(
            dim=dim,
            chunk_size=16,
            heads=8,
            momentum=True,
            batch_size=32
        )
        
        self.episodic_memory = EpisodicMemory(dim, memory_capacity=10000)
        
        # Agentic system
        self.agent_controller = AgentController(
            dim=dim,
            max_iterations=max_iterations,
            vocab_size=vocab_size
        )
        
        # 🎭 TARS Personality
        self.personality = TARSPersonality(
            dim=dim,
            humor_setting=humor_setting,
            honesty_setting=honesty_setting
        )
        
        # Response generator
        self.response_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=4
        )
        
        self.output_projection = nn.Linear(dim, vocab_size)
        
        # Task router
        self.task_router = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # [use_tem, use_recursive]
        )
        
    def set_humor(self, level):
        """
        Set humor level (0-100)
        
        0 = Completely serious
        100 = Maximum wit
        """
        self.personality.set_humor(level)
        print(f"💬 Humor setting: {level}%")
    
    def set_honesty(self, level):
        """
        Set honesty level (0-100)
        
        0 = Diplomatic, gentle
        100 = Brutally direct
        """
        self.personality.set_honesty(level)
        print(f"🎯 Honesty setting: {level}%")
    
    def introduce(self):
        """TARS introduction"""
        intro = f"""
╔═══════════════════════════════════════════════════════════╗
║                         T.A.R.S.                          ║
║          Thinking Autonomous Reasoning System             ║
╠═══════════════════════════════════════════════════════════╣
║  Status: Online                                           ║
║  Version: {self.version}                                  ║
║  Parameters: {sum(p.numel() for p in self.parameters())/1e6:.0f}M ║
║                                                           ║
║  Configuration:                                           ║
║    • Humor Setting: {self.personality.humor_setting}%     ║
║    • Honesty Setting: {self.personality.honesty_setting}% ║
║                                                           ║
║  Capabilities:                                            ║
║    ✓ Pure reasoning (no facts in weights)                 ║
║    ✓ Autonomous planning & execution                      ║
║    ✓ Self-reflection & learning                           ║
║    ✓ External knowledge access (web, RAG)                 ║
║    ✓ Personality-aware responses                          ║
╚═══════════════════════════════════════════════════════════╝
        """
        print(intro)
    
    def forward(
        self,
        query_tokens,
        retrieved_docs=None,
        known_relations=None,
        return_tool_calls=False
    ):
        """
        Standard forward pass for training.
        
        Args:
            query_tokens: [batch, seq_len]
            retrieved_docs: Optional external knowledge
            known_relations: Optional relational facts
            return_tool_calls: Whether to return tool requests
        
        Returns:
            dict with logits, reasoning traces, etc.
        """
        
        batch_size = query_tokens.shape[0]
        
        # Embed tokens
        query_embedded = self.token_embed(query_tokens)
        
        # Get the device from the model parameters
        device = next(self.parameters()).device
        
        # Ensure query is on the correct device
        query_embedded = query_embedded.to(device)
        
        # Understand query
        query_analysis = self.query_understanding(query_embedded)
        
        # Route to appropriate reasoning module
        routing = F.softmax(
            self.task_router(query_analysis['pooled']),
            dim=-1
        )
        use_tem = routing[:, 0] > routing[:, 1]
        
        # Reasoning paths
        if use_tem.any():
            entities = query_embedded
            # Ensure known_relations is on the correct device if provided
            if known_relations is not None:
                known_relations = {k: v.to(device) if torch.is_tensor(v) else v 
                                 for k, v in known_relations.items()}
            tem_output, _ = self.tem_core(entities, known_relations)
            relational_output = tem_output.mean(dim=1)
        else:
            relational_output = None
        
        # Recursive reasoning
        recursive_output = self.recursive_reasoner(
            query_embedded,
            retrieved_context=None
        )
        
        # Combine based on routing
        if relational_output is not None:
            combined = (
                routing[:, 0:1] * relational_output +
                routing[:, 1:2] * recursive_output['final_thought']
            ).unsqueeze(1)
        else:
            combined = recursive_output['final_thought'].unsqueeze(1)
        
        # Generate response
        response = self.response_decoder(query_embedded, combined)
        
        # Add personality
        response = self.personality(response)
        
        logits = self.output_projection(response)
        
        return {
            'logits': logits,
            'routing': routing,
            'recursive_output': recursive_output
        }
    
    def execute_mission(
        self,
        goal,
        tools,
        max_iterations=None,
        enable_reflection=True,
        use_episodic_memory=True,
        verbose=True
    ):
        """
        Execute a mission (TARS-style agent loop).
        
        Args:
            goal: Mission objective (string)
            tools: Available tools dict
            max_iterations: Max steps
            enable_reflection: Self-correction enabled
            use_episodic_memory: Learn from past missions
            verbose: Print progress
        
        Returns:
            dict with success, response, execution trace
        """
        
        max_iterations = max_iterations or self.max_iterations
        
        if verbose:
            print(f"\n🎯 Mission: {goal}")
            print(f"⚙️  Settings: Humor {self.personality.humor_setting}%, Honesty {self.personality.honesty_setting}%")
            print("="*60)
        
        # Tokenize goal
        goal_tokens = self._tokenize(goal)
        goal_embedded = self.token_embed(goal_tokens)
        
        # Run agent loop
        result = self.agent_controller.agent_loop(
            goal_embedded=goal_embedded,
            tools=tools,
            episodic_memory=self.episodic_memory if use_episodic_memory else None,
            working_memory=self.working_memory,
            max_iterations=max_iterations,
            enable_reflection=enable_reflection,
            verbose=verbose
        )
        
        # Generate response with personality
        response_text = self._generate_response(
            goal_embedded,
            result['final_state'],
            result['execution_trace']
        )
        
        # Format with personality
        confidence = result.get('progress', 0.5)
        formatted_response = self.personality.format_response(
            response_text,
            success=result['success'],
            confidence=confidence
        )
        
        result['response'] = formatted_response
        
        if verbose:
            print("\n" + "="*60)
            if result['success']:
                print("✅ Mission accomplished")
            else:
                print("⚠️  Mission incomplete")
            print(f"📊 Iterations: {result['iterations']}")
            print(f"💬 TARS: {formatted_response}")
        
        return result
    
    def quick_answer(self, question, tools, verbose=False):
        """
        Quick answer mode (less iterations, fast response).
        
        Args:
            question: Simple question
            tools: Tools dict
            verbose: Print progress
        
        Returns:
            Answer string
        """
        result = self.execute_mission(
            goal=question,
            tools=tools,
            max_iterations=5,
            enable_reflection=False,
            use_episodic_memory=False,
            verbose=verbose
        )
        
        return result['response']
    
    def _tokenize(self, text):
        """Simple tokenizer (placeholder)"""
        return torch.randint(0, self.vocab_size, (1, min(len(text.split()), 50)))
    
    def _detokenize(self, tokens):
        """Convert tokens to text"""
        return "Mission analysis complete"
    
    def _generate_response(self, goal_embedded, final_state, trace):
        """Generate response from execution trace"""
        
        if not trace:
            return "Unable to complete mission"
        
        # Use recent steps for context
        recent_states = [t['state'] for t in trace[-5:]]
        if recent_states:
            trace_embeddings = torch.stack(recent_states, dim=1)
            
            response_context = torch.cat([
                final_state.unsqueeze(1),
                trace_embeddings
            ], dim=1)
            
            # Decode
            response = self.response_decoder(goal_embedded, response_context)
            
            # Add personality before projecting
            response = self.personality(response)
            
            logits = self.output_projection(response)
            tokens = logits.argmax(dim=-1)
            
            return self._detokenize(tokens)
        
        return "Mission completed successfully"
    
    def save_checkpoint(self, path):
        """Save TARS checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'dim': self.dim,
                'humor_setting': self.personality.humor_setting,
                'honesty_setting': self.personality.honesty_setting
            },
            'version': self.version
        }, path)
        print(f"💾 TARS checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load TARS checkpoint"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        config = checkpoint.get('config', {})
        if 'humor_setting' in config:
            self.personality.set_humor(config['humor_setting'])
        if 'honesty_setting' in config:
            self.personality.set_honesty(config['honesty_setting'])
        
        print(f"📥 TARS checkpoint loaded: {path}")
        return config
    
    def __repr__(self):
        return f"TARS(params={sum(p.numel() for p in self.parameters())/1e6:.0f}M, {self.personality})"