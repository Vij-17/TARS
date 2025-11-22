"""Recursive Reasoning Core - Handles algorithmic and logical reasoning"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecursiveReasoningCore(nn.Module):
    """
    Stores HOW to reason, not WHAT to reason about.
    
    Reasoning operations it learns:
    - Deductive inference (A→B, A ⊢ B)
    - Inductive reasoning (pattern→generalization)
    - Abductive reasoning (effect→best explanation)
    - Analogical mapping (structure transfer)
    - Causal reasoning (intervention→outcome)
    """
    
    def __init__(self, dim=512, reasoning_depth=6):
        super().__init__()
        
        self.dim = dim
        self.reasoning_depth = reasoning_depth
        
        # Shared reasoning cell (applied recursively)
        self.reasoning_cell = nn.ModuleDict({
            # Hypothesis generator
            'hypothesis': nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 2, dim)
            ),
            
            # Contradiction detector
            'contradiction': nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Linear(dim, 1),
                nn.Sigmoid()
            ),
            
            # Causal reasoner
            'causal': nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim, dim)
            ),
            
            # Analogy mapper
            'analogy': nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=1024,
                batch_first=True,
                dropout=0.1
            )
        })
        
        # Reasoning state tracker
        self.state_tracker = nn.GRUCell(dim, dim)
        
        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def reasoning_step(self, thought, context, step_type='hypothesis'):
        """
        Single reasoning step - applied recursively.
        
        Args:
            thought: Current reasoning state [batch, dim]
            context: External information [batch, seq, dim]
            step_type: Type of reasoning operation
        
        Returns:
            new_thought: Updated reasoning state
        """
        
        if step_type == 'hypothesis':
            # Generate hypothesis from current thought
            new_thought = self.reasoning_cell['hypothesis'](thought)
            
        elif step_type == 'contradiction':
            # Check for contradictions with context
            context_pooled = context.mean(dim=1)
            combined = torch.cat([thought, context_pooled], dim=-1)
            contradiction_score = self.reasoning_cell['contradiction'](combined)
            new_thought = thought * (1 - contradiction_score)
            
        elif step_type == 'causal':
            # Causal reasoning
            new_thought = self.reasoning_cell['causal'](thought)
            
        elif step_type == 'analogy':
            # Map structure from context to thought
            combined = torch.cat([thought.unsqueeze(1), context], dim=1)
            new_thought = self.reasoning_cell['analogy'](combined)[:, 0]
        else:
            new_thought = thought
        
        # Update reasoning state
        new_state = self.state_tracker(new_thought, thought)
        
        return new_state
    
    def forward(self, query_encoding, retrieved_context=None):
        """
        Recursive reasoning over multiple steps.
        
        Args:
            query_encoding: Initial problem representation [batch, seq, dim]
            retrieved_context: Retrieved facts/knowledge [batch, seq, dim]
        
        Returns:
            dict with final_thought, reasoning_trace, confidences
        """
        
        # Initialize reasoning state
        thought = query_encoding.mean(dim=1)  # [batch, dim]
        
        context = retrieved_context if retrieved_context is not None else query_encoding
        
        reasoning_trace = []
        confidences = []
        
        # Recursive reasoning loop
        step_types = ['hypothesis', 'causal', 'analogy', 'contradiction']
        
        for step in range(self.reasoning_depth):
            # Determine reasoning operation for this step
            step_type = step_types[step % len(step_types)]
            
            # Apply reasoning step
            thought = self.reasoning_step(thought, context, step_type=step_type)
            
            # Estimate confidence in current reasoning
            conf = self.confidence(thought)
            
            reasoning_trace.append(thought)
            confidences.append(conf)
            
            # Early stopping if high confidence
            if conf.mean() > 0.95:
                break
        
        reasoning_trace = torch.stack(reasoning_trace, dim=1)  # [batch, steps, dim]
        confidences = torch.stack(confidences, dim=1)  # [batch, steps, 1]
        
        return {
            'final_thought': thought,
            'reasoning_trace': reasoning_trace,
            'confidences': confidences
        }