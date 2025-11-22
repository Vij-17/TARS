"""Reflection Module - Self-critique and correction"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectionModule(nn.Module):
    """
    Reflects on action outcomes and corrects course.
    
    Implements self-reflection, critique, and learning from mistakes.
    """
    
    def __init__(self, dim=512, depth=2):
        super().__init__()
        
        self.dim = dim
        
        # Outcome analyzer
        self.outcome_analyzer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=depth
        )
        
        # Success/failure classifier
        self.outcome_classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)  # [exceeded, met, failed]
        )
        
        # Critique generator
        self.critique_generator = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=depth
        )

        # Project outcome (3-class logits) to model dim before feeding to the
        # critique transformer decoder. Without this, the decoder receives a
        # tensor with last-dimension 3 which causes an embedding-dimension
        # mismatch (expected `dim`, e.g. 512).
        self.outcome_proj = nn.Linear(3, dim)
        
        # Plan revision suggester
        self.revision_suggester = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),  # goal + state + critique
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        
        # Learning signal generator
        self.learning_signal = nn.Sequential(
            nn.Linear(dim * 2, 128),  # expected + actual
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Tanh()  # Positive = reinforce, Negative = suppress
        )
        
    def reflect(self, plan, execution_trace, goal, final_state):
        """
        Reflect on entire execution episode.
        
        Args:
            plan: List of planned actions
            execution_trace: List of (action, obs, state) dicts
            goal: Goal embedding
            final_state: Final state achieved
        
        Returns:
            dict with outcome, critique, revision, learning_signals
        """
        
        # Analyze overall outcome
        if not execution_trace:
            return {
                'outcome': torch.zeros(1, 3),
                'critique': torch.zeros(1, self.dim),
                'should_revise': False,
                'revision': None,
                'learning_signals': []
            }
        
        trace_embeddings = torch.stack([
            t['state'] for t in execution_trace
        ], dim=1)
        
        analyzed = self.outcome_analyzer(trace_embeddings)
        outcome = self.outcome_classifier(analyzed.mean(dim=1))
        
        # Generate critique
        critique = self._generate_critique(plan, execution_trace, outcome)
        
        # Check if we need to revise plan
        outcome_type = outcome.argmax(dim=-1).item()
        should_revise = (outcome_type == 2)  # Failed
        
        if should_revise:
            revision = self.revision_suggester(
                torch.cat([goal, final_state, critique], dim=-1)
            )
        else:
            revision = None
        
        # Generate learning signals for each step
        learning_signals = []
        for i, step in enumerate(execution_trace):
            if i + 1 < len(execution_trace):
                expected = step['state']
                actual = execution_trace[i + 1]['state']
                signal = self.learning_signal(
                    torch.cat([expected, actual], dim=-1)
                )
                learning_signals.append({
                    'step': i,
                    'action': step.get('action', 'unknown'),
                    'signal': signal.item(),
                    'reinforce': signal.item() > 0
                })
        
        return {
            'outcome': outcome,
            'critique': critique,
            'should_revise': should_revise,
            'revision': revision,
            'learning_signals': learning_signals
        }
    
    def _generate_critique(self, plan, trace, outcome):
        """Generate critique embedding"""
        
        # Combine plan and trace
        plan_embeddings = torch.stack([s['embedding'] for s in plan], dim=1)
        trace_embeddings = torch.stack([t['state'] for t in trace], dim=1)
        
        combined = torch.cat([plan_embeddings, trace_embeddings], dim=1)
        
        # Generate critique
        # Project outcome logits (shape: [batch, 3]) into model dimension
        # so the transformer decoder receives a tensor with the correct
        # d_model (dim).
        critique_input = self.outcome_proj(outcome).unsqueeze(1)
        critique = self.critique_generator(critique_input, combined)

        return critique.mean(dim=1)