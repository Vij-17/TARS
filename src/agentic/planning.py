"""Planning Module - Generates and verifies action plans"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanningModule(nn.Module):
    """
    Generates executable plans using chain-of-thought reasoning.
    
    Inspired by ReAct, Tree of Thoughts, Plan-and-Solve
    """
    
    def __init__(self, dim=512, max_plan_steps=20):
        super().__init__()
        
        self.dim = dim
        self.max_plan_steps = max_plan_steps
        
        # Action vocabulary
        self.action_types = [
            'web_search',
            'rag_retrieve',
            'calculate',
            'execute_code',
            'reason',
            'ask_clarification',
            'summarize',
            'compare',
            'verify',
            'finish'
        ]
        self.num_actions = len(self.action_types)
        
        # Plan generator (auto-regressive)
        self.plan_generator = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=4
        )
        
        # Action predictor
        self.action_head = nn.Linear(dim, self.num_actions)
        
        # Action parameter generator
        self.param_generator = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        
        # Plan verifier
        self.plan_verifier = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=2
        )
        
        self.feasibility_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Plan optimizer
        self.plan_optimizer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=2
        )
        
    def generate_plan(self, goal_structure, context=None):
        """
        Generate action plan from goal.
        
        Args:
            goal_structure: Output from GoalUnderstanding
            context: Optional context (e.g., from episodic memory)
        
        Returns:
            plan: List of action steps with parameters
        """
        
        goal_encoded = goal_structure['encoded']
        batch_size = goal_encoded.shape[0]
        
        # Initialize plan
        plan_steps = []
        decoder_input = goal_structure['pooled'].unsqueeze(1)
        
        # Auto-regressively generate plan steps
        for step in range(self.max_plan_steps):
            # Generate next step
            step_embedding = self.plan_generator(
                decoder_input,
                goal_encoded
            )[:, -1]
            
            # Predict action type
            action_logits = self.action_head(step_embedding)
            action_idx = action_logits.argmax(dim=-1)
            
            # Get action names
            action_name = [self.action_types[idx.item()] for idx in action_idx]
            
            # Generate action parameters
            params = self.param_generator(step_embedding)
            
            # Store step
            plan_steps.append({
                'action': action_name,
                'action_idx': action_idx,
                'params': params,
                'embedding': step_embedding
            })
            
            # Check if plan complete
            if all(a == 'finish' for a in action_name):
                break
            
            # Append to decoder input
            decoder_input = torch.cat([
                decoder_input,
                step_embedding.unsqueeze(1)
            ], dim=1)
        
        return plan_steps
    
    def verify_plan(self, plan_steps, goal_structure):
        """
        Check if plan is feasible.
        
        Returns:
            feasibility_score: 0-1
            issues: List of potential problems
            step_confidence: Confidence per step
        """
        
        if not plan_steps:
            return {
                'feasibility': torch.tensor(0.0),
                'issues': ['Empty plan'],
                'step_confidence': torch.tensor([])
            }
        
        # Stack plan embeddings
        plan_embeddings = torch.stack(
            [step['embedding'] for step in plan_steps],
            dim=1
        )
        
        # Verify
        verified = self.plan_verifier(plan_embeddings)
        feasibility = self.feasibility_head(verified.mean(dim=1))
        
        # Per-step confidence
        step_confidence = self.feasibility_head(verified).squeeze(-1)
        
        # Identify issues
        issues = []
        for i, conf in enumerate(step_confidence[0]):
            if conf < 0.6:
                action = plan_steps[i]['action'][0]
                issues.append(f"Step {i+1} uncertain: {action} (conf={conf:.2f})")
        
        return {
            'feasibility': feasibility,
            'issues': issues,
            'step_confidence': step_confidence
        }
    
    def optimize_plan(self, plan_steps):
        """
        Improve plan efficiency.
        """
        
        if not plan_steps:
            return plan_steps
        
        plan_embeddings = torch.stack(
            [step['embedding'] for step in plan_steps],
            dim=1
        )
        
        # Optimize
        optimized = self.plan_optimizer(plan_embeddings)
        
        # Generate improved plan
        improved_steps = []
        for i, step in enumerate(plan_steps):
            improved_steps.append({
                'action': step['action'],
                'action_idx': step['action_idx'],
                'params': self.param_generator(optimized[:, i]),
                'embedding': optimized[:, i]
            })
        
        return improved_steps