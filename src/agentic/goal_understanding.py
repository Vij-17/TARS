"""Goal Understanding - Parses and decomposes high-level goals"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalUnderstanding(nn.Module):
    """
    Parses high-level goals and decomposes into sub-goals.
    
    Example:
    Goal: "Book a flight to Tokyo for next month"
    Decomposition:
    1. Find dates next month
    2. Search for flights to Tokyo
    3. Compare prices
    4. Select best option
    5. Complete booking
    """
    
    def __init__(self, dim=512, depth=3, max_subgoals=10):
        super().__init__()
        
        self.dim = dim
        self.max_subgoals = max_subgoals
        
        # Goal encoder
        self.goal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=depth
        )
        
        # Goal type classifier
        self.goal_classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)  # Different goal types
        )
        
        # Complexity estimator
        self.complexity_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0 = simple, 1 = complex
        )
        
        # Sub-goal generator (hierarchical decomposition)
        self.subgoal_generator = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=depth
        )
        
        # Stopping criterion
        self.stop_head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, goal_text_embedded):
        """
        Args:
            goal_text_embedded: [batch, seq_len, dim]
        
        Returns:
            goal_structure: Dict with goal type, complexity, sub-goals
        """
        
        # Encode goal
        encoded = self.goal_encoder(goal_text_embedded)
        pooled = encoded.mean(dim=1)
        
        # Classify goal type
        goal_type = self.goal_classifier(pooled)
        
        # Estimate complexity
        complexity = self.complexity_head(pooled)
        
        # Generate sub-goals if complex
        subgoals = []
        if complexity.mean() > 0.5:
            decoder_input = pooled.unsqueeze(1)
            
            for step in range(self.max_subgoals):
                subgoal = self.subgoal_generator(decoder_input, encoded)
                subgoals.append(subgoal[:, -1])
                
                # Check stopping criterion
                stop_prob = self.stop_head(subgoal[:, -1])
                if stop_prob.mean() > 0.8:
                    break
                
                # Append for next iteration
                decoder_input = torch.cat([decoder_input, subgoal[:, -1:]], dim=1)
            
            subgoals = torch.stack(subgoals, dim=1) if subgoals else None
        else:
            subgoals = None
        
        return {
            'encoded': encoded,
            'pooled': pooled,
            'goal_type': goal_type,
            'complexity': complexity,
            'subgoals': subgoals
        }