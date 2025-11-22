"""State Tracker - Maintains world model and tracks changes"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateTracker(nn.Module):
    """
    Maintains world model and tracks state changes.
    
    Similar to Dreamer (world model learning) and MuZero (hidden state)
    """
    
    def __init__(self, dim=512, num_layers=3):
        super().__init__()
        
        self.dim = dim
        
        # World model (predicts state transitions)
        self.world_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=num_layers
        )
        
        # State encoder (LSTM for sequential state tracking)
        self.state_encoder = nn.LSTM(
            dim, dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Observation integrator
        self.obs_integrator = nn.Sequential(
            nn.Linear(dim * 2, dim),  # prev_state + new_obs
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Surprise detector (unexpected observations)
        self.surprise_detector = nn.Sequential(
            nn.Linear(dim * 2, 128),  # predicted + actual
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Goal progress estimator
        self.progress_estimator = nn.Sequential(
            nn.Linear(dim * 2, 128),  # current_state + goal
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def update_state(self, prev_state, observation, action, goal):
        """
        Update world state based on observation.
        
        Args:
            prev_state: [batch, dim]
            observation: Dict with 'embedding' key
            action: Dict with 'embedding' key
            goal: [batch, dim]
        
        Returns:
            dict with new_state, surprise, progress, predicted_state
        """
        
        obs_embedding = observation['embedding']
        if obs_embedding.dim() == 3:
            obs_embedding = obs_embedding.squeeze(1)
        
        action_embedding = action['embedding']
        if action_embedding.dim() == 3:
            action_embedding = action_embedding.squeeze(1)
        
        # Predict next state (what we expected)
        state_action = torch.cat([
            prev_state.unsqueeze(1),
            action_embedding.unsqueeze(1)
        ], dim=1)
        
        predicted_state = self.world_model(state_action)[:, -1]
        
        # Compute surprise (how unexpected was observation?)
        surprise = self.surprise_detector(
            torch.cat([predicted_state, obs_embedding], dim=-1)
        )
        
        # Integrate observation with previous state
        integrated = self.obs_integrator(
            torch.cat([prev_state, obs_embedding], dim=-1)
        )
        
        # Update state using LSTM
        new_state, _ = self.state_encoder(integrated.unsqueeze(1))
        new_state = new_state.squeeze(1)
        
        # Estimate progress toward goal
        progress = self.progress_estimator(
            torch.cat([new_state, goal], dim=-1)
        )
        
        return {
            'state': new_state,
            'surprise': surprise,
            'progress': progress,
            'predicted_state': predicted_state
        }