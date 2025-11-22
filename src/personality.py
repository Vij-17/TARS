"""TARS Personality Module - Adds wit and character to responses"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class TARSPersonality(nn.Module):
    """
    TARS Personality System
    
    Configurable humor and honesty settings that influence response style.
    Inspired by AI systems that combine intelligence with character.
    """
    
    def __init__(self, dim=512, humor_setting=75, honesty_setting=90):
        super().__init__()
        
        self.dim = dim
        
        # Personality settings (0-100)
        self.humor_setting = min(100, max(0, humor_setting))
        self.honesty_setting = min(100, max(0, honesty_setting))
        
        # Personality encoder
        self.personality_encoder = nn.Sequential(
            nn.Linear(2, 64),  # [humor, honesty]
            nn.ReLU(),
            nn.Linear(64, dim)
        )
        
        # Response style modulator
        self.style_modulator = nn.Sequential(
            nn.Linear(dim * 2, dim),  # content + personality
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        # Wit generator (adds personality to responses)
        self.wit_generator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=1024,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Initialize personality embedding
        self._update_personality_embedding()
    
    def _update_personality_embedding(self):
        """Create embedding from current settings"""
        settings = torch.tensor(
            [[self.humor_setting / 100.0, self.honesty_setting / 100.0]],
            dtype=torch.float32
        )
        self.personality_emb = self.personality_encoder(settings)
    
    def set_humor(self, level):
        """Set humor level (0-100)"""
        self.humor_setting = min(100, max(0, level))
        self._update_personality_embedding()
    
    def set_honesty(self, level):
        """Set honesty level (0-100)"""
        self.honesty_setting = min(100, max(0, level))
        self._update_personality_embedding()
    
    def forward(self, response_embedding, context=None):
        """
        Add personality to response.
        
        Args:
            response_embedding: [batch, seq, dim]
            context: Optional context about the task
        
        Returns:
            personality_response: Response with personality added
        """
        
        # Get personality embedding
        personality = self.personality_emb.to(response_embedding.device)
        
        # Broadcast personality to sequence length
        batch_size, seq_len = response_embedding.shape[:2]
        personality = personality.expand(batch_size, seq_len, -1)
        
        # Modulate response with personality
        combined = torch.cat([response_embedding, personality], dim=-1)
        modulated = self.style_modulator(combined)
        
        # Add wit
        with_personality = self.wit_generator(modulated)
        
        return with_personality
    
    def get_response_prefix(self):
        """Get personality-appropriate response prefix"""
        
        if self.humor_setting > 70:
            prefixes = [
                "",  # Sometimes no prefix for natural flow
                "Alright, ",
                "Let me break this down: ",
                "Here's what we've got: ",
            ]
        elif self.humor_setting > 40:
            prefixes = [
                "",
                "Understood. ",
                "Processing complete. ",
            ]
        else:
            prefixes = ["", "Response: "]
        
        return random.choice(prefixes)
    
    def format_response(self, response_text, success=True, confidence=1.0):
        """
        Format response with personality.
        
        Args:
            response_text: Base response
            success: Whether task succeeded
            confidence: Confidence level (0-1)
        """
        
        prefix = self.get_response_prefix()
        
        # Add honesty-based confidence disclosure
        if self.honesty_setting > 80 and confidence < 0.7:
            honesty_note = f"\n\n(Confidence: {confidence:.0%} - I should mention this is partially uncertain)"
        elif self.honesty_setting > 90 and confidence < 0.5:
            honesty_note = f"\n\n(Honesty setting {self.honesty_setting}%: This answer has low confidence at {confidence:.0%})"
        else:
            honesty_note = ""
        
        # Add humor-based remarks
        if self.humor_setting > 60 and success:
            humor_remarks = [
                "",  # Most of the time, no remark
                " Mission accomplished.",
                " That wasn't so hard.",
            ]
            humor = random.choice(humor_remarks)
        elif self.humor_setting > 60 and not success:
            humor_remarks = [
                "",
                " Well, that could have gone better.",
                " Not quite what we planned.",
            ]
            humor = random.choice(humor_remarks)
        else:
            humor = ""
        
        formatted = f"{prefix}{response_text}{humor}{honesty_note}"
        
        return formatted.strip()
    
    def get_status_message(self, action, iteration, max_iterations):
        """Get personality-appropriate status message"""
        
        progress_pct = (iteration / max_iterations) * 100
        
        if self.humor_setting > 70:
            if progress_pct < 30:
                return f"Starting {action}..."
            elif progress_pct < 70:
                return f"Working on {action}... ({progress_pct:.0f}% through)"
            else:
                return f"Almost done with {action}..."
        else:
            return f"Executing: {action}"
    
    def __repr__(self):
        return f"TARSPersonality(humor={self.humor_setting}%, honesty={self.honesty_setting}%)"