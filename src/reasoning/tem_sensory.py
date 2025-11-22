"""TEM Sensory Module - Factorizes entities into compositional representations"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TEMSensoryModule(nn.Module):
    """
    Converts raw inputs into factorized representations.
    
    Factorization: Each entity gets multiple 'views' or 'features'
    Example: "Paris" → [location_embedding, city_embedding, capital_embedding]
    
    This allows compositional generalization - understanding new combinations
    of known properties.
    """
    
    def __init__(self, input_dim=512, num_factors=4, factor_dim=128):
        super().__init__()
        
        self.num_factors = num_factors
        self.factor_dim = factor_dim
        
        # Create multiple 'factor' projections
        # Each factor captures a different aspect of the entity
        self.factor_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, factor_dim),
                nn.LayerNorm(factor_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_factors)
        ])
        
    def forward(self, x):
        """
        Args:
            x: [batch, num_entities, input_dim] - entity representations
        
        Returns:
            factorized: [batch, num_entities, num_factors, factor_dim]
        """
        factors = []
        for encoder in self.factor_encoders:
            factor = encoder(x)
            factors.append(factor)
        
        # Stack factors: [batch, num_entities, num_factors, factor_dim]
        factorized = torch.stack(factors, dim=2)
        
        return factorized