"""Query Understanding Module"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryUnderstanding(nn.Module):
    """
    Analyzes input queries to determine:
    - Intent (factual, reasoning, creative, etc.)
    - Complexity (simple vs complex)
    - Sub-question decomposition
    - Information needs (retrieval required?)
    """
    
    def __init__(self, dim=512, depth=4):
        super().__init__()
        
        self.dim = dim
        
        # Main encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                norm_first=True
            ),
            num_layers=depth
        )
        
        # Intent classifier
        self.intent_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)  # [factual, reasoning, creative, computational, ...]
        )
        
        # Decomposition head
        self.decompose_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 5)  # Max 5 sub-questions
        )
        
        # Information need classifier
        self.info_need_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)  # [no_retrieval, web_search, rag_lookup]
        )
        
        # Complexity estimator
        self.complexity_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query_tokens):
        """
        Args:
            query_tokens: [batch, seq_len, dim] - embedded query tokens
        
        Returns:
            dict with encoded query, intent, decomposition, info_need, complexity
        """
        # Encode query
        encoded = self.encoder(query_tokens)
        
        # Pool for classification heads
        pooled = encoded.mean(dim=1)  # [batch, dim]
        
        # Classify intent
        intent_logits = self.intent_head(pooled)
        
        # Decomposition (probability of needing N sub-questions)
        decomp_logits = self.decompose_head(pooled)
        
        # Information needs
        info_need_logits = self.info_need_head(pooled)
        
        # Complexity score
        complexity = self.complexity_head(pooled)
        
        return {
            'encoded': encoded,
            'pooled': pooled,
            'intent': intent_logits,
            'decomposition': decomp_logits,
            'info_need': info_need_logits,
            'complexity': complexity
        }