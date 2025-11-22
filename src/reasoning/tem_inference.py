"""TEM Inference Module - Multi-hop reasoning over relational graphs"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TEMInferenceModule(nn.Module):
    """
    Performs multi-step reasoning using the relational memory.
    
    Key capability: Transitive inference
    Example: If A>B and B>C are known, infer A>C
    """
    
    def __init__(self, factor_dim=128, num_reasoning_steps=5):
        super().__init__()
        
        self.num_reasoning_steps = num_reasoning_steps
        self.factor_dim = factor_dim
        
        # Reasoning cell (applied recursively)
        self.reasoning_cell = nn.GRUCell(factor_dim, factor_dim)
        
        # Query processor
        self.query_encoder = nn.Sequential(
            nn.Linear(factor_dim * 2, factor_dim),  # query entity + relation type
            nn.LayerNorm(factor_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention mechanism for entity selection
        self.attention = nn.MultiheadAttention(
            embed_dim=factor_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # Answer decoder
        self.answer_decoder = nn.Sequential(
            nn.Linear(factor_dim, factor_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(factor_dim // 2, 1)  # Score for each entity
        )
        
    def forward(self, entity_embeddings, query_entity, query_relation):
        """
        Answer a query using multi-step reasoning.
        
        Args:
            entity_embeddings: [batch, num_entities, dim] - from relational memory
            query_entity: [batch, dim] - embedding of query entity
            query_relation: [batch, dim] - embedding of relation to infer
        
        Returns:
            answer_scores: [batch, num_entities] - score for each entity being the answer
            reasoning_trace: [batch, num_steps, dim] - trace of reasoning process
        """
        batch_size, num_entities, dim = entity_embeddings.shape
        
        # Encode query
        query = self.query_encoder(
            torch.cat([query_entity, query_relation], dim=-1)
        )
        
        # Multi-step reasoning (like "chain of thought")
        thought = query
        reasoning_trace = []
        
        for step in range(self.num_reasoning_steps):
            # Attend to entities based on current thought
            thought_expanded = thought.unsqueeze(1)  # [batch, 1, dim]
            
            attended, attn_weights = self.attention(
                thought_expanded,
                entity_embeddings,
                entity_embeddings
            )
            
            context = attended.squeeze(1)  # [batch, dim]
            
            # Update thought using GRU
            thought = self.reasoning_cell(context, thought)
            
            reasoning_trace.append(thought)
        
        reasoning_trace = torch.stack(reasoning_trace, dim=1)  # [batch, steps, dim]
        
        # Score each entity as potential answer
        # Expand final thought to compare with all entities
        final_thought = thought.unsqueeze(1).expand(-1, num_entities, -1)
        combined = final_thought + entity_embeddings
        
        answer_scores = self.answer_decoder(combined).squeeze(-1)  # [batch, num_entities]
        
        return answer_scores, reasoning_trace