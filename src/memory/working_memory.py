"""Working Memory - Wrapper around Titans Neural Memory"""

import torch
import torch.nn as nn
from titans_pytorch import NeuralMemory


class WorkingMemory(nn.Module):
    """
    Temporary storage for retrieved facts during problem-solving.
    
    Uses Titans Neural Memory for test-time learning.
    Gets CLEARED after each task - no long-term knowledge storage.
    """
    
    def __init__(
        self,
        dim=512,
        chunk_size=16,
        dim_head=64,
        heads=8,
        momentum=True,
        batch_size=32,
        max_lr=0.1
    ):
        super().__init__()
        
        self.neural_memory = NeuralMemory(
            dim=dim,
            chunk_size=chunk_size,
            dim_head=dim_head,
            heads=heads,
            momentum=momentum,
            batch_size=batch_size,
            default_step_transform_max_lr=max_lr,
            attn_pool_chunks=True,
            qk_rmsnorm=True
        )
        
    def store_facts(self, facts, state=None):
        """
        Store retrieved facts temporarily.
        
        Args:
            facts: [batch, seq_len, dim]
            state: Previous memory state
        
        Returns:
            retrieved: Retrieved information
            new_state: Updated memory state
        """
        retrieved, new_state = self.neural_memory(facts, state=state)
        return retrieved, new_state
    
    def forward(self, facts, state=None):
        """Alias for store_facts"""
        return self.store_facts(facts, state)
    
    def clear(self):
        """Clear working memory (by returning None state)"""
        return None