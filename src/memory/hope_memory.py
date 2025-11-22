"""Hope Memory - A self-modifying architecture with continuum memory (CMS)"""

import torch
import torch.nn as nn
from titans_pytorch import NeuralMemory

class ContinuumMemory(nn.Module):
    """
    Continuum Memory System (CMS) as described in the Nested Learning blog.

    Consists of a spectrum of memory modules, each updating at a different frequency/scale.
    We approximate "frequency" by using different chunk sizes in the Neural Memory.
    - Small chunk size: Updates frequently on local context (Fast memory)
    - Large chunk size: Updates slowly on larger context (Slow memory)
    """

    def __init__(
        self,
        dim,
        chunk_sizes=[4, 16, 64], # Spectrum of timescales
        dim_head=64,
        heads=4,
        batch_size=None,
        momentum=True,
    ):
        super().__init__()
        self.dim = dim
        self.chunk_sizes = chunk_sizes
        self.memories = nn.ModuleList()

        # Create a NeuralMemory for each timescale
        for cs in chunk_sizes:
            self.memories.append(
                NeuralMemory(
                    dim=dim,
                    chunk_size=cs,
                    dim_head=dim_head,
                    heads=heads,
                    momentum=momentum,
                    batch_size=batch_size,
                    attn_pool_chunks=False, # Do not pool chunks, preserve sequence length
                    qk_rmsnorm=True
                )
            )

        # Combine outputs from different timescales
        self.combiner = nn.Linear(dim * len(chunk_sizes), dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, states=None):
        """
        Args:
            x: Input sequence [batch, seq_len, dim]
            states: List of states for each memory level. If None, starts fresh.

        Returns:
            output: Combined memory retrieval [batch, seq_len, dim]
            new_states: List of updated states
        """
        if states is None:
            states = [None] * len(self.memories)

        outputs = []
        new_states = []

        for i, memory in enumerate(self.memories):
            # Each memory processes the input
            # NeuralMemory expects [batch, seq_len, dim]
            # It handles chunking internally based on its chunk_size
            retrieved, new_state = memory(x, state=states[i])
            outputs.append(retrieved)
            new_states.append(new_state)

        # Concatenate outputs from all levels
        # Note: retrieving from NeuralMemory preserves sequence length (usually)
        # if attn_pool_chunks is True, it might be pooled?
        # Let's check NeuralMemory behavior. usually it returns (batch, seq, dim)

        combined = torch.cat(outputs, dim=-1) # [batch, seq_len, dim * num_levels]
        projected = self.combiner(combined)
        output = self.norm(projected + x) # Residual connection

        return output, new_states

class HopeMemory(nn.Module):
    """
    Hope - Self-modifying architecture using Continuum Memory.

    Uses CMS (ContinuumMemory) to handle multiple timescales.
    """

    def __init__(
        self,
        dim=512,
        levels=3,
        base_chunk_size=4,
        heads=8
    ):
        super().__init__()

        # Geometric progression of chunk sizes for different timescales
        # e.g., 4, 16, 64
        chunk_sizes = [base_chunk_size * (4**i) for i in range(levels)]

        self.cms = ContinuumMemory(
            dim=dim,
            chunk_sizes=chunk_sizes,
            heads=heads
        )

    def forward(self, x, state=None):
        return self.cms(x, state)

    def clear(self):
        return None
