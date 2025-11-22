
import torch
import pytest
from src.memory.hope_memory import HopeMemory, ContinuumMemory

def test_continuum_memory_shapes():
    dim = 64
    seq_len = 20
    batch_size = 2

    cms = ContinuumMemory(
        dim=dim,
        chunk_sizes=[4, 8],
        dim_head=32,
        heads=4
    )

    x = torch.randn(batch_size, seq_len, dim)
    output, new_states = cms(x)

    assert output.shape == (batch_size, seq_len, dim)
    assert len(new_states) == 2

def test_hope_memory_initialization():
    dim = 64
    hope = HopeMemory(dim=dim, levels=3, base_chunk_size=2, heads=4)

    assert len(hope.cms.memories) == 3
    # Chunk sizes should be 2, 8, 32
    # hope.cms.memories contains NeuralMemory instances directly
    assert hope.cms.memories[0].chunk_size == 2
    assert hope.cms.memories[1].chunk_size == 8
    assert hope.cms.memories[2].chunk_size == 32

def test_hope_memory_forward():
    dim = 64
    seq_len = 16
    batch_size = 2

    hope = HopeMemory(dim=dim, levels=2, base_chunk_size=4)
    x = torch.randn(batch_size, seq_len, dim)

    output, states = hope(x)

    assert output.shape == (batch_size, seq_len, dim)
    assert len(states) == 2
