
import torch
import pytest
from src.tars_agent import TARS
from src.memory.hope_memory import HopeMemory

def test_tars_initialization_with_hope():
    vocab_size = 1000
    dim = 64

    tars = TARS(vocab_size=vocab_size, dim=dim)

    assert isinstance(tars.working_memory, HopeMemory)
    assert tars.working_memory.cms.dim == dim

def test_tars_forward_pass():
    vocab_size = 1000
    dim = 64
    seq_len = 10
    batch_size = 2

    tars = TARS(vocab_size=vocab_size, dim=dim)

    # Move model to device if CUDA available (but we test on CPU typically)

    query_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = tars(query_tokens)

    assert 'logits' in output
    # TARS is a seq2seq or language model, so it returns logits for the sequence
    assert output['logits'].shape == (batch_size, seq_len, vocab_size)
