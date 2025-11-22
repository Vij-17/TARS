"""Skeleton for LoRA/PEFT updates from an experience buffer.

This module provides a non-blocking skeleton showing how to prepare a small
LoRA update. It is intentionally conservative and does not automatically
download or train large models. Use it as a template for offline jobs.
"""
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def build_training_examples_from_buffer(buffer: List[Dict]) -> List[str]:
    """Convert a list of experience dicts into training texts (prompt+target).

    Buffer entries are expected to have keys like 'prompt' and 'response' or
    be simple Q/A records.
    """
    texts = []
    for ex in buffer:
        if 'prompt' in ex and 'response' in ex:
            texts.append(ex['prompt'] + "\n" + ex['response'])
        elif 'text' in ex:
            texts.append(ex['text'])
        else:
            texts.append(str(ex))
    return texts


def run_lora_update_stub(output_dir: str = "./lora_adapter", examples: Optional[List[str]] = None):
    """Skeleton function that documents the steps for a LoRA update.

    This function does NOT perform training itself. Replace the body with a
    proper PEFT training loop when running on GPU or a suitable environment.
    """
    if examples is None:
        examples = []
    logger.info("Preparing LoRA update")
    logger.info(f"Examples available: {len(examples)}")
    # Steps (manual):
    # 1. Load base model with transformers (8bit or full precision)
    # 2. prepare_model_for_kbit_training
    # 3. create LoraConfig and get_peft_model
    # 4. Tokenize examples and run Trainer or custom loop
    # 5. Save adapter to output_dir
    logger.info("LoRA update skeleton completed (no-op). Replace with real training code to run updates.")


__all__ = ["build_training_examples_from_buffer", "run_lora_update_stub"]
