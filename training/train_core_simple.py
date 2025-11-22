"""Train TARS core reasoning components"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm.auto import tqdm
from datasets import concatenate_datasets
import logging

from src.tars_agent import TARS
from configs.training_config import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningDataset(Dataset):
    """Combined reasoning dataset for TARS"""
    def __init__(self, squad_data, ropes_data, max_length=512):
        self.squad = squad_data
        self.ropes = ropes_data
        self.max_length = max_length
        
        # Combine datasets
        self.combined = []
        
        # Process SQuAD examples
        for example in squad_data:
            self.combined.append({
                'context': example['context'],
                'question': example['question'],
                'answer': example['answers']['text'][0] if example['answers']['text'] else "",
                'type': 'squad'
            })
        
        # Process ROPES examples
        for example in ropes_data:
            self.combined.append({
                'context': example['background'] + " " + example['situation'],
                'question': example['question'],
                'answer': example['answers']['text'][0] if example['answers']['text'] else "",
                'type': 'ropes'
            })
    
    def __len__(self):
        return len(self.combined)
    
    def __getitem__(self, idx):
        item = self.combined[idx]
        
        # Convert to model inputs
        input_text = f"{item['context']} [SEP] {item['question']}"
        target_text = item['answer']
        
        # Create fixed-size tensors
        input_size = 400
        target_size = 100
        
        # Simple tokenization
        input_words = input_text.split()[:input_size]
        target_words = target_text.split()[:target_size]
        
        # Convert to token IDs
        input_ids = torch.tensor([hash(w) % 50000 for w in input_words] + [0] * (input_size - len(input_words)), dtype=torch.long)
        target_ids = torch.tensor([hash(w) % 50000 for w in target_words] + [0] * (target_size - len(target_words)), dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'type': item['type']
        }

def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Forward pass
            input_embedded = model.token_embed(input_ids)
            tem_output, _ = model.tem_core(input_embedded)
            recursive_outputs = model.recursive_reasoner(query_encoding=input_embedded)
            recursive_output = recursive_outputs['final_thought']
            
            # Combine reasoning outputs
            tem_features = tem_output.mean(dim=1)
            combined_reasoning = torch.cat([tem_features, recursive_output], dim=-1)
            
            # Dimension reduction
            dim_reduction = nn.Linear(model.dim * 2, model.dim).to(device)
            combined_reasoning = dim_reduction(combined_reasoning)
            
            # Generate output
            decoder_input = torch.zeros(
                target_ids.size(0),
                target_ids.size(1),
                model.dim,
                device=device
            )
            
            output_seq = model.response_decoder(
                decoder_input,
                combined_reasoning.unsqueeze(1)
            )
            
            # Project to vocabulary
            logits = model.output_projection(output_seq)
            
            # Compute loss
            loss = nn.CrossEntropyLoss(ignore_index=0)(
                logits.view(-1, model.vocab_size),
                target_ids.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    return total_loss / len(dataloader)

def main():
    logger.info("="*60)
    logger.info("Training TARS Core Reasoning")
    logger.info("="*60)
    
    # Load config
    config = TrainingConfig()
    
    # Force CUDA device
    device = torch.device('cuda')
    logger.info(f"Using device: {device}")
    
    # Load datasets
    logger.info("\nLoading datasets...")
    from datasets import load_dataset
    
    squad = load_dataset('squad')['train']
    ropes = load_dataset('ropes')['train']
    
    # Create training dataset
    train_dataset = ReasoningDataset(squad, ropes)
    logger.info(f"Training examples: {len(train_dataset)}")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    logger.info("\nInitializing TARS...")
    model = TARS(
        vocab_size=50000,
        dim=512,
        reasoning_depth=6,
        max_iterations=20,
        humor_setting=50,
        honesty_setting=100
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training loop
    logger.info("\nStarting training...")
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        logger.info("="*40)
        
        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        logger.info(f"Epoch {epoch + 1} - Average loss: {train_loss:.4f}")
        
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = os.path.join('checkpoints', f'tars_epoch_{epoch+1}.pt')
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            logger.info(f"Saved best model checkpoint: {checkpoint_path}")
    
    logger.info("\nTraining complete!")
    logger.info(f"Best loss achieved: {best_loss:.4f}")

if __name__ == "__main__":
    main()