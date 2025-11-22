"""Training script for reasoning components"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import sys
sys.path.append('..')

from src.main_agent import AgenticReasoningEngine
from configs.training_config import TrainingConfig


class ReasoningDataset(Dataset):
    """
    Simple dataset for reasoning tasks.
    In practice, use bAbI, LogiQA, ARC, etc.
    """
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Mock data - replace with real datasets
        input_seq = torch.randint(0, 50000, (50,))
        target_seq = torch.randint(0, 50000, (50,))
        
        return {
            'input': input_seq,
            'target': target_seq
        }


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(
            outputs['logits'].reshape(-1, model.vocab_size),
            targets.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm
        )
        
        # Optimize
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        if batch_idx % config.log_every == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def main():
    print("="*60)
    print("Training Reasoning Components")
    print("="*60)
    
    # Configuration
    config = TrainingConfig()
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    print("\nInitializing model...")
    model = AgenticReasoningEngine(
        vocab_size=50000,
        dim=512,
        reasoning_depth=6
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Dataset
    print("\nPreparing datasets...")
    train_dataset = ReasoningDataset(num_samples=10000)
    val_dataset = ReasoningDataset(num_samples=1000)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training loop
    print("\nStarting training...")
    print("="*60)
    
    num_epochs = config.phase1_config['num_epochs']
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-"*60)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(
                    outputs['logits'].reshape(-1, model.vocab_size),
                    targets.reshape(-1)
                )
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pt"
            model.save_checkpoint(checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    main()