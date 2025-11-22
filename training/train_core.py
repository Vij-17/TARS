"""Train TARS core reasoning components"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm.auto import tqdm
from datasets import concatenate_datasets
from src.tars_agent import TARS
from configs.training_config import TrainingConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningDataset(Dataset):
    """Combined reasoning dataset for TARS"""
    def __init__(self, squad_data, ropes_data, max_length=512):
        self.squad = squad_data
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
        
        # Convert to model inputs (simple tokenization for now)
        input_text = f"{item['context']} [SEP] {item['question']}"
        target_text = item['answer']
        
        # Create fixed-size tensors
        input_size = 400  # Fixed input size
        target_size = 100  # Fixed target size
        
        # Simple tokenization (just for demonstration)
        input_words = input_text.split()[:input_size]
        target_words = target_text.split()[:target_size]
        
        # Convert to token IDs (simple hashing for now)
        input_ids = torch.tensor([hash(w) % 50000 for w in input_words] + [0] * (input_size - len(input_words)), dtype=torch.long)
        target_ids = torch.tensor([hash(w) % 50000 for w in target_words] + [0] * (target_size - len(target_words)), dtype=torch.long)
        
        # Note: We create tensors on CPU here, they will be moved to GPU in the training loop
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'type': item['type']
        }

def train_epoch(model, dataloader, optimizer, device, config, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    # Debug device placement
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move batch data to GPU immediately and validate shapes/types
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            target_ids = batch['target_ids'].to(device, non_blocking=True)
            
            if input_ids.dim() != 2 or target_ids.dim() != 2:
                raise ValueError(f"Expected 2D tensors, got input_ids: {input_ids.dim()}D, target_ids: {target_ids.dim()}D")
            
            # Debug device placement
            logger.info(f"Input device: {input_ids.device}, Target device: {target_ids.device}")
            logger.info(f"Input dtype: {input_ids.dtype}, shape: {input_ids.shape}")
            logger.info(f"Target dtype: {target_ids.dtype}, shape: {target_ids.shape}")
            
            # Create input embeddings and ensure float32 dtype (already on correct device since model is on device)
            input_embedded = model.token_embed(input_ids)  # [batch, seq, dim]
            logger.info(f"Embedded dtype: {input_embedded.dtype}, shape: {input_embedded.shape}, device: {input_embedded.device}")
            
            # Forward pass through reasoning components
            tem_output, _ = model.tem_core(input_embedded)
            logger.info(f"TEM output dtype: {tem_output.dtype}, shape: {tem_output.shape}")
            
            # Pass input_embedded as query_encoding to recursive reasoner
            recursive_outputs = model.recursive_reasoner(query_encoding=input_embedded)
            # Extract final_thought tensor from recursive output dictionary
            recursive_output = recursive_outputs['final_thought']
            logger.info(f"Recursive output dtype: {recursive_output.dtype}, shape: {recursive_output.shape}")
            
            logger.debug(f"TEM output shape: {tem_output.shape}")
            logger.debug(f"Recursive output shape: {recursive_output.shape}")
            
            # Resize target sequence to match output sequence length
            target_seq_len = target_ids.size(1)
            input_seq_len = input_embedded.size(1)
            
            # Add debug for recursive_outputs
            logger.info(f"Recursive outputs keys: {recursive_outputs.keys()}")
            
            # Combine TEM and recursive outputs 
            tem_features = tem_output.mean(dim=1)  # [batch, dim]
            # Debug dimensions
            logger.info(f"TEM features shape: {tem_features.shape}, Recursive shape: {recursive_output.shape}")
            
            combined_reasoning = torch.cat([tem_features, recursive_output], dim=-1)  # [batch, dim*2]
            
            # Use a single dim_reduction layer attached to the model (created in main)
            if hasattr(model, 'dim_reduction'):
                combined_reasoning = model.dim_reduction(combined_reasoning)
            else:
                # fallback (shouldn't happen) - create on device but this won't be in optimizer
                dim_reduction = nn.Linear(model.dim * 2, model.dim).to(device)
                combined_reasoning = dim_reduction(combined_reasoning)  # [batch, dim]
            logger.info(f"Combined reasoning shape after reduction: {combined_reasoning.shape}")
            
            # Initialize decoder input sequence with zeros
            decoder_input = torch.zeros(
                target_ids.size(0),  # batch size
                target_seq_len,      # target sequence length
                model.dim,           # embedding dimension
                device=device,
                dtype=torch.float32
            )
            
            # Generate output sequence
            output_seq = model.response_decoder(
                decoder_input,
                combined_reasoning.unsqueeze(1)  # Add sequence dimension: [batch, 1, dim]
            )
            
            logger.debug(f"Output sequence shape: {output_seq.shape}")
            
            # Project to vocabulary
            logits = model.output_projection(output_seq)  # [batch, seq, vocab]
            logger.debug(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
            logger.debug(f"Logits shape: {logits.shape}")
            
            # Debug target tensor
            logger.info(f"Target dtype: {target_ids.dtype}, shape: {target_ids.shape}")
            
            # Compute loss (ignore padding tokens)
            loss = nn.CrossEntropyLoss(ignore_index=0)(
                logits.view(-1, model.vocab_size),
                target_ids.view(-1)
            )
            
            # Monitor loss value
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"❌ Loss is {loss.item()} at batch {batch_idx}")
                continue
                
                # Backward pass with automatic mixed precision
            optimizer.zero_grad()
            try:
                if device.type == 'cuda' and scaler is not None:
                    # Use the scaler for mixed precision training
                    scaler.scale(loss).backward()
                    
                    # Unscale before clipping
                    scaler.unscale_(optimizer)
                    
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0
                    )
                    
                    # Monitor gradient norms
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logger.error(f"❌ Gradient norm is {grad_norm} at batch {batch_idx}")
                        optimizer.zero_grad()  # Clear bad gradients
                        continue
                    
                    logger.info(f"Gradient norm at batch {batch_idx}: {grad_norm:.4f}")
                    
                    # Step with scaler
                    scaler.step(optimizer)
                    scaler.update()
                    
                else:
                    # Regular backward pass for CPU or if scaler not provided
                    loss.backward()
                    
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0
                    )
                    
                    # Monitor gradient norms
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logger.error(f"❌ Gradient norm is {grad_norm} at batch {batch_idx}")
                        optimizer.zero_grad()  # Clear bad gradients
                        continue
                    
                    logger.info(f"Gradient norm at batch {batch_idx}: {grad_norm:.4f}")
                    optimizer.step()
                    
            except RuntimeError as e:
                if "unscale_() has already been called" in str(e):
                    optimizer.zero_grad()  # Reset the optimizer state
                    if scaler is not None:
                        scaler.update()  # Update scaler state
                continue
            
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'grad_norm': f"{grad_norm:.4f}"
            })
            
            if batch_idx % config.log_every == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Grad norm: {grad_norm:.4f}")
                
        except Exception as e:
            # Log full traceback to help diagnose device/indexing issues
            logger.exception(f"Exception in batch {batch_idx} - skipping batch")
            # Log batch tensor devices for quick inspection
            try:
                keys = list(batch.keys())
                logger.info(f"Batch keys: {keys}")
                if 'input_ids' in batch:
                    logger.info(f"batch['input_ids'].device: {batch['input_ids'].device}")
                if 'target_ids' in batch:
                    logger.info(f"batch['target_ids'].device: {batch['target_ids'].device}")
            except Exception:
                logger.debug('Could not inspect batch devices')
            logger.error("Skipping problematic batch...")
    
    return total_loss / len(dataloader)

def main():
    logger.info("="*60)
    logger.info("Training TARS Core Reasoning")
    logger.info("="*60)
    
    # Load config
    config = TrainingConfig()
    
    # Check for resume checkpoint
    resume_path = os.environ.get('TARS_RESUME_CHECKPOINT')
    start_epoch = 0
    best_loss = float('inf')
    
    # Set device and enable CUDA optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = None
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
        scaler = torch.cuda.amp.GradScaler()
    logger.info(f"Using device: {device}")
    
    # Load datasets
    logger.info("\nLoading datasets...")
    from datasets import load_dataset
    
    squad = load_dataset('squad')['train']
    ropes = load_dataset('ropes')['train']
    
    # Create training dataset
    train_dataset = ReasoningDataset(squad, ropes)
    logger.info(f"Training examples: {len(train_dataset)}")
    
    # Create data loader with GPU-optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if device.type == 'cuda' else False
    )
    
    # Initialize model and move to GPU
    logger.info("\nInitializing TARS...")
    model = TARS(
        vocab_size=50000,
        dim=512,
        reasoning_depth=6,
        max_iterations=20,
        humor_setting=50,  # Neutral for training
        honesty_setting=100  # Maximum honesty for training
    )
    
    # Move model to device and log device info
    model = model.to(device)
    logger.info(f"Model moved to device: {next(model.parameters()).device}")
    
    # Load checkpoint if resuming
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch} with best loss: {best_loss:.4f}")

    # Create and attach a single dim_reduction layer to the model so it's included in optimizer
    if not hasattr(model, 'dim_reduction'):
        model.dim_reduction = nn.Linear(model.dim * 2, model.dim).to(device)
    
    # Enable mixed precision training if on GPU
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Enabled mixed precision training with CUDA")
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Optimizer with larger batch size for GPU
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    # Training loop with improved GPU support
    logger.info("\nStarting training...")
    best_loss = float('inf')
    last_loss = None  # Track last loss for final save
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # Allow overriding number of epochs via environment for quick tests
        epochs = int(os.environ.get('TARS_EPOCHS', config.num_epochs))
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            logger.info("="*40)
            
            # Train with GPU optimizations
            model.train()
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    train_loss = train_epoch(model, train_loader, optimizer, device, config, scaler)
            else:
                train_loss = train_epoch(model, train_loader, optimizer, device, config, None)
            
            last_loss = train_loss
            epoch_time = time.time() - epoch_start_time
            
            # Log detailed epoch statistics
            logger.info(f"Epoch {epoch + 1} Summary:")
            logger.info(f"  Average Loss: {train_loss:.4f}")
            logger.info(f"  Time: {epoch_time:.2f}s")
            logger.info(f"  Samples/sec: {len(train_loader) * config.batch_size / epoch_time:.2f}")
            
            # Save checkpoint if best so far
            if train_loss < best_loss:
                best_loss = train_loss
                checkpoint_path = os.path.join(checkpoint_dir, 
                    f'tars_epoch_{epoch+1}_loss_{train_loss:.4f}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'config': config,
                }, checkpoint_path)
                logger.info(f"✓ Saved best model checkpoint: {checkpoint_path}")
            
            # Regular checkpoint every N epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, 
                    f'tars_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'config': config,
                }, checkpoint_path)
                logger.info(f"✓ Saved periodic checkpoint: {checkpoint_path}")
            
            # Log GPU memory if available
            if device.type == 'cuda':
                max_memory = torch.cuda.max_memory_allocated(device) / 1024**2
                logger.info(f"  Max GPU memory: {max_memory:.1f}MB")
                torch.cuda.reset_peak_memory_stats()
                
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user!")
        
    except Exception as e:
        logger.error(f"\nTraining failed with error: {str(e)}")
        raise
        
    finally:
        # Always try to save the final model
        try:
            final_path = os.path.join(checkpoint_dir, 'tars_final.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'final_loss': last_loss if last_loss is not None else float('inf'),
                'best_loss': best_loss,
                'config': config,
            }, final_path)
            logger.info(f"\n✓ Saved final model to: {final_path}")
        except Exception as e:
            logger.error(f"Failed to save final model: {str(e)}")
    
    logger.info("\nTraining complete!")
    logger.info(f"Best loss achieved: {best_loss:.4f}")

if __name__ == "__main__":
    main()