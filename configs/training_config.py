"""Training configuration"""

class TrainingConfig:
    """Training hyperparameters"""
    
    # General
    device = 'cuda'
    seed = 42
    num_epochs = 50  # Updated to 50 epochs
    
    # Training
    batch_size = 4  # Reduced for memory efficiency
    learning_rate = 2e-4
    weight_decay = 0.01
    max_grad_norm = 1.0
    
    # Optimization
    optimizer = 'adamw'
    scheduler = 'cosine'
    warmup_steps = 1000
    
    # Mixed precision
    use_amp = True
    gradient_accumulation_steps = 4
    
    # Checkpointing
    save_every = 1000
    eval_every = 500
    log_every = 100
    
    # Early stopping
    patience = 5
    min_delta = 1e-4
    
    # Phase 1: Core Reasoning (Week 1)
    phase1_config = {
        'num_epochs': 10,
        'datasets': ['bAbI', 'LogiQA', 'ARC'],
        'focus': 'TEM + Recursive Reasoner'
    }
    
    # Phase 2: Tool Use (Week 2)
    phase2_config = {
        'num_epochs': 10,
        'datasets': ['WebShop', 'ALFWorld'],
        'focus': 'Tool Controller + Action Executor'
    }
    
    # Phase 3: Agentic Training (Week 3)
    phase3_config = {
        'num_epochs': 15,
        'datasets': ['MiniWoB', 'BabyAI'],
        'focus': 'Planning + Reflection'
    }
    
    # Phase 4: Integration (Week 4)
    phase4_config = {
        'num_epochs': 20,
        'datasets': ['Mixed'],
        'focus': 'End-to-end RL'
    }
    
    # Reinforcement Learning
    rl_config = {
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_episode_length': 50
    }