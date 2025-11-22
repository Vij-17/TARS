"""Model configuration"""

class ModelConfig:
    """Configuration for the complete agentic reasoning engine"""
    
    # Model dimensions
    vocab_size = 50000
    dim = 512
    
    # Query Understanding
    query_understanding = {
        'depth': 4,
        'nhead': 8,
        'dim_feedforward': 2048
    }
    
    # TEM Configuration
    tem_config = {
        'input_dim': 512,
        'num_factors': 4,
        'factor_dim': 128,
        'num_relation_types': 10,
        'num_gnn_layers': 3,
        'num_reasoning_steps': 5
    }
    
    # Recursive Reasoner
    recursive_config = {
        'dim': 512,
        'reasoning_depth': 6,
        'num_heads': 8
    }
    
    # Working Memory (Titans)
    working_memory_config = {
        'dim': 512,
        'chunk_size': 16,
        'dim_head': 64,
        'heads': 8,
        'momentum': True,
        'batch_size': 32,
        'default_step_transform_max_lr': 0.1
    }
    
    # Episodic Memory
    episodic_memory_config = {
        'dim': 512,
        'memory_capacity': 10000
    }
    
    # Tool Controller
    tool_config = {
        'dim': 512,
        'num_tools': 5
    }
    
    # Agentic Components
    goal_understanding_config = {
        'dim': 512,
        'depth': 3,
        'max_subgoals': 10
    }
    
    planning_config = {
        'dim': 512,
        'max_plan_steps': 20,
        'num_action_types': 10
    }
    
    action_executor_config = {
        'dim': 512,
        'max_retries': 3
    }
    
    state_tracker_config = {
        'dim': 512,
        'num_layers': 3
    }
    
    reflection_config = {
        'dim': 512,
        'depth': 2
    }
    
    # Response Generator
    response_config = {
        'dim': 512,
        'num_layers': 4,
        'nhead': 8
    }
    
    # Meta-Learning
    meta_learning_config = {
        'dim': 512,
        'adaptation_rate': 0.01
    }