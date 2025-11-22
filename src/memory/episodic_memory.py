"""Episodic Memory - Long-term storage of successful strategies"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicMemory(nn.Module):
    """
    Long-term memory of successful strategies and past experiences.
    
    Different from Working Memory (Titans):
    - Working Memory: Temporary, cleared after task
    - Episodic Memory: Permanent, accumulates over time
    """
    
    def __init__(self, dim=512, memory_capacity=10000):
        super().__init__()
        
        self.dim = dim
        self.capacity = memory_capacity
        
        # Memory storage buffers
        self.register_buffer('episode_keys', torch.zeros(memory_capacity, dim))
        self.register_buffer('episode_values', torch.zeros(memory_capacity, dim))
        self.register_buffer('episode_rewards', torch.zeros(memory_capacity))
        self.register_buffer('episode_count', torch.tensor(0))
        
        # Memory encoder
        self.memory_encoder = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),  # goal + plan + outcome
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        
        # Retrieval module
        self.retrieval_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Relevance scorer
        self.relevance_scorer = nn.Sequential(
            nn.Linear(dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def store_episode(self, goal, plan, outcome, reward):
        """
        Store successful episode for future retrieval.
        
        Args:
            goal: Goal embedding [batch, dim]
            plan: List of action dicts with 'embedding'
            outcome: Final state [batch, dim]
            reward: Success score (0-1)
        """
        
        # Average plan embeddings
        plan_embedding = torch.stack([s['embedding'] for s in plan], dim=0).mean(dim=0)
        
        # Encode episode
        episode_key = self.memory_encoder(
            torch.cat([goal, plan_embedding, outcome], dim=-1)
        )
        
        # Store in circular buffer
        idx = int(self.episode_count.item() % self.capacity)
        self.episode_keys[idx] = episode_key.detach().squeeze()
        self.episode_values[idx] = plan_embedding.detach().squeeze()
        self.episode_rewards[idx] = reward
        
        self.episode_count += 1
    
    def retrieve_similar_episodes(self, current_goal, top_k=5):
        """
        Retrieve similar past episodes.
        
        Args:
            current_goal: Current goal embedding [batch, dim]
            top_k: Number of episodes to retrieve
        
        Returns:
            dict with retrieved plans, rewards, relevance scores
        """
        
        num_episodes = min(int(self.episode_count.item()), self.capacity)
        
        if num_episodes == 0:
            return None
        
        # Attention-based retrieval
        retrieved, attn_weights = self.retrieval_attn(
            current_goal.unsqueeze(0).unsqueeze(0),
            self.episode_keys[:num_episodes].unsqueeze(0),
            self.episode_values[:num_episodes].unsqueeze(0)
        )
        
        # Score relevance of each episode
        relevance = []
        for i in range(min(top_k, num_episodes)):
            score = self.relevance_scorer(
                torch.cat([current_goal, self.episode_keys[i]], dim=-1)
            )
            relevance.append(score.item())
        
        # Get top-k
        if relevance:
            relevance_tensor = torch.tensor(relevance)
            top_indices = relevance_tensor.argsort(descending=True)[:top_k]
            
            return {
                'plans': self.episode_values[top_indices],
                'rewards': self.episode_rewards[top_indices],
                'relevance': relevance_tensor[top_indices]
            }
        
        return None
    
    def clear_all(self):
        """Clear all stored episodes"""
        self.episode_keys.zero_()
        self.episode_values.zero_()
        self.episode_rewards.zero_()
        self.episode_count.zero_()