"""Tool Controller - Decides when and how to use external tools"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToolUseController(nn.Module):
    """
    Decides when and how to use external tools.
    No knowledge in weights - just the strategy of tool use.
    """
    
    def __init__(self, dim=512, num_tools=5):
        super().__init__()
        
        self.dim = dim
        self.num_tools = num_tools
        
        # Tool selector
        self.tool_selector = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_tools)  # [web_search, rag, calculator, code_exec, none]
        )
        
        # Query formatter for each tool
        self.web_query_formatter = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, dim)
        )
        
        self.rag_query_formatter = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, dim)
        )
        
        # Retrieval gate (should we retrieve more info?)
        self.retrieval_gate = nn.Sequential(
            nn.Linear(dim * 2, 128),  # thought + confidence
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, reasoning_state, confidence):
        """
        Decide if we need external information and which tool to use.
        
        Args:
            reasoning_state: [batch, dim]
            confidence: [batch, 1]
        
        Returns:
            dict with should_retrieve, tool_probs, formatted queries
        """
        
        # Ensure proper shapes
        if confidence.dim() == 3:
            confidence = confidence.squeeze(1)
        if confidence.dim() == 1:
            confidence = confidence.unsqueeze(1)
        
        # Should we retrieve?
        gate_input = torch.cat([reasoning_state, confidence], dim=-1)
        should_retrieve = self.retrieval_gate(gate_input)
        
        # Which tool?
        tool_logits = self.tool_selector(reasoning_state)
        tool_probs = F.softmax(tool_logits, dim=-1)
        
        # Format queries for tools
        web_query = self.web_query_formatter(reasoning_state)
        rag_query = self.rag_query_formatter(reasoning_state)
        
        return {
            'should_retrieve': should_retrieve,
            'tool_probs': tool_probs,
            'web_query': web_query,
            'rag_query': rag_query
        }