"""TEM Relational Memory - Learns relationship structures using GNNs"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TEMRelationalMemory(nn.Module):
    """
    Core TEM component: Stores relationships as a graph structure.
    
    Key innovation: Uses graph neural network to learn relational structure
    WITHOUT memorizing specific facts.
    
    Example:
    - Learns: parent_of + parent_of = grandparent_of
    - NOT: "Alice is parent of Bob"
    """
    
    def __init__(self, factor_dim=128, num_relation_types=10, num_gnn_layers=3):
        super().__init__()
        
        self.factor_dim = factor_dim
        self.num_relation_types = num_relation_types
        
        # Relation type embeddings (abstract relation types)
        self.relation_embeddings = nn.Embedding(num_relation_types, factor_dim)
        
        # Graph neural network layers for message passing
        self.gnn_layers = nn.ModuleList([
            GATConv(
                in_channels=factor_dim,
                out_channels=factor_dim,
                heads=4,
                concat=False,
                edge_dim=factor_dim,
                dropout=0.1
            ) for _ in range(num_gnn_layers)
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(factor_dim) for _ in range(num_gnn_layers)
        ])
        
        # Relation predictor (predict if relation exists between entities)
        self.relation_predictor = nn.Sequential(
            nn.Linear(factor_dim * 2, factor_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(factor_dim, num_relation_types)
        )
        
    def forward(self, entity_embeddings, known_relations=None):
        """
        Args:
            entity_embeddings: [batch, num_entities, factor_dim]
            known_relations: Optional [batch, num_edges, 3] where each edge is
                           [entity_i_idx, relation_type, entity_j_idx]
        
        Returns:
            updated_embeddings: [batch, num_entities, factor_dim]
            inferred_relations: [batch, num_pairs, num_relation_types]
        """
        batch_size, num_entities, dim = entity_embeddings.shape
        
        # Build graph
        if known_relations is None:
            # Sparse k-nearest neighbor graph for memory efficiency
            edge_index = self._build_fully_connected_graph(num_entities, batch_size, device=entity_embeddings.device)
            edge_attr = None
        else:
            # Use provided relations
            edge_index = known_relations[:, :, [0, 2]].long()
            edge_type = known_relations[:, :, 1].long()
            edge_attr = self.relation_embeddings(edge_type)
        
        # Flatten batch dimension for GNN processing
        x = entity_embeddings.reshape(-1, dim)
        
        # Message passing through GNN layers
        for gnn_layer, norm in zip(self.gnn_layers, self.norms):
            residual = x
            x = gnn_layer(x, edge_index.reshape(2, -1), edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = x + residual  # Residual connection
        
        # Reshape back to batch
        updated_embeddings = x.reshape(batch_size, num_entities, dim)
        
        # Predict relations between all pairs
        inferred_relations = self._infer_all_relations(updated_embeddings)
        
        return updated_embeddings, inferred_relations
    
    def _build_fully_connected_graph(self, num_nodes, batch_size, device='cpu', k=32):
        """Create k-nearest neighbor graph for each batch to reduce memory"""
        edges = []
        for b in range(batch_size):
            offset = b * num_nodes
            for i in range(num_nodes):
                # Connect to k nearest neighbors (by position)
                for j in range(max(0, i - k//2), min(num_nodes, i + k//2 + 1)):
                    if i != j:
                        edges.append([offset + i, offset + j])
        
        if not edges:
            # Return empty edge index if no edges
            return torch.zeros(2, 0, dtype=torch.long, device=device)
        
        return torch.tensor(edges, dtype=torch.long, device=device).T
    
    def _infer_all_relations(self, embeddings):
        """
        Predict relations between all entity pairs.
        This is where transitive inference happens!
        """
        batch_size, num_entities, dim = embeddings.shape
        
        # Create all pairs
        pairs = []
        for i in range(num_entities):
            for j in range(num_entities):
                if i != j:
                    pair = torch.cat([embeddings[:, i], embeddings[:, j]], dim=-1)
                    pairs.append(pair)
        
        if not pairs:
            return torch.zeros(batch_size, 0, self.num_relation_types, device=embeddings.device)
        
        pairs = torch.stack(pairs, dim=1)  # [batch, num_pairs, dim*2]
        
        # Predict relation types for all pairs
        relation_logits = self.relation_predictor(pairs)
        
        return relation_logits