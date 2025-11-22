"""Complete TEM (Tolman-Eichenbaum Machine) Implementation"""

import torch
import torch.nn as nn
from .tem_sensory import TEMSensoryModule
from .tem_relational import TEMRelationalMemory
from .tem_inference import TEMInferenceModule


class TolmanEichenbaumMachine(nn.Module):
    """
    Full TEM implementation for relational reasoning.
    
    Key properties:
    1. Factorized representations (compositional)
    2. Relational memory (graph structure)
    3. Transitive inference (multi-hop reasoning)
    4. NO factual knowledge in weights - only relational structure!
    """
    
    def __init__(
        self,
        input_dim=512,
        num_factors=4,
        factor_dim=128,
        num_relation_types=10,
        num_gnn_layers=3,
        num_reasoning_steps=5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_factors = num_factors
        self.factor_dim = factor_dim
        
        # Three main components
        self.sensory = TEMSensoryModule(input_dim, num_factors, factor_dim)
        
        self.relational_memory = TEMRelationalMemory(
            factor_dim * num_factors,  # Flattened factors
            num_relation_types,
            num_gnn_layers
        )
        
        self.inference = TEMInferenceModule(
            factor_dim * num_factors,
            num_reasoning_steps
        )
        
    def forward(self, entities, known_relations=None, query=None):
        """
        Args:
            entities: [batch, num_entities, input_dim] - raw entity representations
            known_relations: [batch, num_edges, 3] - known [entity_i, relation, entity_j]
            query: Optional dict with 'entity' and 'relation' for inference
        
        Returns:
            If query provided: answer_scores and reasoning_trace
            Otherwise: updated entity embeddings and inferred relations
        """
        # 1. Factorize entities
        factorized = self.sensory(entities)
        
        # Flatten factors for processing
        batch, num_entities, num_factors, dim = factorized.shape
        factorized_flat = factorized.reshape(batch, num_entities, -1)
        
        # 2. Relational memory processing
        updated_embeddings, inferred_relations = self.relational_memory(
            factorized_flat,
            known_relations
        )
        
        # 3. If query provided, perform inference
        if query is not None:
            answer_scores, reasoning_trace = self.inference(
                updated_embeddings,
                query['entity'],
                query['relation']
            )
            return answer_scores, reasoning_trace
        
        return updated_embeddings, inferred_relations
    
    def extract_relations(self, entities, known_relations=None):
        """
        Extract all inferred relations between entities.
        Useful for building knowledge graphs.
        """
        _, inferred_relations = self.forward(entities, known_relations)
        return inferred_relations