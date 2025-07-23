"""
Neural network implementation for agent decision making.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, MUTATION_VARIANCE, POPULATION_SIZE, NUM_FOOD_POINTS


class AgentNeuralNetwork(nn.Module):
    """
    Feed-forward neural network for agent decision making.
    
    Input: agent's own state + information about other agents
    Output: four decisions (move_forward, rotate_right_pref, rotate_left_pref, rotation_gate)
    """
    
    def __init__(self):
        super(AgentNeuralNetwork, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        
        # Initialize weights randomly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with small random values."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output between 0 and 1
        return x
    
    def get_actions(self, input_tensor):
        """
        Get actions from network output with gated rotation control.
        
        Args:
            input_tensor: Input tensor for the network
            
        Returns:
            tuple: (should_move_forward, should_rotate_right, should_rotate_left)
        """
        with torch.no_grad():
            output = self.forward(input_tensor)
            
            # Output 0: Move forward (sigmoid activation, threshold at 0.5)
            should_move_forward = output[0].item() > 0.5
            
            # Output 3: Rotation gate (sigmoid activation, threshold at 0.5)
            rotation_enabled = output[3].item() > 0.5
            
            # Initialize rotation actions
            should_rotate_right = False
            should_rotate_left = False
            
            # Only process rotation if gate is enabled
            if rotation_enabled:
                # Outputs 1&2: Rotation preferences (softmax activation when gated)
                rotation_outputs = output[1:3]
                rotation_probs = F.softmax(rotation_outputs, dim=0)
                
                # Choose the rotation with the highest probability
                max_rotation_idx = torch.argmax(rotation_probs).item()
                should_rotate_right = max_rotation_idx == 0
                should_rotate_left = max_rotation_idx == 1
            
            return should_move_forward, should_rotate_right, should_rotate_left
    
    def mutate(self, variance=None):
        """
        Mutate the network weights by adding Gaussian noise.
        Used for creating offspring in the evolutionary algorithm.
        
        Args:
            variance: Mutation variance (uses MUTATION_VARIANCE if None)
        """
        if variance is None:
            variance = MUTATION_VARIANCE
            
        with torch.no_grad():
            for param in self.parameters():
                # Add Gaussian noise to each parameter in-place
                noise = torch.normal(0, variance, param.shape)
                param.data += noise
    
    def copy_weights_from(self, other_network):
        """
        Copy weights from another network.
        
        Args:
            other_network: Source network to copy weights from
        """
        with torch.no_grad():
            for self_param, other_param in zip(self.parameters(), other_network.parameters()):
                self_param.data.copy_(other_param.data)
    
    def get_weights_as_dict(self):
        """Get network weights as a dictionary for saving/loading."""
        return {name: param.clone() for name, param in self.named_parameters()}
    
    def set_weights_from_dict(self, weights_dict):
        """Set network weights from a dictionary."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights_dict:
                    param.data.copy_(weights_dict[name])


class AttentionAgentNeuralNetwork(nn.Module):
    """
    Attention-based neural network for agent decision making.
    
    Uses self-attention mechanism where the agent attends to other agents and food points
    as tokens, similar to transformer architecture.
    """
    
    def __init__(self, embed_dim=64, num_heads=4):
        super(AttentionAgentNeuralNetwork, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Token embedding layers
        self.self_embedding = nn.Linear(2, embed_dim)  # [food_count, energy]
        self.agent_embedding = nn.Linear(3, embed_dim)  # [distance, relative_angle, food_diff]
        self.food_embedding = nn.Linear(2, embed_dim)   # [distance, relative_angle]
        
        # Positional/type embeddings to distinguish token types
        self.self_type_embed = nn.Parameter(torch.randn(1, embed_dim))
        self.agent_type_embed = nn.Parameter(torch.randn(1, embed_dim))
        self.food_type_embed = nn.Parameter(torch.randn(1, embed_dim))
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Output projection to action space
        self.output_projection = nn.Linear(embed_dim, OUTPUT_SIZE)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with small random values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.zeros_(module.bias)
        
        # Initialize type embeddings
        nn.init.normal_(self.self_type_embed, mean=0.0, std=0.1)
        nn.init.normal_(self.agent_type_embed, mean=0.0, std=0.1)
        nn.init.normal_(self.food_type_embed, mean=0.0, std=0.1)
    
    def forward(self, tokens_dict, return_attention=False):
        """
        Forward pass through the attention network.
        
        Args:
            tokens_dict: Dictionary containing:
                - 'self': tensor of shape [2] (food_count, energy)
                - 'agents': tensor of shape [N, 3] (distance, relative_angle, food_diff)
                - 'food': tensor of shape [M, 2] (distance, relative_angle)
            return_attention: If True, also return attention weights
        
        Returns:
            torch.Tensor: Output tensor of shape [4] (action probabilities)
            torch.Tensor (optional): Attention weights if return_attention=True
        """
        # Embed tokens
        self_token = self.self_embedding(tokens_dict['self'].unsqueeze(0))  # [1, embed_dim]
        agent_tokens = self.agent_embedding(tokens_dict['agents'])  # [N, embed_dim]
        food_tokens = self.food_embedding(tokens_dict['food'])  # [M, embed_dim]
        
        # Add type embeddings
        self_token = self_token + self.self_type_embed
        agent_tokens = agent_tokens + self.agent_type_embed
        food_tokens = food_tokens + self.food_type_embed
        
        # Concatenate all tokens: [self, agents, food]
        all_tokens = torch.cat([self_token, agent_tokens, food_tokens], dim=0)  # [1+N+M, embed_dim]
        all_tokens = all_tokens.unsqueeze(0)  # Add batch dimension: [1, 1+N+M, embed_dim]
        
        # Self-attention (the self token attends to all tokens including itself)
        attended_tokens, attention_weights = self.attention(all_tokens, all_tokens, all_tokens)
        
        # Apply layer normalization
        attended_tokens = self.layer_norm(attended_tokens)
        
        # Extract the self token (first token) after attention
        self_attended = attended_tokens[0, 0, :]  # [embed_dim]
        
        # Project to action space
        output = self.output_projection(self_attended)  # [4]
        output = torch.sigmoid(output)  # Apply sigmoid activation
        
        if return_attention:
            # Return attention weights from the self token (first token) to all tokens
            # Shape: [1+N+M] where first is self-attention, next N are agent attentions, last M are food attentions
            self_attention_weights = attention_weights[0, 0, :]  # [1+N+M]
            return output, self_attention_weights
        
        return output
    
    def get_actions(self, tokens_dict):
        """
        Get actions from network output with gated rotation control.
        
        Args:
            tokens_dict: Dictionary containing token information
            
        Returns:
            tuple: (should_move_forward, should_rotate_right, should_rotate_left)
        """
        with torch.no_grad():
            output = self.forward(tokens_dict)
            
            # Output 0: Move forward (sigmoid activation, threshold at 0.5)
            should_move_forward = output[0].item() > 0.5
            
            # Output 3: Rotation gate (sigmoid activation, threshold at 0.5)
            rotation_enabled = output[3].item() > 0.5
            
            # Initialize rotation actions
            should_rotate_right = False
            should_rotate_left = False
            
            # Only process rotation if gate is enabled
            if rotation_enabled:
                # Outputs 1&2: Rotation preferences (softmax activation when gated)
                rotation_outputs = output[1:3]
                rotation_probs = F.softmax(rotation_outputs, dim=0)
                
                # Choose the rotation with the highest probability
                max_rotation_idx = torch.argmax(rotation_probs).item()
                should_rotate_right = max_rotation_idx == 0
                should_rotate_left = max_rotation_idx == 1
            
            return should_move_forward, should_rotate_right, should_rotate_left
    
    def mutate(self, variance=None):
        """
        Mutate the network weights by adding Gaussian noise.
        Used for creating offspring in the evolutionary algorithm.
        
        Args:
            variance: Mutation variance (uses MUTATION_VARIANCE if None)
        """
        if variance is None:
            variance = MUTATION_VARIANCE
            
        with torch.no_grad():
            for param in self.parameters():
                # Add Gaussian noise to each parameter in-place
                noise = torch.normal(0, variance, param.shape)
                param.data += noise
    
    def copy_weights_from(self, other_network):
        """
        Copy weights from another network.
        
        Args:
            other_network: Source network to copy weights from
        """
        with torch.no_grad():
            for self_param, other_param in zip(self.parameters(), other_network.parameters()):
                self_param.data.copy_(other_param.data)
    
    def get_weights_as_dict(self):
        """Get network weights as a dictionary for saving/loading."""
        return {name: param.clone() for name, param in self.named_parameters()}
    
    def set_weights_from_dict(self, weights_dict):
        """Set network weights from a dictionary."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights_dict:
                    param.data.copy_(weights_dict[name])


def create_offspring_network(parent_network, mutation_variance=None):
    """
    Create an offspring network by copying parent weights and mutating them.
    
    Args:
        parent_network: Parent network to create offspring from
        mutation_variance: Variance for mutation (uses default if None)
        
    Returns:
        AttentionAgentNeuralNetwork: New mutated offspring network
    """
    offspring = AttentionAgentNeuralNetwork()
    offspring.copy_weights_from(parent_network)
    offspring.mutate(mutation_variance)
    return offspring


def create_crossover_offspring(parent1_network, parent2_network):
    """
    Create an offspring network by crossing over two parent networks.
    Randomly exchanges weights between the two parents.
    
    Args:
        parent1_network: First parent network
        parent2_network: Second parent network
        
    Returns:
        AttentionAgentNeuralNetwork: New offspring network with crossed-over weights
    """
    import torch
    import random
    
    offspring = AttentionAgentNeuralNetwork()
    
    with torch.no_grad():
        # Iterate through all parameters and randomly choose from either parent
        for offspring_param, parent1_param, parent2_param in zip(
            offspring.parameters(), parent1_network.parameters(), parent2_network.parameters()
        ):
            # Create a random mask for crossover
            mask = torch.rand_like(offspring_param) > 0.5
            
            # Use mask to select weights from either parent1 or parent2
            offspring_param.data = torch.where(mask, parent1_param.data, parent2_param.data)
    
    # Note: Mutation is applied separately with configurable probability
    return offspring
