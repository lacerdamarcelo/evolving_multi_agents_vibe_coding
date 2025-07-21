"""
Neural network implementation for agent decision making.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, MUTATION_VARIANCE


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
    
    def mutate(self):
        """
        Mutate the network weights by adding Gaussian noise.
        Used for creating offspring in the evolutionary algorithm.
        """
        with torch.no_grad():
            for param in self.parameters():
                # Add Gaussian noise to each parameter in-place
                noise = torch.normal(0, MUTATION_VARIANCE, param.shape)
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


def create_offspring_network(parent_network):
    """
    Create an offspring network by copying parent weights and mutating them.
    
    Args:
        parent_network: Parent network to create offspring from
        
    Returns:
        AgentNeuralNetwork: New mutated offspring network
    """
    offspring = AgentNeuralNetwork()
    offspring.copy_weights_from(parent_network)
    offspring.mutate()
    return offspring


def create_crossover_offspring(parent1_network, parent2_network):
    """
    Create an offspring network by crossing over two parent networks.
    Randomly exchanges weights between the two parents.
    
    Args:
        parent1_network: First parent network
        parent2_network: Second parent network
        
    Returns:
        AgentNeuralNetwork: New offspring network with crossed-over weights
    """
    import torch
    import random
    
    offspring = AgentNeuralNetwork()
    
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
