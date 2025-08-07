# Attention-Based Neural Network Architecture

This document describes the new attention-based neural network architecture implemented for the evolutionary agent simulation.

## Overview

The neural network architecture has been changed from a traditional Multi-Layer Perceptron (MLP) to an attention-based architecture inspired by transformer models. This allows agents to dynamically attend to relevant information from other agents and food points.

## Architecture Components

### 1. Token-Based Input Representation

Instead of a flattened input vector, the new architecture uses structured tokens:

#### Self Token
- **Dimensions**: 2D vector
- **Content**: `[food_count, energy]`
- **Purpose**: Represents the agent's own state

#### Agent Tokens
- **Dimensions**: N × 3 tensor (where N = POPULATION_SIZE - 1)
- **Content**: `[distance, relative_angle, food_difference]` for each other agent
- **Purpose**: Represents information about other agents in the environment
- **Sorting**: Sorted by distance (closest first)
- **Padding**: Zero-padded to maintain consistent tensor shapes

#### Food Tokens
- **Dimensions**: M × 2 tensor (where M = NUM_FOOD_POINTS)
- **Content**: `[distance, relative_angle]` for each food point
- **Purpose**: Represents information about food points in the environment
- **Sorting**: Sorted by distance (closest first)
- **Padding**: Zero-padded to maintain consistent tensor shapes

### 2. Embedding Layers

Each token type has its own embedding layer to project to a common embedding dimension:

```python
self.self_embedding = nn.Linear(2, embed_dim)    # Self token embedding
self.agent_embedding = nn.Linear(3, embed_dim)   # Agent token embedding
self.food_embedding = nn.Linear(2, embed_dim)    # Food token embedding
```

**Default embedding dimension**: 64

### 3. Type Embeddings

Learnable type embeddings distinguish between different token types:

```python
self.self_type_embed = nn.Parameter(torch.randn(1, embed_dim))
self.agent_type_embed = nn.Parameter(torch.randn(1, embed_dim))
self.food_type_embed = nn.Parameter(torch.randn(1, embed_dim))
```

### 4. Multi-Head Self-Attention

The core attention mechanism:
- **Number of heads**: 4 (configurable)
- **Mechanism**: Self-attention where all tokens attend to all tokens
- **Focus**: The self token's attended representation is used for action prediction

### 5. Layer Normalization

Applied after attention to stabilize training:
```python
self.layer_norm = nn.LayerNorm(embed_dim)
```

### 6. Output Projection

Projects the attended self token to action space:
```python
self.output_projection = nn.Linear(embed_dim, OUTPUT_SIZE)  # OUTPUT_SIZE = 4
```

## Forward Pass Flow

1. **Token Embedding**: Each token type is embedded to common dimension
2. **Type Addition**: Type embeddings are added to distinguish token types
3. **Token Concatenation**: All tokens are concatenated: [self, agents, food]
4. **Self-Attention**: Multi-head attention processes all tokens
5. **Layer Normalization**: Normalizes the attended representations
6. **Self Token Extraction**: Extracts the attended self token (first token)
7. **Action Projection**: Projects to 4D action space with sigmoid activation

## Action Interpretation

The output interpretation remains the same as the original MLP:

- **Output[0]**: Move forward (threshold > 0.5)
- **Output[1]**: Rotate right preference
- **Output[2]**: Rotate left preference  
- **Output[3]**: Rotation gate (threshold > 0.5)

Rotation is only processed if the gate is enabled, then softmax is applied to outputs[1:3] to choose direction.

## Key Advantages

### 1. **Dynamic Attention**
- Agents can dynamically focus on relevant other agents and food points
- Attention weights provide interpretability of decision-making

### 2. **Permutation Invariance**
- Order of agents/food doesn't matter (sorted by distance for consistency)
- More robust to varying numbers of entities

### 3. **Scalability**
- Can handle variable numbers of agents and food points
- Attention mechanism scales better than fully connected layers

### 4. **Biological Plausibility**
- Attention-like mechanisms exist in biological neural networks
- More similar to how animals might process environmental information

### 5. **Interpretability**
- Attention weights show what the agent is "looking at"
- Can analyze which agents/food points influence decisions

## Implementation Details

### Token Generation (Agent.get_perception_tokens)

```python
def get_perception_tokens(self, other_agents, food_points):
    # Self token: [food_count, energy]
    self_token = torch.tensor([self.food_count, self.energy], dtype=torch.float32)
    
    # Agent tokens: [distance, relative_angle, food_diff]
    agent_tokens_list = []
    for other_agent in other_agents:
        if other_agent.alive and other_agent != self:
            # Calculate distance, angle, food difference
            # Sort by distance, pad to fixed size
    
    # Food tokens: [distance, relative_angle]  
    food_tokens_list = []
    for food_point in food_points:
        # Calculate distance and angle
        # Sort by distance, pad to fixed size
    
    return {
        'self': self_token,
        'agents': agent_tokens,
        'food': food_tokens
    }
```

### Network Architecture (AttentionAgentNeuralNetwork)

```python
class AttentionAgentNeuralNetwork(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        # Token embeddings
        self.self_embedding = nn.Linear(2, embed_dim)
        self.agent_embedding = nn.Linear(3, embed_dim)
        self.food_embedding = nn.Linear(2, embed_dim)
        
        # Type embeddings
        self.self_type_embed = nn.Parameter(torch.randn(1, embed_dim))
        self.agent_type_embed = nn.Parameter(torch.randn(1, embed_dim))
        self.food_type_embed = nn.Parameter(torch.randn(1, embed_dim))
        
        # Attention and output
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, OUTPUT_SIZE)
```

## Evolutionary Compatibility

The new architecture maintains full compatibility with the existing evolutionary system:

- **Mutation**: Gaussian noise added to all parameters
- **Crossover**: Random weight exchange between parent networks
- **Saving/Loading**: Complete state dict serialization
- **Offspring Creation**: Same interface as original MLP

## Configuration Parameters

### Neural Network Parameters
- `embed_dim`: Embedding dimension (default: 64)
- `num_heads`: Number of attention heads (default: 4)
- `OUTPUT_SIZE`: Action space dimension (4)

### Token Sizes
- Self token: 2D (food_count, energy)
- Agent tokens: (POPULATION_SIZE-1) × 3
- Food tokens: NUM_FOOD_POINTS × 2

## Performance Considerations

### Computational Complexity
- **Attention**: O(n²) where n = total number of tokens
- **Total tokens**: 1 + (POPULATION_SIZE-1) + NUM_FOOD_POINTS
- **Memory**: Linear in embedding dimension and number of tokens

### Optimization
- Tokens are sorted by distance for consistency
- Zero-padding maintains tensor shapes for efficient computation
- Batch dimension added for PyTorch MultiheadAttention compatibility

## Migration from MLP

The transition from MLP to attention architecture involves:

1. **Agent class**: Uses `AttentionAgentNeuralNetwork` instead of `AgentNeuralNetwork`
2. **Input method**: `get_perception_tokens()` instead of `get_perception_input()`
3. **Action method**: `get_actions(tokens_dict)` instead of `get_actions(input_tensor)`
4. **Offspring functions**: Updated to create `AttentionAgentNeuralNetwork` instances

## Future Extensions

### Possible Enhancements
1. **Multi-layer attention**: Stack multiple attention layers
2. **Positional encoding**: Add learned positional information
3. **Cross-attention**: Separate query/key/value for different token types
4. **Attention visualization**: Tools to visualize attention patterns
5. **Adaptive embedding**: Dynamic embedding dimensions based on input complexity

### Research Opportunities
1. **Attention pattern analysis**: Study what agents learn to attend to
2. **Emergent communication**: Analyze if attention patterns enable implicit communication
3. **Hierarchical attention**: Different attention heads for different purposes
4. **Attention regularization**: Encourage diverse attention patterns

## Conclusion

The attention-based architecture provides a more flexible and interpretable approach to agent decision-making while maintaining compatibility with the existing evolutionary framework. The token-based representation and attention mechanism should enable more sophisticated behavioral strategies to emerge through evolution.
