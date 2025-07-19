# Evolutionary Agent Simulation

An evolutionary simulation featuring 100 neural network-controlled agents that compete for survival in a 2D environment.

## Overview

This simulation implements an evolutionary algorithm where agents with simple neural networks learn to survive by:
- Moving and rotating in a 2D space
- Consuming food to gain energy
- Competing with other agents
- Evolving over 1000 generations

## Features

- **Neural Network Decision Making**: Each agent uses a PyTorch neural network to decide actions
- **Bidirectional Rotation**: Agents can now rotate both left and right using softmax selection
- **Evolutionary Algorithm**: Top 10% of survivors reproduce with mutations
- **Multi-Run Agent Evaluation**: Agents can be evaluated over multiple runs for more reliable performance assessment
- **Real-time Visualization**: Arcade-based graphics showing agents, food, and statistics
- **Configurable Parameters**: Easy to modify simulation parameters

## Requirements

- Python 3.8+
- PyTorch
- Arcade

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the simulation:
```bash
python main.py
```

## Controls

- **SPACE**: Pause/Resume simulation
- **R**: Restart simulation
- **ESC**: Exit

## Simulation Parameters

Key parameters can be modified in `config.py`:

- `POPULATION_SIZE`: Number of agents (default: 100)
- `NUM_GENERATIONS`: Number of evolutionary generations (default: 1000)
- `ITERATIONS_PER_GENERATION`: Steps per generation (default: 1000)
- `AGENT_RADIUS`: Size of agents (default: 5)
- `MOVE_DISTANCE`: Distance moved per step (default: 5)
- `ROTATION_ANGLE`: Rotation angle in degrees (default: 10)
- `INITIAL_ENERGY`: Starting energy (default: 100)
- `ENERGY_FROM_FOOD`: Energy gained from food (default: 10)
- `NUM_FOOD_POINTS`: Number of food points (default: 50)

## Agent Behavior

### Neural Network Architecture
- **Input**: Agent's food count, energy, information about other agents, and food point locations
  - **Own State**: [food_count, energy] (2 values)
  - **Other Agents**: [distance, relative_angle, relative_food_count] × 49 agents (147 values)
  - **Food Points**: [distance, relative_angle] × 20 food points (40 values)
  - **Total Input Size**: 189 values
- **Hidden Layers**: 2 layers with ReLU activation
- **Output**: 4 decisions (move forward, rotate right preference, rotate left preference, rotation gate)
  - **Move Forward**: Sigmoid activation with 0.5 threshold
  - **Rotation Preferences**: Raw sigmoid outputs for left/right preferences
  - **Rotation Gate**: Sigmoid activation that enables/disables rotation
  - **Gated Rotation Logic**: When gate is enabled, softmax is applied to preferences to select direction

### Energy System
- Agents lose 1 energy per iteration when idle
- Agents lose 2 energy per iteration when taking actions
- Agents die when energy reaches 0
- Food consumption restores 10 energy points

### Agent Interactions
- When agents collide, the one with more food points survives
- Winner takes all food points from the loser
- In case of tie, both agents die

### Evolution
- Each generation runs for 1000 iterations
- Top 10% of survivors (by food count) are selected for reproduction
- Offspring are created by copying parent neural networks and adding Gaussian noise
- Population is maintained at 100 agents

## Visualization

- **Blue circles**: Living agents
- **Red lines**: Agent orientation
- **Red dots**: Food points
- **Numbers**: Food count above each agent
- **UI**: Generation, iteration, and statistics display

## Gated Rotation System

The simulation features an advanced gated rotation system that allows agents to selectively control when to rotate, providing more sophisticated and energy-efficient movement capabilities.

### How It Works

- **Four Neural Network Outputs**: 
  1. Move forward (sigmoid activation)
  2. Rotate right preference (sigmoid activation)
  3. Rotate left preference (sigmoid activation)
  4. Rotation gate (sigmoid activation - enables/disables rotation)

- **Gated Control**: The fourth output acts as a gate that enables or disables rotation
- **Selective Rotation**: Agents can choose NOT to rotate when it's unnecessary
- **Competitive Selection**: When rotation is enabled, softmax is applied to preferences to select direction
- **Independent Movement**: Forward movement remains independent and can occur with or without rotation

### Gating Logic

```python
if rotation_gate > 0.5:  # Gate enabled
    # Apply softmax to rotation preferences
    rotation_probs = softmax([right_pref, left_pref])
    # Select direction with highest probability
else:  # Gate disabled
    # No rotation regardless of preferences
    rotate_right = False
    rotate_left = False
```

### Benefits

1. **Selective Control**: Agents can choose when rotation is beneficial vs unnecessary
2. **Energy Efficiency**: Avoid wasteful rotation actions when moving straight is optimal
3. **Realistic Behavior**: Mimics real-world decision making about when to change direction
4. **Enhanced Strategy**: Enables more sophisticated movement patterns and energy management
5. **Better Evolution**: More nuanced control leads to richer behavioral evolution

### Demonstrations

Run the gated rotation demonstration:

```bash
python gated_rotation_demo.py
```

This will show:
- Four-output neural network architecture
- Gate enabling/disabling rotation control
- Comparison of gated vs ungated behavior
- Integration with full environment simulation

Run the original rotation demonstration:

```bash
python rotation_demo.py
```

This shows the basic bidirectional rotation without gating.

## Food Perception System

The simulation features an advanced food perception system that provides agents with direct sensory input about food locations, enabling more sophisticated food-seeking behaviors.

### How It Works

- **Direct Food Awareness**: Agents receive distance and angle information for all food points in the environment
- **Sorted by Proximity**: Food points are automatically sorted by distance (closest first) for optimal decision making
- **Consistent Input Format**: Uses the same distance/angle format as agent perception for consistency
- **Fixed Input Size**: Padded to ensure consistent neural network input dimensions

### Input Structure

```
Neural Network Input (189 values total):
├── Own State (2 values)
│   ├── food_count
│   └── energy
├── Other Agents (147 values = 49 agents × 3 values each)
│   ├── distance
│   ├── relative_angle  
│   └── food_count
└── Food Points (40 values = 20 food points × 2 values each)
    ├── distance
    └── relative_angle
```

### Benefits

1. **Enhanced Navigation**: Agents can now "see" food locations directly instead of wandering randomly
2. **Strategic Planning**: Can evaluate multiple food options and choose optimal targets
3. **Competitive Advantage**: Better food-seeking strategies emerge through evolution
4. **Realistic Behavior**: Mimics how real organisms use sensory input to locate resources
5. **Improved Evolution**: Selection pressure for effective food-seeking behaviors

### Demonstration

Run the food perception demonstration:

```bash
python food_perception_demo.py
```

This will show:
- Complete input structure breakdown
- Food perception accuracy verification
- Decision making with food awareness
- Evolution impact analysis

## Competitive Awareness System

The simulation features an advanced competitive awareness system where agents perceive other agents' food counts relative to their own, enabling strategic decision making and threat assessment.

### How It Works

- **Relative Food Count**: Instead of absolute food counts, agents see the difference between their food count and other agents' food counts
- **Competitive Intelligence**: Agents can assess whether they are winning or losing compared to nearby competitors
- **Strategic Positioning**: Enables agents to make informed decisions about engagement vs avoidance

### Relative Food Count Calculation

```python
relative_food_count = self.food_count - other_agent.food_count
```

- **Positive Values**: Agent has MORE food than the other (competitive advantage)
- **Negative Values**: Agent has LESS food than the other (competitive disadvantage)  
- **Zero Values**: Agents have EQUAL food counts (neutral position)

### Benefits

1. **Threat Assessment**: Agents can identify which nearby agents pose a threat vs which are vulnerable
2. **Strategic Decision Making**: Decisions based on competitive position rather than absolute values
3. **Realistic Competition**: Mimics how real organisms assess relative fitness and threats
4. **Enhanced Evolution**: Selection pressure for competitive awareness and strategic thinking
5. **Dynamic Strategies**: Behavior can adapt based on current competitive landscape

### Competitive Strategies

The relative food count system enables several strategic behaviors:

- **Dominance**: Agents with advantages can pursue aggressive strategies
- **Avoidance**: Disadvantaged agents can avoid stronger competitors
- **Opportunism**: Agents can target weaker competitors while avoiding stronger ones
- **Risk Assessment**: Decisions based on potential gains vs losses in confrontations

### Demonstration

Run the competitive awareness demonstration:

```bash
python relative_food_demo.py
```

This will show:
- Relative food count calculations from different perspectives
- Competitive position analysis (dominant vs vulnerable)
- Strategic implications of competitive awareness
- Evolution impact with competitive intelligence

## File Structure

- `main.py`: Main simulation loop and Arcade visualization
- `agent.py`: Agent class with neural network decision making
- `neural_network.py`: PyTorch neural network implementation
- `environment.py`: Environment management and collision detection
- `evolution.py`: Evolutionary algorithm implementation
- `config.py`: Configuration parameters
- `relative_food_demo.py`: Demonstration of competitive awareness system
- `food_perception_demo.py`: Demonstration of food perception system
- `gated_rotation_demo.py`: Demonstration of gated rotation system
- `rotation_demo.py`: Demonstration of basic bidirectional rotation
- `requirements.txt`: Python dependencies

## Customization

The simulation is highly modular and can be easily customized:

1. **Modify agent behavior**: Edit the neural network architecture in `neural_network.py`
2. **Change environment rules**: Modify collision detection and energy systems in `environment.py`
3. **Adjust evolution parameters**: Update selection and mutation in `evolution.py`
4. **Customize visualization**: Modify colors and display in `main.py`

## Multi-Run Agent Evaluation

The simulation includes an advanced competitive evaluation system that assesses agents in realistic multi-agent scenarios to provide more accurate performance metrics.

### How It Works

The competitive evaluation system efficiently runs full population scenarios where all agents compete together:

1. **Efficient Competitive Scenarios**: All agents compete together in complete simulations (default: 30 runs)
2. **Single-Pass Evaluation**: Each scenario evaluates ALL agents simultaneously, not individually
3. **Realistic Assessment**: Agents face competition for resources, agent-to-agent interactions, and survival conflicts
4. **Batch Performance Tracking**: Individual performance is tracked for all agents within each competitive environment
5. **Optimized Averaging**: Performance metrics are calculated across multiple competitive scenarios:
   - Average food count in competitive scenarios
   - Average final energy in competitive environments
   - Survival rate against competing agents
   - Standard deviation of competitive performance
6. **Competitive Selection**: Top performers are selected based on their ability to succeed against peers

**Efficiency**: Requires only R simulations (R competitive scenarios) instead of N×R simulations (N agents × R runs), making it significantly faster than individual evaluation while providing more realistic assessment.

### Configuration Parameters

Add these parameters to `config.py`:

```python
# Multi-run evaluation parameters
NUM_EVALUATION_RUNS = 10  # Number of runs to average performance over
EVALUATION_ITERATIONS = 100  # Number of iterations per evaluation run
```

### Usage

#### Basic Usage
```python
from evolution import EvolutionManager
from environment import Environment

evolution_manager = EvolutionManager()
environment = Environment()

# Use multi-run evaluation instead of single-run
new_agents, stats, eval_summary = evolution_manager.evolve_generation_with_evaluation(
    environment.agents,
    num_runs=10,        # Optional: override default
    num_iterations=100  # Optional: override default
)
```

#### Running the Demonstration
```bash
python competitive_evaluation_demo.py
```

This will demonstrate:
- Comparison between individual and competitive evaluation methods
- Ranking changes when agents are evaluated competitively
- Performance differences in solo vs competitive scenarios
- Evolution using competitive evaluation

#### Legacy Individual Evaluation
```bash
python evaluation_demo.py
```

This demonstrates the original individual evaluation system for comparison.

### Benefits

1. **More Reliable Selection**: Reduces impact of random environmental factors
2. **Statistical Confidence**: Provides standard deviation and confidence metrics
3. **Consistent Performance**: Selects agents that perform well consistently, not just lucky ones
4. **Better Evolution**: Leads to more stable evolutionary progress
5. **Configurable Precision**: Adjust number of runs vs. computational cost

### Performance Metrics

The evaluation system provides comprehensive metrics:

- **avg_food_count**: Average food collected across all runs
- **avg_final_energy**: Average energy remaining at end of runs
- **survival_rate**: Percentage of runs where agent survived to the end
- **std_food_count**: Standard deviation of food collection (consistency measure)
- **max_food_count**: Best single-run performance
- **min_food_count**: Worst single-run performance

### File Structure (Updated)

- `agent_evaluator.py`: Competitive multi-run evaluation system
- `competitive_evaluation_demo.py`: Demonstration of competitive evaluation system
- `evaluation_demo.py`: Legacy individual evaluation demonstration
- `evolution.py`: Updated with competitive evaluation methods
- `config.py`: Updated with evaluation parameters

## Expected Behavior

Over generations, you should observe:
- Agents learning to move towards food
- Improved survival rates
- Emergence of successful behavioral strategies
- Increasing average fitness scores
- More consistent performance with multi-run evaluation

The simulation demonstrates how simple neural networks can evolve complex behaviors through natural selection, with the multi-run evaluation providing more reliable assessment of agent capabilities.
