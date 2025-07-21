"""
Configuration parameters for the evolutionary agent simulation.
"""

# Environment parameters
ENVIRONMENT_WIDTH = 600
ENVIRONMENT_HEIGHT = 600
BACKGROUND_COLOR = (255, 255, 255)  # White

# Agent parameters
AGENT_RADIUS = 8
AGENT_COLOR = (0, 0, 255)  # Blue
DIRECTION_LINE_COLOR = (255, 0, 0)  # Red
POPULATION_SIZE = 50
INITIAL_ENERGY = 999

# Movement parameters
MOVE_DISTANCE = 2
ROTATION_ANGLE = 20  # degrees
ENERGY_COST_IDLE = 1.0
ENERGY_COST_ACTION = 2.0

# Food parameters
NUM_FOOD_POINTS = 20
FOOD_COLOR = (255, 0, 0)  # Red
FOOD_RADIUS = 5
ENERGY_FROM_FOOD = 50

# Evolution parameters
NUM_GENERATIONS = 1000
ITERATIONS_PER_GENERATION = 500
SURVIVAL_RATE = 0.2  # Top 20%
MUTATION_VARIANCE = 0.05
CROSSOVER_PROPORTION = 0.5  # 50% of offspring created through crossover, 50% through mutation

NUM_EVALUATION_RUNS = 30  # Number of runs to average performance over
EVALUATION_ITERATIONS = 500  # Number of iterations per evaluation run

# Neural network parameters
HIDDEN_LAYER_MULTIPLIER = 3
INPUT_SIZE = 2 + (3 * (POPULATION_SIZE - 1)) + (2 * NUM_FOOD_POINTS)  # own_state + other_agents_info + food_points_info
HIDDEN_SIZE = INPUT_SIZE * HIDDEN_LAYER_MULTIPLIER
OUTPUT_SIZE = 4

# Visualization parameters
FONT_SIZE = 12
TEXT_COLOR = (0, 0, 0)  # Black
FPS = 60
