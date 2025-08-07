"""
Environment class for managing the simulation world.
"""

import random
import math
from agent import Agent
from config import (
    ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, POPULATION_SIZE, NUM_FOOD_POINTS,
    ENERGY_FROM_FOOD, FOOD_RADIUS
)


class FoodPoint:
    """Represents a food point in the environment."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = FOOD_RADIUS


class Environment:
    """
    Manages the simulation environment including agents and food points.
    """
    
    def __init__(self):
        """Initialize the environment."""
        self.width = ENVIRONMENT_WIDTH
        self.height = ENVIRONMENT_HEIGHT
        self.agents = []
        self.food_points = []
        
        # Initialize population and food
        self._initialize_agents()
        self._initialize_food()
    
    def _initialize_agents(self):
        """Initialize the agent population with random positions and orientations."""
        self.agents = []
        for _ in range(POPULATION_SIZE):
            # Random position within environment bounds
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            orientation = random.uniform(0, 360)
            
            agent = Agent(x, y, orientation)
            self.agents.append(agent)
    
    def _initialize_food(self):
        """Initialize food points randomly in the environment."""
        self.food_points = []
        for _ in range(NUM_FOOD_POINTS):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            self.food_points.append(FoodPoint(x, y))
    
    def _spawn_new_food(self):
        """Spawn a new food point at a random location."""
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        return FoodPoint(x, y)
    
    def _keep_agent_in_bounds(self, agent):
        """Keep agent within environment boundaries using clamping."""
        if agent.x < 0:
            agent.x = 0
        elif agent.x > self.width:
            agent.x = self.width
        
        if agent.y < 0:
            agent.y = 0
        elif agent.y > self.height:
            agent.y = self.height
    
    def update(self):
        """
        Update the environment for one iteration.
        This includes agent decision making, movement, and collision handling.
        """
        # Get decisions from all living agents
        agent_actions = {}
        for agent in self.agents:
            if agent.alive:
                should_move, should_rotate_right, should_rotate_left = agent.decide_actions(self.agents, self.food_points)
                agent_actions[agent] = (should_move, should_rotate_right, should_rotate_left)
        
        # Execute actions and update energy
        for agent, (should_move, should_rotate_right, should_rotate_left) in agent_actions.items():
            if not agent.alive:
                continue
            
            # Execute actions (only one rotation can be active due to softmax)
            if should_rotate_right:
                agent.rotate_right()
            elif should_rotate_left:
                agent.rotate_left()
            
            if should_move:
                agent.move_forward()
                self._keep_agent_in_bounds(agent)
            
            # Update energy based on actions taken
            action_taken = should_rotate_right or should_rotate_left or should_move
            agent.update_energy(action_taken)
        
        # Handle collisions
        self._handle_food_collisions()
        self._handle_agent_collisions()
    
    def _handle_food_collisions(self):
        """Handle collisions between agents and food points."""
        food_to_remove = []
        
        for i, food in enumerate(self.food_points):
            for agent in self.agents:
                if agent.alive and agent.collides_with_point(food.x, food.y, food.radius):
                    # Agent consumes food
                    agent.consume_food(ENERGY_FROM_FOOD)
                    food_to_remove.append(i)
                    break
        
        # Remove consumed food and spawn new ones
        for i in reversed(food_to_remove):  # Remove in reverse order to maintain indices
            del self.food_points[i]
            self.food_points.append(self._spawn_new_food())
    
    def _handle_agent_collisions(self):
        """Handle collisions between agents."""
        agents_to_process = [agent for agent in self.agents if agent.alive]
        
        for i in range(len(agents_to_process)):
            for j in range(i + 1, len(agents_to_process)):
                agent1 = agents_to_process[i]
                agent2 = agents_to_process[j]
                
                if agent1.collides_with_agent(agent2):
                    self._resolve_agent_collision(agent1, agent2)
    
    def _resolve_agent_collision(self, agent1, agent2):
        """
        Resolve collision between two agents.
        The agent with more food points survives and takes the other's food count.
        The winner also absorbs energy equivalent to the food points taken.
        In case of tie, both die.
        """
        if not agent1.alive or not agent2.alive:
            return
        
        if agent1.food_count > agent2.food_count:
            # Agent1 wins - takes food points and corresponding energy
            absorbed_food = agent2.food_count
            agent1.food_count += absorbed_food
            agent1.energy += absorbed_food * ENERGY_FROM_FOOD
            agent2.alive = False
        elif agent2.food_count > agent1.food_count:
            # Agent2 wins - takes food points and corresponding energy
            absorbed_food = agent1.food_count
            agent2.food_count += absorbed_food
            agent2.energy += absorbed_food * ENERGY_FROM_FOOD
            agent1.alive = False
        else:
            # Tie - both die
            agent1.alive = False
            agent2.alive = False
    
    def get_living_agents(self):
        """Get list of living agents."""
        return [agent for agent in self.agents if agent.alive]
    
    def get_dead_agents(self):
        """Get list of dead agents."""
        return [agent for agent in self.agents if not agent.alive]
    
    def get_agent_count(self):
        """Get count of living agents."""
        return len(self.get_living_agents())
    
    def get_statistics(self):
        """
        Get simulation statistics.
        
        Returns:
            dict: Dictionary containing various statistics
        """
        living_agents = self.get_living_agents()
        
        if not living_agents:
            return {
                'living_count': 0,
                'dead_count': len(self.agents),
                'avg_energy': 0,
                'avg_food_count': 0,
                'max_food_count': 0,
                'total_food_consumed': 0
            }
        
        energies = [agent.energy for agent in living_agents]
        food_counts = [agent.food_count for agent in living_agents]
        total_food_consumed = sum(agent.food_count for agent in self.agents)
        
        return {
            'living_count': len(living_agents),
            'dead_count': len(self.agents) - len(living_agents),
            'avg_energy': sum(energies) / len(energies),
            'avg_food_count': sum(food_counts) / len(food_counts),
            'max_food_count': max(food_counts),
            'total_food_consumed': total_food_consumed
        }
    
    def reset_with_new_generation(self, new_agents):
        """
        Reset environment with a new generation of agents.
        
        Args:
            new_agents: List of new Agent instances
        """
        self.agents = new_agents
        # Keep existing food points, don't regenerate them
    
    def get_survivors_for_evolution(self):
        """
        Get surviving agents sorted by food count for evolutionary selection.
        
        Returns:
            list: Living agents sorted by food count (descending)
        """
        living_agents = self.get_living_agents()
        return sorted(living_agents, key=lambda agent: agent.food_count, reverse=True)
