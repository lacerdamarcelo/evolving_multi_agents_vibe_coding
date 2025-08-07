"""
Agent class for the evolutionary simulation.
"""

import math
import torch
from neural_network import AttentionAgentNeuralNetwork
from config import (
    AGENT_RADIUS, INITIAL_ENERGY, MOVE_DISTANCE, ROTATION_ANGLE,
    ENERGY_COST_IDLE, ENERGY_COST_ACTION, POPULATION_SIZE, NUM_FOOD_POINTS
)


class Agent:
    """
    Represents an agent in the simulation with neural network-based decision making.
    """
    
    def __init__(self, x=0, y=0, orientation=0):
        """
        Initialize an agent.
        
        Args:
            x: Initial x position
            y: Initial y position
            orientation: Initial orientation in degrees
        """
        self.x = x
        self.y = y
        self.orientation = orientation  # in degrees
        self.energy = INITIAL_ENERGY
        self.food_count = 0
        self.radius = AGENT_RADIUS
        self.alive = True
        
        # Neural network for decision making
        self.neural_network = AttentionAgentNeuralNetwork()
    
    def get_perception_tokens(self, other_agents, food_points):
        """
        Create structured tokens for the attention-based neural network.
        
        Args:
            other_agents: List of other agents in the environment
            food_points: List of food points in the environment
            
        Returns:
            dict: Dictionary containing structured tokens:
                - 'self': tensor of shape [2] (food_count, energy)
                - 'agents': tensor of shape [N, 3] (distance, relative_angle, food_diff)
                - 'food': tensor of shape [M, 2] (distance, relative_angle)
        """
        # Self token: [food_count, energy]
        self_token = torch.tensor([self.food_count, self.energy], dtype=torch.float32)
        
        # Agent tokens: [distance, relative_angle, food_diff]
        agent_tokens_list = []
        
        for other_agent in other_agents:
            if other_agent.alive and other_agent != self:
                # Calculate distance
                dx = other_agent.x - self.x
                dy = other_agent.y - self.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                # Calculate relative angle
                angle_to_other = math.degrees(math.atan2(dy, dx))
                relative_angle = angle_to_other - self.orientation
                
                # Normalize angle to [-180, 180]
                while relative_angle > 180:
                    relative_angle -= 360
                while relative_angle < -180:
                    relative_angle += 360
                
                # Food difference (positive means other agent has more food)
                food_diff = other_agent.food_count - self.food_count
                
                agent_tokens_list.append([distance, relative_angle, food_diff])
        
        # Sort by distance (closest first)
        agent_tokens_list.sort(key=lambda x: x[0])
        
        # Pad to fixed size (POPULATION_SIZE - 1) for consistent tensor shapes
        max_other_agents = POPULATION_SIZE - 1
        while len(agent_tokens_list) < max_other_agents:
            agent_tokens_list.append([0.0, 0.0, 0.0])  # Padding with zeros
        
        # Truncate if too many (shouldn't happen in normal simulation)
        agent_tokens_list = agent_tokens_list[:max_other_agents]
        
        agent_tokens = torch.tensor(agent_tokens_list, dtype=torch.float32)
        
        # Food tokens: [distance, relative_angle]
        food_tokens_list = []
        
        for food_point in food_points:
            # Calculate distance to food point
            dx = food_point.x - self.x
            dy = food_point.y - self.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Calculate relative angle to food point
            angle_to_food = math.degrees(math.atan2(dy, dx))
            relative_angle = angle_to_food - self.orientation
            
            # Normalize angle to [-180, 180]
            while relative_angle > 180:
                relative_angle -= 360
            while relative_angle < -180:
                relative_angle += 360
            
            food_tokens_list.append([distance, relative_angle])
        
        # Sort food points by distance (closest first)
        food_tokens_list.sort(key=lambda x: x[0])
        
        # Pad to fixed size (NUM_FOOD_POINTS) for consistent tensor shapes
        while len(food_tokens_list) < NUM_FOOD_POINTS:
            food_tokens_list.append([0.0, 0.0])  # Padding with zeros
        
        # Truncate if too many (shouldn't happen in normal simulation)
        food_tokens_list = food_tokens_list[:NUM_FOOD_POINTS]
        
        food_tokens = torch.tensor(food_tokens_list, dtype=torch.float32)
        
        return {
            'self': self_token,
            'agents': agent_tokens,
            'food': food_tokens
        }
    
    def decide_actions(self, other_agents, food_points):
        """
        Use neural network to decide on actions.
        
        Args:
            other_agents: List of other agents
            food_points: List of food points
            
        Returns:
            tuple: (should_move_forward, should_rotate_right, should_rotate_left)
        """
        if not self.alive:
            return False, False, False
        
        tokens_dict = self.get_perception_tokens(other_agents, food_points)
        return self.neural_network.get_actions(tokens_dict)
    
    def rotate_right(self):
        """Rotate the agent clockwise by the rotation angle."""
        if not self.alive:
            return
        
        self.orientation += ROTATION_ANGLE
        # Keep orientation in [0, 360) range
        self.orientation = self.orientation % 360
    
    def rotate_left(self):
        """Rotate the agent counter-clockwise by the rotation angle."""
        if not self.alive:
            return
        
        self.orientation -= ROTATION_ANGLE
        # Keep orientation in [0, 360) range
        self.orientation = self.orientation % 360
    
    def move_forward(self):
        """Move the agent forward in its current orientation."""
        if not self.alive:
            return
        
        # Convert orientation to radians
        orientation_rad = math.radians(self.orientation)
        
        # Calculate new position
        self.x += MOVE_DISTANCE * math.cos(orientation_rad)
        self.y += MOVE_DISTANCE * math.sin(orientation_rad)
    
    def update_energy(self, action_taken):
        """
        Update agent's energy based on actions taken.
        
        Args:
            action_taken: True if any action was taken, False if idle
        """
        if not self.alive:
            return
        
        if action_taken:
            self.energy -= ENERGY_COST_ACTION
        else:
            self.energy -= ENERGY_COST_IDLE
        
        # Check if agent dies from lack of energy
        if self.energy <= 0:
            self.alive = False
            self.energy = 0
    
    def consume_food(self, energy_gain):
        """
        Consume food and gain energy.
        
        Args:
            energy_gain: Amount of energy to gain
        """
        if not self.alive:
            return
        
        self.energy += energy_gain
        self.food_count += 1
    
    def get_direction_line_end(self):
        """
        Get the end point of the direction line for visualization.
        
        Returns:
            tuple: (end_x, end_y) coordinates
        """
        orientation_rad = math.radians(self.orientation)
        end_x = self.x + self.radius * math.cos(orientation_rad)
        end_y = self.y + self.radius * math.sin(orientation_rad)
        return end_x, end_y
    
    def distance_to(self, other_agent):
        """
        Calculate distance to another agent.
        
        Args:
            other_agent: Another Agent instance
            
        Returns:
            float: Distance between agents
        """
        dx = self.x - other_agent.x
        dy = self.y - other_agent.y
        return math.sqrt(dx * dx + dy * dy)
    
    def collides_with_point(self, point_x, point_y, point_radius=0):
        """
        Check if agent collides with a point (like food).
        
        Args:
            point_x: X coordinate of the point
            point_y: Y coordinate of the point
            point_radius: Radius of the point (default 0)
            
        Returns:
            bool: True if collision occurs
        """
        dx = self.x - point_x
        dy = self.y - point_y
        distance = math.sqrt(dx * dx + dy * dy)
        return distance <= (self.radius + point_radius)
    
    def collides_with_agent(self, other_agent):
        """
        Check if this agent collides with another agent.
        
        Args:
            other_agent: Another Agent instance
            
        Returns:
            bool: True if collision occurs
        """
        if not self.alive or not other_agent.alive or self == other_agent:
            return False
        
        distance = self.distance_to(other_agent)
        return distance <= (self.radius + other_agent.radius)
