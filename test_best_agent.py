"""
Test script to run the best evolved agent in the 2D environment.
This script loads the saved best agent and runs it in a visual simulation.
"""

import arcade
import math
import time
import sys
import torch
from environment import Environment
from evolution import EvolutionManager
from agent import Agent
from config import (
    ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, BACKGROUND_COLOR, AGENT_COLOR,
    DIRECTION_LINE_COLOR, FOOD_COLOR, INITIAL_ENERGY, TEXT_COLOR, FONT_SIZE, FPS,
    ITERATIONS_PER_GENERATION, POPULATION_SIZE, NUM_FOOD_POINTS
)


class BestAgentTest(arcade.Window):
    """
    Test window to visualize the best evolved agent's performance.
    """
    
    def __init__(self, visualize_attention=False):
        """Initialize the test window."""
        super().__init__(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, "Best Agent Test")
        arcade.set_background_color(BACKGROUND_COLOR)
        
        # Attention visualization settings
        self.visualize_attention = visualize_attention
        self.attention_weights = None
        self.tracked_agent_index = 0  # Agent index 0 is the tracked agent
        
        # Load the best agent
        self.evolution_manager = EvolutionManager()
        self.best_agent = self.evolution_manager.load_best_agent("best_agent.pth")
        
        if self.best_agent is None:
            print("Failed to load best agent! Make sure 'best_agent.pth' exists.")
            print("Run the main evolution simulation first to generate a best agent.")
            self.close()
            return
        
        # Create environment with full scenario (all agents and food points)
        self.environment = Environment()
        
        # Load the best neural network into ALL agents
        for agent in self.environment.agents:
            # Copy the best neural network weights to each agent
            agent.neural_network.copy_weights_from(self.best_agent.neural_network)
            # Reset agent state for testing
            agent.energy = INITIAL_ENERGY  # Full energy
            agent.food_count = 0  # Reset food count
            agent.alive = True
        
        # Keep reference to tracked agent (index 0)
        self.tracked_agent = self.environment.agents[self.tracked_agent_index] if self.environment.agents else None
        
        # Test state
        self.current_iteration = 0
        self.test_running = True
        self.paused = False
        self.max_iterations = ITERATIONS_PER_GENERATION * 2  # Run for 2x normal generation length
        
        # Timing
        self.last_update_time = time.time()
        self.update_interval = 1.0 / FPS  # Target FPS
        
        # Statistics tracking
        self.start_time = time.time()
        self.initial_food_count = 0
        
        print(f"Testing best agent (original fitness: {self.tracked_agent.food_count})")
        print(f"Test will run for {self.max_iterations} iterations")
        print(f"Attention visualization: {'ENABLED' if self.visualize_attention else 'DISABLED'}")
        print("Controls: SPACE - Pause/Resume, A - Toggle attention, R - Restart test, ESC - Exit")
        
    def setup(self):
        """Set up the test."""
        pass
    
    def _get_attention_weights(self):
        """Get attention weights from the tracked agent."""
        if not self.visualize_attention or not self.tracked_agent or not self.tracked_agent.alive:
            return None
        
        try:
            # Get tokens from tracked agent
            tokens_dict = self.tracked_agent.get_perception_tokens(
                self.environment.agents, self.environment.food_points
            )
            
            # Get attention weights from neural network
            with torch.no_grad():
                _, attention_weights = self.tracked_agent.neural_network.forward(
                    tokens_dict, return_attention=True
                )
            
            return attention_weights
        except Exception as e:
            print(f"Error getting attention weights: {e}")
            return None
    
    def _attention_to_color(self, attention_value, base_color=(0, 0, 255)):
        """
        Convert attention value to color.
        
        Args:
            attention_value: Attention weight (0.0 to 1.0)
            base_color: Base color tuple (R, G, B)
            
        Returns:
            tuple: Color tuple (R, G, B)
        """
        # Normalize attention value to [0, 1]
        attention_value = max(0.0, min(1.0, attention_value))
        
        # Create color gradient from blue (low attention) to red (high attention)
        if attention_value < 0.5:
            # Blue to green gradient
            r = 0
            g = int(attention_value * 2 * 255)
            b = int((1.0 - attention_value * 2) * 255)
        else:
            # Green to red gradient
            r = int((attention_value - 0.5) * 2 * 255)
            g = int((1.0 - (attention_value - 0.5) * 2) * 255)
            b = 0
        
        return (r, g, b)
    
    def on_draw(self):
        """Render the test."""
        self.clear()
        
        # Get attention weights if visualization is enabled
        if self.visualize_attention:
            self.attention_weights = self._get_attention_weights()
        
        # Draw food points
        self._draw_food_points()
        
        # Draw all agents
        self._draw_agents()
        
        # Draw UI information
        self._draw_ui()
    
    def _draw_food_points(self):
        """Draw all food points with attention-based coloring."""
        for i, food in enumerate(self.environment.food_points):
            color = FOOD_COLOR  # Default color
            
            # Apply attention-based coloring if enabled
            if self.visualize_attention and self.attention_weights is not None:
                # Food attention weights start after self token (index 0) and agent tokens (POPULATION_SIZE-1)
                food_attention_start_idx = 1 + (POPULATION_SIZE - 1)
                food_attention_idx = food_attention_start_idx + i
                
                if food_attention_idx < len(self.attention_weights):
                    attention_value = self.attention_weights[food_attention_idx].item()
                    color = self._attention_to_color(attention_value)
            
            arcade.draw_circle_filled(food.x, food.y, food.radius, color)
    
    def _draw_agents(self):
        """Draw all agents with attention-based coloring."""
        for i, agent in enumerate(self.environment.agents):
            if agent.alive:
                # Default appearance
                color = AGENT_COLOR
                radius = agent.radius
                line_width = 2
                text_color = TEXT_COLOR
                text_size = FONT_SIZE
                
                # Highlight tracked agent (index 0)
                if i == self.tracked_agent_index:
                    color = (0, 255, 0)  # Green for tracked agent
                    radius = agent.radius + 2
                    line_width = 3
                    text_color = (255, 255, 255)  # White text
                    text_size = FONT_SIZE + 2
                elif self.visualize_attention and self.attention_weights is not None:
                    # Apply attention-based coloring for other agents
                    # Agent attention weights start at index 1 (after self token)
                    # We need to find the correct index for this agent in the attention weights
                    
                    # Create mapping from agent to attention index
                    living_other_agents = [a for a in self.environment.agents if a.alive and a != self.tracked_agent]
                    
                    # Sort by distance to tracked agent (same as in get_perception_tokens)
                    if self.tracked_agent and self.tracked_agent.alive:
                        living_other_agents.sort(key=lambda a: 
                            math.sqrt((a.x - self.tracked_agent.x)**2 + (a.y - self.tracked_agent.y)**2))
                    
                    # Find this agent's position in the sorted list
                    try:
                        agent_idx_in_attention = living_other_agents.index(agent)
                        attention_idx = 1 + agent_idx_in_attention  # +1 to skip self token
                        
                        if attention_idx < len(self.attention_weights):
                            attention_value = self.attention_weights[attention_idx].item()
                            color = self._attention_to_color(attention_value)
                    except (ValueError, IndexError):
                        # Agent not found in list or index out of range, use default color
                        pass
                
                # Draw agent body
                arcade.draw_circle_filled(agent.x, agent.y, radius, color)
                
                # Draw direction line
                end_x, end_y = agent.get_direction_line_end()
                arcade.draw_line(
                    agent.x, agent.y, end_x, end_y, DIRECTION_LINE_COLOR, line_width
                )
                
                # Draw food count above agent
                arcade.draw_text(
                    str(agent.food_count),
                    agent.x - 10, agent.y + radius + 5,
                    text_color, text_size
                )
    
    def _draw_ui(self):
        """Draw user interface information."""
        # Get current statistics
        stats = self.environment.get_statistics()
        
        # Calculate performance metrics
        elapsed_time = time.time() - self.start_time
        total_food_collected = sum(agent.food_count for agent in self.environment.agents)
        population_food_per_minute = (total_food_collected / elapsed_time) * 60 if elapsed_time > 0 else 0
        
        # Prepare text information
        info_lines = [
            f"BEST NEURAL NETWORK TEST - POPULATION SCENARIO",
            f"Iteration: {self.current_iteration + 1}/{self.max_iterations}",
            f"Living Agents: {stats['living_count']}/{stats['living_count'] + stats['dead_count']}",
            f"Population Avg Food: {stats['avg_food_count']:.1f}",
            f"Population Max Food: {stats['max_food_count']}",
            f"Total Food Collected: {total_food_collected}",
            f"Population Food/Min: {population_food_per_minute:.2f}",
            f"Available Food: {len(self.environment.food_points)}",
            f"Elapsed Time: {elapsed_time:.1f}s"
        ]
        
        # Add attention visualization status
        info_lines.append(f"Attention Visualization: {'ON' if self.visualize_attention else 'OFF'}")
        
        # Add pause/running status
        if self.paused:
            info_lines.append("PAUSED - Press SPACE to resume")
        elif not self.test_running:
            info_lines.append("TEST COMPLETE")
        
        # Draw information
        y_offset = ENVIRONMENT_HEIGHT - 30
        for line in info_lines:
            color = (255, 255, 255) if line.startswith("BEST NEURAL NETWORK TEST") else TEXT_COLOR
            font_size = FONT_SIZE + 2 if line.startswith("BEST NEURAL NETWORK TEST") else FONT_SIZE
            arcade.draw_text(
                line, 10, y_offset, color, font_size
            )
            y_offset -= 25 if line.startswith("BEST NEURAL NETWORK TEST") else 20
        
        # Draw controls
        controls = [
            "Controls:",
            "SPACE - Pause/Resume",
            "A - Toggle attention visualization",
            "R - Restart test",
            "ESC - Exit"
        ]
        
        y_offset = 120
        for control in controls:
            arcade.draw_text(
                control, 10, y_offset, TEXT_COLOR, FONT_SIZE - 2
            )
            y_offset -= 15
    
    def on_update(self, delta_time):
        """Update the test."""
        current_time = time.time()
        
        # Control update rate
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Skip update if paused or test complete
        if self.paused or not self.test_running:
            return
        
        # Update environment
        self.environment.update()
        
        # Check if test is complete or all agents are dead
        self.current_iteration += 1
        living_agents = [agent for agent in self.environment.agents if agent.alive]
        
        if self.current_iteration >= self.max_iterations:
            self._end_test()
        elif len(living_agents) == 0:
            # All agents died - restart automatically
            print(f"All agents died at iteration {self.current_iteration}. Restarting simulation...")
            self._restart_test()
    
    def _end_test(self):
        """End the test and display final results."""
        self.test_running = False
        
        # Calculate population-wide statistics
        elapsed_time = time.time() - self.start_time
        stats = self.environment.get_statistics()
        total_food_collected = sum(agent.food_count for agent in self.environment.agents)
        population_food_per_minute = (total_food_collected / elapsed_time) * 60 if elapsed_time > 0 else 0
        living_agents = [agent for agent in self.environment.agents if agent.alive]
        survival_rate = len(living_agents) / len(self.environment.agents) if self.environment.agents else 0
        
        print("=" * 50)
        print("BEST NEURAL NETWORK TEST COMPLETE")
        print("=" * 50)
        print(f"Total iterations: {self.current_iteration}")
        print(f"Elapsed time: {elapsed_time:.1f} seconds")
        print(f"Living agents: {len(living_agents)}/{len(self.environment.agents)} ({survival_rate:.1%})")
        print(f"Total food collected: {total_food_collected}")
        print(f"Average food per agent: {stats['avg_food_count']:.2f}")
        print(f"Maximum food by single agent: {stats['max_food_count']}")
        print(f"Population food collection rate: {population_food_per_minute:.2f} food/minute")
        print(f"Population efficiency: {total_food_collected / self.current_iteration:.4f} food/iteration")
        if living_agents:
            avg_energy = sum(agent.energy for agent in living_agents) / len(living_agents)
            print(f"Average energy of survivors: {avg_energy:.1f}")
        print("=" * 50)
    
    def on_key_press(self, key, modifiers):
        """Handle key presses."""
        if key == arcade.key.SPACE:
            # Toggle pause
            self.paused = not self.paused
            print("Test paused" if self.paused else "Test resumed")
        
        elif key == arcade.key.A:
            # Toggle attention visualization
            self.visualize_attention = not self.visualize_attention
            print(f"Attention visualization {'ENABLED' if self.visualize_attention else 'DISABLED'}")
        
        elif key == arcade.key.R:
            # Restart test
            self._restart_test()
            print("Test restarted")
        
        elif key == arcade.key.ESCAPE:
            # Exit
            self.close()
    
    def _restart_test(self):
        """Restart the test from the beginning."""
        # Reset environment with full scenario
        self.environment = Environment()
        
        # Load the best neural network into ALL agents
        for agent in self.environment.agents:
            # Copy the best neural network weights to each agent
            agent.neural_network.copy_weights_from(self.best_agent.neural_network)
            # Reset agent state for testing
            agent.energy = INITIAL_ENERGY  # Full energy
            agent.food_count = 0  # Reset food count
            agent.alive = True
        
        # Keep reference to first agent as "best_agent" for UI display
        self.best_agent = self.environment.agents[0] if self.environment.agents else None
        
        # Reset test state
        self.current_iteration = 0
        self.test_running = True
        self.paused = False
        self.start_time = time.time()


def main():
    """Main function to run the best agent test."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the best evolved agent with optional attention visualization")
    parser.add_argument("--attention", "-a", action="store_true", 
                       help="Enable attention visualization by default (default: False)")
    args = parser.parse_args()
    
    print("Starting Best Agent Test")
    print("=" * 50)
    print("This script will load and test the best evolved agent.")
    print("Make sure you have run the main evolution simulation first!")
    print("=" * 50)
    
    # Create and run test with optional attention visualization
    test = BestAgentTest(visualize_attention=args.attention)
    if hasattr(test, 'tracked_agent') and test.tracked_agent is not None:  # Only run if agent was loaded successfully
        test.setup()
        arcade.run()


if __name__ == "__main__":
    main()
