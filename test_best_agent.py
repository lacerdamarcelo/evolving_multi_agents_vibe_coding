"""
Test script to run the best evolved agent in the 2D environment.
This script loads the saved best agent and runs it in a visual simulation.
"""

import arcade
import math
import time
from environment import Environment
from evolution import EvolutionManager
from agent import Agent
from config import (
    ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, BACKGROUND_COLOR, AGENT_COLOR,
    DIRECTION_LINE_COLOR, FOOD_COLOR, INITIAL_ENERGY, TEXT_COLOR, FONT_SIZE, FPS,
    ITERATIONS_PER_GENERATION
)


class BestAgentTest(arcade.Window):
    """
    Test window to visualize the best evolved agent's performance.
    """
    
    def __init__(self):
        """Initialize the test window."""
        super().__init__(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, "Best Agent Test")
        arcade.set_background_color(BACKGROUND_COLOR)
        
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
        
        # Keep reference to first agent as "best_agent" for UI display
        self.best_agent = self.environment.agents[0] if self.environment.agents else None
        
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
        
        print(f"Testing best agent (original fitness: {self.best_agent.food_count})")
        print(f"Test will run for {self.max_iterations} iterations")
        print("Controls: SPACE - Pause/Resume, R - Restart test, ESC - Exit")
        
    def setup(self):
        """Set up the test."""
        pass
    
    def on_draw(self):
        """Render the test."""
        self.clear()
        
        # Draw food points
        self._draw_food_points()
        
        # Draw all agents
        self._draw_agents()
        
        # Draw UI information
        self._draw_ui()
    
    def _draw_food_points(self):
        """Draw all food points."""
        for food in self.environment.food_points:
            arcade.draw_circle_filled(
                food.x, food.y, food.radius, FOOD_COLOR
            )
    
    def _draw_agents(self):
        """Draw all agents with the same appearance (all have best neural network)."""
        for agent in self.environment.agents:
            if agent.alive:
                # All agents are equal now - they all have the best neural network
                color = AGENT_COLOR  # Blue for all agents
                radius = agent.radius
                line_width = 2
                text_color = TEXT_COLOR  # Black text
                text_size = FONT_SIZE
                
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
    print("Starting Best Agent Test")
    print("=" * 50)
    print("This script will load and test the best evolved agent.")
    print("Make sure you have run the main evolution simulation first!")
    print("=" * 50)
    
    # Create and run test
    test = BestAgentTest()
    if test.best_agent is not None:  # Only run if agent was loaded successfully
        test.setup()
        arcade.run()


if __name__ == "__main__":
    main()
