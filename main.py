"""
Main simulation file with Arcade visualization for the evolutionary agent simulation.
"""

import arcade
import math
import time
from environment import Environment
from evolution import EvolutionManager
from config import (
    ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, BACKGROUND_COLOR, AGENT_COLOR,
    DIRECTION_LINE_COLOR, FOOD_COLOR, NUM_EVALUATION_RUNS, TEXT_COLOR, FONT_SIZE, FPS,
    NUM_GENERATIONS, ITERATIONS_PER_GENERATION
)


class EvolutionSimulation(arcade.Window):
    """
    Main simulation window using Arcade for visualization.
    """
    
    def __init__(self):
        """Initialize the simulation window."""
        super().__init__(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, "Evolutionary Agent Simulation")
        arcade.set_background_color(BACKGROUND_COLOR)
        
        # Simulation components
        self.environment = Environment()
        self.evolution_manager = EvolutionManager()
        
        # Simulation state
        self.current_generation = 0
        self.current_iteration = 0
        self.simulation_running = True
        self.paused = False
        
        # Timing
        self.last_update_time = time.time()
        self.update_interval = 1.0 / FPS  # Target FPS
        
        # Statistics tracking
        self.generation_start_time = time.time()
        
    def setup(self):
        """Set up the simulation."""
        pass
    
    def on_draw(self):
        """Render the simulation."""
        self.clear()
        
        # Draw food points
        self._draw_food_points()
        
        # Draw agents
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
        """Draw all agents."""
        for agent in self.environment.agents:
            if agent.alive:
                # Draw agent body (blue circle)
                arcade.draw_circle_filled(
                    agent.x, agent.y, agent.radius, AGENT_COLOR
                )
                
                # Draw direction line (red line from center to edge)
                end_x, end_y = agent.get_direction_line_end()
                arcade.draw_line(
                    agent.x, agent.y, end_x, end_y, DIRECTION_LINE_COLOR, 2
                )
                
                # Draw food count above agent
                arcade.draw_text(
                    str(agent.food_count),
                    agent.x - 10, agent.y + agent.radius + 5,
                    TEXT_COLOR, FONT_SIZE
                )
    
    def _draw_ui(self):
        """Draw user interface information."""
        # Get current statistics
        stats = self.environment.get_statistics()
        
        # Prepare text information
        info_lines = [
            f"Generation: {self.current_generation + 1}/{NUM_GENERATIONS}",
            f"Iteration: {self.current_iteration + 1}/{ITERATIONS_PER_GENERATION}",
            f"Living Agents: {stats['living_count']}/{stats['living_count'] + stats['dead_count']}",
            f"Avg Energy: {stats['avg_energy']:.1f}",
            f"Avg Food: {stats['avg_food_count']:.1f}",
            f"Max Food: {stats['max_food_count']}",
            f"Total Food Consumed: {stats['total_food_consumed']}"
        ]
        
        # Add pause/running status
        if self.paused:
            info_lines.append("PAUSED - Press SPACE to resume")
        elif not self.simulation_running:
            info_lines.append("SIMULATION COMPLETE")
        
        # Draw information
        y_offset = ENVIRONMENT_HEIGHT - 30
        for line in info_lines:
            arcade.draw_text(
                line, 10, y_offset, TEXT_COLOR, FONT_SIZE
            )
            y_offset -= 20
        
        # Draw controls
        controls = [
            "Controls:",
            "SPACE - Pause/Resume",
            "R - Restart simulation",
            "ESC - Exit"
        ]
        
        y_offset = 100
        for control in controls:
            arcade.draw_text(
                control, 10, y_offset, TEXT_COLOR, FONT_SIZE - 2
            )
            y_offset -= 15
    
    def on_update(self, delta_time):
        """Update the simulation."""
        current_time = time.time()
        
        # Control update rate
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Skip update if paused or simulation complete
        if self.paused or not self.simulation_running:
            return
        
        # Update environment
        self.environment.update()
        
        # Check if generation is complete
        self.current_iteration += 1
        if self.current_iteration >= ITERATIONS_PER_GENERATION:
            self._end_generation()
    
    def _end_generation(self):
        """End current generation and start next one."""
        # Get survivors and evolve
        survivors = self.environment.get_survivors_for_evolution()
        new_agents, gen_stats, _ = self.evolution_manager.evolve_generation_with_evaluation(self.environment.agents,
                                                                                         num_runs=NUM_EVALUATION_RUNS,
                                                                                         num_iterations=ITERATIONS_PER_GENERATION)
        
        # Print generation summary
        print(f"Generation {self.current_generation + 1} complete:")
        print(f"  Survivors: {gen_stats['survivors']}")
        print(f"  Best fitness: {gen_stats['best_fitness']}")
        print(f"  Avg fitness: {gen_stats['avg_fitness']:.2f}")
        print(f"  Avg energy: {gen_stats['avg_energy']:.1f}")
        print()
        
        # Reset for next generation
        self.environment.reset_with_new_generation(new_agents)
        self.current_generation += 1
        self.current_iteration = 0
        self.generation_start_time = time.time()
        
        # Check if simulation is complete
        if self.current_generation >= NUM_GENERATIONS:
            self._end_simulation()
    
    def _end_simulation(self):
        """End the simulation and display final results."""
        self.simulation_running = False
        
        # Save the best agent
        if self.evolution_manager.save_best_agent("best_agent.pth"):
            print("Best agent saved successfully!")
        
        # Generate evolution plots
        print("Generating evolution plots...")
        self.evolution_manager.generate_evolution_plots("evolution_plots")
        
        # Print final summary
        summary = self.evolution_manager.get_evolution_summary()
        print("=" * 50)
        print("SIMULATION COMPLETE")
        print("=" * 50)
        print(f"Total generations: {summary.get('total_generations', 0)}")
        print(f"Best fitness ever: {summary.get('best_fitness_ever', 0)}")
        print(f"Average best fitness: {summary.get('avg_best_fitness', 0):.2f}")
        print(f"Final generation best: {summary.get('final_generation_best', 0)}")
        print(f"Overall improvement: {summary.get('improvement', 0)}")
        print(f"Avg survivors per generation: {summary.get('avg_survivors_per_gen', 0):.1f}")
        print("=" * 50)
    
    def on_key_press(self, key, modifiers):
        """Handle key presses."""
        if key == arcade.key.SPACE:
            # Toggle pause
            self.paused = not self.paused
            print("Simulation paused" if self.paused else "Simulation resumed")
        
        elif key == arcade.key.R:
            # Restart simulation
            self._restart_simulation()
            print("Simulation restarted")
        
        elif key == arcade.key.ESCAPE:
            # Exit
            self.close()
    
    def _restart_simulation(self):
        """Restart the simulation from the beginning."""
        self.environment = Environment()
        self.evolution_manager = EvolutionManager()
        self.current_generation = 0
        self.current_iteration = 0
        self.simulation_running = True
        self.paused = False
        self.generation_start_time = time.time()


def main():
    """Main function to run the simulation."""
    print("Starting Evolutionary Agent Simulation")
    print("=" * 50)
    print("Controls:")
    print("  SPACE - Pause/Resume simulation")
    print("  R - Restart simulation")
    print("  ESC - Exit")
    print("=" * 50)
    
    # Create and run simulation
    simulation = EvolutionSimulation()
    simulation.setup()
    arcade.run()


if __name__ == "__main__":
    main()
