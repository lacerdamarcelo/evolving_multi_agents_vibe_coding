"""
Evolutionary algorithm implementation for agent evolution.
"""

import random
import numpy as np
import torch
from agent import Agent
from neural_network import create_offspring_network, create_crossover_offspring
from agent_evaluator import AgentEvaluator
from config import (
    POPULATION_SIZE, SURVIVAL_RATE, ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT,
    NUM_EVALUATION_RUNS, EVALUATION_ITERATIONS, CROSSOVER_PROPORTION
)


class EvolutionManager:
    """
    Manages the evolutionary process across generations.
    """
    
    def __init__(self):
        """Initialize the evolution manager."""
        self.generation = 0
        self.generation_stats = []
        self.agent_evaluator = AgentEvaluator()
        self.best_agent_ever = None
        self.best_fitness_ever = -1
        
        # Additional tracking for visualization
        self.best_fitness_history = []  # Track best fitness per generation
        self.population_fitness_history = []  # Track all fitness values per generation
        self.survival_rate_history = []  # Track survival rates per generation
    
    def select_survivors(self, agents):
        """
        Select the top performing agents for reproduction.
        
        Args:
            agents: List of agents sorted by fitness (food count)
            
        Returns:
            list: Selected agents for reproduction
        """
        # Calculate number of survivors
        num_survivors = max(1, int(len(agents) * SURVIVAL_RATE))
        
        # Select top performers
        survivors = agents[:num_survivors]
        
        return survivors
    
    def create_next_generation(self, survivors, best_performer=None):
        """
        Create the next generation from survivors.
        
        Args:
            survivors: List of surviving agents
            best_performer: Best performing agent from the generation (fallback if no survivors)
            
        Returns:
            list: New generation of agents
        """
        if not survivors:
            # If no survivors, use the best performer to generate offspring
            if best_performer:
                print(f"No survivors - using best performer (food: {best_performer.food_count}) to generate offspring")
                new_agents = []
                # Create entire population from the best performer
                for _ in range(POPULATION_SIZE):
                    offspring = self._create_offspring(best_performer)
                    new_agents.append(offspring)
                return new_agents
            else:
                # Fallback: create completely new random population
                print("No survivors and no best performer - creating random population")
                return self._create_random_population()
        
        new_agents = []
        
        # Keep survivors (they get new positions and full energy)
        for survivor in survivors:
            new_agent = self._create_agent_from_survivor(survivor)
            new_agents.append(new_agent)
        
        # Calculate how many offspring to create
        num_offspring_needed = POPULATION_SIZE - len(new_agents)
        
        if len(survivors) > 1:
            # Calculate proportions of crossover vs mutation offspring
            num_crossover_offspring = int(num_offspring_needed * CROSSOVER_PROPORTION)
            num_mutation_offspring = num_offspring_needed - num_crossover_offspring
            
            print(f"Creating {num_crossover_offspring} crossover offspring and {num_mutation_offspring} mutation offspring")
            
            # Create crossover offspring
            for _ in range(num_crossover_offspring):
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                # Ensure different parents
                while parent2 == parent1 and len(survivors) > 1:
                    parent2 = random.choice(survivors)
                offspring = self._create_crossover_offspring(parent1, parent2)
                new_agents.append(offspring)
            
            # Create mutation-only offspring
            for _ in range(num_mutation_offspring):
                parent = random.choice(survivors)
                offspring = self._create_offspring(parent)
                new_agents.append(offspring)
        else:
            # Single survivor - create all offspring through mutation
            print(f"Single survivor: creating {num_offspring_needed} mutation offspring")
            for _ in range(num_offspring_needed):
                parent = random.choice(survivors)
                offspring = self._create_offspring(parent)
                new_agents.append(offspring)
        
        # Apply mutation to all crossover offspring
        crossover_start_idx = len(survivors)
        crossover_end_idx = crossover_start_idx + (int(num_offspring_needed * CROSSOVER_PROPORTION) if len(survivors) > 1 else 0)
        
        for i in range(crossover_start_idx, crossover_end_idx):
            if i < len(new_agents):
                new_agents[i].neural_network.mutate()
        
        return new_agents
    
    def _create_agent_from_survivor(self, survivor):
        """
        Create a new agent based on a survivor (reset position and energy).
        
        Args:
            survivor: Surviving agent
            
        Returns:
            Agent: New agent with survivor's neural network
        """
        # Random new position
        x = random.uniform(0, ENVIRONMENT_WIDTH)
        y = random.uniform(0, ENVIRONMENT_HEIGHT)
        orientation = random.uniform(0, 360)
        
        # Create new agent
        new_agent = Agent(x, y, orientation)
        
        # Copy neural network weights from survivor
        new_agent.neural_network.copy_weights_from(survivor.neural_network)
        
        return new_agent
    
    def _create_offspring(self, parent):
        """
        Create an offspring agent from a parent.
        
        Args:
            parent: Parent agent
            
        Returns:
            Agent: Offspring agent with mutated neural network
        """
        # Random position for offspring
        x = random.uniform(0, ENVIRONMENT_WIDTH)
        y = random.uniform(0, ENVIRONMENT_HEIGHT)
        orientation = random.uniform(0, 360)
        
        # Create new agent
        offspring = Agent(x, y, orientation)
        
        # Create mutated neural network
        offspring.neural_network = create_offspring_network(parent.neural_network)
        
        return offspring
    
    def _create_crossover_offspring(self, parent1, parent2):
        """
        Create an offspring agent from two parents using crossover.
        
        Args:
            parent1: First parent agent
            parent2: Second parent agent
            
        Returns:
            Agent: Offspring agent with crossed-over neural network
        """
        # Random position for offspring
        x = random.uniform(0, ENVIRONMENT_WIDTH)
        y = random.uniform(0, ENVIRONMENT_HEIGHT)
        orientation = random.uniform(0, 360)
        
        # Create new agent
        offspring = Agent(x, y, orientation)
        
        # Create crossed-over neural network
        offspring.neural_network = create_crossover_offspring(parent1.neural_network, parent2.neural_network)
        
        return offspring
    
    def _create_agent_copy(self, agent):
        """
        Create a deep copy of an agent for saving the best agent.
        
        Args:
            agent: Agent to copy
            
        Returns:
            Agent: Deep copy of the agent
        """
        # Create new agent with same position and orientation
        copy_agent = Agent(agent.x, agent.y, agent.orientation)
        copy_agent.energy = agent.energy
        copy_agent.food_count = agent.food_count
        copy_agent.alive = agent.alive
        
        # Copy neural network weights
        copy_agent.neural_network.copy_weights_from(agent.neural_network)
        
        return copy_agent
    
    def _create_random_population(self):
        """
        Create a completely random population (used when no survivors).
        
        Returns:
            list: List of randomly initialized agents
        """
        agents = []
        for _ in range(POPULATION_SIZE):
            x = random.uniform(0, ENVIRONMENT_WIDTH)
            y = random.uniform(0, ENVIRONMENT_HEIGHT)
            orientation = random.uniform(0, 360)
            agents.append(Agent(x, y, orientation))
        return agents
    
    def evolve_generation(self, current_agents):
        """
        Evolve to the next generation.
        
        Args:
            current_agents: Current generation of agents
            
        Returns:
            tuple: (new_agents, generation_stats)
        """
        # Get living agents sorted by fitness
        living_agents = [agent for agent in current_agents if agent.alive]
        living_agents.sort(key=lambda agent: (agent.food_count, agent.energy), reverse=True)
        
        # Find the best performer from the entire generation (including dead agents)
        # Sort all agents by food count, then by energy as tiebreaker
        all_agents_sorted = sorted(current_agents, 
                                 key=lambda agent: (agent.food_count, agent.energy), 
                                 reverse=True)
        best_performer = all_agents_sorted[0] if all_agents_sorted else None
        
        # Update best agent ever
        if best_performer and best_performer.food_count > self.best_fitness_ever:
            self.best_fitness_ever = best_performer.food_count
            self.best_agent_ever = self._create_agent_copy(best_performer)
            print(f"New best agent found! Fitness: {self.best_fitness_ever}")
        
        # Calculate generation statistics
        stats = self._calculate_generation_stats(current_agents, living_agents)
        self.generation_stats.append(stats)
        
        # Select survivors
        survivors = self.select_survivors(living_agents)
        
        # Create next generation, passing the best performer for fallback
        new_agents = self.create_next_generation(survivors, best_performer)
        
        # Increment generation counter
        self.generation += 1
        
        return new_agents, stats
    
    def _calculate_generation_stats(self, all_agents, living_agents):
        """
        Calculate statistics for the current generation.
        
        Args:
            all_agents: All agents in the generation
            living_agents: Living agents sorted by fitness
            
        Returns:
            dict: Generation statistics
        """
        total_food_consumed = sum(agent.food_count for agent in all_agents)
        all_food_counts = [agent.food_count for agent in all_agents]
        best_fitness_overall = max(all_food_counts) if all_food_counts else 0
        
        if not living_agents:
            return {
                'generation': self.generation,
                'survivors': 0,
                'total_agents': len(all_agents),
                'best_fitness': best_fitness_overall,
                'avg_fitness': 0,
                'avg_energy': 0,
                'total_food_consumed': total_food_consumed
            }
        
        food_counts = [agent.food_count for agent in living_agents]
        energies = [agent.energy for agent in living_agents]
        
        return {
            'generation': self.generation,
            'survivors': len(living_agents),
            'total_agents': len(all_agents),
            'best_fitness': max(food_counts),
            'avg_fitness': sum(food_counts) / len(food_counts),
            'avg_energy': sum(energies) / len(energies),
            'total_food_consumed': total_food_consumed
        }
    
    def get_evolution_summary(self):
        """
        Get a summary of the evolutionary progress.
        
        Returns:
            dict: Summary statistics across all generations
        """
        if not self.generation_stats:
            return {}
        
        best_fitnesses = [stats['best_fitness'] for stats in self.generation_stats]
        avg_fitnesses = [stats['avg_fitness'] for stats in self.generation_stats]
        survivor_counts = [stats['survivors'] for stats in self.generation_stats]
        
        return {
            'total_generations': len(self.generation_stats),
            'best_fitness_ever': max(best_fitnesses),
            'avg_best_fitness': sum(best_fitnesses) / len(best_fitnesses),
            'avg_survivors_per_gen': sum(survivor_counts) / len(survivor_counts),
            'final_generation_best': best_fitnesses[-1] if best_fitnesses else 0,
            'improvement': best_fitnesses[-1] - best_fitnesses[0] if len(best_fitnesses) > 1 else 0
        }
    
    def evolve_generation_with_evaluation(self, current_agents, num_runs=None, num_iterations=None):
        """
        Evolve to the next generation using multi-run evaluation for selection.
        
        Args:
            current_agents: Current generation of agents
            num_runs: Number of evaluation runs per agent (default from config)
            num_iterations: Number of iterations per evaluation run (default from config)
            
        Returns:
            tuple: (new_agents, generation_stats, evaluation_summary)
        """
        print(f"\nGeneration {self.generation + 1}: Starting multi-run evaluation...")
        
        # Evaluate all agents over multiple runs
        evaluated_population = self.agent_evaluator.evaluate_population(
            current_agents, num_runs, num_iterations
        )
        
        # Get evaluation summary
        evaluation_summary = self.agent_evaluator.get_evaluation_summary(evaluated_population)
        
        # Find the best performer based on average evaluation performance
        if evaluated_population:
            # Get the best agent from evaluation results (already sorted by average performance)
            best_agent, best_results = evaluated_population[0]
            best_avg_fitness = best_results['avg_food_count']
            
            # Update best agent ever based on average performance
            if best_avg_fitness > self.best_fitness_ever:
                self.best_fitness_ever = best_avg_fitness
                self.best_agent_ever = self._create_agent_copy(best_agent)
                print(f"New best agent found! Average fitness: {self.best_fitness_ever:.2f} "
                      f"(survival rate: {best_results['survival_rate']:.1%}, "
                      f"std: {best_results['std_food_count']:.2f})")
        
        # Select top performers based on average performance
        selected_agents = self.agent_evaluator.select_top_agents(evaluated_population)
        
        # Calculate generation statistics based on original single-run performance
        # (for consistency with existing stats tracking)
        living_agents = [agent for agent in current_agents if agent.alive]
        living_agents.sort(key=lambda agent: agent.food_count, reverse=True)
        stats = self._calculate_generation_stats(current_agents, living_agents)
        
        # Add evaluation-specific stats
        stats['evaluation_summary'] = evaluation_summary
        stats['selected_agents_count'] = len(selected_agents)
        
        self.generation_stats.append(stats)
        
        # Track data for visualization
        self._update_visualization_data(evaluated_population, evaluation_summary)
        
        # Create next generation from selected agents
        new_agents = self.create_next_generation(selected_agents)
        
        # Increment generation counter
        self.generation += 1
        
        print(f"Generation {self.generation} complete with multi-run evaluation.")
        print(f"Population average performance: {evaluation_summary.get('population_avg_food_count', 0):.2f} food")
        print(f"Best agent average performance: {evaluation_summary.get('best_avg_food_count', 0):.2f} food")
        print()
        
        return new_agents, stats, evaluation_summary
    
    def save_best_agent(self, filename="best_agent.pth"):
        """
        Save the best agent to a file.
        
        Args:
            filename: Name of the file to save the best agent
        """
        if self.best_agent_ever is None:
            print("No best agent to save!")
            return False
        
        agent_data = {
            'neural_network_weights': self.best_agent_ever.neural_network.get_weights_as_dict(),
            'fitness': self.best_fitness_ever,
            'generation_found': self.generation,
            'x': self.best_agent_ever.x,
            'y': self.best_agent_ever.y,
            'orientation': self.best_agent_ever.orientation,
            'energy': self.best_agent_ever.energy,
            'food_count': self.best_agent_ever.food_count
        }
        
        torch.save(agent_data, filename)
        print(f"Best agent saved to {filename} (average fitness: {self.best_fitness_ever:.2f})")
        return True
    
    def load_best_agent(self, filename="best_agent.pth"):
        """
        Load the best agent from a file.
        
        Args:
            filename: Name of the file to load the best agent from
            
        Returns:
            Agent: Loaded agent or None if loading failed
        """
        try:
            agent_data = torch.load(filename)
            
            # Create new agent
            agent = Agent(
                agent_data['x'], 
                agent_data['y'], 
                agent_data['orientation']
            )
            agent.energy = agent_data['energy']
            agent.food_count = agent_data['food_count']
            
            # Load neural network weights
            agent.neural_network.set_weights_from_dict(agent_data['neural_network_weights'])
            
            print(f"Best agent loaded from {filename} (average fitness: {agent_data['fitness']:.2f})")
            return agent
            
        except Exception as e:
            print(f"Failed to load best agent from {filename}: {e}")
            return None
    
    def _update_visualization_data(self, evaluated_population, evaluation_summary):
        """
        Update data tracking for visualization.
        
        Args:
            evaluated_population: List of (agent, evaluation_results) tuples
            evaluation_summary: Summary of evaluation results
        """
        # Track best fitness (average) for this generation
        best_avg_fitness = evaluation_summary.get('best_avg_food_count', 0)
        self.best_fitness_history.append(best_avg_fitness)
        
        # Track all fitness values for this generation (for box plots)
        generation_fitness = [results['avg_food_count'] for _, results in evaluated_population]
        self.population_fitness_history.append(generation_fitness)
        
        # Track survival rates for this generation
        generation_survival_rates = [results['survival_rate'] for _, results in evaluated_population]
        avg_survival_rate = sum(generation_survival_rates) / len(generation_survival_rates)
        self.survival_rate_history.append(avg_survival_rate)
    
    def generate_evolution_plots(self, save_dir="evolution_plots"):
        """
        Generate and save evolution visualization plots.
        
        Args:
            save_dir: Directory to save the plots
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        
        # Generate plots
        self._plot_best_fitness_evolution(save_dir)
        self._plot_population_fitness_boxplots(save_dir)
        self._plot_survival_rate_evolution(save_dir)
        self._plot_combined_overview(save_dir)
        
        print(f"Evolution plots saved to '{save_dir}' directory")
    
    def _plot_best_fitness_evolution(self, save_dir):
        """Plot the evolution of best fitness over generations."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        generations = range(1, len(self.best_fitness_history) + 1)
        
        plt.plot(generations, self.best_fitness_history, 'b-', linewidth=2, marker='o', markersize=4)
        plt.title('Evolution of Best Fitness Across Generations', fontsize=14, fontweight='bold')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best Average Fitness (Food Count)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(self.best_fitness_history) > 1:
            z = np.polyfit(generations, self.best_fitness_history, 1)
            p = np.poly1d(z)
            plt.plot(generations, p(generations), "r--", alpha=0.8, linewidth=1, 
                    label=f'Trend (slope: {z[0]:.3f})')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/best_fitness_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_population_fitness_boxplots(self, save_dir):
        """Plot box plots of population fitness across generations."""
        import matplotlib.pyplot as plt
        
        # Sample generations for box plots (to avoid overcrowding)
        total_gens = len(self.population_fitness_history)
        if total_gens > 20:
            # Sample every nth generation to show ~20 box plots
            step = max(1, total_gens // 20)
            sampled_indices = list(range(0, total_gens, step))
            sampled_data = [self.population_fitness_history[i] for i in sampled_indices]
            sampled_labels = [f"Gen {i+1}" for i in sampled_indices]
        else:
            sampled_data = self.population_fitness_history
            sampled_labels = [f"Gen {i+1}" for i in range(total_gens)]
        
        plt.figure(figsize=(15, 8))
        box_plot = plt.boxplot(sampled_data, labels=sampled_labels, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(sampled_data)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title('Population Fitness Distribution Across Generations', fontsize=14, fontweight='bold')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Average Fitness (Food Count)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/population_fitness_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_survival_rate_evolution(self, save_dir):
        """Plot the evolution of survival rates over generations."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        generations = range(1, len(self.survival_rate_history) + 1)
        
        plt.plot(generations, self.survival_rate_history, 'g-', linewidth=2, marker='s', markersize=4)
        plt.title('Evolution of Average Survival Rate Across Generations', fontsize=14, fontweight='bold')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Average Survival Rate', fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(self.survival_rate_history) > 1:
            z = np.polyfit(generations, self.survival_rate_history, 1)
            p = np.poly1d(z)
            plt.plot(generations, p(generations), "r--", alpha=0.8, linewidth=1,
                    label=f'Trend (slope: {z[0]:.4f})')
            plt.legend()
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/survival_rate_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_overview(self, save_dir):
        """Plot a combined overview of all metrics."""
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        generations = range(1, len(self.best_fitness_history) + 1)
        
        # Best fitness evolution
        ax1.plot(generations, self.best_fitness_history, 'b-', linewidth=2, marker='o', markersize=3)
        ax1.set_title('Best Fitness Evolution', fontweight='bold')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Average Fitness')
        ax1.grid(True, alpha=0.3)
        
        # Survival rate evolution
        ax2.plot(generations, self.survival_rate_history, 'g-', linewidth=2, marker='s', markersize=3)
        ax2.set_title('Survival Rate Evolution', fontweight='bold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Average Survival Rate')
        ax2.set_ylim(0, 1.05)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax2.grid(True, alpha=0.3)
        
        # Population fitness statistics over time
        if self.population_fitness_history:
            pop_means = [np.mean(gen_fitness) for gen_fitness in self.population_fitness_history]
            pop_stds = [np.std(gen_fitness) for gen_fitness in self.population_fitness_history]
            
            ax3.plot(generations, pop_means, 'purple', linewidth=2, label='Mean')
            ax3.fill_between(generations, 
                           [m - s for m, s in zip(pop_means, pop_stds)],
                           [m + s for m, s in zip(pop_means, pop_stds)],
                           alpha=0.3, color='purple', label='±1 Std Dev')
            ax3.set_title('Population Fitness Statistics', fontweight='bold')
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Average Fitness')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Fitness improvement rate
        if len(self.best_fitness_history) > 1:
            improvement_rates = []
            for i in range(1, len(self.best_fitness_history)):
                rate = self.best_fitness_history[i] - self.best_fitness_history[i-1]
                improvement_rates.append(rate)
            
            ax4.plot(range(2, len(self.best_fitness_history) + 1), improvement_rates, 
                    'orange', linewidth=2, marker='d', markersize=3)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax4.set_title('Fitness Improvement Rate', fontweight='bold')
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Fitness Change')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Evolution Overview Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/evolution_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def reset(self):
        """Reset the evolution manager for a new run."""
        self.generation = 0
        self.generation_stats = []
        self.best_agent_ever = None
        self.best_fitness_ever = -1
        
        # Reset visualization data
        self.best_fitness_history = []
        self.population_fitness_history = []
        self.survival_rate_history = []
