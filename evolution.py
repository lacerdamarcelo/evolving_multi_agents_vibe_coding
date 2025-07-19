"""
Evolutionary algorithm implementation for agent evolution.
"""

import random
import math
from agent import Agent
from neural_network import create_offspring_network
from agent_evaluator import AgentEvaluator
from config import (
    POPULATION_SIZE, SURVIVAL_RATE, ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT,
    NUM_EVALUATION_RUNS, EVALUATION_ITERATIONS
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
        
        # Create offspring to fill remaining population
        while len(new_agents) < POPULATION_SIZE:
            # Randomly select a parent from survivors
            parent = random.choice(survivors)
            offspring = self._create_offspring(parent)
            new_agents.append(offspring)
        
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
        
        # Create next generation from selected agents
        new_agents = self.create_next_generation(selected_agents)
        
        # Increment generation counter
        self.generation += 1
        
        print(f"Generation {self.generation} complete with multi-run evaluation.")
        print(f"Population average performance: {evaluation_summary.get('population_avg_food_count', 0):.2f} food")
        print(f"Best agent average performance: {evaluation_summary.get('best_avg_food_count', 0):.2f} food")
        print()
        
        return new_agents, stats, evaluation_summary
    
    def reset(self):
        """Reset the evolution manager for a new run."""
        self.generation = 0
        self.generation_stats = []
