"""
Agent evaluation system for multi-run performance assessment.
"""

import copy
import statistics
from environment import Environment
from config import (
    NUM_EVALUATION_RUNS, EVALUATION_ITERATIONS, SURVIVAL_RATE
)


class AgentEvaluator:
    """
    Evaluates agents over multiple runs to determine their average performance.
    """
    
    def __init__(self):
        """Initialize the agent evaluator."""
        self.evaluation_results = {}
    
    def evaluate_agent(self, agent, num_runs=None, num_iterations=None):
        """
        Evaluate a single agent over multiple runs.
        
        Args:
            agent: Agent to evaluate
            num_runs: Number of evaluation runs (default from config)
            num_iterations: Number of iterations per run (default from config)
            
        Returns:
            dict: Evaluation results containing average performance metrics
        """
        if num_runs is None:
            num_runs = NUM_EVALUATION_RUNS
        if num_iterations is None:
            num_iterations = EVALUATION_ITERATIONS
        
        run_results = []
        
        for run_idx in range(num_runs):
            # Create a copy of the agent for this run
            agent_copy = self._create_agent_copy(agent)
            
            # Run evaluation
            result = self._run_single_evaluation(agent_copy, num_iterations)
            run_results.append(result)
        
        # Calculate average performance metrics
        avg_results = self._calculate_average_results(run_results)
        
        return avg_results
    
    def evaluate_population(self, agents, num_runs=None, num_iterations=None):
        """
        Evaluate an entire population of agents in competitive scenarios.
        
        Args:
            agents: List of agents to evaluate
            num_runs: Number of competitive evaluation runs
            num_iterations: Number of iterations per run
            
        Returns:
            list: List of tuples (agent, evaluation_results) sorted by average performance
        """
        if num_runs is None:
            num_runs = NUM_EVALUATION_RUNS
        if num_iterations is None:
            num_iterations = EVALUATION_ITERATIONS
        
        print(f"Evaluating population of {len(agents)} agents in {num_runs} competitive scenarios...")
        
        # Initialize performance tracking for all agents
        agent_performances = [[] for _ in agents]
        
        # Run multiple competitive scenarios
        for run_idx in range(num_runs):
            print(f"  Running competitive scenario {run_idx + 1}/{num_runs}...")
            
            # Run one competitive scenario with all agents
            scenario_results = self._run_competitive_scenario(agents, num_iterations, run_idx)
            
            # Store results for each agent
            for agent_idx, result in enumerate(scenario_results):
                agent_performances[agent_idx].append(result)
        
        # Calculate average performance for each agent
        population_results = []
        for agent_idx, agent in enumerate(agents):
            # Calculate average performance across all scenarios
            avg_results = self._calculate_average_results(agent_performances[agent_idx])
            
            # Store results
            population_results.append((agent, avg_results))
        
        # Sort by average performance (food count, survival rate, then energy as tiebreaker)
        population_results.sort(
            key=lambda x: (x[1]['avg_food_count'], x[1]['survival_rate'], x[1]['avg_final_energy']), 
            reverse=True
        )
        
        return population_results
    
    def select_top_agents(self, evaluated_population, selection_rate=None):
        """
        Select the top performing agents based on their evaluation results.
        
        Args:
            evaluated_population: List of (agent, evaluation_results) tuples
            selection_rate: Fraction of agents to select (default from SURVIVAL_RATE)
            
        Returns:
            list: Selected top-performing agents
        """
        if selection_rate is None:
            selection_rate = SURVIVAL_RATE
        
        num_selected = max(1, int(len(evaluated_population) * selection_rate))
        
        # Take the top performers
        top_performers = evaluated_population[:num_selected]
        
        # Extract just the agents
        selected_agents = [agent for agent, _ in top_performers]
        
        # Print selection summary
        print(f"\nSelected top {num_selected} agents based on average performance:")
        for i, (agent, results) in enumerate(top_performers):
            print(f"  Rank {i+1}: Avg Food: {results['avg_food_count']:.2f}, "
                  f"Avg Energy: {results['avg_final_energy']:.1f}, "
                  f"Success Rate: {results['survival_rate']:.1%}")
        
        return selected_agents
    
    def _create_agent_copy(self, original_agent):
        """
        Create a deep copy of an agent for evaluation.
        
        Args:
            original_agent: Original agent to copy
            
        Returns:
            Agent: Deep copy of the agent
        """
        from config import INITIAL_ENERGY
        
        # Create a new agent with the same initial parameters
        agent_copy = copy.deepcopy(original_agent)
        
        # Reset the agent's state for evaluation
        agent_copy.energy = INITIAL_ENERGY
        agent_copy.food_count = 0
        agent_copy.alive = True
        
        return agent_copy
    
    def _run_competitive_scenario(self, agents, num_iterations, run_idx):
        """
        Run a competitive scenario with all agents competing together.
        
        Args:
            agents: List of all agents to compete
            num_iterations: Number of iterations to run
            run_idx: Index of this run (for random seed variation)
            
        Returns:
            list: Results for each agent in the same order as input
        """
        import random
        from config import ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT
        
        # Set a different random seed for each run to ensure variety
        random.seed(run_idx * 1000)
        
        # Create copies of all agents for this competitive run
        agent_copies = []
        for agent in agents:
            agent_copy = self._create_agent_copy(agent)
            
            # Reset agent position randomly for this run
            agent_copy.x = random.uniform(0, ENVIRONMENT_WIDTH)
            agent_copy.y = random.uniform(0, ENVIRONMENT_HEIGHT)
            agent_copy.orientation = random.uniform(0, 360)
            
            agent_copies.append(agent_copy)
        
        # Create environment with all competing agents
        environment = Environment()
        environment.agents = agent_copies
        
        # Run the competitive scenario
        final_iteration = 0
        for iteration in range(num_iterations):
            # Check if any agents are still alive
            living_agents = [agent for agent in environment.agents if agent.alive]
            if not living_agents:
                break
            
            # Update the environment (all agents compete)
            environment.update()
            final_iteration = iteration
        
        # Collect results for each agent
        results = []
        for agent_copy in agent_copies:
            result = {
                'final_food_count': agent_copy.food_count,
                'final_energy': agent_copy.energy,
                'survived': agent_copy.alive,
                'iterations_survived': final_iteration + 1 if agent_copy.alive else final_iteration
            }
            results.append(result)
        
        # Reset random seed to avoid affecting other parts of the system
        import time
        random.seed(int(time.time()))
        
        return results

    def _run_single_evaluation(self, agent, num_iterations):
        """
        Run a single evaluation of an agent (legacy method for individual evaluation).
        
        Args:
            agent: Agent to evaluate
            num_iterations: Number of iterations to run
            
        Returns:
            dict: Results of the evaluation run
        """
        # Create a minimal environment for evaluation
        # We'll create a simple environment with just this agent and food
        environment = Environment()
        
        # Replace the environment's agents with just our test agent
        environment.agents = [agent]
        
        # Reset agent position randomly
        import random
        from config import ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT
        agent.x = random.uniform(0, ENVIRONMENT_WIDTH)
        agent.y = random.uniform(0, ENVIRONMENT_HEIGHT)
        agent.orientation = random.uniform(0, 360)
        
        # Run the evaluation
        for iteration in range(num_iterations):
            if not agent.alive:
                break
            
            # Update the environment (this will update the agent)
            environment.update()
        
        # Collect results
        result = {
            'final_food_count': agent.food_count,
            'final_energy': agent.energy,
            'survived': agent.alive,
            'iterations_survived': iteration + 1 if agent.alive else iteration
        }
        
        return result
    
    def _calculate_average_results(self, run_results):
        """
        Calculate average results from multiple runs.
        
        Args:
            run_results: List of individual run results
            
        Returns:
            dict: Average performance metrics
        """
        if not run_results:
            return {
                'avg_food_count': 0.0,
                'avg_final_energy': 0.0,
                'survival_rate': 0.0,
                'avg_iterations_survived': 0.0,
                'std_food_count': 0.0,
                'std_final_energy': 0.0,
                'num_runs': 0
            }
        
        # Extract metrics from all runs
        food_counts = [result['final_food_count'] for result in run_results]
        final_energies = [result['final_energy'] for result in run_results]
        survivals = [result['survived'] for result in run_results]
        iterations_survived = [result['iterations_survived'] for result in run_results]
        
        # Calculate averages and standard deviations
        avg_results = {
            'avg_food_count': statistics.mean(food_counts),
            'avg_final_energy': statistics.mean(final_energies),
            'survival_rate': sum(survivals) / len(survivals),
            'avg_iterations_survived': statistics.mean(iterations_survived),
            'std_food_count': statistics.stdev(food_counts) if len(food_counts) > 1 else 0.0,
            'std_final_energy': statistics.stdev(final_energies) if len(final_energies) > 1 else 0.0,
            'num_runs': len(run_results),
            'max_food_count': max(food_counts),
            'min_food_count': min(food_counts)
        }
        
        return avg_results
    
    def get_evaluation_summary(self, evaluated_population):
        """
        Get a summary of the population evaluation.
        
        Args:
            evaluated_population: List of (agent, evaluation_results) tuples
            
        Returns:
            dict: Summary statistics
        """
        if not evaluated_population:
            return {}
        
        # Extract all evaluation results
        all_results = [results for _, results in evaluated_population]
        
        # Calculate population-wide statistics
        avg_food_counts = [results['avg_food_count'] for results in all_results]
        avg_energies = [results['avg_final_energy'] for results in all_results]
        survival_rates = [results['survival_rate'] for results in all_results]
        
        summary = {
            'population_size': len(evaluated_population),
            'best_avg_food_count': max(avg_food_counts),
            'worst_avg_food_count': min(avg_food_counts),
            'population_avg_food_count': statistics.mean(avg_food_counts),
            'population_std_food_count': statistics.stdev(avg_food_counts) if len(avg_food_counts) > 1 else 0.0,
            'best_survival_rate': max(survival_rates),
            'population_avg_survival_rate': statistics.mean(survival_rates),
            'agents_with_perfect_survival': sum(1 for rate in survival_rates if rate >= 1.0)
        }
        
        return summary
