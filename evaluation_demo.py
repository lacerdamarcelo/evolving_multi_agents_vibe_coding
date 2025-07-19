"""
Demonstration script for the multi-run agent evaluation system.
"""

import time
from environment import Environment
from evolution import EvolutionManager
from config import (
    NUM_EVALUATION_RUNS, EVALUATION_ITERATIONS, POPULATION_SIZE,
    NUM_GENERATIONS, ITERATIONS_PER_GENERATION
)


def run_evaluation_demo():
    """
    Demonstrate the multi-run evaluation system.
    """
    print("=" * 60)
    print("MULTI-RUN AGENT EVALUATION DEMONSTRATION")
    print("=" * 60)
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Evaluation runs per agent: {NUM_EVALUATION_RUNS}")
    print(f"Iterations per evaluation run: {EVALUATION_ITERATIONS}")
    print("=" * 60)
    
    # Initialize components
    environment = Environment()
    evolution_manager = EvolutionManager()
    
    # Run a few generations with multi-run evaluation
    num_demo_generations = 3
    
    for generation in range(num_demo_generations):
        print(f"\n{'='*20} GENERATION {generation + 1} {'='*20}")
        
        start_time = time.time()
        
        # Use the new multi-run evaluation method
        new_agents, gen_stats, eval_summary = evolution_manager.evolve_generation_with_evaluation(
            environment.agents,
            num_runs=NUM_EVALUATION_RUNS,
            num_iterations=EVALUATION_ITERATIONS
        )
        
        evaluation_time = time.time() - start_time
        
        # Print detailed results
        print(f"\nGeneration {generation + 1} Results:")
        print(f"  Evaluation time: {evaluation_time:.2f} seconds")
        print(f"  Population size evaluated: {eval_summary['population_size']}")
        print(f"  Best average food count: {eval_summary['best_avg_food_count']:.2f}")
        print(f"  Population average food count: {eval_summary['population_avg_food_count']:.2f}")
        print(f"  Population std food count: {eval_summary['population_std_food_count']:.2f}")
        print(f"  Best survival rate: {eval_summary['best_survival_rate']:.1%}")
        print(f"  Population average survival rate: {eval_summary['population_avg_survival_rate']:.1%}")
        print(f"  Agents with perfect survival: {eval_summary['agents_with_perfect_survival']}")
        print(f"  Selected agents for next generation: {gen_stats['selected_agents_count']}")
        
        # Update environment with new generation
        environment.reset_with_new_generation(new_agents)
        
        print(f"  Next generation created with {len(new_agents)} agents")
    
    # Final summary
    print("\n" + "=" * 60)
    print("EVALUATION DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    final_summary = evolution_manager.get_evolution_summary()
    print(f"Total generations completed: {final_summary.get('total_generations', 0)}")
    print(f"Best fitness achieved: {final_summary.get('best_fitness_ever', 0)}")
    print(f"Average best fitness: {final_summary.get('avg_best_fitness', 0):.2f}")
    print(f"Overall improvement: {final_summary.get('improvement', 0)}")
    
    return evolution_manager, environment


def compare_selection_methods():
    """
    Compare traditional single-run selection vs multi-run evaluation selection.
    """
    print("\n" + "=" * 60)
    print("COMPARING SELECTION METHODS")
    print("=" * 60)
    
    # Create two identical starting populations
    environment1 = Environment()
    environment2 = Environment()
    
    # Copy agents from environment1 to environment2 to ensure identical starting conditions
    import copy
    environment2.agents = copy.deepcopy(environment1.agents)
    
    evolution_manager1 = EvolutionManager()
    evolution_manager2 = EvolutionManager()
    
    print("Running one generation with each method...")
    
    # Method 1: Traditional single-run selection
    print("\n--- Traditional Single-Run Selection ---")
    start_time = time.time()
    new_agents1, stats1 = evolution_manager1.evolve_generation(environment1.agents)
    time1 = time.time() - start_time
    
    print(f"Time taken: {time1:.2f} seconds")
    print(f"Survivors selected: {stats1['survivors']}")
    print(f"Best fitness: {stats1['best_fitness']}")
    print(f"Average fitness: {stats1['avg_fitness']:.2f}")
    
    # Method 2: Multi-run evaluation selection
    print("\n--- Multi-Run Evaluation Selection ---")
    start_time = time.time()
    new_agents2, stats2, eval_summary = evolution_manager2.evolve_generation_with_evaluation(
        environment2.agents,
        num_runs=NUM_EVALUATION_RUNS,
        num_iterations=EVALUATION_ITERATIONS
    )
    time2 = time.time() - start_time
    
    print(f"Time taken: {time2:.2f} seconds")
    print(f"Agents selected: {stats2['selected_agents_count']}")
    print(f"Best average performance: {eval_summary['best_avg_food_count']:.2f}")
    print(f"Population average performance: {eval_summary['population_avg_food_count']:.2f}")
    
    # Comparison
    print(f"\n--- Comparison ---")
    print(f"Time difference: {time2 - time1:.2f} seconds ({time2/time1:.1f}x slower)")
    print(f"Traditional method selected {stats1['survivors']} agents")
    print(f"Multi-run method selected {stats2['selected_agents_count']} agents")
    print(f"Multi-run method provides more reliable performance assessment")
    
    return (evolution_manager1, environment1), (evolution_manager2, environment2)


def main():
    """
    Main demonstration function.
    """
    print("Starting Multi-Run Agent Evaluation Demonstration...")
    
    # Run the basic evaluation demo
    evolution_manager, environment = run_evaluation_demo()
    
    # Compare selection methods
    traditional_setup, multirun_setup = compare_selection_methods()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Benefits of Multi-Run Evaluation:")
    print("1. More reliable agent performance assessment")
    print("2. Reduces impact of random environmental factors")
    print("3. Better selection of consistently performing agents")
    print("4. Provides statistical confidence in agent capabilities")
    print("5. Configurable evaluation parameters for different scenarios")
    
    print(f"\nConfiguration used:")
    print(f"  - Evaluation runs per agent: {NUM_EVALUATION_RUNS}")
    print(f"  - Iterations per evaluation: {EVALUATION_ITERATIONS}")
    print(f"  - Population size: {POPULATION_SIZE}")
    
    print("\nTo use multi-run evaluation in your evolution:")
    print("  evolution_manager.evolve_generation_with_evaluation(agents)")
    print("  instead of:")
    print("  evolution_manager.evolve_generation(agents)")


if __name__ == "__main__":
    main()
