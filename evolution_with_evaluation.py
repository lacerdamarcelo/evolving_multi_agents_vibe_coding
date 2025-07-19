"""
Example script showing how to run evolution with multi-run evaluation.
"""

from environment import Environment
from evolution import EvolutionManager
from config import (
    NUM_EVALUATION_RUNS, EVALUATION_ITERATIONS, NUM_GENERATIONS,
    ITERATIONS_PER_GENERATION
)


def run_evolution_with_evaluation(num_generations=1000, use_evaluation=True):
    """
    Run evolution with optional multi-run evaluation.
    
    Args:
        num_generations: Number of generations to run
        use_evaluation: Whether to use multi-run evaluation or traditional method
    """
    print("=" * 60)
    print(f"RUNNING EVOLUTION {'WITH' if use_evaluation else 'WITHOUT'} MULTI-RUN EVALUATION")
    print("=" * 60)
    
    # Initialize components
    environment = Environment()
    evolution_manager = EvolutionManager()
    
    print(f"Starting population: {len(environment.agents)} agents")
    print(f"Generations to run: {num_generations}")
    
    if use_evaluation:
        print(f"Evaluation runs per agent: {NUM_EVALUATION_RUNS}")
        print(f"Iterations per evaluation: {EVALUATION_ITERATIONS}")
    else:
        print(f"Iterations per generation: {ITERATIONS_PER_GENERATION}")
    
    print("=" * 60)
    
    # Run evolution
    for generation in range(num_generations):
        print(f"\n--- Generation {generation + 1} ---")
        
        if use_evaluation:
            # Use multi-run evaluation
            new_agents, stats, eval_summary = evolution_manager.evolve_generation_with_evaluation(
                environment.agents
            )
            
            # Print evaluation-specific results
            print(f"Best average performance: {eval_summary['best_avg_food_count']:.2f} food")
            print(f"Population average performance: {eval_summary['population_avg_food_count']:.2f} food")
            print(f"Population performance std: {eval_summary['population_std_food_count']:.2f}")
            print(f"Best survival rate: {eval_summary['best_survival_rate']:.1%}")
            print(f"Selected agents: {stats['selected_agents_count']}")
            
        else:
            # Use traditional single-run evolution
            # First, run the generation in the environment
            for iteration in range(ITERATIONS_PER_GENERATION):
                environment.update()
            
            # Then evolve
            new_agents, stats = evolution_manager.evolve_generation(environment.agents)
            
            # Print traditional results
            print(f"Survivors: {stats['survivors']}")
            print(f"Best fitness: {stats['best_fitness']}")
            print(f"Average fitness: {stats['avg_fitness']:.2f}")
        
        # Update environment with new generation
        environment.reset_with_new_generation(new_agents)
        print(f"Next generation: {len(new_agents)} agents")
    
    # Final summary
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    
    summary = evolution_manager.get_evolution_summary()
    print(f"Total generations: {summary.get('total_generations', 0)}")
    print(f"Best fitness ever: {summary.get('best_fitness_ever', 0)}")
    print(f"Average best fitness: {summary.get('avg_best_fitness', 0):.2f}")
    print(f"Overall improvement: {summary.get('improvement', 0)}")
    
    return evolution_manager, environment


def compare_evolution_methods():
    """
    Compare evolution with and without multi-run evaluation.
    """
    print("\n" + "=" * 80)
    print("COMPARING EVOLUTION METHODS")
    print("=" * 80)
    
    import time
    
    # Run traditional evolution
    print("\n1. Traditional Single-Run Evolution:")
    start_time = time.time()
    traditional_manager, traditional_env = run_evolution_with_evaluation(
        num_generations=3, use_evaluation=False
    )
    traditional_time = time.time() - start_time
    
    # Run multi-run evaluation evolution
    print("\n2. Multi-Run Evaluation Evolution:")
    start_time = time.time()
    evaluation_manager, evaluation_env = run_evolution_with_evaluation(
        num_generations=3, use_evaluation=True
    )
    evaluation_time = time.time() - start_time
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    traditional_summary = traditional_manager.get_evolution_summary()
    evaluation_summary = evaluation_manager.get_evolution_summary()
    
    print(f"Traditional Evolution:")
    print(f"  Time taken: {traditional_time:.2f} seconds")
    print(f"  Best fitness: {traditional_summary.get('best_fitness_ever', 0)}")
    print(f"  Average best fitness: {traditional_summary.get('avg_best_fitness', 0):.2f}")
    
    print(f"\nMulti-Run Evaluation Evolution:")
    print(f"  Time taken: {evaluation_time:.2f} seconds")
    print(f"  Best fitness: {evaluation_summary.get('best_fitness_ever', 0)}")
    print(f"  Average best fitness: {evaluation_summary.get('avg_best_fitness', 0):.2f}")
    
    print(f"\nTime overhead: {evaluation_time - traditional_time:.2f} seconds")
    print(f"Speed ratio: {evaluation_time / traditional_time:.1f}x slower")
    
    print("\nMulti-run evaluation provides:")
    print("- More reliable agent selection")
    print("- Statistical confidence in performance")
    print("- Better long-term evolutionary progress")
    print("- Reduced impact of environmental randomness")


def main():
    """
    Main function demonstrating different evolution approaches.
    """
    print("Evolution with Multi-Run Evaluation Demo")
    print("Choose an option:")
    print("1. Run evolution with multi-run evaluation")
    print("2. Run evolution without multi-run evaluation")
    print("3. Compare both methods")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            run_evolution_with_evaluation(num_generations=5, use_evaluation=True)
        elif choice == "2":
            run_evolution_with_evaluation(num_generations=5, use_evaluation=False)
        elif choice == "3":
            compare_evolution_methods()
        else:
            print("Invalid choice. Running comparison by default.")
            compare_evolution_methods()
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Running default comparison...")
        compare_evolution_methods()


if __name__ == "__main__":
    main()
