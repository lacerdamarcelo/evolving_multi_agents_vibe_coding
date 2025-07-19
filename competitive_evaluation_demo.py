"""
Demonstration of the new competitive evaluation system.
This script shows the difference between individual and competitive agent evaluation.
"""

from agent_evaluator import AgentEvaluator
from environment import Environment
from evolution import EvolutionManager
import time

def demonstrate_evaluation_differences():
    """Demonstrate the difference between individual and competitive evaluation."""
    print("=== Competitive vs Individual Evaluation Demonstration ===")
    print()
    
    # Create test environment
    env = Environment()
    print(f"Created environment with {len(env.agents)} agents")
    
    # Run some initial simulation to get varied performance
    print("Running initial simulation to create performance variation...")
    for i in range(50):
        env.update()
    
    initial_stats = env.get_statistics()
    print(f"Initial state: {initial_stats['living_count']} agents alive, "
          f"{initial_stats['total_food_consumed']} total food consumed")
    print()
    
    # Take a subset of agents for demonstration
    test_agents = env.agents[:10]
    evaluator = AgentEvaluator()
    
    print("=== INDIVIDUAL EVALUATION (Legacy Method) ===")
    print("Each agent evaluated alone with only food in environment")
    start_time = time.time()
    
    individual_results = []
    for i, agent in enumerate(test_agents):
        print(f"  Evaluating agent {i+1}/10 individually...")
        result = evaluator.evaluate_agent(agent, num_runs=3, num_iterations=30)
        individual_results.append((agent, result))
    
    individual_time = time.time() - start_time
    
    # Sort by performance
    individual_results.sort(key=lambda x: x[1]['avg_food_count'], reverse=True)
    
    print(f"\nIndividual Evaluation Results (completed in {individual_time:.1f}s):")
    for i, (agent, results) in enumerate(individual_results):
        print(f"  Rank {i+1}: Avg Food: {results['avg_food_count']:.2f}, "
              f"Avg Energy: {results['avg_final_energy']:.1f}, "
              f"Success Rate: {results['survival_rate']:.1%}")
    
    print("\n" + "="*60)
    print("=== COMPETITIVE EVALUATION (Optimized Method) ===")
    print("All agents compete together in realistic scenarios - Single efficient pass!")
    start_time = time.time()
    
    competitive_results = evaluator.evaluate_population(
        test_agents, 
        num_runs=3, 
        num_iterations=30
    )
    
    competitive_time = time.time() - start_time
    
    print(f"\nCompetitive Evaluation Results (completed in {competitive_time:.1f}s):")
    for i, (agent, results) in enumerate(competitive_results):
        print(f"  Rank {i+1}: Avg Food: {results['avg_food_count']:.2f}, "
              f"Avg Energy: {results['avg_final_energy']:.1f}, "
              f"Success Rate: {results['survival_rate']:.1%}")
    
    print("\n" + "="*60)
    print("=== COMPARISON ANALYSIS ===")
    
    # Compare rankings
    individual_rankings = {id(agent): i for i, (agent, _) in enumerate(individual_results)}
    competitive_rankings = {id(agent): i for i, (agent, _) in enumerate(competitive_results)}
    
    ranking_changes = []
    for agent_id in individual_rankings:
        individual_rank = individual_rankings[agent_id]
        competitive_rank = competitive_rankings[agent_id]
        change = individual_rank - competitive_rank
        ranking_changes.append(change)
    
    print(f"Ranking Changes (Individual → Competitive):")
    for i, (agent, _) in enumerate(individual_results):
        agent_id = id(agent)
        individual_rank = individual_rankings[agent_id] + 1
        competitive_rank = competitive_rankings[agent_id] + 1
        change = individual_rank - competitive_rank
        change_str = f"+{change}" if change > 0 else str(change) if change < 0 else "0"
        print(f"  Agent originally ranked #{individual_rank} → now ranked #{competitive_rank} ({change_str})")
    
    # Performance comparison
    individual_avg_food = sum(r[1]['avg_food_count'] for r in individual_results) / len(individual_results)
    competitive_avg_food = sum(r[1]['avg_food_count'] for r in competitive_results) / len(competitive_results)
    
    print(f"\nPerformance Metrics:")
    print(f"  Individual evaluation avg food: {individual_avg_food:.3f}")
    print(f"  Competitive evaluation avg food: {competitive_avg_food:.3f}")
    print(f"  Performance difference: {competitive_avg_food - individual_avg_food:.3f}")
    
    print(f"\nEfficiency Comparison:")
    print(f"  Individual evaluation time: {individual_time:.1f}s (10 agents × 3 runs each = 30 simulations)")
    print(f"  Competitive evaluation time: {competitive_time:.1f}s (3 competitive scenarios = 3 simulations)")
    efficiency_improvement = individual_time / competitive_time if competitive_time > 0 else float('inf')
    print(f"  Efficiency improvement: {efficiency_improvement:.1f}x faster!")
    
    print(f"\nKey Differences:")
    print(f"  • Individual: Agents face no competition, only environmental challenges")
    print(f"  • Competitive: Agents must compete for resources and survive conflicts")
    print(f"  • Individual: Higher average performance (no competition)")
    print(f"  • Competitive: More realistic performance assessment")
    print(f"  • Individual: May select agents good at solo survival")
    print(f"  • Competitive: Selects agents good at competitive survival")
    print(f"  • Individual: Requires N×R simulations (N agents × R runs)")
    print(f"  • Competitive: Requires only R simulations (R competitive scenarios)")

def demonstrate_evolution_with_competitive_evaluation():
    """Demonstrate evolution using competitive evaluation."""
    print("\n" + "="*60)
    print("=== EVOLUTION WITH COMPETITIVE EVALUATION ===")
    print()
    
    # Create fresh environment and evolution manager
    env = Environment()
    evolution_manager = EvolutionManager()
    
    print(f"Starting evolution with {len(env.agents)} agents")
    
    # Run initial simulation
    print("Running initial generation...")
    for i in range(100):
        env.update()
    
    initial_stats = env.get_statistics()
    print(f"Generation 0 results: {initial_stats['living_count']} survivors, "
          f"{initial_stats['total_food_consumed']} total food consumed")
    
    # Evolve using competitive evaluation
    print("\nEvolving to next generation using competitive evaluation...")
    new_agents, gen_stats, eval_summary = evolution_manager.evolve_generation_with_evaluation(
        env.agents,
        num_runs=5,  # More runs for better assessment
        num_iterations=100
    )
    
    print(f"\nEvolution Results:")
    print(f"  Selected agents: {gen_stats['selected_agents_count']}")
    print(f"  Best competitive avg food: {eval_summary['best_avg_food_count']:.2f}")
    print(f"  Population competitive avg food: {eval_summary['population_avg_food_count']:.2f}")
    print(f"  Population survival rate: {eval_summary['population_avg_survival_rate']:.1%}")
    
    # Test new generation
    env.reset_with_new_generation(new_agents)
    print(f"\nGeneration 1 ready with {len(new_agents)} agents")
    
    print("\n✓ Competitive evolution system demonstrated successfully!")

if __name__ == "__main__":
    demonstrate_evaluation_differences()
    demonstrate_evolution_with_competitive_evaluation()
    
    print("\n" + "="*60)
    print("🎉 COMPETITIVE EVALUATION SYSTEM COMPLETE!")
    print("\nKey Benefits:")
    print("• Agents evaluated in realistic competitive scenarios")
    print("• Selection based on performance against peers")
    print("• Emergence of competitive strategies and behaviors")
    print("• More accurate fitness assessment for evolution")
    print("• Better preparation for multi-agent environments")
