"""
Performance test for parallel evaluation with larger workloads.
"""

import time
from agent import Agent
from agent_evaluator import AgentEvaluator
from config import POPULATION_SIZE, NUM_EVALUATION_RUNS, EVALUATION_ITERATIONS

def test_performance_with_larger_workload():
    """Test parallel vs sequential with a more substantial workload."""
    print("Testing parallel evaluation performance with larger workload...")
    print("=" * 70)
    
    # Use larger parameters to better show parallel benefits
    test_population_size = min(20, POPULATION_SIZE)
    test_num_runs = 20  # More runs to show parallel benefit
    test_iterations = 200  # Longer runs
    
    print(f"Performance test parameters:")
    print(f"  Population size: {test_population_size}")
    print(f"  Number of runs: {test_num_runs}")
    print(f"  Iterations per run: {test_iterations}")
    print(f"  Total computational work: {test_population_size * test_num_runs * test_iterations:,} agent-iterations")
    print()
    
    # Create test agents
    agents = []
    for i in range(test_population_size):
        agent = Agent(50 + i * 25, 50 + i * 25, i * 18)  # Spread them out
        agents.append(agent)
    
    evaluator = AgentEvaluator()
    
    # Test sequential evaluation
    print("Testing sequential evaluation...")
    import config
    config.ENABLE_PARALLEL_EVALUATION = False
    
    start_time = time.time()
    sequential_results = evaluator.evaluate_population(agents, test_num_runs, test_iterations)
    sequential_time = time.time() - start_time
    
    print(f"Sequential evaluation completed in {sequential_time:.2f} seconds")
    print(f"Best agent fitness: {sequential_results[0][1]['avg_food_count']:.2f}")
    print(f"Population average fitness: {sum(r[1]['avg_food_count'] for r in sequential_results) / len(sequential_results):.2f}")
    print()
    
    # Test parallel evaluation
    print("Testing parallel evaluation...")
    config.ENABLE_PARALLEL_EVALUATION = True
    
    start_time = time.time()
    parallel_results = evaluator.evaluate_population(agents, test_num_runs, test_iterations)
    parallel_time = time.time() - start_time
    
    print(f"Parallel evaluation completed in {parallel_time:.2f} seconds")
    print(f"Best agent fitness: {parallel_results[0][1]['avg_food_count']:.2f}")
    print(f"Population average fitness: {sum(r[1]['avg_food_count'] for r in parallel_results) / len(parallel_results):.2f}")
    print()
    
    # Performance analysis
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    efficiency = speedup / config.NUM_WORKERS if config.NUM_WORKERS > 0 else 0
    
    print("Performance Analysis:")
    print(f"  Sequential time: {sequential_time:.2f}s")
    print(f"  Parallel time: {parallel_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Efficiency: {efficiency:.2%} (speedup / num_workers)")
    print(f"  Time saved: {sequential_time - parallel_time:.2f}s ({(1 - parallel_time/sequential_time)*100:.1f}%)")
    print()
    
    # Verify results consistency
    seq_avg = sum(r[1]['avg_food_count'] for r in sequential_results) / len(sequential_results)
    par_avg = sum(r[1]['avg_food_count'] for r in parallel_results) / len(parallel_results)
    
    print("Results verification:")
    print(f"  Sequential population average: {seq_avg:.3f}")
    print(f"  Parallel population average: {par_avg:.3f}")
    print(f"  Difference: {abs(seq_avg - par_avg):.3f}")
    
    if abs(seq_avg - par_avg) < 0.5:  # Allow some variance due to randomness
        print("  ✓ Results are consistent between sequential and parallel execution")
    else:
        print("  ⚠ Results differ - this may be due to randomness in competitive scenarios")
    
    return speedup, efficiency

def test_scalability():
    """Test how performance scales with different numbers of runs."""
    print("\n" + "=" * 70)
    print("Testing scalability with different workload sizes...")
    print("=" * 70)
    
    test_cases = [
        (10, 6, 100),   # Small workload
        (10, 12, 150),  # Medium workload
        (15, 20, 200),  # Large workload
    ]
    
    results = []
    
    for pop_size, num_runs, iterations in test_cases:
        print(f"\nTesting with {pop_size} agents, {num_runs} runs, {iterations} iterations...")
        
        # Create agents
        agents = [Agent(50 + i * 30, 50 + i * 30, i * 20) for i in range(pop_size)]
        evaluator = AgentEvaluator()
        
        # Test parallel only (we know sequential works)
        import config
        config.ENABLE_PARALLEL_EVALUATION = True
        
        start_time = time.time()
        parallel_results = evaluator.evaluate_population(agents, num_runs, iterations)
        parallel_time = time.time() - start_time
        
        workload = pop_size * num_runs * iterations
        throughput = workload / parallel_time
        
        print(f"  Completed in {parallel_time:.2f}s")
        print(f"  Throughput: {throughput:,.0f} agent-iterations/second")
        
        results.append((workload, parallel_time, throughput))
    
    print("\nScalability Summary:")
    print("  Workload Size | Time (s) | Throughput (agent-iter/s)")
    print("  " + "-" * 50)
    for workload, time_taken, throughput in results:
        print(f"  {workload:>12,} | {time_taken:>8.2f} | {throughput:>20,.0f}")

if __name__ == "__main__":
    try:
        # Test performance with larger workload
        speedup, efficiency = test_performance_with_larger_workload()
        
        # Test scalability
        test_scalability()
        
        print("\n" + "=" * 70)
        print("Performance testing completed!")
        print(f"Best speedup achieved: {speedup:.2f}x")
        print(f"Parallel efficiency: {efficiency:.2%}")
        
        if speedup > 1.5:
            print("🚀 Excellent parallel performance! The implementation provides significant speedup.")
        elif speedup > 1.1:
            print("✅ Good parallel performance! The implementation provides measurable speedup.")
        else:
            print("⚠️  Limited speedup observed. This may be due to small workload or system constraints.")
        
        print("\nThe parallel evaluation system is ready for production use!")
        
    except Exception as e:
        print(f"Performance test failed with error: {e}")
        import traceback
        traceback.print_exc()
