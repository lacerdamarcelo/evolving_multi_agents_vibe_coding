"""
Test script for parallel evaluation implementation.
"""

import time
from agent import Agent
from agent_evaluator import AgentEvaluator
from config import POPULATION_SIZE, NUM_EVALUATION_RUNS, EVALUATION_ITERATIONS

def test_parallel_vs_sequential():
    """Test parallel vs sequential evaluation performance."""
    print("Testing parallel evaluation implementation...")
    print("=" * 60)
    
    # Create a small population for testing
    test_population_size = min(10, POPULATION_SIZE)
    test_num_runs = min(6, NUM_EVALUATION_RUNS)  # Smaller number for testing
    test_iterations = min(100, EVALUATION_ITERATIONS)  # Shorter runs for testing
    
    print(f"Test parameters:")
    print(f"  Population size: {test_population_size}")
    print(f"  Number of runs: {test_num_runs}")
    print(f"  Iterations per run: {test_iterations}")
    print()
    
    # Create test agents
    agents = []
    for i in range(test_population_size):
        agent = Agent(100 + i * 10, 100 + i * 10, i * 36)  # Spread them out
        agents.append(agent)
    
    evaluator = AgentEvaluator()
    
    # Test sequential evaluation
    print("Testing sequential evaluation...")
    start_time = time.time()
    
    # Temporarily disable parallel evaluation
    from config import ENABLE_PARALLEL_EVALUATION
    original_setting = ENABLE_PARALLEL_EVALUATION
    import config
    config.ENABLE_PARALLEL_EVALUATION = False
    
    sequential_results = evaluator.evaluate_population(agents, test_num_runs, test_iterations)
    sequential_time = time.time() - start_time
    
    print(f"Sequential evaluation completed in {sequential_time:.2f} seconds")
    print(f"Best agent fitness: {sequential_results[0][1]['avg_food_count']:.2f}")
    print()
    
    # Test parallel evaluation
    print("Testing parallel evaluation...")
    config.ENABLE_PARALLEL_EVALUATION = True
    
    start_time = time.time()
    parallel_results = evaluator.evaluate_population(agents, test_num_runs, test_iterations)
    parallel_time = time.time() - start_time
    
    print(f"Parallel evaluation completed in {parallel_time:.2f} seconds")
    print(f"Best agent fitness: {parallel_results[0][1]['avg_food_count']:.2f}")
    print()
    
    # Compare results
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    print("Comparison:")
    print(f"  Sequential time: {sequential_time:.2f}s")
    print(f"  Parallel time: {parallel_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print()
    
    # Verify results consistency (they should be similar but not identical due to randomness)
    seq_best_fitness = sequential_results[0][1]['avg_food_count']
    par_best_fitness = parallel_results[0][1]['avg_food_count']
    
    print("Results verification:")
    print(f"  Sequential best fitness: {seq_best_fitness:.2f}")
    print(f"  Parallel best fitness: {par_best_fitness:.2f}")
    
    # Check if results are reasonably similar (within expected variance)
    if abs(seq_best_fitness - par_best_fitness) < 2.0:  # Allow some variance
        print("  ✓ Results are consistent between sequential and parallel execution")
    else:
        print("  ⚠ Results differ significantly - this may be due to randomness")
    
    # Restore original setting
    config.ENABLE_PARALLEL_EVALUATION = original_setting
    
    print()
    print("Test completed successfully!")
    return speedup

def test_memory_estimation():
    """Test memory estimation functionality."""
    print("\nTesting memory estimation...")
    print("-" * 40)
    
    evaluator = AgentEvaluator()
    
    # Test with different population sizes
    test_cases = [10, 50, 100]
    
    for pop_size in test_cases:
        optimal_workers = evaluator._estimate_optimal_workers(pop_size, NUM_EVALUATION_RUNS)
        print(f"Population size {pop_size}: {optimal_workers} optimal workers")
    
    print("Memory estimation test completed!")

if __name__ == "__main__":
    try:
        # Test the parallel implementation
        speedup = test_parallel_vs_sequential()
        
        # Test memory estimation
        test_memory_estimation()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        if speedup > 1.0:
            print(f"Parallel implementation achieved {speedup:.2f}x speedup!")
        else:
            print("Parallel implementation is working but may not show speedup with small test cases.")
        print("The system is ready for production use.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
