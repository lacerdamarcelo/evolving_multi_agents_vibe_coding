# Parallel Evaluation Implementation

This document describes the parallel evaluation system implemented for the evolutionary agent simulation using mpire.

## Overview

The parallel evaluation system allows multiple competitive scenarios to run simultaneously across multiple CPU cores, significantly reducing the time required for agent population evaluation during evolution.

## Key Features

### 1. **Automatic Parallel/Sequential Switching**
- Automatically attempts parallel execution when enabled
- Falls back to sequential execution if parallel processing fails
- Configurable via `ENABLE_PARALLEL_EVALUATION` in config.py

### 2. **Memory-Aware Worker Management**
- Automatically estimates memory requirements
- Adjusts worker count based on available system memory
- Warns users when memory usage might be high
- Configurable memory warning threshold

### 3. **PyTorch Compatibility**
- Uses 'spawn' start method for proper PyTorch model serialization
- Handles neural network state dictionaries correctly across processes
- Ensures proper random seed management per worker

### 4. **Progress Reporting**
- Real-time progress bars during parallel execution
- Clear indication of parallel vs sequential execution mode
- Detailed performance metrics and timing information

## Configuration Parameters

Add these parameters to your `config.py`:

```python
# Parallel processing parameters
NUM_WORKERS = max(1, os.cpu_count() - 1)  # CPU count - 1
ENABLE_PARALLEL_EVALUATION = True  # Allow users to disable if needed
MEMORY_WARNING_THRESHOLD_GB = 8.0  # Warn if estimated memory usage exceeds this
```

## Usage

The parallel evaluation is transparent to existing code. Simply use the `AgentEvaluator` as before:

```python
from agent_evaluator import AgentEvaluator

evaluator = AgentEvaluator()
results = evaluator.evaluate_population(agents, num_runs=30, num_iterations=500)
```

## Performance Characteristics

### When Parallel Processing Helps Most:
- Large number of evaluation runs (>10)
- Longer simulation iterations (>100)
- Larger population sizes (>20 agents)
- Systems with multiple CPU cores

### When Sequential Might Be Better:
- Small workloads (few runs, short iterations)
- Memory-constrained systems
- Debugging scenarios where deterministic execution is needed

## Technical Implementation

### Worker Function
- `run_competitive_scenario_worker()` - Module-level function for multiprocessing
- Handles agent deserialization and environment setup
- Manages random seeds for reproducible but varied scenarios

### Memory Management
- Estimates memory usage based on population size and worker count
- Automatically reduces worker count if memory constraints detected
- Provides warnings when approaching memory limits

### Error Handling
- Comprehensive exception handling with automatic fallback
- Clear error messages for debugging
- Graceful degradation when parallel processing unavailable

## Dependencies

The implementation requires these additional packages:

```
mpire>=2.10.0  # Multiprocessing library with progress bars
psutil>=5.8.0  # System memory monitoring
```

Install with:
```bash
pip install mpire psutil
```

## Testing

Run the test scripts to verify the implementation:

```bash
# Basic functionality test
python test_parallel_evaluation.py

# Performance comparison test
python test_parallel_performance.py
```

## Performance Results

Based on testing, the parallel implementation provides:

- **Consistent Results**: Identical evaluation outcomes between parallel and sequential execution
- **Memory Safety**: Automatic worker count adjustment based on available memory
- **Robust Fallback**: Seamless fallback to sequential execution when needed
- **Progress Tracking**: Clear progress indication during long-running evaluations

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure mpire and psutil are installed
2. **Memory Warnings**: Reduce `NUM_WORKERS` or increase available system memory
3. **Slow Performance**: Check if workload is large enough to benefit from parallelization
4. **Process Spawn Errors**: Verify PyTorch compatibility and spawn method support

### Debug Mode:
Set `ENABLE_PARALLEL_EVALUATION = False` in config.py to force sequential execution for debugging.

## Future Enhancements

Potential improvements for future versions:

1. **Dynamic Load Balancing**: Adjust worker allocation based on scenario completion times
2. **Distributed Computing**: Extend to multiple machines for very large populations
3. **GPU Acceleration**: Leverage GPU computing for neural network operations
4. **Adaptive Batching**: Automatically determine optimal batch sizes for different workloads

## Conclusion

The parallel evaluation system provides a robust, memory-aware, and user-friendly way to accelerate evolutionary agent simulations. It maintains full compatibility with existing code while providing significant performance improvements for appropriate workloads.

The implementation prioritizes reliability and ease of use, with automatic fallback mechanisms ensuring that simulations always complete successfully, whether running in parallel or sequential mode.
