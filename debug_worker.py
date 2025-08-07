"""
Debug script to understand mpire argument passing.
"""

def debug_worker(worker_args, worker_id=None, shared_objects=None, worker_state=None):
    """Debug worker function to see what arguments are passed."""
    print(f"Worker {worker_id} received:")
    print(f"  worker_args type: {type(worker_args)}")
    print(f"  worker_args: {worker_args}")
    print(f"  worker_id: {worker_id}")
    print(f"  shared_objects: {shared_objects}")
    print(f"  worker_state: {worker_state}")
    return f"Worker {worker_id} processed: {worker_args}"

if __name__ == "__main__":
    from mpire import WorkerPool
    
    # Test data
    test_args = [
        ("data1", 100, 0, {"param": "value1"}),
        ("data2", 100, 1, {"param": "value2"}),
        ("data3", 100, 2, {"param": "value3"})
    ]
    
    print("Testing mpire argument passing...")
    
    with WorkerPool(n_jobs=2, start_method='spawn') as pool:
        results = pool.map(debug_worker, test_args)
    
    print("Results:")
    for result in results:
        print(f"  {result}")
