import torch
import time
import sys
import json
from pathlib import Path

# Add parent directory to path to import both versions
sys.path.append(str(Path(__file__).parent.parent))

# Import both versions
from dynamic_model_builder_original import DynamicModel as OriginalModel
from dynamic_model_builder_optimized import DynamicModel as OptimizedModel

def create_test_configs():
    """Create various test configurations for benchmarking"""
    
    # Simple sequential model (like the JSON we have)
    simple_config = {
        "layers": [
            {"id": "input_data", "name": "Input Layer", "type": "input", "params": {"features": 12}},
            {"id": "linear1", "name": "Linear", "type": "linear", "params": {"in_features": 12, "out_features": 32, "bias": True}, "inputs": ["input_data"]},
            {"id": "relu1", "name": "ReLU", "type": "relu", "params": {}, "inputs": ["linear1"]},
            {"id": "linear2", "name": "Linear", "type": "linear", "params": {"in_features": 32, "out_features": 16, "bias": True}, "inputs": ["relu1"]},
            {"id": "relu2", "name": "ReLU", "type": "relu", "params": {}, "inputs": ["linear2"]},
            {"id": "linear3", "name": "Linear", "type": "linear", "params": {"in_features": 16, "out_features": 1, "bias": True}, "inputs": ["relu2"]},
            {"id": "output_layer", "name": "Output Layer", "type": "output", "params": {"features": 1}, "inputs": ["linear3"]}
        ]
    }
    
    # Complex model with multiple branches (for testing non-sequential path)
    complex_config = {
        "layers": [
            {"id": "input_data", "name": "Input Layer", "type": "input", "params": {"features": 12}},
            {"id": "linear1", "name": "Linear1", "type": "linear", "params": {"in_features": 12, "out_features": 32, "bias": True}, "inputs": ["input_data"]},
            {"id": "linear2", "name": "Linear2", "type": "linear", "params": {"in_features": 12, "out_features": 32, "bias": True}, "inputs": ["input_data"]},
            {"id": "add1", "name": "Add", "type": "Add", "params": {}, "inputs": ["linear1", "linear2"]},
            {"id": "relu1", "name": "ReLU", "type": "relu", "params": {}, "inputs": ["add1"]},
            {"id": "linear3", "name": "Linear3", "type": "linear", "params": {"in_features": 32, "out_features": 1, "bias": True}, "inputs": ["relu1"]},
            {"id": "output_layer", "name": "Output Layer", "type": "output", "params": {"features": 1}, "inputs": ["linear3"]}
        ]
    }
    
    # Deep sequential model (stress test)
    deep_config = {
        "layers": [
            {"id": "input_data", "name": "Input Layer", "type": "input", "params": {"features": 128}}
        ]
    }
    
    # Add many layers
    for i in range(20):
        layer_id = f"linear{i+1}"
        deep_config["layers"].append({
            "id": layer_id,
            "name": f"Linear{i+1}",
            "type": "linear",
            "params": {"in_features": 128, "out_features": 128, "bias": True},
            "inputs": [f"linear{i}" if i > 0 else "input_data"]
        })
        
        relu_id = f"relu{i+1}"
        deep_config["layers"].append({
            "id": relu_id,
            "name": f"ReLU{i+1}",
            "type": "relu",
            "params": {},
            "inputs": [layer_id]
        })
    
    # Final output layer
    deep_config["layers"].append({
        "id": "output_linear",
        "name": "Output Linear",
        "type": "linear",
        "params": {"in_features": 128, "out_features": 1, "bias": True},
        "inputs": ["relu20"]
    })
    deep_config["layers"].append({
        "id": "output_layer",
        "name": "Output Layer",
        "type": "output",
        "params": {"features": 1},
        "inputs": ["output_linear"]
    })
    
    return {
        "simple": simple_config,
        "complex": complex_config,
        "deep": deep_config
    }

def benchmark_model(ModelClass, config, input_tensor, num_iterations=1000):
    """Benchmark a single model configuration"""
    
    # Model creation time
    start_time = time.perf_counter()
    model = ModelClass(config)
    creation_time = time.perf_counter() - start_time
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Forward pass benchmarking
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(input_tensor)
    forward_time = time.perf_counter() - start_time
    
    return {
        "creation_time": creation_time * 1000,  # ms
        "forward_time": forward_time / num_iterations * 1000,  # ms per iteration
        "total_forward_time": forward_time * 1000,  # ms
        "output_shape": output.shape if hasattr(output, 'shape') else "N/A"
    }

def run_benchmark():
    """Run complete benchmark comparing both models"""
    
    print("🚀 Dynamic Model Builder Optimization Benchmark")
    print("=" * 60)
    
    configs = create_test_configs()
    
    # Test different input sizes
    test_cases = [
        ("Simple Model (batch=1, features=12)", configs["simple"], torch.randn(1, 12)),
        ("Simple Model (batch=32, features=12)", configs["simple"], torch.randn(32, 12)),
        ("Simple Model (batch=128, features=12)", configs["simple"], torch.randn(128, 12)),
        ("Complex Model (batch=32, features=12)", configs["complex"], torch.randn(32, 12)),
        ("Deep Model (batch=32, features=128)", configs["deep"], torch.randn(32, 128)),
    ]
    
    results = []
    
    for test_name, config, input_tensor in test_cases:
        print(f"\n📊 Testing: {test_name}")
        print("-" * 40)
        
        try:
            # Benchmark original model
            original_results = benchmark_model(OriginalModel, config, input_tensor)
            print(f"Original  - Creation: {original_results['creation_time']:.3f}ms, Forward: {original_results['forward_time']:.3f}ms/iter")
            
            # Benchmark optimized model
            optimized_results = benchmark_model(OptimizedModel, config, input_tensor)
            print(f"Optimized - Creation: {optimized_results['creation_time']:.3f}ms, Forward: {optimized_results['forward_time']:.3f}ms/iter")
            
            # Calculate improvements
            creation_improvement = ((original_results['creation_time'] - optimized_results['creation_time']) / original_results['creation_time']) * 100
            forward_improvement = ((original_results['forward_time'] - optimized_results['forward_time']) / original_results['forward_time']) * 100
            
            print(f"Improvement - Creation: {creation_improvement:+.1f}%, Forward: {forward_improvement:+.1f}%")
            
            results.append({
                "test_name": test_name,
                "original": original_results,
                "optimized": optimized_results,
                "creation_improvement": creation_improvement,
                "forward_improvement": forward_improvement
            })
            
        except Exception as e:
            print(f"❌ Error in {test_name}: {str(e)}")
            continue
    
    # Summary
    print(f"\n📈 SUMMARY")
    print("=" * 60)
    
    if results:
        avg_creation_improvement = sum(r["creation_improvement"] for r in results) / len(results)
        avg_forward_improvement = sum(r["forward_improvement"] for r in results) / len(results)
        
        print(f"Average Creation Time Improvement: {avg_creation_improvement:+.1f}%")
        print(f"Average Forward Pass Improvement: {avg_forward_improvement:+.1f}%")
        
        # Best improvements
        best_forward = max(results, key=lambda x: x["forward_improvement"])
        print(f"Best Forward Improvement: {best_forward['forward_improvement']:+.1f}% ({best_forward['test_name']})")
    
    return results

if __name__ == "__main__":
    results = run_benchmark() 