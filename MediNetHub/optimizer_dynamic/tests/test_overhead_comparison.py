import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dynamic_model_builder_original import DynamicModel as OriginalModel
from dynamic_model_builder_optimized import DynamicModel as OptimizedModel

class NativeResNet(nn.Module):
    """Native PyTorch implementation for comparison"""
    def __init__(self):
        super(NativeResNet, self).__init__()
        
        # Same architecture as resnet_like_config.json
        self.linear1 = nn.Linear(128, 256, bias=True)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 256, bias=True)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(256, 256, bias=True)
        # skip_connection1: linear3 + relu1
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(256, 256, bias=True)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(256, 256, bias=True)
        # skip_connection2: linear5 + relu3
        self.relu5 = nn.ReLU()
        self.linear6 = nn.Linear(256, 256, bias=True)
        self.relu6 = nn.ReLU()
        self.linear7 = nn.Linear(256, 256, bias=True)
        # skip_connection3: linear7 + relu5
        self.relu7 = nn.ReLU()
        self.output_linear = nn.Linear(256, 10, bias=True)
    
    def forward(self, x):
        # Encoder path
        x1 = self.linear1(x)
        r1 = self.relu1(x1)
        
        x2 = self.linear2(r1)
        r2 = self.relu2(x2)
        
        x3 = self.linear3(r2)
        
        # Skip connection 1: x3 + r1
        skip1 = x3 + r1
        r3 = self.relu3(skip1)
        
        x4 = self.linear4(r3)
        r4 = self.relu4(x4)
        
        x5 = self.linear5(r4)
        
        # Skip connection 2: x5 + r3
        skip2 = x5 + r3
        r5 = self.relu5(skip2)
        
        x6 = self.linear6(r5)
        r6 = self.relu6(x6)
        
        x7 = self.linear7(r6)
        
        # Skip connection 3: x7 + r5
        skip3 = x7 + r5
        r7 = self.relu7(skip3)
        
        output = self.output_linear(r7)
        return output

class NativeSequential(nn.Module):
    """Native PyTorch sequential model for comparison"""
    def __init__(self):
        super(NativeSequential, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 10, bias=True)
        )
    
    def forward(self, x):
        return self.layers(x)

def create_sequential_config():
    """Create sequential model config (no skip connections)"""
    return {
        "layers": [
            {"id": "input_data", "name": "Input Layer", "type": "input", "params": {"features": 128}},
            {"id": "linear1", "name": "Linear1", "type": "linear", "params": {"in_features": 128, "out_features": 256, "bias": True}, "inputs": ["input_data"]},
            {"id": "relu1", "name": "ReLU1", "type": "relu", "params": {}, "inputs": ["linear1"]},
            {"id": "linear2", "name": "Linear2", "type": "linear", "params": {"in_features": 256, "out_features": 256, "bias": True}, "inputs": ["relu1"]},
            {"id": "relu2", "name": "ReLU2", "type": "relu", "params": {}, "inputs": ["linear2"]},
            {"id": "linear3", "name": "Linear3", "type": "linear", "params": {"in_features": 256, "out_features": 256, "bias": True}, "inputs": ["relu2"]},
            {"id": "relu3", "name": "ReLU3", "type": "relu", "params": {}, "inputs": ["linear3"]},
            {"id": "linear4", "name": "Linear4", "type": "linear", "params": {"in_features": 256, "out_features": 256, "bias": True}, "inputs": ["relu3"]},
            {"id": "relu4", "name": "ReLU4", "type": "relu", "params": {}, "inputs": ["linear4"]},
            {"id": "linear5", "name": "Linear5", "type": "linear", "params": {"in_features": 256, "out_features": 256, "bias": True}, "inputs": ["relu4"]},
            {"id": "relu5", "name": "ReLU5", "type": "relu", "params": {}, "inputs": ["linear5"]},
            {"id": "linear6", "name": "Linear6", "type": "linear", "params": {"in_features": 256, "out_features": 256, "bias": True}, "inputs": ["relu5"]},
            {"id": "relu6", "name": "ReLU6", "type": "relu", "params": {}, "inputs": ["linear6"]},
            {"id": "linear7", "name": "Linear7", "type": "linear", "params": {"in_features": 256, "out_features": 256, "bias": True}, "inputs": ["relu6"]},
            {"id": "relu7", "name": "ReLU7", "type": "relu", "params": {}, "inputs": ["linear7"]},
            {"id": "output_linear", "name": "Output Linear", "type": "linear", "params": {"in_features": 256, "out_features": 10, "bias": True}, "inputs": ["relu7"]},
            {"id": "output_layer", "name": "Output Layer", "type": "output", "params": {"features": 10}, "inputs": ["output_linear"]}
        ]
    }

def benchmark_model(model, input_tensor, model_name, num_iterations=1000):
    """Benchmark a single model"""
    
    # Model creation time is already measured outside
    
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
    forward_time = (time.perf_counter() - start_time) / num_iterations * 1000
    
    return {
        "forward_time": forward_time,
        "output_shape": output.shape
    }

def test_overhead_comparison():
    """3-way comparison: Original vs Optimized vs Native PyTorch"""
    
    print("⚡ OVERHEAD COMPARISON TEST")
    print("=" * 50)
    print("🎯 Goal: Measure real overhead of dynamic builders vs native PyTorch")
    print()
    
    # Load ResNet config from JSON
    config_dir = Path(__file__).parent
    resnet_config_path = config_dir / "resnet_like_config.json"
    sequential_config = create_sequential_config()
    
    test_cases = [
        ("Sequential Model", sequential_config, NativeSequential, torch.randn(32, 128)),
        ("ResNet-like (skip connections)", resnet_config_path, NativeResNet, torch.randn(32, 128)),
        ("Large Batch Sequential", sequential_config, NativeSequential, torch.randn(256, 128)),
        ("Large Batch ResNet-like", resnet_config_path, NativeResNet, torch.randn(256, 128)),
    ]
    
    for test_name, config, NativeClass, input_tensor in test_cases:
        print(f"\n🧪 {test_name}")
        print(f"Input shape: {input_tensor.shape}")
        print("-" * 40)
        
        results = {}
        
        try:
            # 1. Original Dynamic Model
            print("1️⃣  Original Dynamic Builder:")
            start_time = time.perf_counter()
            original_model = OriginalModel(config)
            creation_time_orig = (time.perf_counter() - start_time) * 1000
            
            orig_results = benchmark_model(original_model, input_tensor, "Original")
            print(f"   Creation: {creation_time_orig:.2f}ms")
            print(f"   Forward: {orig_results['forward_time']:.3f}ms/iter")
            print(f"   Type: {'Sequential' if getattr(original_model, 'is_sequential', False) else 'Complex'}")
            results['original'] = {**orig_results, 'creation_time': creation_time_orig}
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            continue
            
        try:
            # 2. Optimized Dynamic Model
            print("2️⃣  Optimized Dynamic Builder:")
            start_time = time.perf_counter()
            optimized_model = OptimizedModel(config)
            creation_time_opt = (time.perf_counter() - start_time) * 1000
            
            opt_results = benchmark_model(optimized_model, input_tensor, "Optimized")
            print(f"   Creation: {creation_time_opt:.2f}ms")
            print(f"   Forward: {opt_results['forward_time']:.3f}ms/iter")
            print(f"   Type: {'Sequential' if getattr(optimized_model, 'is_sequential', False) else 'Complex'}")
            results['optimized'] = {**opt_results, 'creation_time': creation_time_opt}
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            continue
            
        try:
            # 3. Native PyTorch Model
            print("3️⃣  Native PyTorch:")
            start_time = time.perf_counter()
            native_model = NativeClass()
            creation_time_native = (time.perf_counter() - start_time) * 1000
            
            native_results = benchmark_model(native_model, input_tensor, "Native")
            print(f"   Creation: {creation_time_native:.2f}ms")
            print(f"   Forward: {native_results['forward_time']:.3f}ms/iter")
            print(f"   Type: Native")
            results['native'] = {**native_results, 'creation_time': creation_time_native}
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            continue
        
        # Calculate comparisons
        if len(results) == 3:
            print(f"\n📊 COMPARISON RESULTS:")
            
            # Optimized vs Original
            forward_improvement = ((results['original']['forward_time'] - results['optimized']['forward_time']) / results['original']['forward_time']) * 100
            print(f"   Optimized vs Original: {forward_improvement:+.1f}% forward improvement")
            
            # Optimized vs Native (overhead)
            forward_overhead_opt = ((results['optimized']['forward_time'] - results['native']['forward_time']) / results['native']['forward_time']) * 100
            print(f"   Optimized vs Native: {forward_overhead_opt:+.1f}% overhead")
            
            # Original vs Native (overhead)
            forward_overhead_orig = ((results['original']['forward_time'] - results['native']['forward_time']) / results['native']['forward_time']) * 100
            print(f"   Original vs Native: {forward_overhead_orig:+.1f}% overhead")
            
            # Relative speeds
            native_speed = results['native']['forward_time']
            opt_speed = results['optimized']['forward_time']
            orig_speed = results['original']['forward_time']
            
            print(f"\n🏃 RELATIVE SPEEDS (lower is better):")
            print(f"   Native PyTorch: {native_speed:.3f}ms (baseline)")
            print(f"   Optimized: {opt_speed:.3f}ms ({opt_speed/native_speed:.2f}x slower)")
            print(f"   Original: {orig_speed:.3f}ms ({orig_speed/native_speed:.2f}x slower)")

if __name__ == "__main__":
    test_overhead_comparison() 