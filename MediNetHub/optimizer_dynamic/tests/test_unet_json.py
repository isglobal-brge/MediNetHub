import torch
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dynamic_model_builder_original import DynamicModel as OriginalModel, create_model_from_config as create_original
from dynamic_model_builder_optimized import DynamicModel as OptimizedModel, create_model_from_config as create_optimized

def test_unet_from_json():
    """Test ResNet-like model with skip connections from JSON configuration"""
    
    print("🏗️ ResNet-like JSON Configuration Test")
    print("=" * 45)
    
    config_dir = Path(__file__).parent
    config_path = config_dir / "resnet_like_config.json"
    
    # Test different input sizes
    test_cases = [
        ("batch=1, features=128", torch.randn(1, 128)),
        ("batch=32, features=128", torch.randn(32, 128)),
        ("batch=128, features=128", torch.randn(128, 128)),
    ]
    
    for test_name, input_tensor in test_cases:
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 25)
        
        try:
            # Test Original Model
            print("Original Model:")
            start_time = time.perf_counter()
            original_model = create_original(config_path)
            creation_time_orig = (time.perf_counter() - start_time) * 1000
            
            print(f"  🏗️  Type: {'Sequential' if getattr(original_model, 'is_sequential', False) else 'Complex'}")
            print(f"  Creation: {creation_time_orig:.2f}ms")
            
            # Test forward pass
            original_model.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = original_model(input_tensor)
                
                # Benchmark
                start_time = time.perf_counter()
                for _ in range(50):  # Fewer iterations for complex model
                    output_orig = original_model(input_tensor)
                forward_time_orig = (time.perf_counter() - start_time) / 50 * 1000
                
            print(f"  Forward: {forward_time_orig:.3f}ms/iter")
            print(f"  Output: {output_orig.shape}")
            
        except Exception as e:
            print(f"  ❌ Original failed: {str(e)}")
            continue
            
        try:
            # Test Optimized Model
            print("Optimized Model:")
            start_time = time.perf_counter()
            optimized_model = create_optimized(config_path)
            creation_time_opt = (time.perf_counter() - start_time) * 1000
            
            print(f"  🏗️  Type: {'Sequential' if getattr(optimized_model, 'is_sequential', False) else 'Complex'}")
            print(f"  Creation: {creation_time_opt:.2f}ms")
            
            # Test forward pass
            optimized_model.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = optimized_model(input_tensor)
                
                # Benchmark
                start_time = time.perf_counter()
                for _ in range(50):
                    output_opt = optimized_model(input_tensor)
                forward_time_opt = (time.perf_counter() - start_time) / 50 * 1000
                
            print(f"  Forward: {forward_time_opt:.3f}ms/iter")
            print(f"  Output: {output_opt.shape}")
            
            # Calculate improvements
            creation_improvement = ((creation_time_orig - creation_time_opt) / creation_time_orig) * 100
            forward_improvement = ((forward_time_orig - forward_time_opt) / forward_time_orig) * 100
            
            print(f"  📈 Improvement:")
            print(f"    Creation: {creation_improvement:+.1f}%")
            print(f"    Forward: {forward_improvement:+.1f}%")
            
        except Exception as e:
            print(f"  ❌ Optimized failed: {str(e)}")
            continue

if __name__ == "__main__":
    test_unet_from_json() 