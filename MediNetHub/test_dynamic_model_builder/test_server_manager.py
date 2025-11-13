import sys
import os
import json
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the cleaner
from json_cleaner import ModelConfigCleaner

# Mock Django timezone for testing
class MockTimezone:
    @staticmethod
    def now():
        from datetime import datetime
        return datetime.now()

# Mock the Django dependency
sys.modules['django.utils'] = type('MockModule', (), {'timezone': MockTimezone})()

# Mock flwr dependencies to avoid import errors
sys.modules['flwr.common'] = type('MockModule', (), {})()
sys.modules['flwr.server.strategy'] = type('MockModule', (), {})()
sys.modules['flwr.server.client_proxy'] = type('MockModule', (), {})()
sys.modules['flwr.server'] = type('MockModule', (), {})()
sys.modules['flwr.server.client_manager'] = type('MockModule', (), {})()
sys.modules['flwr'] = type('MockModule', (), {})()

class MockTrainingJob:
    """Mock training job for testing"""
    def __init__(self):
        self.id = 1
        self.status = 'pending'
        self.logs = ''
        self.metrics_json = None
        self.current_round = 0
        self.progress = 0
        self.accuracy = 0
        self.loss = 0
        
    def save(self):
        print(f"📝 MockTrainingJob.save() called - status: {self.status}")

class MinimalServerManager:
    """Minimal ServerManager for testing JSON processing only"""
    def __init__(self, training_job, model_config):
        print(f"🚀 MinimalServerManager.__init__ called")
        print(f"📋 training_job: {training_job}")
        print(f"📋 model_config type: {type(model_config)}")
        
        self.job = training_job
        self.model_config = model_config
        self.framework = model_config.get('framework', 'pt')
        self.should_stop = False
        
        print(f"🔧 Framework detected: '{self.framework}'")
        
        self.net = None
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the model based on configuration - WITH CLEANER"""
        print(f"🔧 initialize_model called with framework: '{self.framework}'")
        print(f"🔍 model_config keys: {list(self.model_config.keys())}")
        
        if self.framework in ['pt', 'pytorch']:
            # CLEAN the model config first
            print(f"\n🧹 Cleaning model config before creating DynamicModel...")
            cleaned_config = ModelConfigCleaner.clean_model_config(self.model_config)
            
            # Extract cleaned layers
            layers_config = cleaned_config.get('layers', [])
            print(f"🔍 Found {len(layers_config)} CLEANED layers")
            
            print(f"🔧 Would create DynamicModel with CLEANED config: {{'layers': {len(layers_config)} layers}}")
            
            # Instead of creating real model, just simulate it
            self.net = f"MockDynamicModel_with_{len(layers_config)}_CLEANED_layers"
            print(f"✅ Mock model initialized: {self.net}")
            
            # Show final config that would be passed to DynamicModel
            print(f"\n🔍 Final config for DynamicModel:")
            print(json.dumps({"layers": layers_config}, indent=2))
                
        else:
            print(f"❌ Framework '{self.framework}' not supported")
            raise ValueError(f"Framework {self.framework} not supported. Supported: 'pt', 'pytorch'")

def test_with_json_file(json_file_path):
    """Test minimal ServerManager with JSON file"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing MINIMAL ServerManager with JSON file: {json_file_path}")
    print(f"{'='*60}")
    
    try:
        # Load JSON config
        with open(json_file_path, 'r') as f:
            full_config = json.load(f)
        
        print(f"📄 Loaded JSON config")
        print(f"📄 Top-level keys: {list(full_config.keys())}")
        
        # Extract model config (simulate what Django would pass)
        if 'model' in full_config and 'config_json' in full_config['model']:
            model_config = full_config['model']['config_json']
            print(f"📄 Using model.config_json")
        else:
            model_config = full_config
            print(f"📄 Using full config as model_config")
        
        print(f"📄 Model config keys: {list(model_config.keys())}")
        
        # Create mock training job
        mock_job = MockTrainingJob()
        
        # Test MINIMAL ServerManager
        print(f"🔧 Creating MINIMAL ServerManager...")
        server_manager = MinimalServerManager(mock_job, model_config)
        
        print(f"\n✅ Test completed successfully!")
        print(f"✅ ServerManager created: {type(server_manager)}")
        print(f"✅ Mock model created: {server_manager.net}")
        
        return server_manager
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🧪 MINIMAL ServerManager JSON Processing Test")
    print("=" * 60)
    
    # Look for JSON files in the test directory
    test_dir = Path(__file__).parent
    json_files = list(test_dir.glob("*.json"))
    
    if json_files:
        for json_file in json_files:
            test_with_json_file(json_file)
    else:
        print("📄 No JSON files found in test directory.")
        print("📄 Please create a JSON file with your model configuration.")
        print("📄 Example: test_config.json")
        print("\n📝 Create test_config.json with your model configuration and run again!") 