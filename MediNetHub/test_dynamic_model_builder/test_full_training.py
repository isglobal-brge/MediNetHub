import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from clients.dynamic_model_builder import DynamicModel
from json_cleaner import ModelConfigCleaner

def create_dummy_data(batch_size=4, in_channels=12, sequence_length=100):
    """Create dummy data compatible with Conv1d input"""
    print(f"🔧 Creating dummy data: batch_size={batch_size}, in_channels={in_channels}, seq_len={sequence_length}")
    
    # Conv1d expects (N, C, L) where N=batch, C=channels, L=length
    X = torch.randn(batch_size, in_channels, sequence_length)
    
    # Binary classification targets
    y = torch.randint(0, 2, (batch_size,)).float()
    
    print(f"✅ Input shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    
    return X, y

def test_model_creation_and_training(json_file_path: str):
    """Test complete pipeline: JSON → Clean → Model → Training"""
    print(f"\n{'='*80}")
    print(f"🧪 FULL TRAINING TEST with: {json_file_path}")
    print(f"{'='*80}")
    
    try:
        # 1. Load and clean JSON
        print(f"\n📄 Step 1: Loading and cleaning JSON...")
        with open(json_file_path, 'r') as f:
            full_config = json.load(f)
        
        # Extract model config
        if 'model' in full_config and 'config_json' in full_config['model']:
            model_config = full_config['model']['config_json']
        else:
            model_config = full_config
        
        # Clean the config
        cleaned_config = ModelConfigCleaner.clean_model_config(model_config)
        layers_config = cleaned_config.get('layers', [])
        
        print(f"✅ Cleaned config: {len(layers_config)} layers")
        
        # 2. Create real DynamicModel
        print(f"\n🔧 Step 2: Creating REAL DynamicModel...")
        model = DynamicModel({"layers": layers_config})
        print(f"✅ Model created successfully!")
        print(f"✅ Model layers: {list(model.layers.keys())}")
        
        # 3. Create dummy data
        print(f"\n📊 Step 3: Creating dummy data...")
        X, y = create_dummy_data(batch_size=8, in_channels=12, sequence_length=50)
        
        # 4. Test forward pass
        print(f"\n➡️  Step 4: Testing forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(X)
            print(f"✅ Forward pass successful!")
            print(f"✅ Output shape: {output.shape}")
            print(f"✅ Output sample: {output[:3]}")
        
        # 5. Setup training
        print(f"\n🏋️  Step 5: Setting up training...")
        model.train()
        
        # Get loss function from config
        loss_function_name = model_config.get('loss_function', 'mse')
        if loss_function_name == 'bce_with_logits':
            criterion = nn.BCEWithLogitsLoss()
        elif loss_function_name == 'mse':
            criterion = nn.MSELoss()
        else:
            criterion = nn.MSELoss()  # Default
        
        # Get optimizer from config
        optimizer_config = model_config.get('optimizer', {'type': 'adam', 'learning_rate': 0.001})
        if optimizer_config['type'].lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=optimizer_config['learning_rate'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=optimizer_config['learning_rate'])
        
        print(f"✅ Loss function: {criterion}")
        print(f"✅ Optimizer: {optimizer}")
        
        # 6. Training loop
        print(f"\n🔄 Step 6: Running training loop...")
        num_epochs = 5
        
        for epoch in range(num_epochs):
            # Forward pass
            output = model(X)
            
            # Reshape output if needed for loss calculation
            if output.dim() > 1 and output.size(1) == 1:
                output = output.squeeze(1)  # Remove dimension of size 1
            
            # Calculate loss
            loss = criterion(output, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {loss.item():.6f}")
        
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"✅ Final loss: {loss.item():.6f}")
        
        # 7. Test final prediction
        print(f"\n🔮 Step 7: Testing final predictions...")
        model.eval()
        with torch.no_grad():
            final_output = model(X)
            if final_output.dim() > 1 and final_output.size(1) == 1:
                final_output = final_output.squeeze(1)
            
            # Apply sigmoid for binary classification
            if loss_function_name == 'bce_with_logits':
                predictions = torch.sigmoid(final_output)
                binary_preds = (predictions > 0.5).float()
                print(f"✅ Predictions (probabilities): {predictions[:5]}")
                print(f"✅ Binary predictions: {binary_preds[:5]}")
                print(f"✅ Actual targets: {y[:5]}")
            else:
                print(f"✅ Final predictions: {final_output[:5]}")
                print(f"✅ Actual targets: {y[:5]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TRAINING TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 FULL PYTORCH TRAINING TEST")
    print("=" * 80)
    
    # Look for JSON files
    test_dir = Path(__file__).parent
    json_files = list(test_dir.glob("*.json"))
    
    if json_files:
        for json_file in json_files:
            if "cleaned" not in json_file.name:  # Skip already cleaned files
                success = test_model_creation_and_training(str(json_file))
                if success:
                    print(f"\n🎉 SUCCESS: {json_file.name} works completely!")
                else:
                    print(f"\n💥 FAILED: {json_file.name} has issues!")
                break  # Test only first file
    else:
        print("📄 No JSON files found. Create a test JSON file first.") 