import sys
import json
import torch
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from clients.dynamic_model_builder import DynamicModel

def test_model_designer_json():
    """Test que el DynamicModel funciona con el JSON del model_designer"""
    print("🧪 TESTING MODEL DESIGNER JSON COMPATIBILITY")
    print("=" * 80)
    
    # JSON del model_designer (el que proporcionaste)
    model_designer_json = {
        "model_id": "26",
        "job_name": "",
        "job_description": "",
        "config": {
            "train": {
                "epochs": 5,
                "batch_size": 32,
                "rounds": 10,
                "metrics": ["accuracy", "loss"]
            },
            "federated": {
                "name": "FedAvg",
                "parameters": {
                    "fraction_fit": 1,
                    "fraction_eval": 0.3,
                    "min_fit_clients": 1,
                    "min_eval_clients": 1,
                    "min_available_clients": 1
                }
            }
        },
        "model": {
            "id": 26,
            "name": "testf1",
            "framework": "pytorch",
            "description": "",
            "config_json": {
                "name": "testf1",
                "framework": "pytorch",
                "description": "",
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001,
                    "weight_decay": 0
                },
                "loss_function": "bce_with_logits",
                "layers": [
                    {
                        "name": "Input Layer",
                        "type": "input",
                        "params": {"features": 12},
                        "readonly": True
                    },
                    {
                        "name": "Linear",
                        "type": "linear",
                        "params": {
                            "in_features": 12,
                            "out_features": 1,
                            "bias": True,
                            "features": 1
                        }
                    },
                    {
                        "name": "Output Layer",
                        "type": "output",
                        "params": {"features": 1},
                        "readonly": True
                    }
                ],
                "datasets": [
                    {
                        "dataset_name": "heart_failure_clinical_records_dataset",
                        "features_info": {
                            "input_features": 12,
                            "feature_types": {"numeric": 12, "categorical": 0}
                        },
                        "target_info": {
                            "name": "DEATH_EVENT",
                            "type": "binary_classification",
                            "num_classes": 2
                        },
                        "num_columns": 13,
                        "num_rows": 299,
                        "size": 12239,
                        "connection": {
                            "name": "test",
                            "ip": "127.0.0.1",
                            "port": "5000"
                        }
                    }
                ]
            }
        }
    }
    
    try:
        print("📄 Step 1: Testing DynamicModel with model_designer JSON...")
        
        # Create DynamicModel directly with the JSON
        model = DynamicModel(model_designer_json)
        
        print(f"✅ Model created successfully!")
        print(f"✅ Model layers: {list(model.layers.keys())}")
        print(f"✅ Config layers: {[l['id'] for l in model.cleaned_config['layers']]}")
        
        print("\n📊 Step 2: Testing forward pass...")
        
        # Create dummy input data
        batch_size = 4
        input_features = 12
        X = torch.randn(batch_size, input_features)
        
        print(f"Input shape: {X.shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(X)
            print(f"✅ Forward pass successful!")
            print(f"✅ Output shape: {output.shape}")
            print(f"✅ Output sample: {output[:3]}")
            
        print("\n🏋️ Step 3: Testing training...")
        
        # Test training mode
        model.train()
        
        # Create target
        y = torch.randint(0, 2, (batch_size,)).float()
        
        # Loss function
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training step
        output = model(X)
        if output.dim() > 1 and output.size(1) == 1:
            output = output.squeeze(1)
            
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✅ Training step successful!")
        print(f"✅ Loss: {loss.item():.6f}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ El DynamicModel es compatible con el JSON del model_designer")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_designer_json() 