# Experiment Configuration Specification

## Overview

This document defines the JSON structure for Grid Search experiments in MediNetHub. Experiments allow users to test multiple hyperparameter combinations to find optimal model configurations.

**Key Points:**
- The `experiment` key is **server-side only** and should be removed before sending configs to MediNet-Node clients
- Experiments generate multiple training jobs (one per parameter combination)
- Both ML (scikit-learn) and DL (PyTorch) models support experiments

---

## JSON Structure

### Complete Model Config with Experiment

```json
{
  "metadata": {
    "version": "1.0",
    "created_at": "2025-12-14T20:15:00.000Z",
    "model_type": "ml",  // or "dl"
    "framework": "sklearn"  // or "pytorch"
  },
  "basic_info": {
    "name": "FedSVM Kernel Optimization",
    "description": "Grid search to find optimal SVM kernel and regularization"
  },
  "architecture": {
    "ml_algorithm": {
      "type": "FedSVMOptMD",
      "hyperparameters": {
        "kernel": "rbf",
        "c": 2.0,
        "gamma": 0.1,
        "server_eps": 0.01,
        "client_eps": 0.0001,
        "n_random_features": 0
      }
    }
  },
  "training": {
    "metrics": ["accuracy", "f1_score"],
    "epochs": 100,
    "batch_size": 32,
    "rounds": 10
  },
  "dataset": {
    "selected_datasets": []
  },
  "experiment": {
    "enabled": true,
    "name": "FedSVM_Kernel_Grid_Search",
    "description": "Testing different kernel configurations to find optimal SVM parameters",
    "variable_params": {
      "kernel": {
        "type": "categorical",
        "values": ["linear", "rbf", "poly"]
      },
      "c": {
        "type": "numeric",
        "from": 0.1,
        "to": 10.0,
        "step": 0.1,
        "count": 100
      },
      "gamma": {
        "type": "numeric",
        "from": 0.001,
        "to": 1.0,
        "step": 0.001,
        "count": 1000
      }
    },
    "total_jobs": 300000,
    "created_at": "2025-12-14T20:15:00.000Z"
  }
}
```

---

## Experiment Structure Reference

### `experiment` (object)

Top-level experiment configuration. **Must be removed before sending to MediNet-Node clients.**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `enabled` | boolean | Yes | Whether experiment mode is active |
| `name` | string | Yes | Human-readable experiment name |
| `description` | string | No | Detailed description of experiment goals |
| `variable_params` | object | Yes | Parameters to vary (see below) |
| `total_jobs` | integer | Yes | Total number of training jobs (Cartesian product of all param values) |
| `created_at` | string (ISO 8601) | Yes | Timestamp when experiment was created |

### `variable_params` (object)

Map of parameter names to their value specifications. Each parameter can be:

#### Categorical Parameter (Select/Dropdown)

```json
"kernel": {
  "type": "categorical",
  "values": ["linear", "rbf", "poly"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Must be `"categorical"` |
| `values` | array | List of discrete values to test |

#### Numeric Parameter (Range)

```json
"c": {
  "type": "numeric",
  "from": 0.1,
  "to": 10.0,
  "step": 0.1,
  "count": 100
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Must be `"numeric"` |
| `from` | number | Starting value (inclusive) |
| `to` | number | Ending value (inclusive) |
| `step` | number | Increment between values |
| `count` | integer | Total number of values: `floor((to - from) / step) + 1` |

---

## Machine Learning (ML) Experiment Examples

### Example 1: FedSVM Kernel and Regularization

```json
{
  "experiment": {
    "enabled": true,
    "name": "FedSVM_Kernel_C_Search",
    "description": "Find optimal kernel and regularization parameter",
    "variable_params": {
      "kernel": {
        "type": "categorical",
        "values": ["linear", "rbf", "poly"]
      },
      "c": {
        "type": "numeric",
        "from": 0.1,
        "to": 10.0,
        "step": 0.5,
        "count": 20
      }
    },
    "total_jobs": 60,  // 3 kernels × 20 C values
    "created_at": "2025-12-14T20:15:00.000Z"
  }
}
```

### Example 2: Random Forest Depth and Estimators

```json
{
  "experiment": {
    "enabled": true,
    "name": "RandomForest_Optimization",
    "description": "Grid search for optimal tree depth and number of estimators",
    "variable_params": {
      "n_estimators": {
        "type": "numeric",
        "from": 10,
        "to": 100,
        "step": 10,
        "count": 10
      },
      "max_depth": {
        "type": "numeric",
        "from": 5,
        "to": 50,
        "step": 5,
        "count": 10
      },
      "criterion": {
        "type": "categorical",
        "values": ["gini", "entropy"]
      }
    },
    "total_jobs": 200,  // 10 × 10 × 2
    "created_at": "2025-12-14T20:15:00.000Z"
  }
}
```

### Example 3: KNN Neighbors and Weights

```json
{
  "experiment": {
    "enabled": true,
    "name": "KNN_K_Optimization",
    "description": "Test different k values and weighting strategies",
    "variable_params": {
      "n_neighbors": {
        "type": "numeric",
        "from": 3,
        "to": 21,
        "step": 2,
        "count": 10
      },
      "weights": {
        "type": "categorical",
        "values": ["uniform", "distance"]
      },
      "metric": {
        "type": "categorical",
        "values": ["euclidean", "manhattan", "minkowski"]
      }
    },
    "total_jobs": 60,  // 10 × 2 × 3
    "created_at": "2025-12-14T20:15:00.000Z"
  }
}
```

---

## Deep Learning (DL) Experiment Examples

### Example 1: Training Hyperparameters (Epochs, Batch Size, Learning Rate)

```json
{
  "metadata": {
    "model_type": "dl",
    "framework": "pytorch"
  },
  "training": {
    "metrics": ["accuracy", "loss"],
    "epochs": 50,
    "batch_size": 32,
    "rounds": 10,
    "learning_rate": 0.001,
    "optimizer": "adam"
  },
  "experiment": {
    "enabled": true,
    "name": "Training_Hyperparams_Search",
    "description": "Optimize training configuration for convergence speed and accuracy",
    "variable_params": {
      "epochs": {
        "type": "numeric",
        "from": 10,
        "to": 100,
        "step": 10,
        "count": 10
      },
      "batch_size": {
        "type": "categorical",
        "values": [16, 32, 64, 128]
      },
      "learning_rate": {
        "type": "numeric",
        "from": 0.0001,
        "to": 0.01,
        "step": 0.0001,
        "count": 100
      }
    },
    "total_jobs": 4000,  // 10 × 4 × 100
    "created_at": "2025-12-14T20:15:00.000Z"
  }
}
```

### Example 2: Network Architecture (Layer Sizes, Dropout)

```json
{
  "architecture": {
    "layers": [
      {"type": "Dense", "units": 128, "activation": "relu"},
      {"type": "Dropout", "rate": 0.2},
      {"type": "Dense", "units": 64, "activation": "relu"},
      {"type": "Dense", "units": 10, "activation": "softmax"}
    ]
  },
  "experiment": {
    "enabled": true,
    "name": "Network_Architecture_Search",
    "description": "Test different hidden layer sizes and dropout rates",
    "variable_params": {
      "layer_0_units": {
        "type": "categorical",
        "values": [64, 128, 256, 512]
      },
      "layer_1_dropout_rate": {
        "type": "numeric",
        "from": 0.1,
        "to": 0.5,
        "step": 0.1,
        "count": 5
      },
      "layer_2_units": {
        "type": "categorical",
        "values": [32, 64, 128]
      }
    },
    "total_jobs": 60,  // 4 × 5 × 3
    "created_at": "2025-12-14T20:15:00.000Z"
  }
}
```

**Note:** For DL architecture experiments, parameter names use the convention:
- `layer_{index}_{param_name}` (e.g., `layer_0_units`, `layer_1_dropout_rate`)
- This allows the backend to map experiment values to specific layer parameters

### Example 3: Optimizer and Learning Rate Schedule

```json
{
  "training": {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "lr_schedule": "constant"
  },
  "experiment": {
    "enabled": true,
    "name": "Optimizer_LR_Search",
    "description": "Compare optimizers and learning rate schedules",
    "variable_params": {
      "optimizer": {
        "type": "categorical",
        "values": ["adam", "sgd", "rmsprop"]
      },
      "learning_rate": {
        "type": "numeric",
        "from": 0.0001,
        "to": 0.01,
        "step": 0.0001,
        "count": 100
      },
      "lr_schedule": {
        "type": "categorical",
        "values": ["constant", "step_decay", "exponential_decay"]
      }
    },
    "total_jobs": 900,  // 3 × 100 × 3
    "created_at": "2025-12-14T20:15:00.000Z"
  }
}
```

### Example 4: Federated Learning Parameters (Rounds, Client Fraction)

```json
{
  "training": {
    "rounds": 10,
    "client_fraction": 1.0,
    "local_epochs": 5
  },
  "experiment": {
    "enabled": true,
    "name": "Federated_Config_Search",
    "description": "Optimize federated learning communication efficiency",
    "variable_params": {
      "rounds": {
        "type": "numeric",
        "from": 5,
        "to": 50,
        "step": 5,
        "count": 10
      },
      "local_epochs": {
        "type": "categorical",
        "values": [1, 3, 5, 10]
      },
      "client_fraction": {
        "type": "numeric",
        "from": 0.5,
        "to": 1.0,
        "step": 0.1,
        "count": 6
      }
    },
    "total_jobs": 240,  // 10 × 4 × 6
    "created_at": "2025-12-14T20:15:00.000Z"
  }
}
```

---

## Implementation Guidelines

### Frontend (Model Designer)

1. **UI Components:**
   - Checkbox to enable experiment mode
   - For categorical params: Dropdown selector + "Add" button → displays as removable tags
   - For numeric params: From/To/Step inputs → displays value preview

2. **Validation:**
   - Ensure `from <= to`
   - Ensure `step > 0` and `step <= (to - from)`
   - Warn if `total_jobs > 100` (expensive experiment)
   - Block if `total_jobs > 10000` (too expensive)

3. **Total Jobs Calculation:**
   ```javascript
   totalJobs = 1;
   for (param in variable_params) {
     if (param.type === 'categorical') {
       totalJobs *= param.values.length;
     } else if (param.type === 'numeric') {
       count = Math.floor((param.to - param.from) / param.step) + 1;
       totalJobs *= count;
     }
   }
   ```

### Backend (Django Views)

1. **Saving Experiment Config:**
   ```python
   # Save complete config with experiment to ModelConfig
   model_config = ModelConfig.objects.create(
       user=request.user,
       name=data['basic_info']['name'],
       model_type=data['metadata']['model_type'],
       config_json=data  # Includes 'experiment' key
   )
   ```

2. **Creating Training Jobs:**
   ```python
   if 'experiment' in config and config['experiment']['enabled']:
       # Generate all parameter combinations (Cartesian product)
       combinations = generate_combinations(config['experiment']['variable_params'])

       # Create one TrainingJob per combination
       for combo in combinations:
           job_config = copy.deepcopy(config)

           # Apply parameter overrides
           for param_name, param_value in combo.items():
               apply_param_to_config(job_config, param_name, param_value)

           # Remove experiment key before sending to nodes
           del job_config['experiment']

           TrainingJob.objects.create(
               user=request.user,
               model_config=model_config,
               config_json=job_config,  # No 'experiment' key
               experiment_id=f"{model_config.id}_{combo_index}"
           )
   ```

3. **Sending to MediNet-Node:**
   ```python
   # CRITICAL: Remove experiment key before sending
   client_config = copy.deepcopy(job.config_json)
   if 'experiment' in client_config:
       del client_config['experiment']

   response = requests.post(
       f"{node_url}/api/v1/start-client",
       json={
           'model_config': client_config,  # Clean config
           'server_address': server_ip,
           'client_id': client_id
       }
   )
   ```

---

## Database Schema Considerations

### Option 1: Link Jobs to Experiment (Recommended)

```python
class Experiment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    base_model_config = models.ForeignKey(ModelConfig, on_delete=models.CASCADE)
    total_jobs = models.IntegerField()
    completed_jobs = models.IntegerField(default=0)
    status = models.CharField(max_length=20, default='pending')  # pending, running, completed, failed
    created_at = models.DateTimeField(auto_now_add=True)

class TrainingJob(models.Model):
    # Existing fields...
    experiment = models.ForeignKey(Experiment, null=True, blank=True, on_delete=models.CASCADE)
    experiment_combo_index = models.IntegerField(null=True, blank=True)
```

**Benefits:**
- Easy to track experiment progress
- Can display all jobs in experiment as a group
- Can compare results across parameter combinations

### Option 2: Store in ModelConfig Only

```python
# Just store experiment in ModelConfig.config_json
# Simpler but harder to query/filter experiment jobs
```

---

## UI/UX Recommendations

### Experiment Dashboard

Display experiment progress in a dedicated view:

```
Experiment: FedSVM_Kernel_Grid_Search
Status: Running (45/60 jobs completed)
Progress: [███████████░░░░░] 75%

Parameter Combinations:
┌─────────┬─────┬────────┬──────────┐
│ Kernel  │  C  │ Status │ Accuracy │
├─────────┼─────┼────────┼──────────┤
│ linear  │ 0.1 │   ✓    │  0.892   │
│ linear  │ 0.6 │   ✓    │  0.901   │
│ rbf     │ 0.1 │   ⏳   │    -     │
│ rbf     │ 0.6 │  ⏸️    │    -     │
└─────────┴─────┴────────┴──────────┘

Best Configuration:
  Kernel: linear, C: 0.6 → Accuracy: 0.901
```

### Warnings

- **< 50 jobs:** ✅ Safe to run
- **50-100 jobs:** ⚠️ Warning - "This will create 75 training jobs. Continue?"
- **> 100 jobs:** ⛔ Block - "Too many jobs (200). Please reduce parameter ranges."

---

## Security Considerations

1. **Rate Limiting:** Limit experiment creation to prevent abuse (e.g., 5 experiments per user per hour)
2. **Resource Quotas:** Enforce max total_jobs per user (e.g., 500 jobs/day)
3. **Validation:** Sanitize all numeric inputs (from, to, step) to prevent injection
4. **Authorization:** Ensure users can only view/modify their own experiments

---

## Future Enhancements

1. **Random Search:** Support random sampling instead of grid search
2. **Bayesian Optimization:** Use previous results to guide next parameter choices
3. **Early Stopping:** Terminate poor-performing jobs early
4. **Auto-Tuning:** Suggest parameter ranges based on dataset characteristics
5. **Result Visualization:** Interactive charts showing parameter vs. performance
6. **Export Results:** Download experiment results as CSV/JSON

---

## Changelog

- **2025-12-14:** Initial specification created
  - Defined `experiment` structure for ML and DL models
  - Added examples for common use cases
  - Documented implementation guidelines
