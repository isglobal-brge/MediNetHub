# Nuevo Formato de Configuración de Modelo (JSON)

## 1. Visió General

Aquest document descriu la nova estructura estàndard per als fitxers de configuració de models en format JSON. Aquesta estructura unifica tots els paràmetres necessaris per a la definició, entrenament i desplegament d'un model a la plataforma MediNet.

L'objectiu és proporcionar un format consistent que simplifiqui la interacció entre el frontend, el backend i els clients de Flower.

## 2. Estructura General

El JSON principal conté dos nivells d'informació: la configuració completa del model (`model_json`) i paràmetres addicionals per al client.

```json
{
  "model_json": {
    // ... Contingut principal de la configuració ...
  },
  "server_address": "localhost:8080",
  "dataset": "heart_failure_clinical_records_dataset"
}
```

- **`model_json`**: Un objecte que encapsula tota la configuració relacionada amb l'arquitectura i l'entrenament.
- **`server_address`**: L'adreça del servidor Flower al qual el client s'ha de connectar.
- **`dataset`**: L'identificador del dataset que el client ha d'utilitzar per a l'entrenament.

---

## 3. Detall de `model_json`

Aquesta és la secció més important i conté tota la lògica del model.

### 3.1. Secció `train`

Conté els paràmetres generals per a l'execució de l'entrenament.

```json
"train": {
  "epochs": 5,
  "batch_size": 32,
  "rounds": 10,
  "metrics": []
}
```

| Camp | Tipus | Descripció |
|---|---|---|
| `epochs` | Integer | Nombre d'èpoques d'entrenament en cada client per ronda. |
| `batch_size` | Integer | Mida del lot per a l'entrenament. |
| `rounds` | Integer | Nombre total de rondes d'entrenament federat. |
| `metrics`| Array | Llista de mètriques addicionals a calcular (actualment buit). |

### 3.2. Secció `federated`

Defineix l'estratègia d'aprenentatge federat i els seus paràmetres.

```json
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
```
| Camp | Tipus | Descripció |
|---|---|---|
| `name` | String | Nom de l'estratègia federada (ex: "FedAvg"). |
| `parameters`| Object | Paràmetres específics de l'estratègia. |

### 3.3. Secció `model`

Aquesta és la secció central que defineix el model d'aprenentatge automàtic.

#### `metadata`
Informació general sobre el model.

```json
"metadata": {
  "version": "1.0",
  "created_at": "2025-07-09T09:29:24.576Z",
  "model_type": "dl",
  "framework": "pytorch"
}
```

#### `basic_info`
Nom i descripció del model.

```json
"basic_info": {
  "name": "test2",
  "description": ""
}
```

#### `architecture`
Defineix les capes de la xarxa neuronal. És una llista d'objectes, on cada objecte representa una capa.

```json
"architecture": {
  "layers": [
    {
      "name": "Input Layer",
      "type": "input",
      "params": { "features": 12 },
      "readonly": true
    },
    {
      "name": "Linear",
      "type": "linear",
      "params": {
        "in_features": 12,
        "out_features": 64,
        "bias": true,
        "features": 64
      }
    },
    // ... més capes ...
    {
      "name": "Output Layer",
      "type": "output",
      "params": { "features": 1 },
      "readonly": true
    }
  ]
}
```

#### `training`
Conté els paràmetres específics per a l'entrenament del model, com l'optimitzador i la funció de pèrdua.

**Nota:** Aquesta secció pot contenir paràmetres redundants amb la secció `train` de nivell superior (com `epochs`, `batch_size`, `rounds`). La secció `model.training` ha de tenir prioritat.

```json
"training": {
  "optimizer": {
    "type": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0
  },
  "loss_function": "bce_with_logits",
  "metrics": [ "accuracy", "f1_score" ],
  "epochs": 100,
  "batch_size": 32,
  "rounds": 10
}
```

#### `dataset`
Descriu els datasets que s'utilitzaran per a l'entrenament.

```json
"dataset": {
  "selected_datasets": [
    {
      "dataset_name": "heart_failure_clinical_records_dataset",
      "features_info": {
        "input_features": 12,
        "feature_types": { "numeric": 12, "categorical": 0 }
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
```

---

## 4. Exemple Complet

A continuació es mostra un exemple complet del fitxer JSON, corresponent a `debug_received_from_server.json`.

```json
{
  "model_json": {
    "train": {
      "epochs": 5,
      "batch_size": 32,
      "rounds": 10,
      "metrics": []
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
    },
    "model": {
      "metadata": {
        "version": "1.0",
        "created_at": "2025-07-09T09:29:24.576Z",
        "model_type": "dl",
        "framework": "pytorch"
      },
      "basic_info": {
        "name": "test2",
        "description": ""
      },
      "architecture": {
        "layers": [
          {
            "name": "Input Layer",
            "type": "input",
            "params": {
              "features": 12
            },
            "readonly": true
          },
          {
            "name": "Linear",
            "type": "linear",
            "params": {
              "in_features": 12,
              "out_features": 64,
              "bias": true,
              "features": 64
            }
          },
          {
            "name": "Linear",
            "type": "linear",
            "params": {
              "in_features": 64,
              "out_features": 1,
              "bias": true,
              "features": 1
            }
          },
          {
            "name": "Output Layer",
            "type": "output",
            "params": {
              "features": 1
            },
            "readonly": true
          }
        ]
      },
      "training": {
        "optimizer": {
          "type": "adam",
          "learning_rate": 0.001,
          "weight_decay": 0
        },
        "loss_function": "bce_with_logits",
        "metrics": [
          "accuracy",
          "f1_score"
        ],
        "epochs": 100,
        "batch_size": 32,
        "rounds": 10
      },
      "dataset": {
        "selected_datasets": [
          {
            "dataset_name": "heart_failure_clinical_records_dataset",
            "features_info": {
              "input_features": 12,
              "feature_types": {
                "numeric": 12,
                "categorical": 0
              }
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
  },
  "server_address": "localhost:8080",
  "dataset": "heart_failure_clinical_records_dataset"
}
``` 