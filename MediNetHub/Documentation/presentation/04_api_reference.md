# MediNet - Referència de l'API REST

## Introducció

L'API REST de MediNet proporciona endpoints per gestionar totes les funcionalitats de la plataforma d'aprenentatge federat. Aquesta documentació detalla tots els endpoints disponibles, paràmetres, respostes i exemples d'ús.

## Autenticació

Tots els endpoints requereixen autenticació excepte els endpoints públics. MediNet suporta autenticació basada en sessió de Django.

### Headers Requerits
```http
Content-Type: application/json
X-CSRFToken: [csrf_token]
Cookie: sessionid=[session_id]
```

## Endpoints de l'API

### 1. Gestió de Models

#### GET /api/get-model-configs/
Obté la llista de configuracions de models de l'usuari.

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "models": [
    {
      "id": 1,
      "name": "Classificador Cardiopaties",
      "created_at": "2024-01-15 10:30"
    }
  ]
}
```

#### POST /api/save-model-config/
Guarda una nova configuració de model.

**Request Body**
```json
{
  "name": "Nou Model CNN",
  "description": "Model convolucional per imatges mèdiques",
  "config": {
    "layers": [
      {
        "type": "conv2d",
        "filters": 32,
        "kernel_size": 3
      }
    ],
    "optimizer": "adam",
    "learning_rate": 0.001
  }
}
```

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "id": 5,
  "message": "Model configuration saved successfully!"
}
```

#### GET /api/get-model-config/{model_id}/
Obté la configuració d'un model específic.

**Paràmetres de Ruta**
- `model_id` (integer): ID del model

**Resposta d'Èxit (200)**
```json
{
  "id": 1,
  "name": "Classificador Cardiopaties",
  "description": "Model per detectar cardiopaties",
  "framework": "pt",
  "config_json": {
    "layers": [...],
    "optimizer": "adam"
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

#### POST /api/delete-model-config/{model_id}/
Elimina una configuració de model.

**Paràmetres de Ruta**
- `model_id` (integer): ID del model

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "message": "Model configuration \"Classificador Cardiopaties\" deleted."
}
```

### 2. Gestió d'Entrenament

#### POST /api/start-training/
Inicia un nou entrenament federat.

**Request Body**
```json
{
  "model_id": 1,
  "job_name": "Entrenament Cardiopaties v1",
  "description": "Primer entrenament del model",
  "rounds": 10,
  "min_clients": 2,
  "selected_datasets": [
    {
      "dataset_name": "cardio_data",
      "connection": {
        "name": "Hospital A",
        "ip": "192.168.1.100",
        "port": 8080
      }
    }
  ]
}
```

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "job_id": 15,
  "message": "Training started successfully!",
  "server_address": "localhost:8080"
}
```

#### GET /api/get-job-metrics/{job_id}/
Obté les mètriques d'un entrenament en temps real.

**Paràmetres de Ruta**
- `job_id` (integer): ID del job d'entrenament

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "metrics": [
    {
      "round": 1,
      "loss": 0.8234,
      "accuracy": 0.6754,
      "precision": 0.6890,
      "recall": 0.6621,
      "f1": 0.6753,
      "timestamp": "2024-01-15T10:35:00Z"
    }
  ],
  "job_status": "running",
  "progress": 30,
  "current_round": 3,
  "total_rounds": 10
}
```

#### POST /api/stop-training/
Atura un entrenament en curs.

**Request Body**
```json
{
  "job_id": 15
}
```

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "message": "Training stopped successfully"
}
```

### 3. Gestió de Connexions

#### POST /api/validate-connection/
Valida els paràmetres d'una connexió.

**Query Parameters**
- `ip` (string): Adreça IP
- `port` (string): Port de connexió

**Resposta d'Èxit (200)**
```json
{
  "valid_ip": true,
  "valid_port": true,
  "valid": true
}
```

#### GET /api/test-connection/
Testa la connectivitat amb un servidor.

**Query Parameters**
- `ip` (string): Adreça IP
- `port` (string): Port de connexió

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "message": "Connection successful!"
}
```

### 4. Gestió de Datasets

#### GET /api/preview-dataset/{dataset_id}/
Obté una previsualització d'un dataset.

**Paràmetres de Ruta**
- `dataset_id` (string): ID del dataset (format: connection_id_dataset_name)

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "preview": {
    "columns": ["edad", "genero", "presion_arterial", "riesgo_cv"],
    "data": [
      [45, "M", 140, 0.12],
      [62, "F", 160, 0.35],
      [38, "M", 120, 0.08]
    ]
  }
}
```

#### GET /api/dataset-stats/{dataset_id}/
Obté estadístiques descriptives d'un dataset.

**Paràmetres de Ruta**
- `dataset_id` (string): ID del dataset

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "stats": {
    "numeric_stats": {
      "edad": {
        "min": 18,
        "max": 85,
        "mean": 54.3,
        "std": 12.8,
        "quartiles": [38, 52, 67]
      }
    },
    "categorical_stats": {
      "genero": {
        "M": 585,
        "F": 665
      }
    }
  }
}
```

#### POST /api/store-selected-datasets/
Emmagatzema els datasets seleccionats per entrenament.

**Request Body**
```json
{
  "datasets": [
    {
      "dataset_name": "cardio_data",
      "connection": {
        "name": "Hospital A",
        "ip": "192.168.1.100",
        "port": 8080
      },
      "rows": 1250,
      "columns": 15,
      "class_label": "riesgo_cv"
    }
  ]
}
```

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "message": "Datasets stored successfully",
  "count": 1
}
```

#### POST /api/check-dataset-status/
Comprova si un dataset està seleccionat.

**Request Body**
```json
{
  "dataset": {
    "dataset_name": "cardio_data",
    "connection": {
      "name": "Hospital A",
      "ip": "192.168.1.100",
      "port": 8080
    }
  }
}
```

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "is_selected": true
}
```

#### POST /api/remove-selected-dataset/
Elimina un dataset de la selecció.

**Request Body**
```json
{
  "dataset": {
    "dataset_name": "cardio_data",
    "connection": {
      "name": "Hospital A",
      "ip": "192.168.1.100",
      "port": 8080
    }
  }
}
```

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "message": "Dataset removed successfully"
}
```

### 5. Sistema de Notificacions

#### GET /api/notifications-count/
Obté el nombre de notificacions no llegides.

**Resposta d'Èxit (200)**
```json
{
  "count": 3
}
```

#### GET /api/recent-notifications/
Obté les notificacions més recents.

**Resposta d'Èxit (200)**
```json
{
  "notifications": [
    {
      "id": 1,
      "title": "Entrenament Completat",
      "message": "El teu model ha completat l'entrenament exitosament",
      "link": "/dashboard/15/",
      "is_read": false,
      "created_at": "15 jan, 10:30"
    }
  ]
}
```

#### POST /api/mark-notification-read/
Marca una notificació com llegida.

**Request Body**
```json
{
  "notification_id": 1
}
```

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "message": "Notification marked as read"
}
```

### 6. Gestió de Projectes

#### POST /api/switch-project/
Canvia el projecte actiu.

**Request Body**
```json
{
  "project_id": 2
}
```

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "message": "Switched to project \"Projecte Cardiopaties\"",
  "project": {
    "id": 2,
    "name": "Projecte Cardiopaties",
    "color": "#2e7d32"
  }
}
```

#### GET /api/projects/
Obté la llista de projectes de l'usuari.

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "projects": [
    {
      "id": 1,
      "name": "Projecte Diabetes",
      "description": "Models per diagnòstic de diabetes",
      "color": "#1976d2",
      "created_at": "2024-01-10T09:00:00Z"
    }
  ]
}
```

### 7. Gestió d'Usuaris

#### GET /api/user-profile/
Obté el perfil de l'usuari actual.

**Resposta d'Èxit (200)**
```json
{
  "id": 1,
  "username": "doctor_martinez",
  "email": "martinez@hospital.cat",
  "first_name": "Joan",
  "last_name": "Martinez",
  "profile": {
    "organization": "Hospital de Barcelona",
    "bio": "Especialista en cardiologia",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

#### PUT /api/user-profile/
Actualitza el perfil de l'usuari.

**Request Body**
```json
{
  "first_name": "Joan",
  "last_name": "Martinez García",
  "email": "j.martinez@hospital.cat",
  "profile": {
    "organization": "Hospital Clínic de Barcelona",
    "bio": "Cardiòleg especialitzat en IA mèdica"
  }
}
```

**Resposta d'Èxit (200)**
```json
{
  "success": true,
  "message": "Profile updated successfully"
}
```

## Codis d'Error

### Errors Comuns

#### 400 Bad Request
```json
{
  "success": false,
  "error": "Invalid JSON data"
}
```

#### 401 Unauthorized
```json
{
  "success": false,
  "error": "Authentication required"
}
```

#### 403 Forbidden
```json
{
  "success": false,
  "error": "Permission denied"
}
```

#### 404 Not Found
```json
{
  "success": false,
  "error": "Resource not found"
}
```

#### 500 Internal Server Error
```json
{
  "success": false,
  "error": "Internal server error"
}
```

### Errors Específics

#### Model Not Found
```json
{
  "success": false,
  "error": "Model not found or access denied"
}
```

#### Training Job Failed
```json
{
  "success": false,
  "error": "Training job failed to start",
  "details": {
    "reason": "No datasets selected",
    "code": "NO_DATASETS"
  }
}
```

#### Connection Error
```json
{
  "success": false,
  "error": "Error communicating with hospital server",
  "details": {
    "timeout": true,
    "host": "192.168.1.100",
    "port": 8080
  }
}
```

## Rate Limiting

L'API implementa rate limiting per prevenir abús:

- **Requests generals**: 100 requests per minut per usuari
- **Entrenament**: 5 entrenaments simultanis per usuari
- **Validació de connexions**: 20 requests per minut

### Headers de Rate Limiting
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1640995200
```

## Webhooks (Futur)

MediNet planeja suportar webhooks per notificar esdeveniments:

### Esdeveniments Disponibles
- `training.started`
- `training.completed`
- `training.failed`
- `model.created`
- `connection.established`

### Format de Webhook
```json
{
  "event": "training.completed",
  "timestamp": "2024-01-15T11:00:00Z",
  "data": {
    "job_id": 15,
    "model_name": "Classificador Cardiopaties",
    "final_accuracy": 0.8456
  }
}
```

## SDK i Clients

### Python SDK (Planificat)
```python
from medinet_client import MediNetClient

client = MediNetClient(api_key='your_api_key')
models = client.models.list()
training_job = client.training.start(model_id=1, datasets=[...])
```

### JavaScript SDK (Planificat)
```javascript
import { MediNetClient } from 'medinet-js';

const client = new MediNetClient('your_api_key');
const models = await client.models.list();
const job = await client.training.start({
  modelId: 1,
  datasets: [...]
});
```

## Exemples d'Ús

### Workflow Complet d'Entrenament

1. **Crear un model**
```bash
curl -X POST http://localhost:8000/api/save-model-config/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Model Prova",
    "config": {...}
  }'
```

2. **Seleccionar datasets**
```bash
curl -X POST http://localhost:8000/api/store-selected-datasets/ \
  -H "Content-Type: application/json" \
  -d '{"datasets": [...]}'
```

3. **Iniciar entrenament**
```bash
curl -X POST http://localhost:8000/api/start-training/ \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": 1,
    "job_name": "Test Training"
  }'
```

4. **Monitoritzar progrés**
```bash
curl http://localhost:8000/api/get-job-metrics/1/
```

## Millors Pràctiques

### Seguretat
- Sempre utilitza HTTPS en producció
- Valida tots els inputs
- Implementa rate limiting adequat
- Utilitza tokens de sessió segurs

### Rendiment
- Utilitza paginació per llistes grans
- Implementa caching quan sigui apropiat
- Comprimeix respostes JSON
- Utilitza ETags per cache de clients

### Gestió d'Errors
- Proporciona missatges d'error clars
- Inclou codis d'error específics
- Log tots els errors per debugging
- Implementa retry logic per errors temporals

---

Aquesta documentació de l'API està en constant evolució. Per a la versió més actualitzada, consulta la documentació interactiva disponible a `/api/docs/` quan estigui disponible. 