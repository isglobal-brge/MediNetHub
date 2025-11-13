# Plan de Desarrollo: Sistema de Tracking de Clientes en Tiempo Real

## 1. Resumen Ejecutivo

Este documento describe el plan de desarrollo para implementar un sistema de tracking de clientes en tiempo real que permita al `client_dashboard.html` monitorear el estado y métricas de los clientes durante el entrenamiento federado.

### Objetivos Principales:
- Asignar IDs únicos a cada cliente cuando se inicia el entrenamiento
- Mapear nombres de conexión del usuario a IDs únicos de cliente
- Trackear métricas y estado de clientes en tiempo real
- Mostrar información actualizada en el dashboard de clientes

## 2. Arquitectura del Sistema

### 2.1 Flujo de Datos
```
datasets.html (conexiones) → TrainingJob.clients_config → Server → TrainingJob.clients_status → client_dashboard.html
```

### 2.2 Componentes Principales
1. **TrainingJob.clients_status**: Almacena IDs únicos, configuración y métricas de clientes
2. **TrainingJob.clients_config**: Configuración inicial de clientes
3. **Server Process**: Recolecta métricas de clientes y actualiza clients_status
4. **Client Dashboard**: Visualiza datos en tiempo real

## 3. Cambios en la Base de Datos

### 3.1 Modificaciones en `TrainingJob`
**Archivo**: `webapp/models.py`

Vamos a aprovechar el campo `clients_status` existente y añadir un nuevo campo para la configuración:

```python
class TrainingJob(models.Model):
    # ... campos existentes ...
    
    # CAMPO EXISTENTE: Ya tenemos clients_status para métricas
    clients_status = models.JSONField(default=dict, blank=True, null=True)  # Existente
    
    # NUEVO: Configuración de clientes para el job (mapeo de IDs)
    clients_config = models.JSONField(default=dict, help_text="Configuración de clientes para este job")
```

### 3.2 Estructura de Datos en `clients_status`
El campo `clients_status` almacenará toda la información de tracking:

```json
{
  "client_abc12345": {
    "client_id": "client_abc12345",
    "connection_name": "Hospital Barcelona",
    "connection_ip": "192.168.1.10",
    "connection_port": 5000,
    "dataset_name": "heart_failure_dataset",
    "status": "training",
    "current_round": 5,
    "accuracy": 0.85,
    "loss": 0.23,
    "train_samples": 1000,
    "test_samples": 200,
    "response_time": 2.5,
    "last_seen": "2024-01-15T10:30:00Z",
    "created_at": "2024-01-15T10:00:00Z"
  },
  "client_def67890": {
    // ... otro cliente
  }
}
```

### 3.3 Estructura de Datos en `clients_config`
El campo `clients_config` almacenará la configuración inicial:

```json
{
  "client_abc12345": {
    "connection_name": "Hospital Barcelona",
    "connection_ip": "192.168.1.10", 
    "connection_port": 5000,
    "dataset_name": "heart_failure_dataset"
  }
}
```

## 4. Implementación por Fases

### Fase 1: Generación de IDs de Cliente (Semana 1)

#### 4.1 Backend - Vista de Inicio de Entrenamiento
**Archivo**: `webapp/views.py`

```python
import uuid
from django.utils import timezone
from django.db import transaction
import logging

logger = logging.getLogger(__name__)

def start_training(request):
    """Modificar para generar IDs de cliente al iniciar entrenamiento"""
    
    if request.method == 'POST':
        # ... lógica existente ...
        
        # NUEVO: Generar IDs únicos para cada conexión seleccionada
        selected_datasets = request.session.get('selected_datasets', [])
        
        clients_config = {}
        clients_status = {}
        
        for dataset in selected_datasets:
            connection_info = dataset['connection']
            
            # Generar ID único para el cliente
            client_id = f"client_{uuid.uuid4().hex[:8]}"
            
            # Guardar configuración del cliente
            clients_config[client_id] = {
                'connection_name': connection_info['name'],
                'connection_ip': connection_info['ip'],
                'connection_port': connection_info['port'],
                'dataset_name': dataset['dataset_name']
            }
            
            # Inicializar estado del cliente
            clients_status[client_id] = {
                'client_id': client_id,
                'connection_name': connection_info['name'],
                'connection_ip': connection_info['ip'],
                'connection_port': connection_info['port'],
                'dataset_name': dataset['dataset_name'],
                'status': 'pending',
                'current_round': 0,
                'accuracy': None,
                'loss': None,
                'train_samples': 0,
                'test_samples': 0,
                'response_time': None,
                'last_seen': None,
                'created_at': timezone.now().isoformat()
            }
        
        # Guardar configuración y estado inicial en el job
        training_job.clients_config = clients_config
        training_job.clients_status = clients_status
        training_job.save()
        
        # ... resto de la lógica ...
```

#### 4.2 Modificación del Proceso del Servidor
**Archivo**: `webapp/server_process.py`

```python
def start_flower_server(job_id, config_data):
    """Modificar para incluir IDs de cliente en la configuración"""
    
    job = TrainingJob.objects.get(id=job_id)
    
    # Obtener configuración de clientes
    clients_config = job.clients_config
    
    # Modificar config_data para incluir client_ids
    config_data['clients_mapping'] = clients_config
    
    # ... resto de la lógica del servidor ...
```

### Fase 2: Tracking de Clientes Ultra-Simplificado (1.5 días)

**Arquitectura**: Cliente API recibe ID → guarda temporal → incluye en métricas → server mapea ID→nombre → dashboard muestra nombre

#### 4.3 Modificación Client API - Recibir y Guardar ID
**Archivo**: `clients/torch_client.py` o similar

```python
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assigned_client_id = None  # Variable temporal para el ID
        
    def set_client_id(self, client_id):
        """Método para asignar ID desde la configuración"""
        self.assigned_client_id = client_id
        logger.info(f"Client assigned ID: {client_id}")
    
    def fit(self, parameters, config):
        """Añadir client_id a las métricas devueltas"""
        # ... lógica de entrenamiento existente ...
        
        # NUEVA LÍNEA: Incluir client_id en métricas
        metrics = {
            'accuracy': accuracy,
            'loss': loss,
            'train_samples': len(trainloader.dataset),
            'client_id': self.assigned_client_id  # ← KEY: ID en métricas
        }
        
        return parameters, len(trainloader.dataset), metrics
```

#### 4.4 Server Tracking - Una Función Simple
**Archivo**: `webapp/server_fn/strategies.py`

```python
def update_client_tracking(server_manager, server_round, results):
    """
    Actualizar métricas usando client_id incluido en las métricas del cliente
    
    Args:
        server_manager: ServerManager instance (tiene job con mappings)
        server_round (int): Número del round actual  
        results: Lista de (ClientProxy, FitRes) de Flower
    """
    try:
        clients_status = server_manager.job.clients_status or {}
        
        if not clients_status:
            logger.warning(f"No client mappings in job {server_manager.job.id}")
            return
        
        # Actualizar cada cliente que devolvió resultados
        for client_proxy, fit_res in results:
            # Obtener client_id desde las métricas del cliente
            client_id = fit_res.metrics.get('client_id')
            
            if client_id and client_id in clients_status:
                # Actualizar métricas directamente usando el mapping existente
                clients_status[client_id].update({
                    'current_round': server_round,
                    'status': 'training',
                    'accuracy': fit_res.metrics.get('accuracy', 0),
                    'loss': fit_res.metrics.get('loss', 0),
                    'train_samples': fit_res.metrics.get('train_samples', 0),
                    'last_seen': timezone.now().isoformat()
                })
                
                logger.info(f"Updated client {client_id} (→ {clients_status[client_id]['connection_name']})")
            else:
                logger.warning(f"Client ID {client_id} not found in mappings")
        
        # Una sola escritura a BD por round
        server_manager.job.clients_status = clients_status
        server_manager.job.save(update_fields=['clients_status'])
        
    except Exception as e:
        logger.error(f"Error updating client tracking: {e}")
```

#### 4.5 Integración en Estrategia Existente
**Archivo**: `webapp/server_fn/strategies.py` (modificación en FedAvgModelStrategy)

```python
class FedAvgModelStrategy(FedAvg):
    # ... código existente ...
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model weights and save checkpoint + CLIENT TRACKING"""
        
        # ... todo el código existente hasta línea ~335 ...
        
        # NUEVA LÍNEA: Actualizar tracking de clientes
        update_client_tracking(self.server_manager, server_round, results)
        
        # ... resto del código existente ...
        return aggregated_parameters, aggregated_metrics
```

#### 4.6 Paso de IDs a Clientes (configuración inicial)
**Archivo**: `webapp/server_process.py` o configuración de clientes

```python
def configure_clients(job):
    """Configurar clientes con sus IDs asignados"""
    clients_status = job.clients_status or {}
    
    # Para cada cliente configurado, pasar su ID
    for client_id, client_info in clients_status.items():
        # Aquí configurarías el cliente con su ID
        # Esto depende de cómo inicias tus clientes federados
        client_config = {
            'client_id': client_id,
            'server_address': 'localhost:8080',
            'connection_name': client_info['connection_name']
        }
        
        logger.info(f"Configured client {client_info['connection_name']} with ID {client_id}")
```

### Fase 3: APIs de Dashboard (0.5 días)

#### 4.7 API Endpoints (YA IMPLEMENTADOS)
**Archivo**: `webapp/views.py`

```python
 @login_required
 def get_clients_data(request, job_id):
     """API para obtener datos de clientes en tiempo real"""
     
     try:
         job = TrainingJob.objects.get(id=job_id, user=request.user)
         clients_status = job.clients_status or {}
         
         clients_data = []
         for client_id, client_info in clients_status.items():
             clients_data.append({
                 'id': client_info['client_id'],
                 'name': client_info['connection_name'],
                 'ip': client_info['connection_ip'],
                 'port': client_info['connection_port'],
                 'dataset_name': client_info['dataset_name'],
                 'status': client_info['status'],
                 'current_round': client_info['current_round'],
                 'accuracy': client_info['accuracy'],
                 'loss': client_info['loss'],
                 'train_samples': client_info['train_samples'],
                 'test_samples': client_info['test_samples'],
                 'response_time': client_info['response_time'],
                 'last_seen': client_info['last_seen'],
             })
        
        # Calcular estadísticas generales
        overview_stats = {
            'total_clients': len(clients_data),
            'active_clients': len([c for c in clients_data if c['status'] in ['connected', 'training']]),
            'warning_clients': len([c for c in clients_data if c['status'] in ['error', 'offline']]),
            'avg_accuracy': sum(c['accuracy'] for c in clients_data if c['accuracy']) / len([c for c in clients_data if c['accuracy']]) if any(c['accuracy'] for c in clients_data) else 0
        }
        
        return JsonResponse({
            'success': True,
            'clients': clients_data,
            'overview_stats': overview_stats
        })
        
    except TrainingJob.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Job not found'}, status=404)

 @login_required
 def get_client_performance_data(request, job_id, client_id):
     """API para obtener datos de rendimiento histórico de un cliente"""
     
     try:
         job = TrainingJob.objects.get(id=job_id, user=request.user)
         clients_status = job.clients_status or {}
         
         if client_id not in clients_status:
             return JsonResponse({'success': False, 'error': 'Client not found'}, status=404)
         
         client_info = clients_status[client_id]
         current_round = client_info['current_round']
         
         # Aquí podrías implementar un sistema de métricas históricas
         # Por ahora, devolvemos datos simulados basados en el round actual
         performance_data = {
             'labels': [f'Round {i}' for i in range(1, current_round + 1)],
             'accuracy': [0.5 + (i * 0.05) for i in range(current_round)],
             'loss': [1.0 - (i * 0.1) for i in range(current_round)]
         }
         
         return JsonResponse({
             'success': True,
             'performance_data': performance_data,
             'client_info': client_info
         })
         
     except TrainingJob.DoesNotExist:
         return JsonResponse({'success': False, 'error': 'Job not found'}, status=404)
```

#### 4.5 URLs
**Archivo**: `webapp/urls.py`

```python
urlpatterns = [
    # ... URLs existentes ...
    
    # APIs para client dashboard
    path('api/jobs/<int:job_id>/clients/', views.get_clients_data, name='get_clients_data'),
    path('api/jobs/<int:job_id>/clients/<str:client_id>/performance/', views.get_client_performance_data, name='get_client_performance_data'),
]
```

### Fase 4: Dashboard - Conectar con APIs Reales (0.5 días)

**Cambio principal**: Reemplazar datos dummy con llamadas a APIs reales.

#### 4.9 Actualizar JavaScript del Dashboard
**Archivo**: `templates/webapp/client_dashboard.html`

```javascript
// Añadir al script existente del dashboard

// Variables globales para tracking
let clientsUpdateInterval;
let currentJobId = {{ job_id }};

// Función para actualizar datos de clientes (REEMPLAZA datos dummy)
function updateClientsData() {
    fetch(`/api/jobs/${currentJobId}/clients/`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateOverviewStats(data.overview_stats);
                updateClientsList(data.clients);
            }
        })
        .catch(error => console.error('Error updating clients data:', error));
}

// Actualizar estadísticas generales
function updateOverviewStats(stats) {
    document.getElementById('total-clients').textContent = stats.total_clients;
    document.getElementById('active-clients').textContent = stats.active_clients;
    document.getElementById('warning-clients').textContent = stats.warning_clients;
    document.getElementById('avg-accuracy').textContent = stats.avg_accuracy.toFixed(1) + '%';
}

// Crear/actualizar elementos de cliente dinámicamente
function updateClientsList(clients) {
    const clientsSidebar = document.querySelector('.client-sidebar');
    
    // Limpiar lista existente (después del header)
    const header = clientsSidebar.querySelector('.client-sidebar-header');
    clientsSidebar.innerHTML = '';
    clientsSidebar.appendChild(header);
    
    // Añadir clientes reales
    clients.forEach(client => {
        const clientElement = createClientElement(client);
        clientsSidebar.appendChild(clientElement);
    });
}

function createClientElement(client) {
    const div = document.createElement('div');
    div.className = 'client-list-item';
    div.setAttribute('data-client-id', client.id);
    
    div.innerHTML = `
        <div class="d-flex align-items-center">
            <span class="client-status status-${client.status}"></span>
            <div>
                <div class="fw-bold">${client.name}</div>
                <small class="text-muted">
                    ${client.status !== 'offline' ? `${client.accuracy}%` : 'Offline'}
                </small>
            </div>
        </div>
        <div class="text-end">
            <small>Round ${client.current_round}</small>
        </div>
    `;
    
    div.addEventListener('click', () => selectClient(client.id));
    return div;
}

// Inicializar con datos reales
document.addEventListener('DOMContentLoaded', function() {
    // Actualización inicial
    updateClientsData();
    
    // Polling cada 5 segundos
    clientsUpdateInterval = setInterval(updateClientsData, 5000);
});
```

### Cronograma Ultra-Simplificado

| **Fase** | **Duración** | **Tareas** | **Archivos** |
|----------|--------------|------------|--------------|
| **Fase 2** | **1.5 días** | Client tracking en server | `strategies.py`, `torch_client.py` |
| **Fase 3** | **0.5 días** | APIs de dashboard | `views.py`, `urls.py` |
| **Fase 4** | **0.5 días** | Conectar dashboard real | `client_dashboard.html` |

**Total: 2.5 días** para funcionalidad completa

### Resultado Final
- Dashboard muestra **"Hospital Barcelona: 85%"** en lugar de **"192.168.1.10: 85%"**
- Tracking en tiempo real de cada cliente individual
- Una función + una línea + dos APIs = solución completa

## 5. Configuración y Despliegue

### 5.1 Migraciones de Base de Datos
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5.2 Variables de Configuración
```python
# settings.py
CLIENT_TRACKING_UPDATE_INTERVAL = 5  # segundos
CLIENT_TIMEOUT_THRESHOLD = 30  # segundos para marcar como offline
```

## 6. Testing Estrategia Completa

### 6.1 Tests Unitarios
```python
# tests/test_client_tracking.py
def test_client_id_generation():
    """Test unique ID generation in start_training"""
    # Test que IDs son únicos y formato correcto
    
def test_flower_client_mapping():
    """Test get_client_id() implementation"""
    # Test mapping client_proxy → client_id
    
def test_metrics_update_atomic():
    """Test update_client_metrics with proper data types"""
    # Test validación datos y atomic updates
```

### 6.2 Tests de Integración Flower
```python
# tests/test_flower_integration.py  
def test_strategy_client_tracking():
    """Test TrackingFedAvg strategy with real Flower clients"""
    # Simular FL round completo con 3 clients
    # Verificar clients_status se actualiza correctamente
    
def test_client_disconnect_handling():
    """Test behavior when client disconnects mid-training"""
    # Simular client timeout/disconnect
    # Verificar estado se marca como 'offline'
```

### 6.3 Tests Frontend
```python
# tests/test_dashboard_api.py
def test_clients_data_api():
    """Test API returns correct data structure"""
    
def test_adaptive_polling():
    """Test JavaScript polling adapts to job status"""
    # Mock responses, verificar intervals cambian
    
def test_dashboard_error_handling():
    """Test dashboard handles API errors gracefully"""
```

### 6.4 Tests de Carga FL
```python
# tests/test_federated_load.py
def test_multiple_clients_performance():
    """Test system performance with 20+ clients"""
    # Verificar response times < 200ms
    # Memory usage estable
    
def test_large_model_tracking():
    """Test tracking with large models (>100MB)"""
    # Verificar no timeouts en updates
```

## 7. Consideraciones de Rendimiento

### 7.1 Arquitectura Federada - Sin Concurrencia
**Realidad**: En federated learning, **solo el server process** actualiza `clients_status`:
- Clients envían resultados → Server recibe → Server actualiza BD
- **No hay race conditions** entre clients (no escriben directamente)
- Server procesa clients **secuencialmente** en cada round

### 7.2 Optimizaciones Reales
**Frontend - Polling Adaptativo**:
```javascript
const POLLING_INTERVALS = {
    training: 3000,   // 3s durante rounds activos
    waiting: 8000,    // 8s entre rounds  
    completed: 30000  // 30s cuando job terminado
};

function getJobStatus(clients) {
    if (clients.some(c => c.status === 'training')) return 'training';
    if (clients.every(c => c.status === 'completed')) return 'completed';
    return 'waiting';
}
```

**Backend - Caché Inteligente**:
```python
from django.core.cache import cache

@login_required
def get_clients_data(request, job_id):
    cache_key = f"clients_data_{job_id}"
    
    # Caché solo si job no está en training activo
    job = TrainingJob.objects.get(id=job_id)
    if job.status != 'running':
        cached_data = cache.get(cache_key)
        if cached_data:
            return JsonResponse(cached_data)
    
    # Generar datos frescos
    clients_data = generate_clients_data(job)
    
    # Caché por 10s si no está training
    if job.status != 'running':
        cache.set(cache_key, clients_data, 10)
    
    return JsonResponse(clients_data)
```

### 7.3 Escalabilidad Federada
- **Límite práctico**: ~50-100 clients por job (Flower limitation)
- **JSONField size**: Max ~1MB per job (reasonable for 100 clients)
- **Network**: Bottleneck real es bandwidth clients ↔ server
- **Cleanup**: Solo jobs antiguos, no concurrency issues

### 7.4 Monitoreo Específico FL
```python
# Métricas importantes para FL
def calculate_fl_stats(clients_status):
    active_clients = [c for c in clients_status.values() if c['status'] == 'training']
    
    return {
        'round_completion': len([c for c in clients_status.values() if c['current_round'] > 0]),
        'stragglers': len([c for c in clients_status.values() if c['response_time'] > 30]),
        'avg_samples': sum(c.get('train_samples', 0) for c in clients_status.values()) / len(clients_status),
        'federation_health': len(active_clients) / len(clients_status) * 100
    }
```

## 8. Cronograma de Implementación Revisado

| Semana | Fase | Entregables | Riesgos |
|--------|------|-------------|---------|
| 1 | **Generación de IDs + Migración** | Modificación start_training, campo clients_config, migración Django | Bajo |
| 2 | **Flower Integration** | get_client_id() implementado, TrackingFedAvg strategy completa | **Alto** - Flower docs |
| 3 | **APIs + Security** | APIs con auth, validación, error handling robusto | Medio |
| 4 | **Frontend Dashboard** | JavaScript adaptativo, UI responsive, polling inteligente | Bajo |  
| 5 | **Testing FL Integration** | Tests con Flower real, load testing, edge cases | **Alto** - FL complexity |
| 6 | **Performance + Deployment** | Optimizaciones, monitoring, production ready | Medio |

### Cambios vs Plan Original:
- **+2 semanas** para testing FL robusto y performance tuning
- **Fase 2 crítica**: Flower integration es el mayor riesgo técnico
- **Testing separado**: FL testing requiere setup específico
- **Security añadida**: APIs necesitan auth/validation proper

## 9. Criterios de Éxito

- [ ] IDs únicos se generan correctamente al iniciar entrenamiento
- [ ] Métricas de clientes se actualizan en tiempo real
- [ ] Dashboard muestra información actualizada cada 5 segundos
- [ ] No hay degradación de rendimiento en el entrenamiento
- [ ] Sistema es robusto ante desconexiones de clientes

## 10. Riesgos y Mitigaciones FL-Específicos

| Riesgo | Probabilidad | Impacto | Mitigación FL |
|--------|--------------|---------|---------------|
| **Flower client mapping falla** | **Alta** | **Crítico** | Implementar múltiples estrategias ID mapping, fallbacks |
| **Client stragglers** | Alta | Alto | Timeout configurables, status 'slow' vs 'offline' |
| **Network partitions** | Media | Alto | Graceful degradation, client reconnection |
| **Large model updates** | Media | Medio | Streaming updates, compression, timeout increases |
| **Dashboard overwhelm** | Baja | Medio | Adaptive polling, client limit warnings |
| **JSONField size explosion** | Baja | Alto | Size limits, data rotation, archival strategy |

### Mitigaciones Específicas FL:
```python
# Robust client identification
def get_client_id_robust(self, client_proxy):
    strategies = [
        lambda: client_proxy.cid,
        lambda: getattr(client_proxy, 'client_id', None),
        lambda: f"client_{hash(str(client_proxy))}",
        lambda: f"client_{uuid.uuid4().hex[:8]}"
    ]
    
    for strategy in strategies:
        try:
            client_id = strategy()
            if client_id:
                return str(client_id)
        except:
            continue
    
    raise ValueError("Cannot identify client")
```

## 11. Documentación Adicional

- Guía de usuario para el client dashboard
- Documentación de APIs
- Guía de troubleshooting para administradores 