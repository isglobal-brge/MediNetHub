# Dataset Details View - Plan de Desarrollo

## Resumen Ejecutivo

**Objetivo**: Implementar vista de detalles extensible para datasets con arquitectura preparada para múltiples tipos de datos (tabular, imágenes, texto) manteniendo compatibilidad con sistema actual.

**Estrategia**: Backend extensible + Frontend con dummies inteligentes + Cliente temporal (SQLiteUploader se mantendrá hasta refactor de infraestructura de clientes).

## Arquitectura Técnica

### 1. Extensiones Backend

#### 1.1 Modelo Dataset (models.py)
```python
class Dataset(models.Model):
    # Campos existentes mantenidos...
    connection = models.ForeignKey(Connection, on_delete=models.CASCADE, related_name='datasets')
    dataset_name = models.CharField(max_length=255)
    class_label = models.CharField(max_length=255)
    num_columns = models.IntegerField(default=0)
    num_rows = models.IntegerField(default=0)
    size = models.IntegerField(default=0)
    
    # NUEVOS CAMPOS EXTENSIBLES
    dataset_type = models.CharField(
        max_length=20,
        choices=[
            ('tabular', 'Tabular Data'),
            ('image_classification', 'Image Classification'),
            ('image_segmentation', 'Image Segmentation'),
            ('text', 'Text/NLP'),
            ('time_series', 'Time Series'),
        ],
        default='tabular',
        help_text="Tipo de dataset para métricas específicas"
    )
    
    # Metadata extensible (reemplaza campos específicos futuros)
    extended_metadata = models.JSONField(
        default=dict, 
        blank=True, 
        null=True,
        help_text="Metadata específica según tipo de dataset"
    )
    
    # Campos de fecha mantenidos
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

#### 1.2 Factory Pattern para Métricas (nueva clase)
```python
# webapp/utils/dataset_metrics.py
class DatasetMetricsCalculator:
    @staticmethod
    def get_calculator(dataset_type):
        calculators = {
            'tabular': TabularMetricsCalculator(),
            'image_classification': ImageClassificationMetricsCalculator(),
            'image_segmentation': ImageSegmentationMetricsCalculator(),
            'text': TextMetricsCalculator(),
            'time_series': TimeSeriesMetricsCalculator(),
        }
        return calculators.get(dataset_type, TabularMetricsCalculator())

class TabularMetricsCalculator:
    def calculate_metrics(self, dataset):
        # Lógica actual + balance de clases detallado
        return {
            'type': 'tabular',
            'class_balance': self._calculate_class_balance(dataset),
            'feature_analysis': self._analyze_features(dataset),
            'data_quality': self._assess_data_quality(dataset)
        }

class ImageClassificationMetricsCalculator:
    def calculate_metrics(self, dataset):
        # DUMMY DATA para desarrollo - reemplazar con lógica real en futuro refactor
        return {
            'type': 'image_classification',
            'class_balance': {'clase_1': 45, 'clase_2': 35, 'clase_3': 20},  # Dummy
            'image_stats': {
                'avg_resolution': '512x512',
                'formats': {'DICOM': 70, 'PNG': 20, 'JPEG': 10},
                'color_channels': 1
            },
            'is_dummy': True  # Flag importante para UI
        }
```

#### 1.3 Vista de Detalles (views.py)
```python
def dataset_detail_view(request, dataset_id):
    """
    Vista polimórfica para detalles de dataset según su tipo
    """
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        # Verificar permisos (futuro: validar acceso por usuario/proyecto)
        # if not user_has_access_to_dataset(request.user, dataset):
        #     return HttpResponseForbidden()
        
        # Factory pattern para cálculo de métricas
        calculator = DatasetMetricsCalculator.get_calculator(dataset.dataset_type)
        detailed_metrics = calculator.calculate_metrics(dataset)
        
        context = {
            'dataset': dataset,
            'metrics': detailed_metrics,
            'dataset_type': dataset.dataset_type,
            'is_dummy_data': detailed_metrics.get('is_dummy', False),
            'back_url': request.META.get('HTTP_REFERER', reverse('datasets'))
        }
        
        return render(request, 'webapp/dataset_details.html', context)
        
    except Exception as e:
        logger.error(f"Error loading dataset details {dataset_id}: {str(e)}")
        messages.error(request, "Error loading dataset details")
        return redirect('datasets')
```

#### 1.4 URL Pattern (urls.py)
```python
# Agregar a webapp/urls.py
path('dataset-details/<int:dataset_id>/', views.dataset_detail_view, name='dataset_details'),
```

### 2. Frontend - Diseño ASCII de la Vista

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              🔍 Dataset Details                                 │
│                                                                                 │
│  ← Back to Datasets                                    🔗 Share URL             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  📊 Heart Failure Clinical Dataset                                             │
│  📍 Hospital Sant Joan - 192.168.1.100:5000                                   │
│  🏷️  Tabular Data                                         ⚠️  Preview Mode*     │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────── 📋 Basic Information ────────────────┐ ┌── ⚖️ Class Balance ───┐
│                                                     │ │                       │
│  📈 Samples: 299                                    │ │  ● Death Event        │
│  📊 Features: 12                                    │ │    ├─ Yes: 96 (32%)   │
│  💾 Size: 15.2 KB                                   │ │    └─ No: 203 (68%)   │
│  🎯 Target: DEATH_EVENT                             │ │                       │
│  📝 Type: Binary Classification                     │ │  📊 Balance Ratio     │
│                                                     │ │     2.1:1 (Moderate) │
│                                                     │ │                       │
│                                                     │ │  💡 Recommendation    │
│                                                     │ │     Consider SMOTE    │
└─────────────────────────────────────────────────────┘ └───────────────────────┘

┌────────────────────── 🔢 Feature Analysis ──────────────────────────┐
│                                                                      │
│  📊 Feature Types                    🎯 Target Distribution           │
│  ├─ Numeric: 10                     ┌─────────────────────────────┐   │
│  └─ Categorical: 2                  │ ████████░░ 68% No Death     │   │
│                                     │ ████░░░░░░ 32% Death        │   │
│  📈 Value Ranges                    └─────────────────────────────┘   │
│  ├─ Age: 40-95 years                                                 │
│  ├─ Ejection Fraction: 14-80%       🎲 Data Quality Score            │
│  └─ Creatinine: 0.5-9.4 mg/dL       ████████░░ 83/100               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────── 🎯 ML Training Suitability ───────────────────┐
│                                                                   │
│  ✅ Sample size adequate for binary classification                │
│  ⚠️  Moderate class imbalance - consider balancing techniques     │
│  ✅ Feature variety supports deep learning architectures          │
│  ✅ Data quality sufficient for production models                 │
│                                                                   │
│  🏆 Suitability Score: 8.2/10                                    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌─────────────────── 💡 Recommended Strategies ───────────────────┐
│                                                                  │
│  🎯 Model Types: Neural Networks, Random Forest, SVM            │
│  ⚖️  Balancing: SMOTE, Class Weights, Undersampling             │
│  🔀 Cross-Validation: Stratified K-Fold (k=5)                   │
│  📊 Metrics: Precision, Recall, F1-Score, AUC-ROC               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

                          [🚫 Close]  [➕ Add to Training]
```

### 3. Vista para Imágenes (Dummy Mode)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              🔍 Dataset Details                                 │
│                                                                                 │
│  ← Back to Datasets                                    🔗 Share URL             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  🖼️  Brain MRI Tumor Segmentation                                              │
│  📍 Hospital Clínico - 192.168.1.101:5000                                     │
│  🏷️  Image Segmentation                              ⚠️  Preview Mode (Dummy)   │
└─────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│  ℹ️  Preview Mode: Full metrics will be available after client infrastructure │
│     update. Current data is simulated for development purposes.               │
└───────────────────────────────────────────────────────────────────────────────┘

┌─────────────── 📋 Basic Information ────────────────┐ ┌── 🎨 Mask Classes ────┐
│                                                     │ │                       │
│  🖼️  Images: 1,500                                  │ │  ● Background         │
│  📐 Resolution: 256x256x155                         │ │    └─ 85%             │
│  💾 Size: 2.1 GB                                    │ │  ● Necrotic Core      │
│  🎯 Task: Tumor Segmentation                        │ │    └─ 5%              │
│  📝 Modalities: T1, T2, FLAIR, T1ce                │ │  ● Peritumoral Edema  │
│                                                     │ │    └─ 7%              │
│                                                     │ │  ● Enhancing Tumor    │
│                                                     │ │    └─ 3%              │
└─────────────────────────────────────────────────────┘ └───────────────────────┘

┌────────────────────── 🖼️ Image Analysis (Simulated) ──────────────────────────┐
│                                                                                │
│  📊 Format Distribution              🎯 Segmentation Quality                   │
│  ├─ DICOM: 70%                      ┌─────────────────────────────────────┐   │
│  ├─ NIfTI: 25%                      │ [████████░░] Mask Coverage: 82%     │   │
│  └─ PNG: 5%                         │ [███████░░░] Annotation Quality: 74%│   │
│                                     │ [█████████░] Consistency Score: 91% │   │
│  🌈 Intensity Ranges                └─────────────────────────────────────┘   │
│  ├─ T1: 0-4095 HU                                                             │
│  ├─ T2: 0-3841 HU                   🎲 Dataset Quality Score                  │
│  └─ FLAIR: 0-4022 HU                ███████░░░ 79/100 (Good)                 │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────── 🎯 Segmentation Suitability (Simulated) ───────────────────┐
│                                                                                │
│  ✅ Image resolution adequate for U-Net architectures                          │revisando
│  ⚠️  Class imbalance typical for medical segmentation                          │
│  ✅ Multi-modal data supports advanced segmentation models                     │
│  ⚠️  Small enhancing tumor class may need specialized loss functions           │
│                                                                                │
│  🏆 Suitability Score: 7.8/10                                                 │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────── 💡 Recommended Strategies (Simulated) ────────────────────┐
│                                                                               │
│  🎯 Model Types: U-Net, Attention U-Net, nnU-Net                             │
│  ⚖️  Loss Functions: Dice Loss, Focal Loss, Combined CE+Dice                  │
│  🔀 Augmentation: Rotation, Elastic Deformation, Intensity Scaling            │
│  📊 Metrics: Dice Score, Hausdorff Distance, Sensitivity, Specificity         │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘

                          [🚫 Close]  [➕ Add to Training]
```

### 4. Implementación Template

#### 4.1 Template Principal (dataset_details.html)
```html
{% extends 'base.html' %}
{% load humanize %}

{% block title %}Dataset Details - {{ dataset.dataset_name }}{% endblock %}

{% block content %}
<div class="container py-4">
    <!-- Header con navegación -->
    <div class="row mb-4">
        <div class="col">
            <div class="d-flex justify-content-between align-items-center">
                <div class="d-flex align-items-center">
                    <a href="{{ back_url }}" class="btn btn-outline-secondary me-3">
                        <i class="fas fa-arrow-left me-2"></i>Back
                    </a>
                    <div>
                        <h2 class="mb-0">
                            <i class="fas fa-search me-2"></i>Dataset Details
                        </h2>
                        <p class="text-muted mb-0">Comprehensive analysis and metrics</p>
                    </div>
                </div>
                <button class="btn btn-outline-primary" onclick="shareDatasetURL()">
                    <i class="fas fa-share-alt me-2"></i>Share URL
                </button>
            </div>
        </div>
    </div>

    <!-- Dataset Header Card -->
    <div class="card mb-4 border-0 shadow-sm">
        <div class="card-body">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <h3 class="mb-1">
                        {% if dataset_type == 'tabular' %}
                            <i class="fas fa-table text-primary me-2"></i>
                        {% elif dataset_type == 'image_classification' or dataset_type == 'image_segmentation' %}
                            <i class="fas fa-images text-success me-2"></i>
                        {% elif dataset_type == 'text' %}
                            <i class="fas fa-file-alt text-info me-2"></i>
                        {% endif %}
                        {{ dataset.dataset_name }}
                    </h3>
                    <p class="text-muted mb-2">
                        <i class="fas fa-hospital me-1"></i>
                        {{ dataset.connection.name }} - {{ dataset.connection.ip }}:{{ dataset.connection.port }}
                    </p>
                    <span class="badge bg-secondary">{{ dataset.get_dataset_type_display }}</span>
                </div>
                <div class="col-md-4 text-end">
                    {% if is_dummy_data %}
                        <div class="alert alert-warning mb-0">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            <strong>Preview Mode</strong><br>
                            <small>Full metrics available after infrastructure update</small>
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>

    <!-- Dummy Data Warning (si aplica) -->
    {% if is_dummy_data %}
    <div class="alert alert-info mb-4">
        <div class="d-flex align-items-center">
            <i class="fas fa-info-circle me-3 fa-2x"></i>
            <div>
                <h6 class="mb-1">Preview Mode Active</h6>
                <p class="mb-0">Current metrics are simulated for development purposes. 
                Full dataset analysis will be available after client infrastructure update.</p>
            </div>
        </div>
    </div>
    {% endif %}

    <!-- Métricas según tipo de dataset -->
    {% if dataset_type == 'tabular' %}
        {% include 'webapp/dataset_details/tabular_metrics.html' %}
    {% elif dataset_type == 'image_classification' %}
        {% include 'webapp/dataset_details/image_classification_metrics.html' %}
    {% elif dataset_type == 'image_segmentation' %}
        {% include 'webapp/dataset_details/image_segmentation_metrics.html' %}
    {% elif dataset_type == 'text' %}
        {% include 'webapp/dataset_details/text_metrics.html' %}
    {% else %}
        {% include 'webapp/dataset_details/default_metrics.html' %}
    {% endif %}

    <!-- Action Buttons -->
    <div class="text-center mt-4">
        <button type="button" class="btn btn-secondary me-3" onclick="window.history.back()">
            <i class="fas fa-times me-2"></i>Close
        </button>
        <button type="button" class="btn btn-primary" onclick="addToTraining('{{ dataset.id }}')">
            <i class="fas fa-plus me-2"></i>Add to Training
        </button>
    </div>
</div>
{% endblock %}
```

## 5. Fases de Implementación

### Fase 1: Backend Extensible (1-2 horas)
1. Migración de modelos para agregar campos extensibles
2. Factory pattern para calculadores de métricas
3. Vista polimórfica básica

### Fase 2: Frontend con Dummies (2-3 horas)
1. Template principal responsive
2. Includes específicos por tipo de dataset
3. Datos dummy realistas para imagen/texto

### Fase 3: Integración y Testing (1 hora)
1. URLs y navegación
2. Testing con datos existentes
3. Documentación de componentes temporales

### Fase 4: Futuro Refactor (Post-infraestructura)
1. Reemplazar calculadores dummy con lógica real
2. Actualizar cliente para enviar metadata extendida
3. Implementar métricas avanzadas específicas

## 6. Consideraciones Técnicas

### Compatibilidad Backwards
- Campos existentes mantenidos
- Default values para nuevos campos
- Migración automática sin pérdida de datos

### Performance
- Lazy loading de métricas complejas
- Caching de cálculos costosos
- Paginación para datasets grandes

### Seguridad
- Validación de permisos por dataset
- URLs no adivinables (considerar slugs)
- Rate limiting para vista de detalles

## 7. Notas para Desarrollo

### TODOs Críticos
- [ ] **TEMPORAL**: Cliente SQLiteUploader - reemplazar en refactor de infraestructura
- [ ] **FUTURO**: Implementar lógica real en calculadores de imagen/texto
- [ ] **SECURITY**: Agregar validación de permisos por usuario/proyecto
- [ ] **PERFORMANCE**: Implementar caching para métricas calculadas

### Estructura de Archivos
```
webapp/
├── templates/webapp/
│   ├── dataset_details.html                    # Template principal
│   └── dataset_details/
│       ├── tabular_metrics.html               # Métricas tabulares (real)
│       ├── image_classification_metrics.html  # Métricas imágenes (dummy)
│       ├── image_segmentation_metrics.html    # Métricas segmentación (dummy)
│       ├── text_metrics.html                  # Métricas texto (dummy)
│       └── default_metrics.html               # Fallback
├── utils/
│   └── dataset_metrics.py                     # Factory pattern y calculadores
└── static/css/
    └── dataset_details.css                     # Estilos específicos
```

---

## 8. Plan de Testing Manual (QA)

### 8.1 Pre-requisitos para Testing
```bash
# Datos de prueba necesarios
# 1. Dataset tabular existente (Heart Failure)
# 2. Dataset simulado de imágenes (crear entrada dummy)
# 3. Dataset simulado de texto (crear entrada dummy)
```

### 8.2 Tests de Funcionalidad Core

#### Test 1: Vista de Dataset Tabular (Real Data)
**Objetivo**: Verificar que funciona con datos reales existentes

**Pasos**:
1. Navegar a `/datasets/` 
2. Hacer clic en "View Details" de cualquier dataset tabular
3. Verificar URL: `/dataset-details/{id}/`
4. **Validar elementos**:
   - ✅ Header con nombre del dataset y botón "Back"
   - ✅ Card de información básica (samples, features, size, target)
   - ✅ Sección "Class Balance" con distribución real
   - ✅ Sección "Feature Analysis" con tipos de features
   - ✅ Gráfico de distribución del target
   - ✅ "ML Training Suitability" con score calculado
   - ✅ "Recommended Strategies" específicas
   - ✅ Botones "Close" y "Add to Training"

**Logging esperado en consola del servidor**:
```
INFO: Loading dataset details for ID: {dataset_id}
INFO: Dataset type detected: tabular
INFO: Calculating tabular metrics for dataset: {dataset_name}
INFO: Class balance calculated: {class_distribution}
INFO: Feature analysis completed: {feature_summary}
```

**Criterios de éxito**:
- Tiempo de carga < 2 segundos
- Todas las métricas muestran datos reales (no dummy)
- Sin errores 500 o JavaScript en consola
- Navegación "Back" funciona correctamente

#### Test 2: Vista de Dataset de Imágenes (Dummy Mode)
**Objetivo**: Verificar modo preview con datos simulados

**Setup previo**:
```python
# Crear dataset dummy en Django admin:
# Name: "Brain MRI Tumor Scans"
# Type: "image_segmentation" 
# Connection: cualquier conexión existente
```

**Pasos**:
1. Acceder a dataset con `dataset_type = 'image_segmentation'`
2. **Validar warning de Preview Mode**:
   - ✅ Banner naranja con texto "Preview Mode (Dummy)"
   - ✅ Alert box explicativo sobre datos simulados
3. **Validar métricas simuladas**:
   - ✅ Información básica: "1,500 images", "256x256x155", "2.1 GB"
   - ✅ Mask Classes con porcentajes: "Background 85%", etc.
   - ✅ Format Distribution: "DICOM: 70%", "NIfTI: 25%", "PNG: 5%"
   - ✅ Segmentation Quality bars simuladas
   - ✅ Suitability Score: "7.8/10"

**Logging esperado**:
```
INFO: Loading dataset details for ID: {dataset_id}
INFO: Dataset type detected: image_segmentation
WARNING: Using dummy data calculator for image_segmentation
INFO: Dummy metrics generated for preview mode
```

**Criterios de éxito**:
- Warning claramente visible y entendible
- Datos dummy realistas y coherentes
- Flag `is_dummy: true` presente en métricas
- Métricas específicas para segmentación médica

#### Test 3: Navegación y UX
**Objetivo**: Verificar experiencia de usuario completa

**Pasos**:
1. **Test de navegación**:
   - Desde `/datasets/` → Click "View Details" → Verificar breadcrumb
   - Botón "Back" → Debe volver a lista de datasets
   - URL sharing → Copiar URL y abrir en nueva pestaña
2. **Test de responsividad**:
   - Redimensionar ventana → Layout debe adaptarse
   - Móvil (F12 → responsive) → Cards deben stackearse
3. **Test de botones de acción**:
   - "Share URL" → Debe copiar URL al clipboard
   - "Add to Training" → Debe mostrar modal o redirigir

**Logging esperado**:
```
INFO: Dataset details accessed via referrer: /datasets/
INFO: Back URL set: /datasets/
INFO: Share URL requested for dataset: {dataset_id}
```

### 8.3 Tests de Error Handling

#### Test 4: Dataset No Existente
**Objetivo**: Verificar manejo de errores 404

**Pasos**:
1. Acceder a URL: `/dataset-details/99999/`
2. **Validar comportamiento**:
   - ✅ Debe mostrar página 404 o redirigir a `/datasets/`
   - ✅ Mensaje de error user-friendly
   - ✅ No debe crashear la aplicación

**Logging esperado**:
```
ERROR: Dataset with ID 99999 not found
INFO: Redirecting to datasets list
```

#### Test 5: Tipo de Dataset No Soportado
**Objetivo**: Verificar fallback para tipos desconocidos

**Setup**:
```python
# Crear dataset con dataset_type = 'unknown_type'
```

**Pasos**:
1. Acceder al dataset con tipo no válido
2. **Validar fallback**:
   - ✅ Debe usar calculador por defecto (TabularMetricsCalculator)
   - ✅ Template debe renderizar métricas básicas
   - ✅ Warning sobre tipo no reconocido

### 8.4 Tests de Performance

#### Test 6: Carga con Dataset Grande
**Objetivo**: Verificar performance con datasets de gran tamaño

**Setup**:
```python
# Modificar temporalmente un dataset para simular gran tamaño:
dataset.num_rows = 100000
dataset.size = 50000000  # 50MB
dataset.save()
```

**Pasos**:
1. Acceder a dataset grande
2. **Medir performance**:
   - ✅ Tiempo de carga < 3 segundos
   - ✅ Métricas calculadas correctamente
   - ✅ Sin timeouts en servidor

**Logging esperado**:
```
INFO: Processing large dataset: 100,000 rows, 50MB
INFO: Metrics calculation completed in: {time}ms
```

### 8.5 Tests de Integración

#### Test 7: Integración con Sistema Actual
**Objetivo**: Verificar compatibilidad con funcionalidad existente

**Pasos**:
1. **Test de datasets existentes**:
   - Verificar que todos los datasets en `/datasets/` tienen link "View Details"
   - Acceder a cada tipo y confirmar que cargan
2. **Test de datos existentes**:
   - Confirmar que campos existentes se muestran correctamente
   - Verificar que `class_label` se usa como target
   - Validar que `num_rows`, `num_columns`, `size` son precisos

#### Test 8: Test de Compatibilidad Backwards
**Objetivo**: Asegurar que cambios no rompen funcionalidad existente

**Pasos**:
1. **Antes de implementar cambios**:
   - Tomar screenshot de `/datasets/`
   - Anotar funcionalidad actual
2. **Después de implementar**:
   - Verificar que `/datasets/` funciona igual
   - Confirmar que no hay errores en logs
   - Validar que performance no se degradó

### 8.6 Checklist de Testing Completo

**Funcionalidad Core**:
- [ ] Dataset tabular muestra métricas reales
- [ ] Dataset imágenes muestra warning de preview mode
- [ ] Navegación "Back" funciona
- [ ] Botón "Share URL" copia URL correcta
- [ ] Layout responsive en móvil
- [ ] Tiempo de carga < 2 segundos

**Error Handling**:
- [ ] 404 para dataset inexistente
- [ ] Fallback para tipo desconocido
- [ ] No crashes con datos malformados
- [ ] Mensajes de error user-friendly

**Integración**:
- [ ] Compatible con datasets existentes
- [ ] Links desde `/datasets/` funcionan
- [ ] No regresiones en funcionalidad actual
- [ ] Logs informativos sin errores

**Performance**:
- [ ] Datasets grandes cargan en <3s
- [ ] Sin memory leaks en navegación repetida
- [ ] Métricas calculadas eficientemente

### 8.7 Comandos de Debug para QA

```bash
# Ver logs del servidor durante testing
tail -f webapp/debug.log | grep "dataset_detail"

# Verificar métricas calculadas en Django shell
python manage.py shell
>>> from webapp.models import Dataset
>>> from webapp.utils.dataset_metrics import DatasetMetricsCalculator
>>> d = Dataset.objects.get(id=1)
>>> calc = DatasetMetricsCalculator.get_calculator(d.dataset_type)
>>> metrics = calc.calculate_metrics(d)
>>> print(metrics)

# Test de URLs
curl -I http://localhost:8000/dataset-details/1/
curl -I http://localhost:8000/dataset-details/99999/
```

### 8.8 Criterios de Aceptación Final

**Must Have**:
- ✅ Vista funciona para todos los tipos de dataset
- ✅ Datos dummy claramente identificados como preview
- ✅ Performance aceptable (<3s para cualquier dataset)
- ✅ Sin errores 500 o JavaScript
- ✅ Navegación intuitiva y sin broken links

**Nice to Have**:
- ✅ Animaciones suaves en carga
- ✅ Tooltips explicativos en métricas
- ✅ Copy-paste funcionando en "Share URL"
- ✅ Métricas visualmente atractivas

**Blockers**:
- ❌ Crashes con datasets existentes
- ❌ Pérdida de funcionalidad actual
- ❌ Datos sensibles expuestos incorrectamente
- ❌ Performance inaceptable (>5s)

---

**Estado**: ✅ Listo para implementación
**Prioridad**: Alta  
**Tiempo estimado**: 4-6 horas desarrollo completo
**Tiempo estimado testing**: 2-3 horas testing manual completo
**Dependencias**: Ninguna (compatible con sistema actual)