### Plan de Desarrollo: Gestión y Almacenamiento de Modelos

#### 1. Resumen Ejecutivo

La propuesta busca mejorar la plataforma añadiendo un sistema avanzado de gestión de modelos entrenados. Esto incluye guardar "checkpoints" de modelos durante el entrenamiento, ofrecer una interfaz dedicada para inspeccionar y descargar rondas específicas, y proporcionar a los usuarios herramientas para gestionar el espacio de almacenamiento que consumen.

#### 2. Arquitectura de Almacenamiento

Se ha decidido utilizar un enfoque híbrido, siguiendo las mejores prácticas de Django:

*   **Sistema de Archivos:** Todos los artefactos de modelo (pesos `.pth` y arquitectura `.json`) se almacenarán en el sistema de archivos, dentro del directorio `media/` de Django. Esto garantiza un alto rendimiento para la lectura/escritura de archivos grandes.
*   **Base de Datos:** La base de datos solo almacenará metadatos sobre los artefactos (rutas, tamaños, etc.) y la configuración del entrenamiento. Esto mantiene la base de datos ligera y rápida.

La estructura de carpetas será la siguiente para garantizar el aislamiento y la fácil gestión:
```
media/
└── models/
    └── user_<ID_DEL_USUARIO>/
        └── job_<ID_DEL_JOB>/
            ├── architecture.json
            ├── checkpoint_round_<N>.pth
            └── final_model.pth
```

#### 3. Cambios en la Base de Datos (`webapp/models.py`)

Se realizarán las siguientes modificaciones en los modelos de la aplicación:

**A. Modelo `TrainingJob` (Modificación):**
```python
class TrainingJob(models.Model):
    # ... campos existentes ...
    config_json = models.JSONField(default=dict)
    
    # NUEVO CAMPO: Frecuencia de guardado de checkpoints.
    # 0 o nulo significa guardar solo el modelo final.
    save_frequency = models.PositiveIntegerField(
        default=0, 
        null=True, 
        blank=True,
        help_text="Guardar un checkpoint cada X rondas. 0 para guardar solo al final."
    )
```

**B. Nuevo Modelo `ModelArtifact`:**
Se creará un nuevo modelo para registrar cada archivo físico guardado.

```python
# Función para generar la ruta de subida dinámicamente
def get_artifact_upload_path(instance, filename):
    user_id = instance.job.user.id
    job_id = instance.job.id
    return f'models/user_{user_id}/job_{job_id}/{filename}'

class ModelArtifact(models.Model):
    job = models.ForeignKey(TrainingJob, on_delete=models.CASCADE, related_name='artifacts')
    
    # Tipo de artefacto: 'architecture', 'checkpoint', 'final_model'
    artifact_type = models.CharField(max_length=20)
    
    # Para checkpoints, el número de ronda correspondiente.
    round_number = models.PositiveIntegerField(null=True, blank=True)

    # El archivo físico en el sistema de archivos.
    file = models.FileField(upload_to=get_artifact_upload_path)
    file_size = models.BigIntegerField() # Se calculará en bytes al guardar.
    
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        # Lógica para calcular y guardar file_size antes de guardar
        if self.file and not self.file_size:
            self.file_size = self.file.size
        super().save(*args, **kwargs)
```

**C. Nuevo Modelo `UserProfile`:**
Para gestionar el almacenamiento total por usuario.

```python
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    storage_used = models.BigIntegerField(default=0, help_text="Almacenamiento total usado en bytes.")

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)
    instance.profile.save()
```

#### 4. Plan de Implementación por Fases

---

#### Fase 1: Funcionalidad Central de Checkpoints

**Objetivo:** Implementar la lógica para guardar la arquitectura y los checkpoints según la `save_frequency`.

1.  **Migración de BD:** Aplicar los cambios de los modelos definidos anteriormente.
2.  **Lógica del Servidor Flower (`server_process.py` o similar):**
    *   **Al inicio del Job:**
        *   Leer el `config_json` del `TrainingJob`.
        *   Extraer la sección de la arquitectura.
        *   Guardarla en un archivo temporal `architecture.json`.
        *   Crear una instancia del nuevo modelo `ModelArtifact` con `artifact_type='architecture'`, apuntando a este archivo. Esto lo registrará en la BD y lo moverá a la ruta correcta (`media/...`).
        *   Actualizar el `storage_used` del usuario con el tamaño del `architecture.json`.
    *   **Durante el Entrenamiento (Callback de Flower):**
        *   Después de cada ronda de agregación, comprobar `if job.save_frequency > 0 and current_round % job.save_frequency == 0:`.
        *   Si se cumple, guardar los pesos del modelo en un archivo temporal `checkpoint_round_<N>.pth`.
        *   Crear una instancia de `ModelArtifact` con `artifact_type='checkpoint'`, `round_number`, etc.
        *   Actualizar el `storage_used` del usuario.
    *   **Al final del Job:**
        *   Guardar siempre el modelo final (`final_model.pth`).
        *   Crear una instancia de `ModelArtifact` con `artifact_type='final_model'`.
        *   Actualizar el `storage_used` del usuario.
3.  **Frontend (Formulario de Entrenamiento):**
    *   Añadir el campo `save_frequency` al formulario para que el usuario pueda definirlo.

---

#### Fase 2: Interfaz de Gestión y Descarga

**Objetivo:** Crear una vista dedicada para que el usuario pueda interactuar con los artefactos de un job.

1.  **URL y Vista (`webapp/views.py`):**
    *   Crear URL `/jobs/<int:job_id>/manage/` y vista `manage_job_artifacts_view`.
    *   La vista consultará `ModelArtifact.objects.filter(job=job_id)`.
    *   Agrupará los artefactos (la arquitectura por un lado, los checkpoints por otro) y los pasará a la plantilla.

2.  **Plantilla HTML (`manage_job_artifacts.html`):**
    *   Mostrará una tabla de artefactos, cada uno con su tipo, ronda (si aplica), tamaño y botones de **Descargar** y **Borrar**.

3.  **Lógica de Borrado:**
    *   El botón "Borrar" llamará a una vista que:
        *   Obtiene el `ModelArtifact` a borrar.
        *   Resta `file_size` del `storage_used` del usuario.
        *   Llama a `artifact.file.delete()` para borrar el archivo del disco.
        *   Llama a `artifact.delete()` para borrar el registro de la BD.

4.  **Actualización de `job_detail.html`:**
    *   El botón "Download Model" ahora se llamará "Manage & Download Models" y enlazará a esta nueva vista.

---

#### Fase 3: Visión de Almacenamiento Global

**Objetivo:** Mostrar al usuario su consumo total de espacio.

1.  **Frontend (Página de Perfil):**
    *   En `profile.html`, mostrar el campo `user.profile.storage_used`, usando un `templatetag` para formatearlo a KB/MB/GB.

2.  **Tareas de Sincronización (Opcional, pero recomendado):**
    *   Crear un `management command` de Django que pueda ser ejecutado periódicamente (con un cron job) para recalcular el espacio usado por cada usuario. Esto serviría como una medida de seguridad para corregir cualquier posible desincronización entre la BD y el sistema de archivos. 