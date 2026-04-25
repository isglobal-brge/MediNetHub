from django.db import models
from django.contrib.auth.models import User
from cryptography.fernet import Fernet
from django.conf import settings
import base64
from django.db.models.signals import post_save
from django.dispatch import receiver

class EncryptedTextField(models.TextField):
    """
    Custom field that automatically encrypts/decrypts text data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_fernet(self):
        """Get Fernet cipher instance using the key from settings"""
        key = settings.FERNET_KEYS[0]
        # Ensure key is bytes
        if isinstance(key, str):
            key = key.encode()
        return Fernet(key)
    
    def from_db_value(self, value, expression, connection):
        """Decrypt value when reading from database"""
        if value is None:
            return value
        try:
            fernet = self.get_fernet()
            decrypted = fernet.decrypt(value.encode())
            return decrypted.decode()
        except:
            # If decryption fails, return the original value
            return value
    
    def to_python(self, value):
        """Convert value to Python string"""
        if isinstance(value, str) or value is None:
            return value
        return str(value)
    
    def get_prep_value(self, value):
        """Encrypt value before saving to database"""
        if value is None:
            return value
        try:
            fernet = self.get_fernet()
            encrypted = fernet.encrypt(value.encode())
            return encrypted.decode()
        except:
            # If encryption fails, return the original value
            return value

class Project(models.Model):
    """
    Project model to organize hospital connections and datasets
    """
    COLOR_CHOICES = [
        ('#1976d2', 'Blue'),      # Material Blue
        ('#2e7d32', 'Green'),     # Material Green  
        ('#f57c00', 'Orange'),    # Material Orange
        ('#0288d1', 'Light Blue'), # Material Light Blue
        ('#7b1fa2', 'Purple'),    # Material Purple
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='projects')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    color = models.CharField(max_length=7, choices=COLOR_CHOICES, default='#1976d2')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['user', 'name']
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} ({self.user.username})"

class UserProfile(models.Model):
    """
    Extension of the User model to add additional information
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    organization = models.CharField(max_length=100, blank=True, null=True)
    bio = models.TextField(blank=True, null=True)
    storage_used = models.BigIntegerField(default=0, help_text="Almacenamiento total usado en bytes.")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}'s Profile"

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)
    instance.profile.save()


class ModelConfig(models.Model):
    """
    Model to store neural network configurations
    """
    MODEL_TYPE_CHOICES = (
        ('dl', 'Deep Learning'),
        ('ml', 'Machine Learning'),
    )
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='model_configs')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    framework = models.CharField(max_length=50, default='pt')  # 'pt' for PyTorch
    model_type = models.CharField(max_length=2, choices=MODEL_TYPE_CHOICES, default='dl')
    config_json = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class TrainingJob(models.Model):
    """
    Model to store information about training jobs
    """
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('server_ready', 'Server Ready'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    )
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='training_jobs')
    model_config = models.ForeignKey(ModelConfig, on_delete=models.CASCADE, related_name='training_jobs')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    dataset_id = models.CharField(max_length=255, blank=True, null=True)  # Mantenim el camp original
    dataset_ids = models.JSONField(default=list, blank=True, null=True)  # Nou camp per múltiples datasets
    metrics_file = models.CharField(max_length=255, blank=True, null=True)
    metrics_json = models.JSONField(default=list, blank=True, null=True)  # Emmagatzemar les mètriques en JSON
    model_file_path = models.CharField(max_length=255, blank=True, null=True)
    config_json = models.JSONField(default=dict, blank=True, null=True)  # Configuració en JSON
    progress = models.IntegerField(default=0)  # Progrés en percentatge (0-100)
    current_round = models.IntegerField(default=0)  # Ronda actual d'entrenament
    total_rounds = models.IntegerField(default=0)  # Total de rondes a completar
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # Estat simulat dels clients
    clients_status = models.JSONField(default=dict, blank=True, null=True)  # Estat dels clients en JSON
    # Configuración de clientes para el job (mapeo de IDs)
    clients_config = models.JSONField(default=dict, blank=True, null=True, help_text="Configuración de clientes para este job")
    logs = models.TextField(blank=True, null=True)  # Logs de l'entrenament
    server_pid = models.IntegerField(null=True, blank=True, help_text='PID of the Flower server process')
    training_duration = models.FloatField(null=True, blank=True, help_text='Total training duration in seconds')
    
    # NUEVO CAMPO: Frecuencia de guardado de checkpoints.
    # 0 o nulo significa guardar solo el modelo final.
    save_frequency = models.PositiveIntegerField(default=0, null=True, blank=True, help_text="Save a checkpoint every X rounds. 0 to save only the final model.")

    # Differential-privacy accounting: populated when Flower reports final metrics.
    # None means training has not completed or DP was not active.
    privacy_epsilon = models.FloatField(
        null=True, blank=True,
        help_text="Accumulated ε reported by the Node after training (lower = stronger privacy).",
    )
    privacy_delta = models.FloatField(
        null=True, blank=True,
        help_text="δ value used when computing ε (typically 1e-5).",
    )

    def __str__(self):
        return f"{self.name} ({self.status})"

class Connection(models.Model):
    """
    Model to store connection information for federated learning
    """
    name = models.CharField(max_length=100)
    ip = models.CharField(max_length=45)  # IPv6 can be longer
    port = models.IntegerField()
    username = models.CharField(max_length=100, blank=True, null=True)
    password = EncryptedTextField(blank=True, null=True)
    api_key = EncryptedTextField(blank=True, null=True)  # API key for secure authentication
    active = models.BooleanField(default=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='connections')
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='connections', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.ip}:{self.port})"

class Dataset(models.Model):
    """
    Model to store dataset information
    """
    connection = models.ForeignKey(Connection, on_delete=models.CASCADE, related_name='datasets')
    dataset_name = models.CharField(max_length=255)
    class_label = models.CharField(max_length=255)
    num_columns = models.IntegerField(default=0)
    num_rows = models.IntegerField(default=0)
    size = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.dataset_name} ({self.connection.name})"

class Model(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    config = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name

class Notification(models.Model):
    """
    Model to store notifications for users
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    title = models.CharField(max_length=100)
    message = models.TextField()
    link = models.CharField(max_length=200, blank=True, null=True)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title} - {self.user.username}"


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
    file_size = models.BigIntegerField(help_text="Size of the file in bytes.") # Se calculará en bytes al guardar.

    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        # Lógica para calcular y guardar file_size antes de guardar
        if self.file and hasattr(self.file, 'size'):
            self.file_size = self.file.size
        super().save(*args, **kwargs)

    def __str__(self):
        return f'{self.artifact_type} for job {self.job.name} ({self.file.name})'


class Experiment(models.Model):
    """
    Model to store hyperparameter tuning experiments.
    An experiment groups multiple TrainingJobs with different parameter combinations.
    """
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='experiments')
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    base_model_config = models.ForeignKey(
        ModelConfig,
        on_delete=models.CASCADE,
        related_name='experiments',
        help_text='Base model configuration to use for all jobs in this experiment'
    )

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    # Grid search configuration stored as JSON
    # Example: {"kernel": ["rbf", "linear"], "C": [0.1, 1.0, 10.0], "gamma": [0.001, 0.01]}
    parameter_grid = models.JSONField(
        default=dict,
        help_text='Parameter grid for grid search, e.g. {"kernel": ["rbf", "linear"], "C": [0.1, 1.0]}'
    )

    # List of TrainingJob IDs belonging to this experiment
    # Jobs are fetched on-demand using get_jobs() method
    job_ids = models.JSONField(
        default=list,
        help_text='List of TrainingJob IDs in this experiment'
    )

    # Track the best performing job
    best_job_id = models.IntegerField(null=True, blank=True)
    best_accuracy = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.status})"

    @property
    def total_jobs(self):
        """Total number of jobs in this experiment"""
        return len(self.job_ids)

    @property
    def completed_jobs(self):
        """Number of completed jobs"""
        return TrainingJob.objects.filter(
            id__in=self.job_ids,
            status='completed'
        ).count()

    @property
    def progress(self):
        """Progress percentage (0-100)"""
        if self.total_jobs == 0:
            return 0
        return int((self.completed_jobs / self.total_jobs) * 100)

    def get_jobs(self):
        """
        Fetch all TrainingJobs belonging to this experiment.
        Returns QuerySet ordered by best accuracy (descending).
        """
        return TrainingJob.objects.filter(
            id__in=self.job_ids
        ).order_by('-metrics_json__best_accuracy')

    def update_best_job(self):
        """
        Update best_job_id and best_accuracy based on completed jobs.
        Should be called whenever a job completes.
        """
        # Find the best completed job
        best_job = self.get_jobs().filter(
            status='completed'
        ).order_by('-metrics_json__best_accuracy').first()

        if best_job:
            # Extract best accuracy from metrics_json
            # Assuming metrics_json has a structure like: {"best_accuracy": 92.34, ...}
            if best_job.metrics_json and 'best_accuracy' in best_job.metrics_json:
                self.best_job_id = best_job.id
                self.best_accuracy = best_job.metrics_json['best_accuracy']
                self.save()


class ExperimentJobConfig(models.Model):
    """
    Links a TrainingJob to an Experiment and stores its specific parameter combination.
    This model tracks what hyperparameters were used for each job in the experiment.
    """
    experiment = models.ForeignKey(
        Experiment,
        on_delete=models.CASCADE,
        related_name='job_configs'
    )
    job = models.OneToOneField(
        TrainingJob,
        on_delete=models.CASCADE,
        related_name='experiment_config'
    )

    # Specific parameter combination for this job
    # Example: {"kernel": "rbf", "C": 1.0, "gamma": 0.001}
    parameters = models.JSONField(
        default=dict,
        help_text='Specific parameter combination for this job'
    )

    # Ranking within the experiment (1 = best, 2 = second best, etc.)
    # Updated when jobs complete and rankings change
    rank = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ['rank', '-job__metrics_json__best_accuracy']

    def __str__(self):
        return f"{self.experiment.name} - Job {self.job.id} (Rank: {self.rank or 'N/A'})"
