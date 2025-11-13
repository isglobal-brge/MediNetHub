from django.contrib import admin
from .models import UserProfile, ModelConfig, TrainingJob, Connection, Dataset

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'organization', 'created_at')
    search_fields = ('user__username', 'organization')

@admin.register(ModelConfig)
class ModelConfigAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'framework', 'created_at')
    list_filter = ('framework', 'created_at')
    search_fields = ('name', 'description', 'user__username')

@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'model_config', 'status', 'started_at', 'completed_at')
    list_filter = ('status', 'created_at')
    search_fields = ('name', 'description', 'user__username')

@admin.register(Connection)
class ConnectionAdmin(admin.ModelAdmin):
    list_display = ('name', 'ip', 'port', 'active', 'user')
    list_filter = ('active', 'created_at')
    search_fields = ('name', 'ip', 'user__username')

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('dataset_name', 'connection', 'num_rows', 'num_columns', 'class_label')
    list_filter = ('connection',)
    search_fields = ('dataset_name', 'class_label')
