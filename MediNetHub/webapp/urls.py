from django.urls import path
from django.contrib.auth import views as auth_views
from . import base_views
from . import dataset_views
from . import modelStudio_views
from . import training_views

urlpatterns = [
    # Main pages
    path('', base_views.home, name='home'),
    path('panel/', base_views.user_dashboard, name='user_dashboard'),
    path('datasets/', dataset_views.datasets, name='datasets'),
    path('dataset-details/<str:dataset_id>/', dataset_views.dataset_detail_view, name='dataset_details'),
    path('models/', modelStudio_views.models_list, name='models_list'),
    path('model-designer/', modelStudio_views.model_designer, name='model_designer'),
    path('model-designer/<int:model_id>/', modelStudio_views.model_designer, name='edit_model'),
    path('model-designer-advanced/', modelStudio_views.model_designer_advanced, name='model_designer_advanced'),
    path('model-designer-advanced/<int:model_id>/', modelStudio_views.model_designer_advanced, name='edit_model_advanced'),
    path('model-studio/', modelStudio_views.model_studio, name='model_studio'),
    path('training/', training_views.training, name='training'),
    path('dashboard/<int:job_id>/', training_views.dashboard, name='dashboard'),
    path('dashboard/<int:job_id>/clients/', training_views.client_dashboard, name='client_dashboard'),
    path('notifications/', base_views.notifications, name='notifications'),
    path('go-to-training/<int:model_id>/', modelStudio_views.go_to_training, name='go_to_training'),
    path('ml-model-designer/', modelStudio_views.ml_model_designer, name='ml_model_designer'),
    path('ml-model-designer/<int:model_id>/', modelStudio_views.ml_model_designer, name='edit_ml_model'),

    # Authentication
    path('login/', auth_views.LoginView.as_view(template_name='webapp/login.html'), name='login'),
    path('logout/', base_views.logout_view, name='logout'),
    path('register/', base_views.register, name='register'),
    path('profile/', base_views.profile, name='profile'),
    
    # Rutas para Jobs
    path('jobs/<int:job_id>/', training_views.job_detail, name='job_detail'),
    path('jobs/<int:job_id>/manage/', training_views.manage_job_artifacts, name='manage_job_artifacts'),
    path('jobs/<int:job_id>/delete/', training_views.delete_job, name='delete_job'),
    
    # API endpoints
    path('api/validate-connection/', dataset_views.validate_connection, name='validate_connection'),
    path('api/test-connection/', dataset_views.test_connection, name='test_connection'),
    path('api/save-model-config/', modelStudio_views.save_model_config, name='api_save_model_config'),
    path('api/get-model-config/<int:model_id>/', modelStudio_views.get_model_config, name='api_get_model_config'),
    path('api/delete-model-config/<int:model_id>/', modelStudio_views.delete_model_config, name='api_delete_model_config'),
    path('api/start-training/', training_views.start_training, name='api_start_training'),
    path('api/get-model-configs/', modelStudio_views.get_model_configs, name='get-model-configs'),
    path('api/store-selected-datasets/', dataset_views.store_selected_datasets, name='store_selected_datasets'),
    path('api/check-dataset-status/', dataset_views.check_dataset_status, name='check_dataset_status'),
    path('api/remove-selected-dataset/', dataset_views.remove_selected_dataset, name='remove_selected_dataset'),
    path('api/job-details/<int:job_id>/', training_views.api_job_details, name='api_job_details'),
    path('api/client-status/<int:job_id>/', training_views.client_status, name='api_client_status'),
    path('api/get-job-metrics/<int:job_id>/', training_views.get_job_metrics, name='api_get_job_metrics'),
    path('api/get-notifications-count/', base_views.get_notifications_count, name='get_notifications_count'),
    path('api/get-recent-notifications/', base_views.get_recent_notifications, name='get_recent_notifications'),
    path('api/switch-project/', base_views.switch_project_api, name='switch_project_api'),
    
    path('api/update-job-status/<int:job_id>/', training_views.update_job_status, name='update_job_status'),
    path('api/download-model/<int:job_id>/', training_views.download_model, name='download_model'),
    path('api/download-metrics/<int:job_id>/', training_views.download_metrics, name='download_metrics'),
    
    # APIs para client dashboard
    path('api/jobs/<int:job_id>/clients/', training_views.get_clients_data, name='get_clients_data'),
    path('api/jobs/<int:job_id>/clients/<str:client_id>/performance/', training_views.get_client_performance_data, name='get_client_performance_data'),
]