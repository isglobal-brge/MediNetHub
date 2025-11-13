def run_flower_server_process(job_id):
    """Wrapper to run Flower server with Django setup"""
    import os
    import django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medinet.settings')
    django.setup()
    
    # Import after Django setup to avoid AppRegistryNotReady
    from webapp.server_fn.server import start_flower_server
    from webapp.models import TrainingJob
    
    # Reconstruct the job object after Django is configured
    job = TrainingJob.objects.get(id=job_id)
    
    # FASE 1: Incluir configuración de clientes en el config_data
    # Obtener configuración de clientes
    clients_config = job.clients_config or {}
    clients_status = job.clients_status or {}
    
    # Modificar config_data para incluir client_ids (preparación para Fase 2)
    config_data = job.config_json.copy()
    config_data['clients_mapping'] = clients_config
    
    start_flower_server(job) 