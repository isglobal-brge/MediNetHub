class _Tee:
    """Duplicate a stream to the console and a per-job log file."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def run_flower_server_process(job_id):
    """Wrapper to run Flower server with Django setup"""
    import os
    import sys
    import django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medinet.settings')
    django.setup()

    from django.conf import settings

    # All child output (prints + Flower/gRPC logs) goes to logs/flower_server_job_<id>.log
    log_dir = getattr(settings, 'LOGS_DIR', settings.BASE_DIR / 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, f'flower_server_job_{job_id}.log'),
                    'a', encoding='utf-8', errors='backslashreplace', buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    # Import after Django setup to avoid AppRegistryNotReady
    from webapp.server_fn.server import start_flower_server
    from webapp.models import TrainingJob

    # Reconstruct the job object after Django is configured
    job = TrainingJob.objects.get(id=job_id)

    # FASE 1: Incluir configuración de clientes en el config_data
    clients_config = job.clients_config or {}
    clients_status = job.clients_status or {}

    # Modificar config_data para incluir client_ids (preparación para Fase 2)
    config_data = job.config_json.copy()
    config_data['clients_mapping'] = clients_config

    start_flower_server(job)
