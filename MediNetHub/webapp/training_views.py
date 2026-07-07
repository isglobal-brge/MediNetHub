from __future__ import annotations
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse, Http404
from django.views.decorators.http import require_POST
from .models import ModelConfig, TrainingJob, Connection
from webapp.server_process import run_flower_server_process
from webapp.server_fn.server import get_ca_certificate, is_ssl_enabled
import json
import time
import threading
import random
from django.utils import timezone
import requests
from django.utils import timezone
from multiprocessing import Process
from .decorators import user_rate_limit
import socket
import os
import logging

logger = logging.getLogger(__name__)

def clean_model_json_for_ml(model_json):
    """
    Clean model JSON by removing DL-specific fields when model is ML type.

    Args:
        model_json (dict): Original model configuration

    Returns:
        dict: Cleaned model configuration
    """
    if not isinstance(model_json, dict):
        return model_json

    model_type = model_json.get('model', {}).get('metadata', {}).get('model_type')

    if model_type != 'ml':
        return model_json

    cleaned = model_json.copy()

    # Remove DL-specific fields from train section
    if 'train' in cleaned:
        train = cleaned['train'].copy()
        train.pop('epochs', None)
        train.pop('batch_size', None)
        cleaned['train'] = train

    # Remove DL-specific fields from model.training section
    if 'model' in cleaned and 'training' in cleaned['model']:
        training = cleaned['model']['training'].copy()
        training.pop('epochs', None)
        training.pop('batch_size', None)
        cleaned['model']['training'] = training

    return cleaned


def extract_dataset_id_from_model(model_json):
    """
    Extract dataset ID from model configuration.
    
    Args:
        model_json (dict): Model configuration JSON
        
    Returns:
        int: Dataset ID or None if not found
    """
    try:
        if isinstance(model_json, dict):
            # Check model.dataset.selected_datasets[0] structure
            model_config = model_json.get('model', {})
            dataset_config = model_config.get('dataset', {})
            selected_datasets = dataset_config.get('selected_datasets', [])
            
            if selected_datasets and len(selected_datasets) > 0:
                first_dataset = selected_datasets[0]
                if isinstance(first_dataset, dict):
                    dataset_id = first_dataset.get('dataset_id')
                    if dataset_id:
                        return int(dataset_id)
            
            # Check direct dataset_id reference
            dataset_id = model_json.get('dataset_id')
            if dataset_id:
                return int(dataset_id)
                
    except Exception as e:
        logger.error(f"Error extracting dataset ID from model config: {str(e)}")
        return None
    
    return None



def create_center_specific_config(center_datasets, base_config):
    """
    Create configuration containing ONLY data for specific center.
    Critical for federated learning security - prevents credential/data leakage between centers.
    """
    import copy
    
    print(f"Creating center-specific config for {len(center_datasets)} datasets")
    
    center_config = copy.deepcopy(base_config)

    # Include only datasets from this specific center (NO other center data).
    # dataset_id IS included: it is the center's own LOCAL id (not a credential —
    # the connection info below is what stays stripped) and the Node needs it to
    # resolve which local dataset to train on. Omitting it makes every center fall
    # back to the model's baked-in id (Node 1's), so other centers get
    # "Access denied" on a dataset they don't own.
    center_selected = [{
        'dataset_id': ds.get('dataset_id'),
        'dataset_name': ds['dataset_name'],
        'features_info': ds['features_info'],
        'target_info': ds['target_info'],
        'num_columns': ds.get('num_columns', 0),
        'num_rows': ds.get('num_rows', 0),
        'size': ds.get('size', 0),
        # SECURITY: NO connection info included to prevent credential leakage
    } for ds in center_datasets]

    center_config['dataset'] = {'selected_datasets': center_selected}
    # The Node reads its dataset from model.dataset.selected_datasets, so scope
    # THAT nested block to this center too (not just the top-level one).
    if isinstance(center_config.get('model'), dict):
        center_config['model']['dataset'] = {'selected_datasets': center_selected}

    # Log security compliance
    for ds in center_datasets:
        print(f"[FEDERATED] Including dataset '{ds['dataset_name']}' for this center only")

    return center_config


def prepare_center_authentication(connection):
    """
    Prepare authentication headers and credentials for center-specific API communication.
    Follows the same pattern as the datasets function with enhanced security.
    """
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'MediNet-WebApp/1.0'
    }
    auth = None
    
    print(f"[AUTH] Preparing authentication for {connection.name}:")
    print(f"  - IP: {connection.ip}")
    print(f"  - Port: {connection.port}")
    print(f"  - Username: {connection.username if connection.username else 'Not set'}")
    print(f"  - Password: {'Set' if connection.password else 'Not set'}")
    print(f"  - API Key: {'Set' if connection.api_key else 'Not set'}")

    # API Key authentication (preferred for external APIs)
    if connection.api_key:
        headers['Authorization'] = f'Bearer {connection.api_key}'
        print("[AUTH] Using API Key authentication")

    # Basic HTTP authentication fallback
    elif connection.username and connection.password:
        auth = (connection.username, connection.password)
        print("[AUTH] Using Basic HTTP authentication")

    else:
        print("WARNING: [AUTH] No authentication method available")
    
    return headers, auth



def is_running_in_docker():
    """
    Detect if we're running inside a Docker container.

    Returns:
        bool: True if running in Docker, False otherwise
    """
    # Check for .dockerenv file (most common indicator)
    if os.path.exists('/.dockerenv'):
        return True

    # Check cgroup for docker
    try:
        with open('/proc/1/cgroup', 'r') as f:
            return 'docker' in f.read()
    except:
        pass

    return False


def get_server_ip_for_clients():
    """
    Get the server IP address that external clients can use to connect.
    Priority order:
    1. FLOWER_SERVER_IP environment variable (manually configured)
    2. Auto-detection based on Docker/non-Docker environment

    Returns:
        str: IP address that clients should use to connect to this server
    """
    try:
        # PRIORITY 1: Check for manually configured IP in environment variable
        configured_ip = os.environ.get('FLOWER_SERVER_IP')
        if configured_ip:
            return configured_ip

        # PRIORITY 2: Auto-detect based on environment
        in_docker = is_running_in_docker()
        if in_docker:
            # DOCKER MODE: Get host machine IP for containers to connect
            print("[DOCKER] Attempting to detect host machine IP...")

            # Try method 1: Connect to external address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(2)
            try:
                s.connect(('8.8.8.8', 80))
                host_ip = s.getsockname()[0]
                return host_ip
            except Exception as e:
                print(f"[DOCKER] Socket method failed: {e}")
            finally:
                s.close()

            # Try method 2: Get gateway IP (Docker host)
            try:
                gateway_ip = os.popen("ip route | grep default | awk '{print $3}'").read().strip()
                if gateway_ip and gateway_ip != '':
                    print(f"[DOCKER] Using gateway IP: {gateway_ip}")
                    return gateway_ip
            except Exception as e:
                print(f"[DOCKER] Gateway detection failed: {e}")

            # Fallback for Docker
            print("[DOCKER] Using default Docker gateway: 172.17.0.1")
            return '172.17.0.1'
        else:
            # NON-DOCKER MODE: Get the local network IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(2)
            try:
                s.connect(('8.8.8.8', 80))
                local_ip = s.getsockname()[0]
                print(f"[LOCAL] Using local network IP: {local_ip}")
                return local_ip
            except Exception:
                print("WARNING: [LOCAL] Using fallback: 127.0.0.1")
                return '127.0.0.1'
            finally:
                s.close()

    except Exception as e:
        print(f"[ERROR] Critical error detecting server IP: {e}")
        return 'localhost'


@login_required
def training(request):
    """
    Training view - allows configuring and starting federated training jobs
    """
    model_configs = ModelConfig.objects.filter(user=request.user)
    model_id = request.session.get('model_id')

    model_json = {}

    if model_id:
        try:
            model_config = ModelConfig.objects.get(id=model_id, user=request.user)
            model_json = model_config.config_json
        except ModelConfig.DoesNotExist:
            messages.error(request, 'The specified model was not found.')
            
    connections = Connection.objects.filter(user=request.user, active=True)

    training_jobs = TrainingJob.objects.filter(
        user=request.user
    ).order_by('-created_at')[:10]

    NUM_ROUNDS_HELPER ="Number of federated learning rounds to perform. Each round involves training on selected clients and aggregating their models."
    FRACTION_FIT_HELPER = "Fraction of available clients that will be selected for training in each round (0.1 = 10%, 1.0 = 100%)."
    FRACTION_EVALUATE_HELPER = "Fraction of available clients that will be selected for evaluation in each round (0.1 = 10%, 1.0 = 100%)."
    MIN_FIT_CLIENTS_HELPER = "Minimum number of clients required to participate in training for each round to proceed."
    MIN_EVALUATE_CLIENTS_HELPER = "Minimum number of clients required to participate in evaluation for each round to proceed."
    MIN_AVAILABLE_CLIENTS_HELPER = "Minimum number of clients that must be available before starting the federated training process."
    STRATEGY_HELPER = "Federated learning strategy that determines how client models are aggregated. FedAvg averages model weights, FedProx adds a proximal term to handle client heterogeneity."

    available_metrics = [
        # Basic metrics (calculated by default)
        {'name': 'accuracy', 'description': 'General model accuracy', 'category': 'basic'},
        {'name': 'loss', 'description': 'Training loss', 'category': 'basic'},
        {'name': 'precision', 'description': 'Precision', 'category': 'basic'},
        {'name': 'recall', 'description': 'Recall', 'category': 'basic'},
        {'name': 'f1', 'description': 'F1 Score', 'category': 'basic'},
        
        # Segmentation metrics
        {'name': 'mean_iou', 'description': 'Mean Intersection over Union', 'category': 'segmentation'},
        {'name': 'dice_score', 'description': 'Dice coefficient for segmentation', 'category': 'segmentation'},
        {'name': 'hausdorff_distance', 'description': 'Hausdorff distance for boundary accuracy', 'category': 'segmentation'},
        
        # Medical-specific metrics
        {'name': 'sensitivity', 'description': 'True Positive Rate (Sensitivity)', 'category': 'medical'},
        {'name': 'specificity', 'description': 'True Negative Rate (Specificity)', 'category': 'medical'},
        {'name': 'npv', 'description': 'Negative Predictive Value', 'category': 'medical'},
        
        # Detection/Classification metrics
        {'name': 'auc_roc', 'description': 'Area Under ROC Curve', 'category': 'detection'},
        {'name': 'auc_pr', 'description': 'Area Under Precision-Recall Curve', 'category': 'detection'},
        {'name': 'mean_ap', 'description': 'Mean Average Precision for object detection', 'category': 'detection'},
        
        # Regression/Time series metrics
        {'name': 'mae', 'description': 'Mean Absolute Error', 'category': 'regression'},
        {'name': 'rmse', 'description': 'Root Mean Square Error', 'category': 'regression'},
        {'name': 'mape', 'description': 'Mean Absolute Percentage Error', 'category': 'regression'},
    ]

    selected_datasets = request.session.get('selected_datasets', [])
    context = {
        'model_json': json.dumps(model_json),
        'model_configs': model_configs,
        'connections': connections,
        'selected_datasets': selected_datasets,  
        'training_jobs': training_jobs,
        'available_metrics': available_metrics,
        'model_id': model_id, 
        'NUM_ROUNDS_HELPER': NUM_ROUNDS_HELPER,
        'FRACTION_FIT_HELPER': FRACTION_FIT_HELPER,
        'FRACTION_EVALUATE_HELPER': FRACTION_EVALUATE_HELPER,
        'MIN_FIT_CLIENTS_HELPER': MIN_FIT_CLIENTS_HELPER,
        'MIN_EVALUATE_CLIENTS_HELPER': MIN_EVALUATE_CLIENTS_HELPER,
        'MIN_AVAILABLE_CLIENTS_HELPER': MIN_AVAILABLE_CLIENTS_HELPER,
        'STRATEGY_HELPER': STRATEGY_HELPER,
    }

    return render(request, 'webapp/training.html', context)


@login_required
def update_job_status(request, job_id):
    """
    API endpoint to manually update a job's status
    """
    if request.method == 'POST':
        try:
            job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
            data = json.loads(request.body)
            
            new_status = data.get('status')
            if new_status not in [status for status, _ in TrainingJob.STATUS_CHOICES]:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid status'
                })
            
            job.status = new_status

            if new_status == 'running' and not job.started_at:
                job.started_at = timezone.now()
            elif new_status in ['completed', 'failed', 'cancelled'] and not job.completed_at:
                job.completed_at = timezone.now()
            
            job.save()
            
            return JsonResponse({
                'success': True,
                'message': f'Job status updated to {new_status}'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    })


@login_required
def get_job_metrics(request, job_id):
    """
    API endpoint to get metrics for a training job
    """
    try:
        job = TrainingJob.objects.get(id=job_id, user=request.user)
    except TrainingJob.DoesNotExist:
        return JsonResponse({'error': 'Job not found'}, status=404)

    try:
        metrics = []
        if job.metrics_json:
            if isinstance(job.metrics_json, str):
                try:
                    metrics = json.loads(job.metrics_json)
                except json.JSONDecodeError:
                    metrics = []
            else:
                metrics = job.metrics_json

        progress = job.progress
        if job.status == 'completed' and progress != 100:
            progress = 100

        return JsonResponse({
            'success': True,
            'metrics': metrics,
            'job_status': job.status,
            'progress': progress,
            'current_round': job.current_round,
            'total_rounds': job.total_rounds
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@login_required
def client_status(request, job_id):
    """
    API endpoint to get client status for a training job
    """
    try:
        job = TrainingJob.objects.get(id=job_id, user=request.user)
    except TrainingJob.DoesNotExist:
        return JsonResponse({'error': 'Job not found'}, status=404)

    try:
        clients = job.clients_status or {}
        #clients = json.loads(job.clients_status) if job.clients_status else {}

        return JsonResponse({
            'success': True,
            'clients': clients,
            'client_count': len(clients),
            'job_status': job.status,
            'is_complete': job.status in ['completed', 'failed', 'cancelled']
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@login_required
def job_detail(request, job_id):
    """
    Display the details of a training job in HTML format
    """
    try:
        job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
        
        # Prepare datasets display with full (sanitized) metadata
        datasets = []
        if job.dataset_ids:
            try:
                # The field might be a JSONField (list) or a TextField (string)
                if isinstance(job.dataset_ids, str):
                    datasets_list = json.loads(job.dataset_ids)
                else:
                    datasets_list = job.dataset_ids

                # Sanitize for display: remove sensitive info but keep the rest
                for dataset in datasets_list:
                    if isinstance(dataset, dict) and 'connection' in dataset:
                        # Create sanitized connection info (remove sensitive IP/port)
                        sanitized_connection = {'name': dataset['connection'].get('name', 'Unknown')}
                        # Preserve all dataset information except sensitive connection details
                        dataset['connection'] = sanitized_connection
                datasets = datasets_list
            except (json.JSONDecodeError, TypeError):
                datasets = [] # Handle cases with malformed data

        # Extract final metrics for display
        final_metrics_data = {}
        if job.metrics_json:
            try:
                # The data can be a pre-parsed dict (JSONField) or a JSON string
                metrics_data = job.metrics_json if isinstance(job.metrics_json, dict) else json.loads(job.metrics_json)
                
                # The stored data could be a list of metrics (per round) or a single dict
                if isinstance(metrics_data, list):
                    if metrics_data: # If list is not empty, get the last dictionary
                        final_metrics_data = metrics_data[-1]
                elif isinstance(metrics_data, dict):
                    # Sometimes the actual metrics are nested inside a 'metrics' key
                    final_metrics_data = metrics_data.get('metrics', metrics_data)

            except (json.JSONDecodeError, TypeError):
                # If parsing fails or data is malformed, final_metrics_data remains empty
                pass

    except TrainingJob.DoesNotExist:
        messages.error(request, 'Training job not found.')
        return redirect('training')

    context = {
        'job': job,
        'datasets': datasets,
        'final_metrics_data': final_metrics_data
    }
    return render(request, 'webapp/job_detail.html', context)


@login_required
def download_model(request, job_id):
    """
    Allow users to download trained models
    """
    try:
        job = TrainingJob.objects.get(id=job_id)

        if job.user != request.user:
            messages.error(request, "You don't have permission to access this model.")
            return redirect('training')

        if job.status != 'completed':
            messages.error(request, "This model has not yet completed training.")
            return redirect('job_detail', job_id=job_id)

        response = HttpResponse(content_type='application/json')
        response['Content-Disposition'] = f'attachment; filename="medinet_model_{job_id}.json"'

        model_data = {
            'model_id': job_id,
            'name': job.name,
            'created_at': job.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'completed_at': job.updated_at.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': json.loads(job.config_json) if job.config_json else {},
            'datasets': json.loads(job.dataset_ids) if job.dataset_ids else [],
            'metrics': json.loads(job.metrics_json) if job.metrics_json else {},
            'accuracy': job.accuracy,
            'loss': job.loss,
            'precision': job.precision,
            'recall': job.recall,
            'f1': job.f1,
            'total_rounds': job.total_rounds,
        }
        
        # In a real app, you would include the actual model weights here
        model_data['model_weights'] = "dummy_weights_placeholder"
        
        json.dump(model_data, response, indent=4)
        return response
        
    except TrainingJob.DoesNotExist:
        messages.error(request, "El treball d'entrenament no existeix.")
        return redirect('training')
    
    except Exception as e:
        messages.error(request, f"S'ha produït un error en descarregar el model: {str(e)}")
        return redirect('job_detail', job_id=job_id)



@login_required
def download_metrics(request, job_id):
    """
    Download training metrics for a job
    """
    try:
        job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
        
        metrics = []
        if job.metrics_json:
            try:
                metrics = json.loads(job.metrics_json)
            except json.JSONDecodeError:
                metrics = []
        
        response = HttpResponse(
            json.dumps(metrics, indent=2),
            content_type='application/json'
        )
        response['Content-Disposition'] = f'attachment; filename="metrics_{job_id}.json"'
        return response
        
    except Exception as e:
        messages.error(request, f'Error downloading metrics: {str(e)}')
        return redirect('job_detail', job_id=job_id)


def find_free_flower_port(start=8080, end=8099):
    """Pick a free Flower-server port, skipping ports claimed by active jobs and
    ports with a live listener. Returns None if the whole range is in use."""
    claimed = set()
    for job in TrainingJob.objects.filter(status__in=['pending', 'server_ready', 'running']):
        try:
            p = (job.config_json or {}).get('server', {}).get('port')
            if p:
                claimed.add(int(p))
        except (TypeError, ValueError, AttributeError):
            pass
    for port in range(start, end + 1):
        if port in claimed:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                continue  # already listening
        return port
    return None


@login_required
@user_rate_limit(rate=settings.RATE_LIMITS['START_TRAINING'], method='POST', block=True)
def start_training(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            model_config_id = data.get('model_id')
            name = data.get('job_name', '')
            description = data.get('job_description', '')
            dataset_id = data.get('dataset', '')
            config = data.get('config', {})
            
            if not model_config_id:
                return JsonResponse({'success': False, 'error': 'Model configuration is required'})
            
            if not name:
                return JsonResponse({'success': False, 'error': 'Job name is required'})
            
            model_config = get_object_or_404(ModelConfig, id=model_config_id, user=request.user)
            
            if 'model' not in config:
                config['model'] = model_config.config_json
            
            selected_datasets = request.session.get('selected_datasets', [])
            total_rounds = config.get('train', {}).get('rounds', 10)
            
            # FASE 1: Generar IDs únicos para cada cliente
            clients_config = {}
            clients_status = {}
            
            print(f"START_TRAINING: Generating IDs for {len(selected_datasets)} datasets")
            
            for i, dataset in enumerate(selected_datasets):
                connection_info = dataset.get('connection', {})

                import uuid
                client_id = f"client_{uuid.uuid4().hex[:8]}"

                print(f"GENERATED ID: {client_id} -> {connection_info.get('name', 'Unknown')} ({connection_info.get('ip', 'No IP')})")

                clients_config[client_id] = {
                    'connection_name': connection_info.get('name', ''),
                    'connection_ip': connection_info.get('ip', ''),
                    'connection_port': connection_info.get('port', 0),
                    'dataset_name': dataset.get('dataset_name', '')
                }

                clients_status[client_id] = {
                    'client_id': client_id,
                    'connection_name': connection_info.get('name', ''),
                    'connection_ip': connection_info.get('ip', ''),
                    'connection_port': connection_info.get('port', 0),
                    'dataset_name': dataset.get('dataset_name', ''),
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
                
            print(f"CLIENTS_CONFIG: {clients_config}")
            print(f"CLIENTS_STATUS initialized with {len(clients_status)} clients")
            
            model_config = clean_model_json_for_ml(model_config)

            # Dedicated port per job so concurrent trainings don't collide on 8080.
            # The client reads this same field to know where to connect.
            server_cfg = config.setdefault('server', {})
            server_cfg.setdefault('host', '0.0.0.0')
            if not server_cfg.get('port'):
                free_port = find_free_flower_port()
                if free_port is None:
                    return JsonResponse({'success': False, 'error': 'No hay puertos libres (8080-8099) para el servidor de entrenamiento. Espera a que termine algún entrenamiento en curso.'})
                server_cfg['port'] = free_port

            training_job = TrainingJob.objects.create(
                user=request.user,
                model_config=model_config,
                name=name,
                description=description,
                dataset_id=dataset_id,
                dataset_ids=selected_datasets,
                config_json=config,
                total_rounds=total_rounds,
                status='pending',
                progress=0,
                clients_config=clients_config,
                clients_status=clients_status
            )
            
            # Use real Flower server in separate process
            server_process = Process(target=run_flower_server_process, args=(training_job.id,))
            server_process.start()
            
            # Store process PID in training job for cleanup
            training_job.server_pid = server_process.pid
            training_job.save()
            print(f"Server process started with PID: {server_process.pid}")
            
            # Activate clients in background thread (non-blocking — returns HTTP response immediately)
            activation_thread = threading.Thread(
                target=activate_clients_for_training,
                args=(training_job, server_process),
                daemon=True,
            )
            activation_thread.start()

            def monitor_server_process():
                try:
                    server_process.join()
                    print("[INFO] Flower server process completed")
                except Exception as e:
                    print(f"[ERROR] server process: {str(e)}")
                    # Update job status if process fails
                    training_job.refresh_from_db()
                    if training_job.status not in ['completed', 'failed']:
                        training_job.status = 'failed'
                        training_job.logs = f"Server process error: {str(e)}"
                        training_job.save()
                        # Kill the server process if it's still running
                        if server_process.is_alive():
                            print(f"[INFO] Terminating server process PID: {server_process.pid}")
                            server_process.terminate()
                            server_process.join(timeout=5)
                            if server_process.is_alive():
                                print(f"[INFO] Force killing server process PID: {server_process.pid}")
                                server_process.kill()
            
            monitor_thread = threading.Thread(target=monitor_server_process)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Client activation is now handled inside the server process
            
            return JsonResponse({
                'success': True,
                'id': training_job.id,
                'message': 'Training job started successfully!'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid JSON data'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


@login_required
def dashboard(request, job_id):
    """
    Real-time training dashboard for a specific job
    """
    job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
    
    context = {
        'job': job,
        'job_id': job_id
    }
    
    return render(request, 'webapp/dashboard.html', context)


@login_required
@require_POST
def request_budget_reset_proxy(request):
    """
    Proxy a researcher's budget-reset request from Hub to the relevant MediNetNode.

    Expects JSON body:
        connection_ip   (str)  — Node IP
        connection_port (int)  — Node port
        dataset_id      (int)  — Node-side dataset PK (from budget_exhausted_nodes)
        reason          (str)  — Researcher's justification (max 1000 chars)

    Returns the Node's JSON response verbatim (201 on success, 4xx on error).
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    connection_ip = body.get('connection_ip', '').strip()
    connection_port = body.get('connection_port')
    dataset_id = body.get('dataset_id')
    reason = body.get('reason', '').strip()

    if not connection_ip or not connection_port:
        return JsonResponse({'error': 'connection_ip and connection_port are required'}, status=400)
    if not isinstance(dataset_id, int):
        return JsonResponse({'error': 'dataset_id must be an integer'}, status=400)
    if not reason:
        return JsonResponse({'error': 'reason is required'}, status=400)
    if len(reason) > 1000:
        return JsonResponse({'error': 'reason must be 1000 characters or fewer'}, status=400)

    # Look up the Connection object so we can authenticate with the Node's API key
    connection = Connection.objects.filter(
        ip=connection_ip,
        port=connection_port,
        user=request.user,
        active=True,
    ).first()

    if not connection:
        return JsonResponse({'error': 'Connection not found'}, status=404)

    if not connection.api_key:
        return JsonResponse({'error': 'Connection has no API key configured'}, status=400)

    node_url = f"{settings.MEDINET_NODE_SCHEME}://{connection_ip}:{connection_port}/api/v2/budget-reset/"
    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': connection.api_key,
        'User-Agent': 'MediNet-Hub/1.0',
    }
    payload = {'dataset_id': dataset_id, 'reason': reason}

    try:
        response = requests.post(node_url, json=payload, headers=headers, timeout=10)
        try:
            data = response.json()
        except Exception:
            data = {'raw': response.text[:500]}
        return JsonResponse(data, status=response.status_code)
    except requests.exceptions.ConnectionError:
        return JsonResponse({'error': 'Could not connect to Node'}, status=503)
    except requests.exceptions.Timeout:
        return JsonResponse({'error': 'Node request timed out'}, status=504)
    except requests.exceptions.RequestException as exc:
        return JsonResponse({'error': f'Request failed: {exc}'}, status=502)


@login_required
def budget_status_proxy(request):
    """
    Proxy GET /api/v2/budget-status/ from the Hub to the relevant MediNetNode.

    Query parameters:
        connection_ip   (str)  — Node IP
        connection_port (int)  — Node port

    Returns the Node's JSON response verbatim (list of per-dataset budgets).
    """
    connection_ip = request.GET.get('connection_ip', '').strip()
    connection_port = request.GET.get('connection_port', '').strip()

    if not connection_ip or not connection_port:
        return JsonResponse({'error': 'connection_ip and connection_port are required'}, status=400)

    try:
        connection_port_int = int(connection_port)
    except (ValueError, TypeError):
        return JsonResponse({'error': 'connection_port must be an integer'}, status=400)

    connection = Connection.objects.filter(
        ip=connection_ip,
        port=connection_port_int,
        user=request.user,
        active=True,
    ).first()

    if not connection:
        return JsonResponse({'error': 'Connection not found'}, status=404)
    if not connection.api_key:
        return JsonResponse({'error': 'Connection has no API key configured'}, status=400)

    node_url = f"{settings.MEDINET_NODE_SCHEME}://{connection_ip}:{connection_port_int}/api/v2/budget-status/"
    headers = {
        'X-API-Key': connection.api_key,
        'User-Agent': 'MediNet-Hub/1.0',
    }

    try:
        response = requests.get(node_url, headers=headers, timeout=10)
        try:
            data = response.json()
        except Exception:
            data = {'raw': response.text[:500]}
        return JsonResponse(data, status=response.status_code)
    except requests.exceptions.ConnectionError:
        return JsonResponse({'error': 'Could not connect to Node'}, status=503)
    except requests.exceptions.Timeout:
        return JsonResponse({'error': 'Node request timed out'}, status=504)
    except requests.exceptions.RequestException as exc:
        return JsonResponse({'error': f'Request failed: {exc}'}, status=502)


@login_required
@require_POST
def estimate_epsilon_proxy(request):
    """
    Proxy POST /api/v2/estimate-epsilon/ from the Hub to the relevant MediNetNode.

    Expects JSON body:
        connection_ip   (str)  — Node IP
        connection_port (int)  — Node port
        dataset_name    (str)  — Name of the dataset on the Node
        model_json      (obj)  — Training config (same shape as start-client)

    Returns the Node's JSON response: {estimated_epsilon, delta, dataset_id, dataset_size}
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    connection_ip = body.get('connection_ip', '').strip()
    connection_port = body.get('connection_port')
    dataset_name = body.get('dataset_name', '').strip()
    model_json = body.get('model_json')

    if not connection_ip or not connection_port:
        return JsonResponse({'error': 'connection_ip and connection_port are required'}, status=400)
    if not dataset_name:
        return JsonResponse({'error': 'dataset_name is required'}, status=400)

    try:
        connection_port_int = int(connection_port)
    except (ValueError, TypeError):
        return JsonResponse({'error': 'connection_port must be an integer'}, status=400)

    connection = Connection.objects.filter(
        ip=connection_ip,
        port=connection_port_int,
        user=request.user,
        active=True,
    ).first()

    if not connection:
        return JsonResponse({'error': 'Connection not found'}, status=404)
    if not connection.api_key:
        return JsonResponse({'error': 'Connection has no API key configured'}, status=400)

    node_url = f"{settings.MEDINET_NODE_SCHEME}://{connection_ip}:{connection_port_int}/api/v2/estimate-epsilon/"
    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': connection.api_key,
        'User-Agent': 'MediNet-Hub/1.0',
    }
    payload = {'dataset_name': dataset_name, 'model_json': model_json}

    try:
        response = requests.post(node_url, json=payload, headers=headers, timeout=10)
        try:
            data = response.json()
        except Exception:
            data = {'raw': response.text[:500]}
        return JsonResponse(data, status=response.status_code)
    except requests.exceptions.ConnectionError:
        return JsonResponse({'error': 'Could not connect to Node'}, status=503)
    except requests.exceptions.Timeout:
        return JsonResponse({'error': 'Node request timed out'}, status=504)
    except requests.exceptions.RequestException as exc:
        return JsonResponse({'error': f'Request failed: {exc}'}, status=502)


@login_required
def delete_job(request, job_id):
    """
    Delete a training job
    """
    try:
        job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
        job_name = job.name
        
        if job.status in ['running', 'pending']:
            messages.error(request, 'Cannot delete a running or pending training job.')
            return redirect('job_detail', job_id=job_id)
        
        job.delete()
        messages.success(request, f'Training job "{job_name}" deleted successfully.')
        return redirect('training')
        
    except Exception as e:
        messages.error(request, f'Error deleting job: {str(e)}')
        return redirect('job_detail', job_id=job_id)


@login_required
def api_job_details(request, job_id):
    """
    API endpoint to get job details
    """
    try:
        job = TrainingJob.objects.get(id=job_id, user=request.user)
    except TrainingJob.DoesNotExist:
        return JsonResponse({'error': 'Job not found'}, status=404)

    try:
        metrics = []
        if job.metrics_json:
            try:
                metrics = json.loads(job.metrics_json)
            except json.JSONDecodeError:
                metrics = []

        duration = None
        if job.started_at and job.completed_at:
            duration = (job.completed_at - job.started_at).total_seconds()
        elif job.started_at:
            duration = (timezone.now() - job.started_at).total_seconds()

        data = {
            'id': job.id,
            'name': job.name,
            'status': job.status,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'duration': duration,
            'metrics': metrics,
            'model_name': job.model_config.name if job.model_config else 'Unknown'
        }

        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def manage_job_artifacts(request, job_id):
    """
    Manage job artifacts (models, logs, metrics)
    """
    job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'delete_artifacts':
            # In a real implementation, this would delete actual model files
            messages.success(request, 'Job artifacts would be deleted (not implemented)')
        elif action == 'archive_job':
            job.archived = True
            job.save()
            messages.success(request, f'Job "{job.name}" archived successfully.')
    
    return redirect('job_detail', job_id=job_id)


@login_required
def client_dashboard(request, job_id):
    """
    Client performance dashboard view
    """
    try:
        job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
        
        # DEBUG: Log real data from database
        print(f"CLIENT_DASHBOARD: Job {job_id} requested")
        print(f"Job status: {job.status}")
        print(f"Job clients_status type: {type(job.clients_status)}")
        print(f"Job clients_status: {job.clients_status}")
        print(f"Job clients_config: {job.clients_config}")

        # Get real client data from database
        clients = []
        clients_status = job.clients_status or {}
        clients_config = job.clients_config or {}
        
        print(f"Processing {len(clients_status)} clients from database")
        
        for client_id, status_data in clients_status.items():
            config_data = clients_config.get(client_id, {})
            connection_name = config_data.get('connection_name', 'Unknown')
            connection_ip = config_data.get('connection_ip', 'unknown')
            
            # Get current metrics - data is directly in status_data
            accuracy = status_data.get('accuracy', 0) * 100 if status_data.get('accuracy') else 0
            loss = status_data.get('loss', 0)
            
            client_data = {
                'id': client_id,
                'name': connection_name,
                'description': f'Connection: {connection_name} ({connection_ip})',
                'status': status_data.get('status', 'unknown'),
                'accuracy': round(accuracy, 1),
                'accuracy_decimal': accuracy / 100,
                'loss': round(loss, 3),
                'train_samples': status_data.get('train_samples', 0),
                'test_samples': status_data.get('test_samples', 0),
                'response_time': 1.2,  # TODO: Calculate real response time
                'ip': connection_ip,
                'last_seen': status_data.get('last_seen', 'Unknown'),
                'rounds': job.current_round or 0,
                'trend': 'stable',  # TODO: Calculate trend
                'trend_value': 0,
                'rounds_history': status_data.get('rounds_history', {})
            }
            
            clients.append(client_data)
            print(f"Client processed: {client_id} -> {connection_name} | Status: {client_data['status']} | Acc: {client_data['accuracy']}%")
        
        print(f"Total clients processed: {len(clients)}")
        
        # Estadísticas generales usando datos reales
        total_clients = len(clients)
        active_clients = len([c for c in clients if c['status'] != 'offline'])
        warning_clients = len([c for c in clients if c['status'] == 'warning'])
        active_client_accuracies = [c['accuracy'] for c in clients if c['status'] != 'offline']
        avg_accuracy = sum(active_client_accuracies) / len(active_client_accuracies) if active_client_accuracies else 0
        
        overview_stats = {
            'total_clients': total_clients,
            'active_clients': active_clients,
            'warning_clients': warning_clients,
            'avg_accuracy': round(avg_accuracy, 1)
        }
        
        # Datos del gráfico de rendimiento usando datos reales de la BD
        performance_chart_data = {'labels': [], 'accuracy': [], 'loss': []}
        
        # Extract real performance data from clients' rounds_history
        if clients:
            max_rounds = 0
            for client in clients:
                client_status = clients_status.get(client['id'], {})
                rounds_history = client_status.get('rounds_history', {})
                max_rounds = max(max_rounds, len(rounds_history))
            
            # Build performance data averaging across all clients per round
            for round_num in range(1, max_rounds + 1):
                round_accuracies = []
                round_losses = []
                
                for client in clients:
                    client_status = clients_status.get(client['id'], {})
                    rounds_history = client_status.get('rounds_history', {})
                    round_data = rounds_history.get(str(round_num), {})
                    
                    if round_data:
                        round_accuracies.append(round_data.get('accuracy', 0))
                        round_losses.append(round_data.get('loss', 0))
                
                if round_accuracies:  # Only add if we have data
                    performance_chart_data['labels'].append(f'R{round_num}')
                    performance_chart_data['accuracy'].append(sum(round_accuracies) / len(round_accuracies))
                    performance_chart_data['loss'].append(sum(round_losses) / len(round_losses))
        
        print(f"PERFORMANCE_CHART_DATA: {len(performance_chart_data['labels'])} rounds of real data")
        
        context = {
            'job': job,
            'job_id': job_id,
            'clients': clients,
            'overview_stats': overview_stats,
            'performance_chart_data': performance_chart_data,
            'total_rounds': job.total_rounds or 10
        }
        
        print(f"CONTEXT SENT: {len(clients)} clients, avg_accuracy: {avg_accuracy:.1f}%")
        print(f"SAMPLE CLIENT DATA: {clients[0] if clients else 'No clients'}")
        print(f"CHART DATA SAMPLE: Accuracy R1: {performance_chart_data['accuracy'][0] if performance_chart_data['accuracy'] else 'No data'}")
        
        return render(request, 'webapp/client_dashboard.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading client dashboard: {str(e)}')
        return redirect('dashboard', job_id=job_id)
    


@login_required
def get_clients_data(request, job_id):
    """
    API para obtener datos de clientes en tiempo real
    """
    try:
        job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
        
        client_data = []
        if job.client_data:
            try:
                client_data = json.loads(job.client_data)
            except json.JSONDecodeError:
                client_data = []
        
        return JsonResponse({
            'success': True,
            'clients': client_data,
            'job_status': job.status
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@login_required
def get_client_performance_data(request, job_id, client_id):
    """
    API para obtener datos de rendimiento histórico de un cliente
    """
    try:
        job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
        
        # In a real implementation, this would return actual performance data
        # For now, return placeholder data
        performance_data = {
            'client_id': client_id,
            'rounds': list(range(1, 11)),
            'accuracy': [0.6 + i * 0.03 + random.uniform(-0.02, 0.02) for i in range(10)],
            'loss': [1.2 - i * 0.08 + random.uniform(-0.05, 0.05) for i in range(10)],
            'training_time': [120 + random.uniform(-20, 20) for _ in range(10)]
        }
        
        return JsonResponse({
            'success': True,
            'performance_data': performance_data
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


def activate_clients_for_training(training_job, server_process=None):
    """
    Activate clients after starting the Flower server with secure federated learning.
    Enhanced with credential isolation and new authenticated API.
    """
    try:
        # Wait for server to be ready (with timeout and failure check)
        print(f"Waiting for server to be ready. Current status: {training_job.status}")
        timeout = 60  # 30 seconds timeout
        start_time = time.time()
        
        while training_job.status not in ['server_ready', 'failed', 'cancelled']:
            if time.time() - start_time > timeout:
                print(f"Timeout waiting for server to be ready")
                TrainingJob.objects.filter(id=training_job.id).update(
                    status='failed', logs="Timeout waiting for server to start")
                return
                
            time.sleep(5)
            training_job.refresh_from_db()
            print(f"Checking status: {training_job.status}")
        
        # Check if server failed to start
        if training_job.status in ['failed', 'cancelled']:
            print(f"ERROR: Server failed to start. Status: {training_job.status}")
            return
        
        # Give additional time for server to fully start listening
        print(f"Server ready, waiting additional 5 seconds for full startup...")
        time.sleep(5)
        
        print("[SECURE] activate_clients_for_training - using new authenticated API")
        print(f"training_job: {training_job}")
        
        if not training_job.dataset_ids:
            print("WARNING: No dataset_ids found in training job")
            return
        
        if isinstance(training_job.dataset_ids, str):
            selected_datasets = json.loads(training_job.dataset_ids)
        else:
            selected_datasets = training_job.dataset_ids
            
        print(f"Selected_datasets: {selected_datasets}")
        
        unique_connections = {}
        
        # Extract unique connections
        for dataset in selected_datasets:
            conn = dataset['connection']
            conn_key = f"{conn['ip']}:{conn['port']}"
            if conn_key not in unique_connections:
                unique_connections[conn_key] = conn
                
        print(f"Unique_connections: {unique_connections}")
        print(f"[FEDERATED] Will activate {len(unique_connections)} centers with credential isolation")
        
        # Track client activation results
        activated_clients = []
        failed_clients = []
        budget_exhausted_nodes = []

        # Get server address for clients (automatically detects Docker vs non-Docker)
        server_ip = get_server_ip_for_clients()
        server_port = 8080  # Default port

        if 'server' in training_job.config_json:
            server_config = training_job.config_json['server']
            server_port = server_config.get('port', 8080)

        client_server_address = f"{server_ip}:{server_port}"
        print(f"[SERVER_ADDRESS] Clients will connect to: {client_server_address}")

        # Activate each client with secure, center-specific configuration
        for conn_key, conn in unique_connections.items():
            print(f"[SECURE] Activating client: {conn['name']} ({conn_key})")
            
            # 🆔 FIND CLIENT_ID: Buscar client_id por IP en clients_config  
            client_id = None
            clients_config = training_job.clients_config or {}
            
            for cid, client_info in clients_config.items():
                if client_info['connection_ip'] == conn['ip'] and client_info['connection_port'] == conn['port']:
                    client_id = cid
                    break
            
            if not client_id:
                print(f"ERROR: CLIENT_ID not found for {conn['name']} ({conn['ip']}:{conn['port']})")
                failed_clients.append(f"{conn['name']} (No client_id)")
                continue
                
            print(f"FOUND CLIENT_ID: {conn['name']} -> {client_id}")
            
            # 🔐 SECURITY: Get center-specific credentials from database
            try:
                connection_obj = Connection.objects.get(
                    ip=conn['ip'],
                    port=conn['port'], 
                    user=training_job.user
                )
                print(f"[AUTH] Retrieved credentials for {connection_obj.name}")
            except Connection.DoesNotExist:
                print(f"ERROR: [AUTH] No credentials found for {conn['name']} ({conn['ip']}:{conn['port']})")
                failed_clients.append(f"{conn['name']} (No credentials in database)")
                continue
            
            # 🔐 SECURITY: Prepare center-specific authentication
            auth_config = prepare_center_authentication(connection_obj)
            
            # 🔐 SECURITY: Filter datasets for THIS center only (prevent cross-center data leakage)
            center_datasets = [
                ds for ds in selected_datasets 
                if ds['connection']['ip'] == conn['ip'] and ds['connection']['port'] == conn['port']
            ]
            
            if not center_datasets:
                print(f"No datasets found for center {conn['name']}")
                failed_clients.append(f"{conn['name']} (No datasets for this center)")
                continue
            
            print(f"[FEDERATED] Center {conn['name']} will receive {len(center_datasets)} datasets (isolated)")

            # SECURITY: Create center-specific config (NO cross-center data)
            center_specific_config = create_center_specific_config(center_datasets, training_job.config_json)
            
            # Build secure client configuration
            client_config = {
                "model_json": center_specific_config,
                "server_address": client_server_address,
                "dataset": center_datasets[0]['dataset_name'],  # Primary dataset for this center
                "client_id": client_id,
                "center_datasets": [ds['dataset_name'] for ds in center_datasets]  # All datasets for this center
            }

            # Add SSL/TLS CA certificate if SSL is enabled
            if is_ssl_enabled():
                ca_cert = get_ca_certificate()
                if ca_cert:
                    client_config["ca_cert"] = ca_cert
                    client_config["ssl_enabled"] = True
                    print("[SSL] Including CA certificate for secure connection")
                else:
                    client_config["ssl_enabled"] = False
                    print("WARNING: [SSL] SSL enabled but CA certificate not available")
            else:
                client_config["ssl_enabled"] = False
                print("[SSL] SSL disabled, client will connect without TLS")

            print(f"[SECURE] Client will connect to: {client_server_address}")
            print(f"[SECURE] Sending center-specific config with {len(center_datasets)} datasets")
            print(f"[SECURE] Center datasets: {[ds['dataset_name'] for ds in center_datasets]}")
            print(f"[SECURE] SSL enabled: {client_config.get('ssl_enabled', False)}")

            # Validate port in allowed range (5000-5099)
            if not (5000 <= int(conn['port']) <= 5099):
                print(f"ERROR: Port {conn['port']} not in allowed range (5000-5099) for {conn['name']}")
                failed_clients.append(f"{conn['name']} (Invalid port)")
                continue
            
            # 🚀 Use authenticated /api/v2/start-client endpoint
            client_url = f"{settings.MEDINET_NODE_SCHEME}://{conn['ip']}:{conn['port']}/api/v2/start-client"
            print(f"[API] Making authenticated request to: {client_url}")
            # Redact credentials — API keys must never reach logs/console.
            safe_headers = {k: ('***' if k.lower() in ('x-api-key', 'authorization') else v)
                            for k, v in auth_config.headers.items()}
            print(f"[API] Headers: {safe_headers}")
            
            try:
                # Make authenticated request with center-specific credentials
                response = requests.post(
                    client_url,
                    json=client_config,
                    headers=auth_config.headers,
                    auth=auth_config.basic_auth,
                    # cold node may pay a first-time opacus/torch import
                    timeout=75
                )
                
                print(f"[API] Response status: {response.status_code}")
                print(f"[API] Response headers: {dict(response.headers)}")
                
                if response.content:
                    try:
                        print(f"[API] Response content: {response.text[:500]}...")
                    except:
                        print("[API] Response content: [Could not decode]")
                
                if response.status_code == 200:
                    print(f"[SUCCESS] Client {conn['name']} activated with secure API")
                    activated_clients.append(conn['name'])
                else:
                    print(f"ERROR: [ERROR] Failed to activate client {conn['name']}: HTTP {response.status_code}")
                    try:
                        error_detail = response.json() if response.content else {}
                        print(f"ERROR: [ERROR] Response detail: {error_detail}")
                        # Detect Node budget exhaustion so Hub can surface the reset UI
                        if response.status_code == 403 and error_detail.get('budget_exhausted'):
                            node_dataset_id = error_detail.get('dataset_id')
                            # Find dataset_name for this connection from selected_datasets
                            node_dataset_name = next(
                                (ds.get('dataset_name', '') for ds in selected_datasets
                                 if ds.get('connection', {}).get('ip') == conn['ip']
                                 and ds.get('connection', {}).get('port') == conn['port']),
                                ''
                            )
                            budget_exhausted_nodes.append({
                                'connection_name': conn['name'],
                                'connection_ip': conn['ip'],
                                'connection_port': conn['port'],
                                'dataset_id': node_dataset_id,
                                'dataset_name': node_dataset_name,
                            })
                            print(f"WARNING: [BUDGET] Budget exhausted for {conn['name']} dataset_id={node_dataset_id}")
                    except Exception:
                        print(f"ERROR: [ERROR] Response text: {response.text[:200]}")
                    failed_clients.append(f"{conn['name']} (HTTP {response.status_code})")
                    
            except requests.exceptions.HTTPError as e:
                print(f"ERROR: [HTTP] HTTP error for client {conn['name']}: {str(e)}")
                failed_clients.append(f"{conn['name']} (HTTP Error: {str(e)})")
            except requests.exceptions.ConnectionError as e:
                print(f"ERROR: [CONN] Connection error for client {conn['name']}: {str(e)}")
                failed_clients.append(f"{conn['name']} (Connection Error)")
            except requests.exceptions.Timeout as e:
                print(f"ERROR: [TIMEOUT] Request timeout for client {conn['name']}: {str(e)}")
                failed_clients.append(f"{conn['name']} (Timeout)")
            except requests.exceptions.RequestException as e:
                print(f"ERROR: [REQUEST] Request error for client {conn['name']}: {str(e)}")
                failed_clients.append(f"{conn['name']} (Request Error: {str(e)})")
        
        # Update training job status based on client activation results
        total_clients = len(unique_connections)
        activated_count = len(activated_clients)
        failed_count = len(failed_clients)
        
        print(f"[SUMMARY] Client activation: {activated_count}/{total_clients} centers activated")
        
        if budget_exhausted_nodes:
            training_job.budget_exhausted_nodes = budget_exhausted_nodes
            training_job.save(update_fields=['budget_exhausted_nodes'])

        # Targeted updates only: this thread's instance is stale by now (the
        # Flower process may already have advanced status to running/completed),
        # so a full save() would silently resurrect an old status.
        if activated_count == 0:
            # No clients activated - mark as failed and kill server
            TrainingJob.objects.filter(id=training_job.id).update(
                status='failed',
                logs=f"Failed to activate any federated learning centers. Errors: {'; '.join(failed_clients)}",
            )
            print("ERROR: Training job marked as FAILED - no centers activated")

            # Kill the server process since no clients can connect
            if server_process and server_process.is_alive():
                print("Terminating server process due to center activation failure")
                server_process.terminate()
                server_process.join(timeout=5)
                if server_process.is_alive():
                    print("Force killing server process")
                    server_process.kill()
            
        elif failed_count > 0:
            # Partial activation. If fewer centers came up than the strategy needs
            # (min_available_clients), the Flower server would wait forever and then
            # leak its process/port — so fail the job and kill the server, same as
            # the 0-clients case. Only continue when enough centers are up.
            min_needed = (training_job.config_json.get('federated', {})
                          .get('parameters', {}).get('min_available_clients', 1))
            if activated_count < min_needed:
                TrainingJob.objects.filter(id=training_job.id).update(
                    status='failed',
                    logs=(f"Only {activated_count}/{total_clients} centers activated, "
                          f"below required {min_needed}. Errors: {'; '.join(failed_clients)}"),
                )
                print(f"ERROR: Insufficient centers ({activated_count}<{min_needed}) - marking FAILED")
                if server_process and server_process.is_alive():
                    print("Terminating server process due to insufficient centers")
                    server_process.terminate()
                    server_process.join(timeout=5)
                    if server_process.is_alive():
                        server_process.kill()
            else:
                warning_msg = f"Warning: {failed_count}/{total_clients} centers failed to activate: {'; '.join(failed_clients)}"
                TrainingJob.objects.filter(id=training_job.id).update(logs=warning_msg)
                print(f"WARNING: Federated training continuing with {activated_count} centers. {warning_msg}")

        else:
            # All clients activated successfully
            success_msg = f"All {activated_count} federated learning centers activated successfully: {', '.join(activated_clients)}"
            TrainingJob.objects.filter(id=training_job.id).update(logs=success_msg)
            print("All federated learning centers activated with secure authentication")
                
    except Exception as e:
        print(f"ERROR: Error activating federated learning clients: {str(e)}")
        import traceback
        print(f"ERROR: Full traceback: {traceback.format_exc()}")
        # Mark training job as failed due to client activation error
        TrainingJob.objects.filter(id=training_job.id).update(
            status='failed',
            logs=f"Federated client activation failed: {str(e)}",
        )
        print("ERROR: Training job marked as FAILED due to client activation error")
        
        
def prepare_center_authentication(connection):
    """
    Prepare authentication headers and credentials for center-specific API communication.
    Follows the same pattern as the datasets function with enhanced security.
    """
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'MediNet-WebApp/1.0'
    }
    auth = None
    
    print(f"[AUTH] Preparing authentication for {connection.name}:")
    print(f"  - IP: {connection.ip}")
    print(f"  - Port: {connection.port}")
    print(f"  - Username: {connection.username if connection.username else 'Not set'}")
    print(f"  - Password: {'Set' if connection.password else 'Not set'}")
    print(f"  - API Key: {'Set' if connection.api_key else 'Not set'}")

    # API Key authentication (primary)
    if connection.api_key:
        headers['X-API-Key'] = connection.api_key
        print(f"[AUTH] Using API key authentication (X-API-Key header)")

    # Basic authentication (secondary/fallback)
    if connection.username and connection.password:
        auth = (connection.username, connection.password)
        print(f"[AUTH] Using basic authentication with username: {connection.username}")
    
    class AuthConfig:
        def __init__(self, headers, basic_auth):
            self.headers = headers
            self.basic_auth = basic_auth

    return AuthConfig(headers, auth)


@login_required
@user_rate_limit('training')
def experiment_detail(request, experiment_id):
    """
    Display experiment detail with all jobs comparison.
    Currently uses dummy data until Experiment model is implemented.
    """
    # Dummy experiment data (will be replaced with database query later)
    experiments_data = {
        1: {
            'id': 1,
            'name': 'FedSVM Kernel Grid Search',
            'description': 'Testing RBF kernel with gamma variations',
            'status': 'completed',
            'status_color': 'success',
            'progress': 100,
            'created': '2024-01-15',
            'total_jobs': 9,
            'completed_jobs': 9,
            'best_accuracy': 92.34,
            'jobs': [
                {'id': 101, 'name': 'rbf_gamma_0.001', 'accuracy': 92.34, 'loss': 0.1245, 'f1': 0.9156, 'params': {'kernel': 'rbf', 'gamma': 0.001, 'C': 1.0}, 'rank': 1},
                {'id': 102, 'name': 'rbf_gamma_0.01', 'accuracy': 91.56, 'loss': 0.1389, 'f1': 0.9078, 'params': {'kernel': 'rbf', 'gamma': 0.01, 'C': 1.0}, 'rank': 2},
                {'id': 103, 'name': 'rbf_gamma_0.1', 'accuracy': 90.87, 'loss': 0.1523, 'f1': 0.8998, 'params': {'kernel': 'rbf', 'gamma': 0.1, 'C': 1.0}, 'rank': 3},
                {'id': 104, 'name': 'rbf_gamma_1.0', 'accuracy': 89.23, 'loss': 0.1876, 'f1': 0.8845, 'params': {'kernel': 'rbf', 'gamma': 1.0, 'C': 1.0}, 'rank': 4},
                {'id': 105, 'name': 'linear_C_0.1', 'accuracy': 88.45, 'loss': 0.1945, 'f1': 0.8767, 'params': {'kernel': 'linear', 'C': 0.1}, 'rank': 5},
                {'id': 106, 'name': 'linear_C_1.0', 'accuracy': 87.34, 'loss': 0.2123, 'f1': 0.8656, 'params': {'kernel': 'linear', 'C': 1.0}, 'rank': 6},
                {'id': 107, 'name': 'poly_degree_2', 'accuracy': 86.12, 'loss': 0.2289, 'f1': 0.8534, 'params': {'kernel': 'poly', 'degree': 2, 'C': 1.0}, 'rank': 7},
                {'id': 108, 'name': 'poly_degree_3', 'accuracy': 85.34, 'loss': 0.2456, 'f1': 0.8456, 'params': {'kernel': 'poly', 'degree': 3, 'C': 1.0}, 'rank': 8},
                {'id': 109, 'name': 'sigmoid', 'accuracy': 84.01, 'loss': 0.2678, 'f1': 0.8323, 'params': {'kernel': 'sigmoid', 'C': 1.0}, 'rank': 9}
            ]
        },
        2: {
            'id': 2,
            'name': 'FedSVM Regularization Tuning',
            'description': 'Optimizing C parameter for RBF kernel',
            'status': 'running',
            'status_color': 'primary',
            'progress': 67,
            'created': '2024-01-16',
            'total_jobs': 6,
            'completed_jobs': 4,
            'best_accuracy': 91.45,
            'jobs': [
                {'id': 201, 'name': 'C_0.01', 'accuracy': 89.23, 'loss': 0.1876, 'f1': 0.8845, 'params': {'kernel': 'rbf', 'gamma': 0.001, 'C': 0.01}, 'rank': 3},
                {'id': 202, 'name': 'C_0.1', 'accuracy': 91.45, 'loss': 0.1423, 'f1': 0.9067, 'params': {'kernel': 'rbf', 'gamma': 0.001, 'C': 0.1}, 'rank': 1},
                {'id': 203, 'name': 'C_1.0', 'accuracy': 90.98, 'loss': 0.1534, 'f1': 0.9012, 'params': {'kernel': 'rbf', 'gamma': 0.001, 'C': 1.0}, 'rank': 2},
                {'id': 204, 'name': 'C_10.0', 'accuracy': 88.67, 'loss': 0.1912, 'f1': 0.8789, 'params': {'kernel': 'rbf', 'gamma': 0.001, 'C': 10.0}, 'rank': 4},
            ]
        },
        3: {
            'id': 3,
            'name': 'Multi-kernel Comparison',
            'description': 'Comparing different kernel types with optimal parameters',
            'status': 'pending',
            'status_color': 'secondary',
            'progress': 0,
            'created': '2024-01-17',
            'total_jobs': 4,
            'completed_jobs': 0,
            'best_accuracy': None,
            'jobs': []
        }
    }

    # Get experiment data (or 404 if not found)
    if experiment_id not in experiments_data:
        messages.error(request, f'Experiment with ID {experiment_id} not found.')
        return redirect('training')

    experiment = experiments_data[experiment_id]

    # Prepare jobs data as JSON for JavaScript
    jobs_json = json.dumps(experiment['jobs'])

    context = {
        'experiment': experiment,
        'jobs': experiment['jobs'],
        'jobs_json': jobs_json,
    }

    return render(request, 'webapp/experiment_detail.html', context)