from __future__ import annotations
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from .models import ModelConfig, TrainingJob, Connection, Notification
import json
import time
import threading
import random
from django.utils import timezone
import requests
from django.views.decorators.http import require_http_methods
from multiprocessing import Process
from .decorators import user_rate_limit
from datetime import datetime, timedelta
from .base_views import create_notification, sanitize_config_for_client, create_center_specific_config
import socket
import os

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
    Handles both Docker and non-Docker environments.

    Returns:
        str: IP address that clients should use to connect to this server
    """
    try:
        # Detect if running in Docker
        in_docker = is_running_in_docker()
        print(f"🐳 [ENVIRONMENT] Running in Docker: {in_docker}")

        # if in_docker:
        #     # DOCKER MODE: Get host machine IP for containers to connect
        #     # Try to get the actual IP address of the host machine
        #     s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #     s.settimeout(0)
        #     try:
        #         # Connect to external address to determine our IP
        #         s.connect(('10.254.254.254', 1))
        #         host_ip = s.getsockname()[0]
        #     except Exception:
        #         # Fallback: try to get gateway IP (Docker host)
        #         host_ip = os.popen("ip route | grep default | awk '{print $3}'").read().strip()
        #         if not host_ip:
        #             host_ip = '172.17.0.1'  # Default Docker gateway
        #     finally:
        #         s.close()
        #
        #     print(f"🌐 [DOCKER] Using host IP for clients: {host_ip}")
        #     return host_ip
        # else:
        # NON-DOCKER MODE: Get the local network IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            # Connect to an external address to determine our IP
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = '127.0.0.1'
        finally:
            s.close()

        print(f"🌐 [LOCAL] Using local network IP for clients: {local_ip}")
        return local_ip

    except Exception as e:
        print(f"⚠️ [ERROR] Error detecting server IP: {e}, using localhost")
        return 'localhost'


@login_required
def training(request):
    """
    Training view - allows configuring and starting federated training jobs
    """
    # Get user's model configurations
    model_configs = ModelConfig.objects.filter(user=request.user)
    # Recuperar el modelo guardado desde la sesión (solución clara)
    model_id = request.session.get('model_id')

    model_json = {}

    if model_id:
        try:
            model_config = ModelConfig.objects.get(id=model_id, user=request.user)
            model_json = model_config.config_json
        except ModelConfig.DoesNotExist:
            messages.error(request, 'The specified model was not found.')
            
    connections = Connection.objects.filter(user=request.user, active=True)

    # Obtener los trabajos de entrenamiento del usuario
    training_jobs = TrainingJob.objects.filter(
        user=request.user
    ).order_by('-created_at')[:10]

    # Helpers with detailed descriptions
    NUM_ROUNDS_HELPER = "Number of federated learning rounds to perform. Each round involves training on selected clients and aggregating their models."
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

    # Get selected datasets from session
    selected_datasets = request.session.get('selected_datasets', [])
    print("🔍 DEBUG: selected_datasets: ", selected_datasets)
    context = {
        'model_json': json.dumps(model_json),  
        'model_configs': model_configs,
        'connections': connections,
        'selected_datasets': selected_datasets,  
        'training_jobs': training_jobs,
        'available_metrics': available_metrics,
        'model_id': model_id,  # ✅ Agregar model_id al context
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
            
            # Actualitzar estat
            job.status = new_status
            
            # Actualitzar timestamps
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
        job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
        
        # Obtenir mètriques
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
        job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
        
        # Obtenir estat de clients
        clients = job.clients_status or {}
        #clients = json.loads(job.clients_status) if job.clients_status else {}
        
        return JsonResponse({
            'success': True,
            'clients': clients,
            'client_count': len(clients),
            'job_status': job.status,  # Inclou l'estat per saber si ha acabat
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
        
        # Check that the user has access to this job
        if job.user != request.user:
            messages.error(request, "You don't have permission to access this model.")
            return redirect('training')
        
        # Check that the job is completed
        if job.status != 'completed':
            messages.error(request, "This model has not yet completed training.")
            return redirect('job_detail', job_id=job_id)
        
        # Generate a simple model file with the configuration and metrics
        response = HttpResponse(content_type='application/json')
        response['Content-Disposition'] = f'attachment; filename="medinet_model_{job_id}.json"'
        
        # Create a model representation with metadata and metrics
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
            
            # Get model config
            model_config = get_object_or_404(ModelConfig, id=model_config_id, user=request.user)
            
            if 'model' not in config:
                config['model'] = model_config.config_json
            
            selected_datasets = request.session.get('selected_datasets', [])
            total_rounds = config.get('train', {}).get('rounds', 10)
            
            # FASE 1: Generar IDs únicos para cada cliente
            clients_config = {}
            clients_status = {}
            
            print(f"🎯 START_TRAINING: Generating IDs for {len(selected_datasets)} datasets")
            
            for i, dataset in enumerate(selected_datasets):
                connection_info = dataset.get('connection', {})
                
                # Generar ID único para el cliente
                import uuid
                client_id = f"client_{uuid.uuid4().hex[:8]}"
                
                print(f"🆔 GENERATED ID: {client_id} → {connection_info.get('name', 'Unknown')} ({connection_info.get('ip', 'No IP')})")
                
                # Guardar configuración del cliente
                clients_config[client_id] = {
                    'connection_name': connection_info.get('name', ''),
                    'connection_ip': connection_info.get('ip', ''),
                    'connection_port': connection_info.get('port', 0),
                    'dataset_name': dataset.get('dataset_name', '')
                }
                
                # Inicializar estado del cliente
                from django.utils import timezone
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
                
            print(f"📋 CLIENTS_CONFIG: {clients_config}")
            print(f"📊 CLIENTS_STATUS initialized with {len(clients_status)} clients")
            
            training_job = TrainingJob.objects.create(
                user=request.user,
                model_config=model_config,
                name=name,
                description=description,
                dataset_id=dataset_id,
                dataset_ids=selected_datasets,  # Store selected datasets
                config_json=config,             # Store full config
                total_rounds=total_rounds,      # Store total rounds
                status='pending',               # Initial status
                progress=0,                     # Initial progress
                clients_config=clients_config,  # NUEVO: Configuración de clientes
                clients_status=clients_status   # NUEVO: Estado inicial de clientes
            )
            
            # Use real Flower server in separate process
            from webapp.server_process import run_flower_server_process
            server_process = Process(target=run_flower_server_process, args=(training_job.id,))
            server_process.start()
            
            # Store process PID in training job for cleanup
            training_job.server_pid = server_process.pid
            training_job.save()
            print(f"🔧 Server process started with PID: {server_process.pid}")
            
            # Monitor process in background thread
            activate_clients_for_training(training_job, server_process)
            def monitor_server_process():
                try:
                    server_process.join()  # Wait for process to finish
                    print(f"✅ Flower server process completed")
                except Exception as e:
                    print(f"❌ Error in server process: {str(e)}")
                    # Update job status if process fails
                    training_job.refresh_from_db()
                    if training_job.status not in ['completed', 'failed']:
                        training_job.status = 'failed'
                        training_job.logs = f"Server process error: {str(e)}"
                        training_job.save()
                        # Kill the server process if it's still running
                        if server_process.is_alive():
                            print(f"🔪 Terminating server process PID: {server_process.pid}")
                            server_process.terminate()
                            server_process.join(timeout=5)
                            if server_process.is_alive():
                                print(f"🔪 Force killing server process PID: {server_process.pid}")
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
        job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
        
        # Parse metrics
        metrics = []
        if job.metrics_json:
            try:
                metrics = json.loads(job.metrics_json)
            except json.JSONDecodeError:
                metrics = []
        
        # Calculate training duration
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
            # Archive the job
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
        
        # 🔍 DEBUG: Log real data from database
        print(f"🌐 CLIENT_DASHBOARD: Job {job_id} requested")
        print(f"📋 Job status: {job.status}")
        print(f"📋 Job clients_status type: {type(job.clients_status)}")
        print(f"📋 Job clients_status: {job.clients_status}")
        print(f"📋 Job clients_config: {job.clients_config}")
        
        # 📊 Get real client data from database
        clients = []
        clients_status = job.clients_status or {}
        clients_config = job.clients_config or {}
        
        print(f"📊 Processing {len(clients_status)} clients from database")
        
        for client_id, status_data in clients_status.items():
            # Get connection info from clients_config
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
            print(f"📊 Client processed: {client_id} → {connection_name} | Status: {client_data['status']} | Acc: {client_data['accuracy']}%")
        
        print(f"📊 Total clients processed: {len(clients)}")
        
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
            # Get the maximum number of rounds from any client
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
        
        print(f"📊 PERFORMANCE_CHART_DATA: {len(performance_chart_data['labels'])} rounds of real data")
        
        context = {
            'job': job,
            'job_id': job_id,
            'clients': clients,  # Use real clients instead of dummy
            'overview_stats': overview_stats,
            'performance_chart_data': performance_chart_data,
            'total_rounds': job.total_rounds or 10
        }
        
        print(f"📊 CONTEXT SENT: {len(clients)} clients, avg_accuracy: {avg_accuracy:.1f}%")
        print(f"📊 SAMPLE CLIENT DATA: {clients[0] if clients else 'No clients'}")
        print(f"📊 CHART DATA SAMPLE: Accuracy R1: {performance_chart_data['accuracy'][0] if performance_chart_data['accuracy'] else 'No data'}")
        
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
        
        # Parse client data
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
        print(f"🔍 Waiting for server to be ready. Current status: {training_job.status}")
        timeout = 60  # 30 seconds timeout
        start_time = time.time()
        
        while training_job.status not in ['server_ready', 'failed', 'cancelled']:
            if time.time() - start_time > timeout:
                print(f"⏰ Timeout waiting for server to be ready")
                training_job.status = 'failed'
                training_job.logs = "Timeout waiting for server to start"
                training_job.save()
                return
                
            time.sleep(5)
            training_job.refresh_from_db()
            print(f"🔍 Checking status: {training_job.status}")
        
        # Check if server failed to start
        if training_job.status in ['failed', 'cancelled']:
            print(f"❌ Server failed to start. Status: {training_job.status}")
            return
        
        # Give additional time for server to fully start listening
        print(f"Server ready, waiting additional 5 seconds for full startup...")
        time.sleep(5)
        
        print(f"[SECURE] activate_clients_for_training - using new authenticated API")
        print(f"training_job: {training_job}")
        
        if not training_job.dataset_ids:
            print(f"⚠️ No dataset_ids found in training job")
            return
        
        # Parse selected datasets
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

        # Get server address for clients (automatically detects Docker vs non-Docker)
        server_ip = get_server_ip_for_clients()
        server_port = 8080  # Default port

        if 'server' in training_job.config_json:
            server_config = training_job.config_json['server']
            server_port = server_config.get('port', 8080)

        client_server_address = f"{server_ip}:{server_port}"
        print(f"🌐 [SERVER_ADDRESS] Clients will connect to: {client_server_address}")

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
                print(f"❌ CLIENT_ID not found for {conn['name']} ({conn['ip']}:{conn['port']})")
                failed_clients.append(f"{conn['name']} (No client_id)")
                continue
                
            print(f"🆔 FOUND CLIENT_ID: {conn['name']} → {client_id}")
            
            # 🔐 SECURITY: Get center-specific credentials from database
            try:
                connection_obj = Connection.objects.get(
                    ip=conn['ip'],
                    port=conn['port'], 
                    user=training_job.user
                )
                print(f"✅ [AUTH] Retrieved credentials for {connection_obj.name}")
            except Connection.DoesNotExist:
                print(f"❌ [AUTH] No credentials found for {conn['name']} ({conn['ip']}:{conn['port']})")
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
                print(f"⚠️ No datasets found for center {conn['name']}")
                failed_clients.append(f"{conn['name']} (No datasets for this center)")
                continue
            
            print(f"🎯 [FEDERATED] Center {conn['name']} will receive {len(center_datasets)} datasets (isolated)")
            
            # 🔐 SECURITY: Create center-specific config (NO cross-center data)
            center_specific_config = create_center_specific_config(center_datasets, training_job.config_json)
            
            # Build secure client configuration
            client_config = {
                "model_json": center_specific_config,
                "server_address": client_server_address,
                "dataset": center_datasets[0]['dataset_name'],  # Primary dataset for this center
                "client_id": client_id,
                "center_datasets": [ds['dataset_name'] for ds in center_datasets]  # All datasets for this center
            }
            
            print(f"📋 [SECURE] Client will connect to: {client_server_address}")
            print(f"📋 [SECURE] Sending center-specific config with {len(center_datasets)} datasets")
            print(f"📋 [SECURE] Center datasets: {[ds['dataset_name'] for ds in center_datasets]}")

            # Validate port in allowed range (5000-5099)
            if not (5000 <= int(conn['port']) <= 5099):
                print(f"❌ Port {conn['port']} not in allowed range (5000-5099) for {conn['name']}")
                failed_clients.append(f"{conn['name']} (Invalid port)")
                continue
            
            # 🚀 NEW API: Use authenticated /api/v1/start-client endpoint
            client_url = f"http://{conn['ip']}:{conn['port']}/api/v1/start-client"
            print(f"🌐 [API] Making authenticated request to: {client_url}")
            print(f"📋 [API] Headers: {auth_config.headers}")
            
            try:
                # Make authenticated request with center-specific credentials
                response = requests.post(
                    client_url,
                    json=client_config,
                    headers=auth_config.headers,
                    auth=auth_config.basic_auth,
                    timeout=10
                )
                
                print(f"📡 [API] Response status: {response.status_code}")
                print(f"📡 [API] Response headers: {dict(response.headers)}")
                
                if response.content:
                    try:
                        print(f"📄 [API] Response content: {response.text[:500]}...")
                    except:
                        print(f"📄 [API] Response content: [Could not decode]")
                
                if response.status_code == 200:
                    print(f"✅ [SUCCESS] Client {conn['name']} activated with secure API")
                    activated_clients.append(conn['name'])
                else:
                    print(f"❌ [ERROR] Failed to activate client {conn['name']}: HTTP {response.status_code}")
                    try:
                        error_detail = response.json() if response.content else response.text
                        print(f"❌ [ERROR] Response detail: {error_detail}")
                    except:
                        print(f"❌ [ERROR] Response text: {response.text[:200]}")
                    failed_clients.append(f"{conn['name']} (HTTP {response.status_code})")
                    
            except requests.exceptions.HTTPError as e:
                print(f"❌ [HTTP] HTTP error for client {conn['name']}: {str(e)}")
                failed_clients.append(f"{conn['name']} (HTTP Error: {str(e)})")
            except requests.exceptions.ConnectionError as e:
                print(f"❌ [CONN] Connection error for client {conn['name']}: {str(e)}")
                failed_clients.append(f"{conn['name']} (Connection Error)")
            except requests.exceptions.Timeout as e:
                print(f"❌ [TIMEOUT] Request timeout for client {conn['name']}: {str(e)}")
                failed_clients.append(f"{conn['name']} (Timeout)")
            except requests.exceptions.RequestException as e:
                print(f"❌ [REQUEST] Request error for client {conn['name']}: {str(e)}")
                failed_clients.append(f"{conn['name']} (Request Error: {str(e)})")
        
        # Update training job status based on client activation results
        total_clients = len(unique_connections)
        activated_count = len(activated_clients)
        failed_count = len(failed_clients)
        
        print(f"📊 [SUMMARY] Client activation: {activated_count}/{total_clients} centers activated")
        
        if activated_count == 0:
            # No clients activated - mark as failed and kill server
            training_job.status = 'failed'
            training_job.logs = f"Failed to activate any federated learning centers. Errors: {'; '.join(failed_clients)}"
            training_job.save()
            print(f"❌ Training job marked as FAILED - no centers activated")
            
            # Kill the server process since no clients can connect
            if server_process and server_process.is_alive():
                print(f"🔪 Terminating server process due to center activation failure")
                server_process.terminate()
                server_process.join(timeout=5)
                if server_process.is_alive():
                    print(f"🔪 Force killing server process")
                    server_process.kill()
            
        elif failed_count > 0:
            # Some clients failed - add warning to logs but continue
            warning_msg = f"Warning: {failed_count}/{total_clients} centers failed to activate: {'; '.join(failed_clients)}"
            training_job.logs = warning_msg
            training_job.save()
            print(f"⚠️ Federated training continuing with {activated_count} centers. {warning_msg}")
            
        else:
            # All clients activated successfully
            success_msg = f"All {activated_count} federated learning centers activated successfully: {', '.join(activated_clients)}"
            training_job.logs = success_msg
            training_job.save()
            print(f"✅ All federated learning centers activated with secure authentication")
                
    except Exception as e:
        print(f"❌ Error activating federated learning clients: {str(e)}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        # Mark training job as failed due to client activation error
        training_job.status = 'failed'
        training_job.logs = f"Federated client activation failed: {str(e)}"
        training_job.save()
        print(f"❌ Training job marked as FAILED due to client activation error")
        
        
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
    
    print(f"🔍 [AUTH] Preparing authentication for {connection.name}:")
    print(f"  - IP: {connection.ip}")
    print(f"  - Port: {connection.port}")
    print(f"  - Username: {connection.username if connection.username else 'Not set'}")
    print(f"  - Password: {'Set' if connection.password else 'Not set'}")
    print(f"  - API Key: {'Set' if connection.api_key else 'Not set'}")
    
    # API Key authentication (primary)
    if connection.api_key:
        headers['X-API-Key'] = connection.api_key
        print(f"🔑 [AUTH] Using API key authentication (X-API-Key header)")
    
    # Basic authentication (secondary/fallback)
    if connection.username and connection.password:
        auth = (connection.username, connection.password)
        print(f"👤 [AUTH] Using basic authentication with username: {connection.username}")
    
    class AuthConfig:
        def __init__(self, headers, basic_auth):
            self.headers = headers
            self.basic_auth = basic_auth
    
    return AuthConfig(headers, auth)