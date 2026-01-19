from __future__ import annotations
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .models import ModelConfig, TrainingJob, Model
import json
import logging
from .decorators import user_rate_limit
from .helpers import parameter_helpers
from .training_params import layer_types, optimizer_types, loss_types, strategy_types
from .base_views import sanitize_config_for_client


@login_required
def models_list(request):
    """
    Models management view - shows list of user's models with actions
    """
    logger = logging.getLogger(__name__)
    
    # Get user's model configurations
    model_configs = ModelConfig.objects.filter(user=request.user).order_by('-created_at')
    
    # Handle model deletion
    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'delete_model':
            model_id = request.POST.get('model_id')
            try:
                model = get_object_or_404(ModelConfig, id=model_id, user=request.user)
                model_name = model.name
                model.delete()
                messages.success(request, f'Model "{model_name}" deleted successfully.')
            except Exception as e:

                messages.error(request, f'Error deleting model: {str(e)}')
            return redirect('models_list')
    
    # Process model configs to extract useful info for display
    processed_models = []
    for model in model_configs:
        config = model.config_json or {}
        
        # Extract layer count (excluding input/output if they exist)
        # Try different possible locations for layers
        layers = []
        if 'layers' in config:
            layers = config['layers']
        elif 'model' in config and 'layers' in config['model']:
            layers = config['model']['layers']
        
        # Simple count: total layers minus input and output (first and last)
        layer_count = max(0, len(layers) - 2) if len(layers) >= 2 else 0
        
        # Extract optimizer and loss info
        optimizer_type = config.get('optimizer', {}).get('type', 'N/A')
        loss_type = config.get('loss', {}).get('type', 'N/A')
        
        processed_model = {
            'id': model.id,
            'name': model.name,
            'description': model.description,
            'framework': model.framework,
            'created_at': model.created_at,
            'updated_at': model.updated_at,
            'config_json': model.config_json,
            'layer_count': layer_count,
            'total_layers': len(layers),
            'optimizer_type': optimizer_type,
            'loss_type': loss_type,
            'layer_preview': [layer.get('type', 'Unknown') for layer in layers[:5]],
            'has_more_layers': len(layers) > 5,
            'extra_layers_count': max(0, len(layers) - 5)
        }
        processed_models.append(processed_model)
    
    try:
        models_json = json.dumps({str(model['id']): model for model in processed_models}, default=str)
    except Exception as e:
        logger.error(f"JSON serialization error: {str(e)}")
        models_json = "{}"
    
    context = {
        'model_configs': processed_models,
        'models_json': models_json,
    }
    
    return render(request, 'webapp/models_list.html', context)


@login_required
def model_designer(request):
    """
    Model designer view - allows creating and editing model configurations
    """
    # Check if we're in edit mode
    edit_model_id = request.GET.get('edit')
    edit_model = None
    edit_mode = False
    edit_model_json = {}  # Initialize as an empty dictionary

    if edit_model_id:
        try:
            edit_model = ModelConfig.objects.get(id=edit_model_id, user=request.user)
            edit_mode = True

            config_to_parse = edit_model.config_json
            if config_to_parse:
                try:
                    # Ensure we have a dictionary
                    config_dict = json.loads(config_to_parse) if isinstance(config_to_parse, str) else config_to_parse
                    edit_model_json = sanitize_config_for_client(config_dict)
                except json.JSONDecodeError:
                    edit_model_json = {"error": "Invalid configuration format."}

        except ModelConfig.DoesNotExist:
            messages.error(request, 'Model not found or access denied.')
            return redirect('models_list')

    # Get user's model configurations - ONLY DL models or legacy models without type
    model_configs = ModelConfig.objects.filter(
        user=request.user.id
    ).exclude(
        model_type='ml'  # Exclude ML models
    ).order_by('-created_at')

    models_list = [{
        'id': model.id,
        'name': model.name,
        'created_at': model.created_at.strftime('%Y-%m-%d %H:%M'),
    } for model in model_configs]

    selected_datasets = request.session.get('selected_datasets', [])
    
    context = {
        'model_configs': models_list,
        'layer_types': layer_types,
        'optimizer_types': optimizer_types,
        'loss_types': loss_types,
        'strategy_types': strategy_types,
        'previous_step_completed': True,
        'selected_datasets': selected_datasets,
        'edit_mode': edit_mode,
        'edit_model': edit_model,
        'edit_model_json': edit_model_json,
    }

    # Añadir todos los textos de ayuda al contexto
    context.update(parameter_helpers)

    return render(request, 'webapp/model_designer.html', context)


@login_required
def ml_model_designer(request, model_id=None):
    """
    View for the Machine Learning model designer.
    Handles both creation of new models and editing of existing models.
    """
    model = None
    if model_id:
        model = get_object_or_404(ModelConfig, pk=model_id, user=request.user)

    model_configs_qs = ModelConfig.objects.filter(user=request.user, model_type='ml').order_by('-updated_at')

    # Create a list of dicts for JSON serialization
    models_json_list = [
        {
            'id': config.id,
            'name': config.name,
            'description': config.description,
            'config_json': config.config_json
        } for config in model_configs_qs
    ]

    context = {
        'model_configs': model_configs_qs, # For looping in template's modal
        'models_json': json.dumps(models_json_list), # For use in javascript
        'edit_mode': model is not None,
        'edit_model': model,
        'edit_model_json': json.dumps(model.config_json) if model and model.config_json else 'null',
        'selected_datasets': request.session.get('selected_datasets', [])  # Add selected datasets
    }
    return render(request, 'webapp/ml_model_designer.html', context)


@login_required
def model_designer_advanced(request, model_id=None):
    """
    Advanced Model Designer view - for creating models with residual connections (ResNet, U-Net, etc.)
    """
    model = None
    edit_mode = False
    edit_model_json = {}

    if model_id:
        model = get_object_or_404(ModelConfig, pk=model_id, user=request.user)
        edit_mode = True
        if model.config_json:
            edit_model_json = model.config_json if isinstance(model.config_json, dict) else json.loads(model.config_json)

    # Get user's advanced model configurations
    model_configs = ModelConfig.objects.filter(
        user=request.user
    ).exclude(
        model_type='ml'
    ).order_by('-created_at')

    models_list = [{
        'id': config.id,
        'name': config.name,
        'created_at': config.created_at.strftime('%Y-%m-%d %H:%M'),
    } for config in model_configs]

    context = {
        'model_configs': models_list,
        'edit_mode': edit_mode,
        'edit_model': model,
        'edit_model_json': json.dumps(edit_model_json) if edit_model_json else 'null',
        'selected_datasets': request.session.get('selected_datasets', [])
    }

    return render(request, 'webapp/model_designer_advanced.html', context)


@login_required
def model_studio(request):
    """
    Model Studio view - unified interface for browsing, designing, and comparing models
    """
    # Get user's model configurations for the browse tab
    model_configs = ModelConfig.objects.filter(user=request.user).order_by('-created_at')
            
    # Model Comparison functionality
    # Get only completed training jobs for the current user
    available_jobs = TrainingJob.objects.filter(
        user_id=request.user.id,
        status='completed'
    ).order_by('-created_at')
    
    # Get selected job IDs from URL parameters
    selected_job_ids = request.GET.getlist('job_ids', [])
    selected_job_ids = [int(id) for id in selected_job_ids if id.isdigit()]
    
    # Limitar a màxim 2 models per comparació directa
    selected_job_ids = selected_job_ids[:2]
    
    selected_models = []
    if selected_job_ids:
        for job_id in selected_job_ids:
            try:
                # Get job ensuring it belongs to current user
                job = TrainingJob.objects.get(id=job_id, user_id=request.user.id)
                
                # Get latest metrics
                metrics = {}
                if job.metrics_json:
                    try:
                        all_metrics = json.loads(job.metrics_json)
                        # Keep the raw metrics for the table
                        metrics = all_metrics[-1] if all_metrics else {}
                                
                    except (json.JSONDecodeError, IndexError):
                        metrics = {}
                
                # Get full configuration
                config = {}
                if job.config_json:
                    try:
                        if isinstance(job.config_json, str):
                            config = json.loads(job.config_json)
                        else:
                            config = job.config_json
                    except json.JSONDecodeError:
                        config = {}
                
                # Get model architecture (from config)
                model_architecture = []
                if isinstance(config, dict) and 'model' in config:
                    model_config = config.get('model', {})
                    # Get layers if they exist
                    if isinstance(model_config, dict) and 'layers' in model_config:
                        model_architecture = model_config.get('layers', [])
                
                # Get training parameters
                training_params = {}
                if isinstance(config, dict):
                    # Get relevant training parameters
                    training_params = {
                        'optimizer': config.get('optimizer', {}),
                        'loss': config.get('loss', {}),
                        'train': config.get('train', {}),
                        'fed_config': config.get('fed_config', {})
                    }
                
                # Calculate training time
                training_time = "N/A"
                if job.completed_at and job.started_at:
                    duration = job.completed_at - job.started_at
                    minutes, seconds = divmod(duration.total_seconds(), 60)
                    training_time = f"{int(minutes)}m {int(seconds)}s"
                
                selected_models.append({
                    'id': job.id,
                    'name': job.name,
                    'model_name': job.model_config.name if hasattr(job, 'model_config') and job.model_config else 'Unknown',
                    'training_time': training_time,
                    'metrics': metrics,
                    'architecture': model_architecture,
                    'config': config,
                    'training_params': training_params
                })
            except TrainingJob.DoesNotExist:
                # Ignore IDs that don't exist or don't belong to user
                continue
    
    context = {
        'models': model_configs,
        'total_models': model_configs.count(),
        # Model comparison data
        'available_jobs': available_jobs,
        'selected_job_ids': selected_job_ids,
        'selected_models': selected_models
    }
    
    return render(request, 'webapp/model_studio.html', context)


@login_required
@user_rate_limit(rate=settings.RATE_LIMITS['GENERAL_ACTIONS'], method='POST', block=True)
def save_model_config(request):
    """
    API endpoint to save a model configuration
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Get config from unified structure
            config_json = data.get('config', {})
            
            # Extract from unified structure
            name = config_json.get('basic_info', {}).get('name', '')
            description = config_json.get('basic_info', {}).get('description', '')
            model_type = config_json.get('metadata', {}).get('model_type', 'dl')
            framework = config_json.get('metadata', {}).get('framework', 'pytorch')
            
            if not name:
                return JsonResponse({'success': False, 'error': 'Model name is required'})

            # Check if updating existing model
            model_id = data.get('id')
            if model_id:
                # Update existing model
                try:
                    model_config = ModelConfig.objects.get(id=model_id, user=request.user)
                    model_config.name = name
                    model_config.description = description
                    model_config.config_json = config_json
                    # Use extracted framework and model_type
                    model_config.framework = framework
                    model_config.model_type = model_type
                    model_config.save()
                    created = False
                except ModelConfig.DoesNotExist:
                    return JsonResponse({'success': False, 'error': 'Model not found or access denied'})
            else:
                # Create new model config
                model_config, created = ModelConfig.objects.update_or_create(
                    user=request.user,
                    name=name,
                    defaults={
                        'description': description,
                        'config_json': config_json,
                        # Use extracted framework and model_type
                        'framework': framework,
                        'model_type': model_type
                    }
                )
            
            return JsonResponse({
                'success': True, 
                'model_id': model_config.id,
                'message': 'Model configuration saved successfully!'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid JSON data'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


@login_required
def get_model_config(request, model_id):
    config = get_object_or_404(ModelConfig, pk=model_id)
    # Check ownership
    if config.user != request.user:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    config_data = {
        'id': config.id,
        'name': config.name,
        'description': config.description,
        'framework': config.framework,
        'model_type': config.model_type,
        'config_json': config.config_json, # Assuming it's JSON serializable
        'created_at': config.created_at,
        'updated_at': config.updated_at,
    }
    return JsonResponse(config_data)


@login_required
def delete_model_config(request, model_id):
    config = get_object_or_404(ModelConfig, pk=model_id)
    # Check ownership
    if config.user != request.user:
        return JsonResponse({'error': 'Permission denied'}, status=403)

    if request.method in ['DELETE', 'POST']:
        config.delete()
        return JsonResponse({'success': True, 'message': 'Model deleted successfully'})
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)


@login_required
def save_model(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            name = data.get('name', 'Untitled Model')
            config = data.get('config', '{}')
            
            # Guardar el model a la base de dades
            model = Model.objects.create(
                user=request.user,
                name=name,
                config=config
            )
            
            return JsonResponse({
                'success': True,
                'model_id': model.id
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Mètode no permès'
    })


@login_required
def go_to_training(request, model_id):
    try:
        model_config = get_object_or_404(ModelConfig, id=model_id, user=request.user)
        # Aquí guardas claramente el model_id en sesión
        request.session['model_id'] = model_id
        messages.success(request, f'Model "{model_config.name}" carregat correctament')
        return redirect('training')  # Limpio, sin parámetros GET
    except:
        messages.error(request, 'The specified model does not exist or you do not have permission to access it')
        return redirect('model_studio')


@login_required
def get_model_configs(request):
    try:
        # Get filter parameter for model type compatibility
        model_type_filter = request.GET.get('type', 'dl')  # Default to DL for backwards compatibility
        
        if model_type_filter == 'ml':
            # Only ML models
            models = ModelConfig.objects.filter(user=request.user, model_type='ml').order_by('-created_at')
        elif model_type_filter == 'dl':
            # Only DL models (exclude ML models)
            models = ModelConfig.objects.filter(user=request.user).exclude(model_type='ml').order_by('-created_at')
        else:
            # All models (fallback)
            models = ModelConfig.objects.filter(user=request.user).order_by('-created_at')
        
        models_list = [{
            'id': model.id,
            'name': model.name,
            'created_at': model.created_at.strftime('%Y-%m-%d %H:%M'),
        } for model in models]
        
        return JsonResponse({
            'success': True,
            'models': models_list
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })