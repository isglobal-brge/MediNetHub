from __future__ import annotations
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .models import Connection, Dataset, Project
from .forms import ConnectionForm
import json
import re
import ipaddress
import time
from django.utils import timezone
import requests
from django.views.decorators.http import require_http_methods
from .decorators import user_rate_limit
from .base_views import sanitize_config_for_client, create_center_specific_config, prepare_center_authentication


@login_required
def datasets(request):
    """
    Manage connections and view discovered datasets.
    Connections are now stored in the database.
    Discovered datasets are temporarily stored in session.
    """
    # Get user's projects 
    projects = Project.objects.filter(user=request.user).order_by('name')
    
    # Get selected project from session
    selected_project_id = request.session.get('selected_project_id')
    selected_project = None
    
    if selected_project_id:
        try:
            selected_project = Project.objects.get(id=selected_project_id, user=request.user)
        except Project.DoesNotExist:
            # Clear invalid project from session
            del request.session['selected_project_id']
            selected_project_id = None
    
    
    # Fetch connections belonging to the current user and selected project
    if selected_project:
        connections = Connection.objects.filter(user=request.user, project=selected_project).order_by('name')
    else:
        # Si no hay proyecto seleccionado, muestra conexiones sin proyecto
        connections = Connection.objects.filter(user=request.user, project__isnull=True).order_by('name')
    selected_datasets = request.session.get('selected_datasets', [])
    
    # Use forms for adding/editing connections
    connection_form = ConnectionForm()
    edit_connection_form = None
    connection_to_edit_id = request.GET.get('edit_id')

    if connection_to_edit_id:
        try:
            connection_to_edit = get_object_or_404(Connection, pk=int(connection_to_edit_id), user=request.user)
            edit_connection_form = ConnectionForm(instance=connection_to_edit)
        except ValueError:
             messages.error(request, "Invalid connection ID for editing.")

    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'create_project':
            name = request.POST.get('project_name', '').strip()
            description = request.POST.get('project_description', '').strip()
            color = request.POST.get('project_color', '#1976d2')
            
            if name and not Project.objects.filter(user=request.user, name=name).exists():
                project = Project.objects.create(
                    user=request.user, 
                    name=name, 
                    description=description if description else None,
                    color=color
                )
                # Auto-select the new project
                request.session['selected_project_id'] = project.id
                
                messages.success(request, f'Project "{name}" created successfully')
            else:
                messages.error(request, 'Invalid project name or project already exists')
            return redirect('datasets')
        
        elif action == 'add_connection':
            form = ConnectionForm(request.POST)
            if form.is_valid():
                try:
                    connection = form.save(commit=False)
                    connection.user = request.user
                    connection.project = selected_project
                    
                    # Assign password from form to encrypted field
                    raw_password = form.cleaned_data.get('password')
                    if raw_password:
                        connection.password = raw_password
                    
                    # Assign API key from form to encrypted field
                    api_key = form.cleaned_data.get('api_key')  # Fixed: use 'api_key' not 'api-key'
                    if api_key:
                        connection.api_key = api_key
                    
                    connection.save()
                    messages.success(request, f'Connection "{connection.name}" added successfully.')
                except Exception as e:
                    messages.error(request, f'Error saving connection: {str(e)}')
            else:
                 for field, errors in form.errors.items():
                     for error in errors:
                         messages.error(request, f"{form.fields[field].label}: {error}")
            return redirect('datasets')

        elif action == 'edit_connection':
            connection_id = request.POST.get('connection_id')
            connection = get_object_or_404(Connection, id=connection_id, user=request.user)
            form = ConnectionForm(request.POST, instance=connection)
            if form.is_valid():
                try:
                    connection = form.save(commit=False)
                    
                    # Handle password correctly for editing
                    raw_password = form.cleaned_data.get('password')
                    if raw_password:
                        # Direct assignment to the encrypted field - django-fernet-fields handles encryption
                        connection.password = raw_password
                            
                    connection.save()
                    messages.success(request, f'Connection "{connection.name}" updated successfully.')
                except Exception as e:
                    messages.error(request, f'Error updating connection: {e}')
            else:
                for field, errors in form.errors.items():
                         for error in errors:
                             messages.error(request, f"{form.fields[field].label}: {error}")
            return redirect('datasets')

        elif action == 'delete_connection':
            connection_id = int(request.POST.get('connection_id'))
            try:
                connection = get_object_or_404(Connection, pk=connection_id, user=request.user)
                connection_name = connection.name
                connection_ip = connection.ip
                connection_port = connection.port

                # Eliminar datasets associats de la sessió
                selected_datasets = request.session.get('selected_datasets', [])
                updated_datasets = [
                    ds for ds in selected_datasets
                    if not (
                        ds['connection']['ip'] == connection_ip and
                        ds['connection']['port'] == connection_port
                    )
                ]
                
                # Actualitzar la sessió amb els datasets filtrats
                request.session['selected_datasets'] = updated_datasets
                request.session.modified = True
                
                # Eliminar la connexió
                connection.delete()
                
                # Eliminar datasets de la sessió de datasets disponibles
                if 'datasets' in request.session:
                    request.session['datasets'].pop(str(connection_id), None)
                    request.session.modified = True
                
                messages.success(request, f'Connection "{connection_name}" and its associated datasets have been deleted successfully.')
            except Connection.DoesNotExist:
                messages.error(request, "Connection not found or permission denied.")
            except Exception as e:
                messages.error(request, f"Error deleting connection: {str(e)}")
            return redirect('datasets')

        elif action == 'fetch_datasets':
            connection_id = int(request.POST.get('connection_id'))
            try:
                connection = get_object_or_404(Connection, pk=connection_id, user=request.user)
                # Validar puerto en rango permitido (5000-5099)
                if not (5000 <= connection.port <= 5099):
                    messages.error(request, f'Port {connection.port} is not allowed. Use ports 5000-5099')
                    return redirect('datasets')
                
                url_scheme = "http" 
                fetch_url = f"{url_scheme}://{connection.ip}:{connection.port}/api/v1/get-data-info"
                
                # Prepare authentication headers matching test_api_researcher.py format
                headers = {
                    'Content-Type': 'application/json',
                    'User-Agent': 'MediNet-WebApp/1.0'
                }
                auth = None
             
                
                if connection.api_key:
                    headers['X-API-Key'] = connection.api_key
                
                if connection.username and connection.password:
                    auth = (connection.username, connection.password)
   
                
                try:
                    response = requests.get(fetch_url, headers=headers, auth=auth, timeout=10)
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    
                    data = response.json()     
                    if 'datasets' not in request.session:
                        request.session['datasets'] = {}
                    if not isinstance(data, dict):
                        raise ValueError(f"Expected dictionary, got {type(data)}")
                    
                    # Check for required fields (adapted to your API structure)
                    required_fields = ['dataset_id', 'dataset_name', 'patient_count', 'file_size', 'num_columns']
                    missing_required = [field for field in required_fields if field not in data]
                    if missing_required:
                        raise ValueError(f"Missing required fields: {', '.join(missing_required)}")
                    
                    # Validate that all fields are lists
                    non_list_fields = [field for field in data.keys() if not isinstance(data[field], list)]
                    if non_list_fields:
                        raise ValueError(f"Expected lists for fields: {', '.join(non_list_fields)}")
                    
                    # Validate list lengths are consistent
                    lengths = {field: len(data[field]) for field in data.keys()}
                    unique_lengths = set(lengths.values())
                    if len(unique_lengths) > 1:
                        raise ValueError(f"Inconsistent field lengths: {lengths}")
                    
                    # Check if we have at least one dataset
                    dataset_count = len(data['dataset_name'])
                    if dataset_count == 0:
                        messages.info(request, f'No datasets available from "{connection.name}".')
                    else:
                        request.session['datasets'][str(connection_id)] = data
                        request.session.modified = True
                        messages.success(request, f'Successfully synchronized {dataset_count} dataset(s) from "{connection.name}".')
                        
                except ValueError as e:
                    messages.error(request, f'Data validation error from "{connection.name}": {str(e)}')
                    
                except requests.exceptions.HTTPError as e:
                    error_msg = f'HTTP Error {response.status_code}'
                    try:
                        error_data = response.json()
                        error_msg += f': {error_data}'
                    except:
                        error_msg += f': {response.text[:200]}'
                    
                    messages.error(request, f'HTTP Error communicating with "{connection.name}": {error_msg}')
                    
                except requests.exceptions.ConnectionError as e:
                    error_msg = f'Connection failed - check if server is running on {connection.ip}:{connection.port}'
                    messages.error(request, f'{error_msg}')
                    
                except requests.exceptions.Timeout as e:
                    error_msg = f'Request timeout after 10 seconds'
                    messages.error(request, f'Timeout communicating with "{connection.name}": {error_msg}')
                    
                except requests.exceptions.RequestException as e:
                    messages.error(request, f'Error communicating with "{connection.name}": {str(e)}')
                         
            except Connection.DoesNotExist:
                messages.error(request, "Connection not found or permission denied.")
            return redirect('datasets')
            
        elif action == 'clear_session_data':
            # 🧹 Clear corrupted session data
            try:
                if 'datasets' in request.session:
                    corrupted_count = len(request.session['datasets'])
                    request.session['datasets'] = {}
                    request.session.modified = True
                    print(f"🧹 [DEBUG] Cleared {corrupted_count} dataset entries from session")
                    messages.success(request, f'Cleared session data. Re-fetch datasets from connections to reload.')
                else:
                    messages.info(request, 'No session data to clear.')
            except Exception as e:
                messages.error(request, f'Error clearing session data: {str(e)}')
            return redirect('datasets')

    datasets = []
    session_datasets = request.session.get('datasets', {})
    if selected_project:
        active_connections = Connection.objects.filter(user=request.user, active=True, project=selected_project)
    else:
        active_connections = Connection.objects.filter(user=request.user, active=True, project__isnull=True)
    connection_map = {str(c.id): c for c in active_connections}

    for conn_id_str, conn_data in session_datasets.items():
         if conn_id_str in connection_map:
            connection_obj = connection_map[conn_id_str]
            
            try:
                # Required fields for datasets (adapted to your API structure)
                required_fields = ['dataset_id', 'dataset_name', 'patient_count', 'file_size', 'num_columns']
                
                # Check if basic required fields exist and are lists
                missing_fields = []
                for field in required_fields:
                    if field not in conn_data:
                        missing_fields.append(field)
                    elif not isinstance(conn_data[field], list):
                        missing_fields.append(f"{field} (not a list)")
                
                if missing_fields:
                    # Remove corrupted data from session
                    if 'datasets' in request.session and conn_id_str in request.session['datasets']:
                        del request.session['datasets'][conn_id_str]
                        request.session.modified = True
                    continue
                
                # Get lengths and validate consistency
                lengths = {field: len(conn_data[field]) for field in required_fields}
                
                if len(set(lengths.values())) > 1:
                    raise ValueError(f"Inconsistent array lengths: {lengths}")
                
                num_datasets = lengths['dataset_name']
                if num_datasets == 0:
                    continue

                # Process each dataset with robust error handling
                for i in range(num_datasets):
                    try:
                        
                        # Safely get fields from your API structure
                        dataset_id = conn_data['dataset_id'][i] if i < len(conn_data['dataset_id']) else i
                        dataset_name = conn_data['dataset_name'][i] if i < len(conn_data['dataset_name']) else f"unknown_dataset_{i}"
                        
                        num_columns = conn_data['num_columns'][i] if i < len(conn_data['num_columns']) else 0
                        patient_count = conn_data['patient_count'][i] if i < len(conn_data['patient_count']) else 0
                        file_size = conn_data['file_size'][i] if i < len(conn_data['file_size']) else 0
                        
                        # Optional fields with safe defaults
                        medical_domain = conn_data.get('medical_domain', [None] * num_datasets)[i] if i < len(conn_data.get('medical_domain', [])) else 'Unknown'
                        data_type = conn_data.get('data_type', [None] * num_datasets)[i] if i < len(conn_data.get('data_type', [])) else 'Tabular Data'
                        description = conn_data.get('description', [None] * num_datasets)[i] if i < len(conn_data.get('description', [])) else ''
                        target_column = conn_data.get('target_column', [None] * num_datasets)[i] if i < len(conn_data.get('target_column', [])) else 'unknown'
                        
                        # Parse rich metadata if available
                        metadata = {}
                        if 'metadata' in conn_data and i < len(conn_data['metadata']) and conn_data['metadata'][i]:
                            try:
                                if isinstance(conn_data['metadata'][i], str):
                                    metadata = json.loads(conn_data['metadata'][i])
                                elif isinstance(conn_data['metadata'][i], dict):
                                    metadata = conn_data['metadata'][i]
                            except (json.JSONDecodeError, TypeError) as e:
                              messages.error(request,"⚠️ [DEBUG] Failed to parse metadata for {dataset_name}: {e}")
                        
                        # Extract target_info from metadata or create default structure
                        target_info = {
                            'name': target_column,
                            'type': 'unknown',
                            'num_classes': 0
                        }

                        # Extract target_info from metadata if available
                        if metadata and 'statistical_summary' in metadata and 'target_info' in metadata['statistical_summary']:
                            target_metadata = metadata['statistical_summary']['target_info']
                            target_info = {
                                'name': target_metadata.get('column_name', target_column),
                                'type': target_metadata.get('task_type', 'unknown'),
                                'task_subtype': target_metadata.get('task_subtype', 'unknown'),
                                'data_type': target_metadata.get('data_type', 'unknown'),
                                'num_classes': target_metadata.get('num_classes', 0),
                                'classes': target_metadata.get('classes', []),
                                'output_neurons': target_metadata.get('output_neurons', 0),
                                'recommended_activation': target_metadata.get('recommended_activation', 'softmax'),
                                'recommended_loss': target_metadata.get('recommended_loss', 'CrossEntropyLoss')
                            }

                        print(f"📊 [DEBUG] Dataset: {dataset_name}, Target Info: {target_info}")
                        # Create features_info from metadata if available
                        features_info = {'input_features': num_columns - 1, 'feature_types': {'numeric': num_columns - 1, 'categorical': 0}}
                        if metadata and 'statistical_summary' in metadata and 'column_types' in metadata['statistical_summary']:
                            column_types = metadata['statistical_summary']['column_types']
                            numeric_count = sum(1 for t in column_types.values() if t == 'numeric')
                            categorical_count = len(column_types) - numeric_count
                            features_info = {
                                'input_features': len(column_types) - (1 if target_column in column_types else 0),
                                'feature_types': {'numeric': numeric_count, 'categorical': categorical_count}
                            }
                        
                        # Generate a safe ID by sanitizing the dataset name
                        safe_dataset_name = dataset_name.replace(' ', '-').replace('_', '-').lower()
                        # Remove any characters that aren't alphanumeric or hyphens
                        import re
                        safe_dataset_name = re.sub(r'[^a-z0-9-]', '', safe_dataset_name)
                        dataset_info = {
                            'id': f"ds-{conn_id_str}-{safe_dataset_name}",
                            'connection': connection_obj,
                            'dataset_id': dataset_id,
                            'dataset_name': dataset_name,
                            'medical_domain': medical_domain,
                            'data_type': data_type,
                            'description': description,
                            'num_columns': num_columns,
                            'num_rows': patient_count,  # Using patient_count as num_rows for compatibility
                            'patient_count': patient_count,
                            'size': file_size,
                            'file_size': file_size,
                            'target_column': target_column,
                            'metadata': metadata,
                            'features_info': features_info,
                            'target_info': target_info
                        }
                        
                        # Check if selected
                        dataset_info['is_selected'] = any(
                            sd['connection']['ip'] == connection_obj.ip and 
                            sd['dataset_name'] == dataset_info['dataset_name']
                            for sd in selected_datasets
                        )

                        datasets.append(dataset_info)
                        
                    except Exception as e:
                        continue  # Skip this dataset but continue with others
                        
            except Exception as e:
                # Remove corrupted data from session
                if 'datasets' in request.session and conn_id_str in request.session['datasets']:
                    del request.session['datasets'][conn_id_str]
                    request.session.modified = True
                    messages.warning(request, f"Data format error from connection {connection_obj.name}. Corrupted data has been cleaned. Please re-sync the connection.")
                else:
                    messages.warning(request, f"Data format error from connection {connection_obj.name}: {str(e)}")
                continue 

    context = {
        'connections': connections,
        'datasets': datasets,
        'connection_form': connection_form,
        'edit_connection_form': edit_connection_form,
        'connection_to_edit_id': connection_to_edit_id,
        'has_selected_datasets': len(selected_datasets) > 0,
        'projects': projects,
        'selected_project': selected_project,
        'project_colors': Project.COLOR_CHOICES,
    }

    return render(request, 'webapp/datasets.html', context)


def validate_connection(request):
    """
    API endpoint to validate connection details
    """
    ip = request.GET.get('ip', '')
    port = request.GET.get('port', '')
    
    # Validate IP
    valid_ip = False
    try:
        ipaddress.ip_address(ip)
        valid_ip = True
    except ValueError:
        valid_ip = False
    
    # Validate port
    valid_port = False
    try:
        port_num = int(port)
        valid_port = 1 <= port_num <= 65535
    except ValueError:
        valid_port = False
    
    return JsonResponse({
        'valid_ip': valid_ip,
        'valid_port': valid_port,
        'valid': valid_ip and valid_port
    })


@login_required
@user_rate_limit(rate=settings.RATE_LIMITS['CONNECTION_TEST'], method='POST', block=True)
def test_connection(request):
    """
    API endpoint to test connection to a server
    """
    ip = request.GET.get('ip', '')
    port = request.GET.get('port', '')
    
    # This will be implemented to actually test the connection
    # For now, we just validate the format
    
    # Validate IP
    valid_ip = False
    try:
        ipaddress.ip_address(ip)
        valid_ip = True
    except ValueError:
        valid_ip = False
    
    # Validate port
    valid_port = False
    try:
        port_num = int(port)
        valid_port = 1 <= port_num <= 65535
    except ValueError:
        valid_port = False
    
    success = valid_ip and valid_port
    
    return JsonResponse({
        'success': success,
        'message': 'Connection successful!' if success else 'Invalid IP or port'
    })



@login_required
def store_selected_datasets(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            if 'selected_datasets' not in request.session:
                request.session['selected_datasets'] = []
            
            current_datasets = request.session['selected_datasets']
            new_dataset = data['dataset']
            
            # Validar que el dataset té l'estructura correcta
            if not isinstance(new_dataset, dict):
                return JsonResponse({'success': False, 'error': 'Invalid dataset format: must be a dictionary'})
            
            required_fields = ['dataset_id', 'dataset_name', 'features_info', 'target_info', 'num_columns', 'num_rows', 'size', 'connection']
            missing_fields = [field for field in required_fields if field not in new_dataset]
            if missing_fields:
                return JsonResponse({
                    'success': False, 
                    'error': f'Invalid dataset format: missing fields {", ".join(missing_fields)}'
                })
            
            # Comprovar si ja existeix la combinació dataset+connexió
            exists = any(
                ds['dataset_name'] == new_dataset['dataset_name'] and 
                ds['connection']['ip'] == new_dataset['connection']['ip'] and
                ds['connection']['port'] == new_dataset['connection']['port']
                for ds in current_datasets
            )
            
            if not exists:
                # Crear una còpia neta del dataset amb només els camps necessaris
                clean_dataset = {
                    'dataset_id': new_dataset['dataset_id'],
                    'dataset_name': new_dataset['dataset_name'],
                    'features_info': new_dataset['features_info'],
                    'target_info': new_dataset['target_info'],
                    'num_columns': new_dataset['num_columns'],
                    'num_rows': new_dataset['num_rows'],
                    'size': new_dataset['size'],
                    'connection': {
                        'name': new_dataset['connection']['name'],
                        'ip': new_dataset['connection']['ip'],
                        'port': new_dataset['connection']['port']
                    }
                }
                
                current_datasets.append(clean_dataset)
                request.session['selected_datasets'] = current_datasets
                request.session.modified = True
                
                return JsonResponse({
                    'success': True,
                    'message': f'Dataset {clean_dataset["dataset_name"]} from {clean_dataset["connection"]["name"]} added successfully'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': f'Dataset {new_dataset["dataset_name"]} from this connection is already selected'
                })
                
        except KeyError as e:
            return JsonResponse({
                'success': False,
                'error': f'Missing required field: {str(e)}'
            })
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    })


@login_required
def check_dataset_status(request):
    try:
        data = json.loads(request.body)
        dataset_info = data.get('dataset', {})
        
        # Get selected datasets from session
        selected_datasets = request.session.get('selected_datasets', [])
        
        # Check if dataset is already selected
        is_selected = any(
            ds['dataset_name'] == dataset_info['dataset_name'] and
            ds['connection']['name'] == dataset_info['connection']['name'] and
            ds['connection']['ip'] == dataset_info['connection']['ip'] and
            ds['connection']['port'] == dataset_info['connection']['port']
            for ds in selected_datasets
        )
        
        return JsonResponse({
            'success': True,
            'is_selected': is_selected
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


@require_http_methods(["POST"])
@user_rate_limit(rate=settings.RATE_LIMITS['MANAGE_DATASETS'], method='POST', block=True)
def remove_selected_dataset(request):
    try:
        data = json.loads(request.body)
        dataset_info = data.get('dataset', {})
        
        # Get current selected datasets
        selected_datasets = request.session.get('selected_datasets', [])
        
        # Remove the dataset if it exists using all comparison fields
        selected_datasets = [
            ds for ds in selected_datasets
            if not (
                ds['dataset_name'] == dataset_info['dataset_name'] and
                ds['connection']['name'] == dataset_info['connection']['name'] and
                ds['connection']['ip'] == dataset_info['connection']['ip'] and
                ds['connection']['port'] == dataset_info['connection']['port']
            )
        ]
        
        # Update session
        request.session['selected_datasets'] = selected_datasets
        request.session.modified = True
        
        return JsonResponse({
            'success': True,
            'message': 'Dataset removed successfully'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


@login_required
def dataset_detail_view(request, dataset_id):
    """
    Vista polimórfica para mostrar detalles detallados de un dataset desde session data
    dataset_id format: "ds-connection_id-dataset_name" (e.g., "ds-123-hr-failure")
    """

    try:
        # Parse new dataset_id format: ds-connection_id-dataset_name
        if not dataset_id.startswith('ds-'):
            messages.error(request, "Invalid dataset identifier format")
            return redirect('datasets')
        
        # Remove 'ds-' prefix and split by '-'
        id_without_prefix = dataset_id[3:]  # Remove 'ds-'
        parts = id_without_prefix.split('-', 1)  # Split on first dash
        
        if len(parts) != 2:
            messages.error(request, "Invalid dataset identifier format")
            return redirect('datasets')
        
        conn_id_str, safe_dataset_name = parts

        session_datasets = request.session.get('datasets', {})
        
        # Verify connection_id exists in session
        if conn_id_str not in session_datasets:
            messages.error(request, "Dataset not found or session expired. Please refresh datasets.")
            return redirect('datasets')
        
        # Get connection object and verify user access
        try:
            connection_id = int(conn_id_str)
            connection = get_object_or_404(Connection, id=connection_id, user=request.user)
        except (ValueError, Connection.DoesNotExist):
            messages.error(request, "Connection not found or access denied")
            return redirect('datasets')
        
        # Find dataset in session data by searching all datasets for matching safe name
        conn_data = session_datasets[conn_id_str]
        dataset_index = None
        actual_dataset_name = None
        
        try:
            dataset_names = conn_data.get('dataset_name', [])
            
            # Find the dataset by converting each name to safe format and comparing
            import re
            for i, name in enumerate(dataset_names):
                # Apply same sanitization as in datasets view
                safe_name = name.replace(' ', '-').replace('_', '-').lower()
                safe_name = re.sub(r'[^a-z0-9-]', '', safe_name)
                
                
                if safe_name == safe_dataset_name:
                    dataset_index = i
                    actual_dataset_name = name
                    break
            if dataset_index is None:
                messages.error(request, f"Dataset with identifier '{safe_dataset_name}' not found in connection data")
                return redirect('datasets')
                
        except (KeyError, ValueError) as e:
            messages.error(request, "Invalid dataset data structure")
            return redirect('datasets')
        
        # Build dataset object from session data (adapted for new API structure)
        try:

            # Extract basic data using the new API structure
            dataset_id_val = conn_data.get('dataset_id', [None])[dataset_index] if isinstance(conn_data.get('dataset_id', []), list) else conn_data.get('dataset_id')
            medical_domain = conn_data.get('medical_domain', [None])[dataset_index] if isinstance(conn_data.get('medical_domain', []), list) else conn_data.get('medical_domain', 'General')
            data_type = conn_data.get('data_type', ['Tabular Data'])[dataset_index] if isinstance(conn_data.get('data_type', []), list) else conn_data.get('data_type', 'Tabular Data')
            description = conn_data.get('description', [None])[dataset_index] if isinstance(conn_data.get('description', []), list) else conn_data.get('description', '')
            target_column = conn_data.get('target_column', [None])[dataset_index] if isinstance(conn_data.get('target_column', []), list) else conn_data.get('target_column', 'unknown')
            num_columns = conn_data.get('num_columns', [0])[dataset_index] if isinstance(conn_data.get('num_columns', []), list) else conn_data.get('num_columns', 0)
            patient_count = conn_data.get('patient_count', [0])[dataset_index] if isinstance(conn_data.get('patient_count', []), list) else conn_data.get('patient_count', 0)
            file_size = conn_data.get('file_size', [0])[dataset_index] if isinstance(conn_data.get('file_size', []), list) else conn_data.get('file_size', 0)
            metadata_raw = conn_data.get('metadata', [{}])[dataset_index] if isinstance(conn_data.get('metadata', []), list) else conn_data.get('metadata', {})

            # Parse metadata if it's a JSON string
            metadata = {}
            if isinstance(metadata_raw, str):
                try:
                    metadata = json.loads(metadata_raw)
                except json.JSONDecodeError:
                    metadata = {}
            elif isinstance(metadata_raw, dict):
                metadata = metadata_raw

            # Extract rich metadata fields from statistical_summary
            statistical_summary = metadata.get('statistical_summary', {})

            # Extract column types
            column_types = statistical_summary.get('column_types', {})
            # Extract statistical summary for features (nested dict)
            features_statistics = statistical_summary.get('statistical_summary', {})
            # Extract target_info with complete information
            target_info_raw = statistical_summary.get('target_info', {})
            target_info = {
                'column_name': target_info_raw.get('column_name', target_column),
                'data_type': target_info_raw.get('data_type', 'unknown'),
                'task_type': target_info_raw.get('task_type', 'unknown'),
                'task_subtype': target_info_raw.get('task_subtype', 'unknown'),
                'num_classes': target_info_raw.get('num_classes', 0),
                'classes': target_info_raw.get('classes', []),
                'output_neurons': target_info_raw.get('output_neurons', 0),
                'recommended_activation': target_info_raw.get('recommended_activation', 'softmax'),
                'recommended_loss': target_info_raw.get('recommended_loss', 'CrossEntropyLoss')
            }

            # Extract features info
            numeric_features = sum(1 for dtype in column_types.values() if dtype == 'numeric')
            categorical_features = sum(1 for dtype in column_types.values() if dtype == 'categorical')
            total_features = len(column_types) - (1 if target_column in column_types else 0)

            features_info = {
                'input_features': total_features,
                'numeric_count': numeric_features,
                'categorical_count': categorical_features,
                'feature_types': column_types
            }

            # Extract quality metrics
            quality_score = metadata.get('quality_score', None)
            completeness_percentage = metadata.get('completeness_percentage', None)

            # Extract missing values info
            missing_values = metadata.get('missing_values', {})

            # Extract data distribution
            data_distribution = metadata.get('data_distribution', {})

            # Extract timestamps
            generated_at = metadata.get('generated_at', None)
            updated_at = metadata.get('updated_at', None)

            print(f"Extracted target_info: {target_info}")
            print(f"Quality Score: {quality_score}, Completeness: {completeness_percentage}%")

            dataset_obj = {
                'id': dataset_id,
                'dataset_id': dataset_id_val,
                'dataset_name': actual_dataset_name,
                'connection': connection,
                'medical_domain': medical_domain,
                'data_type': data_type,
                'description': description,
                'target_column': target_column,
                'num_columns': num_columns,
                'patient_count': patient_count,
                'file_size': file_size,
                'metadata': metadata,
                # New rich fields
                'column_types': column_types,
                'target_info': target_info,
                'features_info': features_info,
                'features_statistics': features_statistics,  # Statistical summary for each feature
                'quality_score': quality_score,
                'completeness_percentage': completeness_percentage,
                'missing_values': missing_values,
                'data_distribution': data_distribution,
                'generated_at': generated_at,
                'updated_at': updated_at
            }
        except Exception as e:
            print(f"Error processing dataset data: {str(e)}")
            import traceback
            print(traceback.format_exc())
            messages.error(request, f"Error processing dataset data: {str(e)}")
            return redirect('datasets')
        context = {
            'dataset': dataset_obj
        }
        return render(request, 'webapp/dataset_details.html', context)
        
    except Exception as e:
        messages.error(request, f"Error loading dataset details: {str(e)}")
        return redirect('datasets')