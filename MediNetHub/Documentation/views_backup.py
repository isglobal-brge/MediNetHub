from __future__ import annotations
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, authenticate, logout, update_session_auth_hash
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from .models import UserProfile, ModelConfig, TrainingJob, Connection, Dataset, Model, Notification, Project
from .forms import (
    UserProfileForm,
    ConnectionForm,
    UserUpdateForm,
    ProfileUpdateForm,
    CustomPasswordChangeForm,
)
import json
import re
import ipaddress
import threading
import random
import time
from django.utils import timezone
import requests
# from .server_fn.server import start_flower_server  # Moved inside run_flower_server_process
from django.views.decorators.http import require_http_methods
from multiprocessing import Process
from .decorators import ip_rate_limit, user_rate_limit
from datetime import datetime, timedelta
from .helpers import parameter_helpers
from .training_params import layer_types, optimizer_types, loss_types, strategy_types

def sanitize_config_for_client(config):
    """
    Remove sensitive information (connection details) from config before sending to client.
    The client doesn't need DB connection info since it only receives processed data.
    """
    import copy
    sanitized = copy.deepcopy(config)
    
    # Remove connection information from datasets
    if 'dataset' in sanitized and 'selected_datasets' in sanitized['dataset']:
        for dataset in sanitized['dataset']['selected_datasets']:
            if 'connection' in dataset:
                # Keep only non-sensitive connection info if needed
                dataset['connection'] = {
                    'name': dataset['connection'].get('name', 'unknown')
                    # Removed: ip, port, user, password
                }
                print(f"🔒 Removed sensitive connection info for dataset: {dataset.get('dataset_name', 'unknown')}")
    
    return sanitized

def create_center_specific_config(center_datasets, base_config):
    """
    Create configuration containing ONLY data for specific center.
    Critical for federated learning security - prevents credential/data leakage between centers.
    """
    import copy
    
    print(f"🎯 Creating center-specific config for {len(center_datasets)} datasets")
    
    center_config = copy.deepcopy(base_config)
    
    # Include only datasets from this specific center (NO other center data)
    center_config['dataset'] = {
        'selected_datasets': [{
            'dataset_name': ds['dataset_name'],
            'features_info': ds['features_info'],
            'target_info': ds['target_info'],
            'num_columns': ds.get('num_columns', 0),
            'num_rows': ds.get('num_rows', 0),
            'size': ds.get('size', 0),
            # ✅ SECURITY: NO connection info included to prevent credential leakage
        } for ds in center_datasets]
    }
    
    # Log security compliance
    for ds in center_datasets:
        print(f"🔐 [FEDERATED] Including dataset '{ds['dataset_name']}' for this center only")
    
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
    
    print(f"🔍 [AUTH] Preparing authentication for {connection.name}:")
    print(f"  - IP: {connection.ip}")
    print(f"  - Port: {connection.port}")
    print(f"  - Username: {connection.username if connection.username else 'Not set'}")
    print(f"  - Password: {'Set' if connection.password else 'Not set'}")
    print(f"  - API Key: {'Set' if connection.api_key else 'Not set'}")
    
    # API Key authentication (primary)
    if connection.api_key:
        headers['X-API-Key'] = connection.api_key
        headers['X-Client-IP'] = '127.0.0.1'  # Using localhost as client IP
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

def home(request):
    """
    Home page view
    """
    return render(request, 'webapp/home.html')

@ip_rate_limit(rate=settings.RATE_LIMITS['REGISTER'], method='POST', block=True)
def register(request):
    """
    User registration view
    """
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            try:
                user = form.save()
                # UserProfile is automatically created by the signal, no need to create it manually
                
                # Set additional user fields
                user.first_name = form.cleaned_data.get('first_name', '')
                user.last_name = form.cleaned_data.get('last_name', '')
                user.email = form.cleaned_data.get('email', '')
                user.save()
                
                # Log the user in
                username = form.cleaned_data.get('username')
                password = form.cleaned_data.get('password1')
                user = authenticate(username=username, password=password)
                login(request, user)
                messages.success(request, f'Account created for {username}!')
                return redirect('user_dashboard')
            except Exception as e:
                print(f"Error during registration: {e}")
                messages.error(request, 'There was an error creating your account. Please try again.')
    else:
        form = UserCreationForm()
    return render(request, 'webapp/register.html', {'form': form})

@login_required
@user_rate_limit(rate=settings.RATE_LIMITS['GENERAL_ACTIONS'], method='POST', block=True)
def profile(request):
    # Ensure a UserProfile object exists for the user.
    profile, created = UserProfile.objects.get_or_create(user=request.user)

    if request.method == 'POST':
        # Determine which form is being submitted
        if 'update_profile' in request.POST:
            user_form = UserUpdateForm(request.POST, instance=request.user)
            profile_form = ProfileUpdateForm(request.POST, instance=profile)
            if user_form.is_valid() and profile_form.is_valid():
                user_form.save()
                profile_form.save()
                messages.success(request, 'Your profile has been updated successfully!')
                return redirect('profile')
        
        elif 'change_password' in request.POST:
            password_form = CustomPasswordChangeForm(request.user, request.POST)
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)  # Important!
                messages.success(request, 'Your password was successfully updated!')
                return redirect('profile')
            else:
                messages.error(request, 'Please correct the error below.')

    # For GET request, instantiate the forms with current user data
    user_form = UserUpdateForm(instance=request.user)
    profile_form = ProfileUpdateForm(instance=profile)
    password_form = CustomPasswordChangeForm(request.user)

    context = {
        'user_form': user_form,
        'profile_form': profile_form,
        'password_form': password_form,
        'profile': profile
    }
    return render(request, 'webapp/profile.html', context)

@login_required
@user_rate_limit(rate=settings.RATE_LIMITS['MANAGE_DATASETS'], method='POST', block=True)
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
                    print(f"🔍 [DEBUG] Connection: {connection}")
                    
                    # Assign password from form to encrypted field
                    raw_password = form.cleaned_data.get('password')
                    if raw_password:
                        connection.password = raw_password
                        print(f"✅ [DEBUG] Password assigned")
                    
                    # Assign API key from form to encrypted field
                    api_key = form.cleaned_data.get('api_key')  # Fixed: use 'api_key' not 'api-key'
                    if api_key:
                        connection.api_key = api_key
                        print(f"✅ [DEBUG] API Key assigned")
                    
                    connection.save()
                    print(f"✅ [DEBUG] Connection saved successfully with ID: {connection.id}")
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
                
                # --- IMPORTANT: Use HTTPS if available/required --- 
                # url_scheme = "https" if connection.port == 443 else "http" # Example logic
                url_scheme = "http" # KEEPING HTTP FOR DEVELOPMENT as per audit context
                fetch_url = f"{url_scheme}://{connection.ip}:{connection.port}/api/v1/get-data-info"
                
                # Prepare authentication headers matching test_api_researcher.py format
                headers = {
                    'Content-Type': 'application/json',
                    'User-Agent': 'MediNet-WebApp/1.0'
                }
                auth = None
             
                
                if connection.api_key:
                    headers['X-API-Key'] = connection.api_key
                    headers['X-Client-IP'] = '127.0.0.1'  # Using localhost as client IP
                
                if connection.username and connection.password:
                    auth = (connection.username, connection.password)
   
                
                try:
                    response = requests.get(fetch_url, headers=headers, auth=auth, timeout=10)
                    
                    print(f"📡 [DEBUG] Response status: {response.status_code}")
                    print(f"📡 [DEBUG] Response headers: {dict(response.headers)}")
                    
                    if response.content:
                        try:
                            print(f"📄 [DEBUG] Raw response content: {response.text[:500]}...")
                        except:
                            print(f"📄 [DEBUG] Raw response content: [Could not decode response text]")
                    
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    
                    data = response.json()
                    print(f"✅ [DEBUG] Parsed JSON data from {connection.name}:")
                    print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    if isinstance(data, dict) and len(data) < 5:  # Only print small responses
                        print(f"  Data: {data}")
                    else:
                        print(f"  Data size: {len(str(data))} characters")
                    
                    if 'datasets' not in request.session:
                        request.session['datasets'] = {}
                    # ✅ Enhanced validation of received data structure
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
                        print(f"⚠️ [DEBUG] No datasets returned from {connection.name}")
                        messages.info(request, f'No datasets available from "{connection.name}".')
                    else:
                        print(f"✅ [DEBUG] Validated {dataset_count} datasets from {connection.name}")
                        request.session['datasets'][str(connection_id)] = data
                        request.session.modified = True
                        messages.success(request, f'Successfully synchronized {dataset_count} dataset(s) from "{connection.name}".')
                        
                except ValueError as e:
                    print(f"❌ [DEBUG] Data validation error for {connection.name}: {e}")
                    messages.error(request, f'Data validation error from "{connection.name}": {str(e)}')
                    
                except requests.exceptions.HTTPError as e:
                    error_msg = f'HTTP Error {response.status_code}'
                    try:
                        error_data = response.json()
                        error_msg += f': {error_data}'
                    except:
                        error_msg += f': {response.text[:200]}'
                    
                    print(f"❌ [DEBUG] HTTP Error for {connection.name}: {error_msg}")
                    messages.error(request, f'HTTP Error communicating with "{connection.name}": {error_msg}')
                    
                except requests.exceptions.ConnectionError as e:
                    error_msg = f'Connection failed - check if server is running on {connection.ip}:{connection.port}'
                    print(f"❌ [DEBUG] Connection Error for {connection.name}: {str(e)}")
                    messages.error(request, f'{error_msg}')
                    
                except requests.exceptions.Timeout as e:
                    error_msg = f'Request timeout after 10 seconds'
                    print(f"❌ [DEBUG] Timeout Error for {connection.name}: {str(e)}")
                    messages.error(request, f'Timeout communicating with "{connection.name}": {error_msg}')
                    
                except requests.exceptions.RequestException as e:
                    print(f"❌ [DEBUG] Request Exception for {connection.name}: {str(e)}")
                    messages.error(request, f'Error communicating with "{connection.name}": {str(e)}')
                    
                except json.JSONDecodeError as e:
                    print(f"❌ [DEBUG] JSON Decode Error for {connection.name}: {str(e)}")
                    print(f"❌ [DEBUG] Response was: {response.text[:200]}")
                    messages.error(request, f'Received invalid JSON from "{connection.name}". Check server response format.')
                    
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
                print(f"❌ [DEBUG] Error clearing session data: {e}")
                messages.error(request, f'Error clearing session data: {str(e)}')
            return redirect('datasets')

    # Prepare datasets list from session data for rendering
    datasets = []
    session_datasets = request.session.get('datasets', {})
    # Use the same connection filtering logic as above
    if selected_project:
        active_connections = Connection.objects.filter(user=request.user, active=True, project=selected_project)
    else:
        active_connections = Connection.objects.filter(user=request.user, active=True, project__isnull=True)
    connection_map = {str(c.id): c for c in active_connections}

    for conn_id_str, conn_data in session_datasets.items():
         if conn_id_str in connection_map:
            connection_obj = connection_map[conn_id_str]
            
            try:
                # 🔍 Validate data structure before processing
                print(f"🔍 [DEBUG] Processing data from connection {connection_obj.name}")
                print(f"📊 [DEBUG] Available keys in conn_data: {list(conn_data.keys())}")
                print(f"📊 [DEBUG] Raw conn_data structure: {type(conn_data)}")
                
                # Check if conn_data is valid
                if not isinstance(conn_data, dict):
                    print(f"❌ [DEBUG] conn_data is not a dictionary: {type(conn_data)}")
                    continue
                
                if not conn_data:
                    print(f"⚠️ [DEBUG] conn_data is empty for connection {connection_obj.name}")
                    continue
                
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
                    print(f"❌ [DEBUG] Connection {connection_obj.name} has corrupted data. Missing/invalid fields: {', '.join(missing_fields)}")
                    print(f"❌ [DEBUG] Removing corrupted data from session for connection {connection_obj.name}")
                    # Remove corrupted data from session
                    if 'datasets' in request.session and conn_id_str in request.session['datasets']:
                        del request.session['datasets'][conn_id_str]
                        request.session.modified = True
                    continue
                
                # Get lengths and validate consistency
                lengths = {field: len(conn_data[field]) for field in required_fields}
                print(f"📏 [DEBUG] Field lengths: {lengths}")
                
                if len(set(lengths.values())) > 1:
                    raise ValueError(f"Inconsistent array lengths: {lengths}")
                
                num_datasets = lengths['dataset_name']
                if num_datasets == 0:
                    print(f"⚠️ [DEBUG] No datasets found for connection {connection_obj.name}")
                    continue
                
                # Process each dataset with robust error handling
                for i in range(num_datasets):
                    try:
                        print(f"🔄 [DEBUG] Processing dataset {i+1}/{num_datasets}")
                        
                        # Safely get fields from your API structure
                        dataset_id = conn_data['dataset_id'][i] if i < len(conn_data['dataset_id']) else i
                        dataset_name = conn_data['dataset_name'][i] if i < len(conn_data['dataset_name']) else f"unknown_dataset_{i}"
                        
                        # Debug specific dataset
                        if 'hr_failure' in dataset_name.lower():
                            print(f"🔍 [DEBUG-HR] Found hr_failure dataset:")
                            print(f"  - dataset_id: {dataset_id}")
                            print(f"  - dataset_name: {dataset_name}")
                            print(f"  - Connection: {connection_obj.name}")
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
                                print(f"✅ [DEBUG] Parsed metadata for {dataset_name}")
                            except (json.JSONDecodeError, TypeError) as e:
                                print(f"⚠️ [DEBUG] Failed to parse metadata for {dataset_name}: {e}")
                        
                        # Create compatible target_info from your structure
                        target_info = {
                            'name': target_column,
                            'type': 'binary_classification' if target_column else 'unknown',
                            'num_classes': 2  # Default assumption
                        }
                        
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
                        print(f"✅ [DEBUG] Successfully processed dataset: {dataset_name}")
                        
                    except Exception as e:
                        print(f"❌ [DEBUG] Error processing dataset {i}: {e}")
                        continue  # Skip this dataset but continue with others
                        
            except Exception as e:
                print(f"❌ [DEBUG] Error processing connection {connection_obj.name}: {e}")
                print(f"🧹 [DEBUG] Attempting to clean corrupted data for connection {connection_obj.name}")
                # Remove corrupted data from session
                if 'datasets' in request.session and conn_id_str in request.session['datasets']:
                    del request.session['datasets'][conn_id_str]
                    request.session.modified = True
                    print(f"✅ [DEBUG] Removed corrupted session data for connection {connection_obj.name}")
                    messages.warning(request, f"Data format error from connection {connection_obj.name}. Corrupted data has been cleaned. Please re-sync the connection.")
                else:
                    messages.warning(request, f"Data format error from connection {connection_obj.name}: {str(e)}")
                continue # Skip this connection's data if malformed

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

@login_required
@user_rate_limit(rate=settings.RATE_LIMITS['GENERAL_ACTIONS'], method='POST', block=True)
def models_list(request):
    """
    Models management view - shows list of user's models with actions
    """
    import logging
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
    
    logger.error(f"🔍 DEBUG: Rendering template")
    print(f"🔍 DEBUG: Rendering template")
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
    
    print("🔍 DEBUG: selected_datasets: ", request.session.get('selected_datasets', []))
    context = {
        'model_configs': models_list,
        'layer_types': layer_types,
        'optimizer_types': optimizer_types,
        'loss_types': loss_types,
        'strategy_types': strategy_types,
        'previous_step_completed': True,
        'selected_datasets': request.session.get('selected_datasets', []),
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

# Diccionari per guardar informació de simulació d'entrenament

def create_notification(user, title, message, link=None):
    """
    Helper function to create a notification for a user
    """
    # Only create notifications for logged-in users
    if user and user.is_authenticated:
        notification = Notification.objects.create(
            user=user,
            title=title,
            message=message,
            link=link
        )
        return notification
    return None


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

def download_metrics(request, job_id):
    pass

@login_required
@user_rate_limit(rate=settings.RATE_LIMITS['CONNECTION_TEST'], method='POST', block=True)
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
    
    if request.method == 'POST': # Ensure it's a POST request for deletion
        try:
            config_name = config.name
            config.delete()
            return JsonResponse({'success': True, 'message': f'Model configuration "{config_name}" deleted.'})
        except Exception as e:
            # Log exception e
            return JsonResponse({'error': f'Error deleting configuration: {str(e)}'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method. Use POST.'}, status=405)

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
            
            print(f"🎯 START_TRAINING: Generando IDs para {len(selected_datasets)} datasets")
            
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
def dataset_stats(request, dataset_id):
    """API endpoint to get statistics for a dataset"""
    # Similar a preview_dataset, pero con estadísticas
    connection_id = dataset_id.split('_')[1] if '_' in dataset_id else None
    
    if not connection_id:
        return JsonResponse({'success': False, 'error': 'ID de dataset inválido'})
    
    try:
        connection = Connection.objects.get(id=connection_id, user=request.user)
        
        # Estadísticas de ejemplo
        stats_data = {
            'numeric_stats': {
                'edad': {
                    'min': 18, 'max': 85, 'mean': 54.3, 'std': 12.8,
                    'quartiles': [38, 52, 67]
                },
                'glucosa': {
                    'min': 65, 'max': 300, 'mean': 110.5, 'std': 35.2,
                    'quartiles': [92, 105, 125]
                },
                'imc': {
                    'min': 16.5, 'max': 42.1, 'mean': 26.8, 'std': 4.6,
                    'quartiles': [22.7, 25.4, 29.8]
                },
                'riesgo_cv': {
                    'min': 0.01, 'max': 0.85, 'mean': 0.25, 'std': 0.18,
                    'quartiles': [0.1, 0.22, 0.38]
                }
            },
            'categorical_stats': {
                'genero': {
                    'M': 585, 'F': 665
                },
                'hipertension': {
                    'True': 432, 'False': 818
                },
                'cardiopatia': {
                    'True': 285, 'False': 965
                },
                'tabaquismo': {
                    'Nunca': 625, 'Exfumador': 310, 'Fumador': 315
                }
            }
        }
        
        return JsonResponse({'success': True, 'stats': stats_data})
        
    except Connection.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Conexión no encontrada'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

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

def logout_view(request):
    """
    Custom logout view to ensure template rendering
    """
    if 'connections' in request.session:
        del request.session['connections']
    if 'datasets' in request.session:
        del request.session['datasets']
    logout(request)
    return render(request, 'webapp/logout.html')

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

@login_required
@user_rate_limit(rate=settings.RATE_LIMITS['MANAGE_DATASETS'], method='POST', block=True)
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
def user_dashboard(request):
    """
    Main user dashboard view - shows overview of user activity and quick actions
    """
    # Get recent training jobs
    recent_jobs = TrainingJob.objects.filter(user=request.user).order_by('-created_at')[:5]
    
    # Get recent notifications
    recent_notifications = Notification.objects.filter(user=request.user, is_read=False).order_by('-created_at')[:3]
    
    # Get user statistics
    total_models = ModelConfig.objects.filter(user=request.user).count()
    total_jobs = TrainingJob.objects.filter(user=request.user).count()
    completed_jobs = TrainingJob.objects.filter(user=request.user, status='completed').count()
    active_connections = Connection.objects.filter(user=request.user, active=True).count()
    
    # Get current active training jobs
    active_jobs = TrainingJob.objects.filter(
        user=request.user, 
        status__in=['pending', 'running', 'server_ready']
    ).order_by('-created_at')[:3]
    
    context = {
        'recent_jobs': recent_jobs,
        'recent_notifications': recent_notifications,
        'stats': {
            'total_models': total_models,
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'active_connections': active_connections,
            'success_rate': round((completed_jobs / total_jobs * 100) if total_jobs > 0 else 0, 1)
        },
        'active_jobs': active_jobs,
        'is_dashboard': True
    }
    
    return render(request, 'webapp/dashboard_home.html', context)

@login_required
def dashboard(request, job_id):
    """
    Display the real-time training dashboard for a specific job
    """
    try:
        job = TrainingJob.objects.get(id=job_id)
        
        # Check that the user has access to this job
        if job.user != request.user:
            messages.error(request, "No tens permís per accedir a aquest treball d'entrenament.")
            return redirect('training')
        
        # Prepare the context
        context = {
            'job_id': job_id,
            'job': job,
        }
        
        return render(request, 'webapp/dashboard.html', context)
    
    except TrainingJob.DoesNotExist:
        messages.error(request, "El treball d'entrenament no existeix.")
        return redirect('training')
    
    except Exception as e:
        messages.error(request, f"S'ha produït un error en accedir al dashboard: {str(e)}")
        return redirect('training')

@login_required
def notifications(request):
    """
    Display all notifications for a user
    """
    notifications = Notification.objects.filter(user=request.user)
    
    # Mark all as read
    if request.method == 'POST' and 'mark_all_read' in request.POST:
        notifications.update(is_read=True)
        messages.success(request, "Totes les notificacions han estat marcades com a llegides.")
        return redirect('notifications')
    
    # Delete all
    if request.method == 'POST' and 'delete_all' in request.POST:
        notifications.delete()
        messages.success(request, "Totes les notificacions han estat eliminades.")
        return redirect('notifications')
    
    # Mark individual as read
    if request.method == 'POST' and 'mark_read' in request.POST:
        notification_id = request.POST.get('notification_id')
        try:
            notification = Notification.objects.get(id=notification_id, user=request.user)
            notification.is_read = True
            notification.save()
        except Notification.DoesNotExist:
            pass
        return redirect('notifications')
    
    # Delete individual
    if request.method == 'POST' and 'delete' in request.POST:
        notification_id = request.POST.get('notification_id')
        try:
            notification = Notification.objects.get(id=notification_id, user=request.user)
            notification.delete()
        except Notification.DoesNotExist:
            pass
        return redirect('notifications')
    
    return render(request, 'webapp/notifications.html', {'notifications': notifications})

# Afegim un processador de context per a les notificacions
def notifications_processor(request):
    """
    Context processor that adds unread notification count to all templates
    """
    unread_count = 0
    if request.user.is_authenticated:
        unread_count = Notification.objects.filter(user=request.user, is_read=False).count()
    
    return {
        'unread_notifications_count': unread_count
    }

@login_required
def delete_job(request, job_id):
    """
    Delete a training job, ensuring the user owns it.
    """
    job = get_object_or_404(TrainingJob, pk=job_id) # Get the job first
    
    # Check ownership
    if job.user != request.user:
        messages.error(request, "You do not have permission to delete this job.")
        return redirect('training') # Or wherever appropriate
        
    try:
        job_name = job.name
        
        # Kill server process if it's still running
        if job.server_pid:
            try:
                import psutil
                if psutil.pid_exists(job.server_pid):
                    process = psutil.Process(job.server_pid)
                    if process.is_running():
                        print(f"🔪 Terminating server process PID: {job.server_pid} for job deletion")
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            print(f"🔪 Force killing server process PID: {job.server_pid}")
                            process.kill()
            except ImportError:
                print("⚠️ psutil not available - cannot kill server process")
            except Exception as e:
                print(f"⚠️ Error killing server process: {e}")
        
        job.delete()
        messages.success(request, f"El treball '{job_name}' ha estat eliminat correctament.")
    except Exception as e:
        # Log the exception e
        messages.error(request, f"Error en eliminar el treball: {str(e)}")
    
    return redirect('training')

@login_required
def get_notifications_count(request):
    """API endpoint to get the number of unread notifications"""
    count = Notification.objects.filter(user=request.user, is_read=False).count()
    return JsonResponse({'count': count})

@login_required
def get_recent_notifications(request):
    """API endpoint to get the most recent notifications"""
    notifications = Notification.objects.filter(user=request.user).order_by('-created_at')[:5]
    
    notifications_data = []
    for notification in notifications:
        notifications_data.append({
            'id': notification.id,
            'title': notification.title,
            'message': notification.message,
            'link': notification.link,
            'is_read': notification.is_read,
            'created_at': notification.created_at.strftime('%j %b, %H:%M')
        })
    
    return JsonResponse({'notifications': notifications_data})

@login_required
def api_job_details(request, job_id):
    try:
        job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
        
        # Obtenim les últimes mètriques si existeixen
        last_metrics = {}
        if job.metrics_json:
            try:
                if isinstance(job.metrics_json, str):
                    metrics_list = json.loads(job.metrics_json)
                else:
                    metrics_list = job.metrics_json
                last_metrics = metrics_list[-1] if metrics_list else {}
            except (json.JSONDecodeError, IndexError, TypeError):
                last_metrics = {}
        
        # ✅ Assegurar que el progress és 100 si està completat
        progress = job.progress
        if job.status == 'completed' and progress != 100:
            progress = 100
        
        return JsonResponse({
            'success': True,
            'job': {
                'id': job.id,
                'name': job.name,
                'status': job.status,
                'progress': progress,
                'current_round': job.current_round,
                'total_rounds': job.total_rounds,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'metrics': last_metrics,
                'config_json': job.config_json or '{}'
            }
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

@require_http_methods(["POST"])
@user_rate_limit(rate=settings.RATE_LIMITS['MANAGE_DATASETS'], method='POST', block=True)
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
        
        # Update any active training jobs
        active_jobs = TrainingJob.objects.filter(
            user=request.user,
            status__in=['pending', 'in_progress']
        )
        
        for job in active_jobs:
            try:
                job_datasets = json.loads(job.dataset_ids) if job.dataset_ids else []
                job_datasets = [
                    ds for ds in job_datasets
                    if not (
                        ds['dataset_name'] == dataset_info['dataset_name'] and
                        ds['connection']['name'] == dataset_info['connection']['name'] and
                        ds['connection']['ip'] == dataset_info['connection']['ip'] and
                        ds['connection']['port'] == dataset_info['connection']['port']
                    )
                ]
                job.dataset_ids = json.dumps(job_datasets)
                job.save()
            except Exception as e:
                print(f"Error updating job {job.id}: {str(e)}")
        
        return JsonResponse({
            'success': True,
            'message': 'Dataset removed successfully'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


def activate_clients_for_training(training_job, server_process=None):
    """
    Activate clients after starting the Flower server with secure federated learning.
    Enhanced with credential isolation and new authenticated API.
    """
    try:
        # Wait for server to be ready (with timeout and failure check)
        print(f"🔍 Waiting for server to be ready. Current status: {training_job.status}")
        timeout = 30  # 30 seconds timeout
        start_time = time.time()
        
        while training_job.status not in ['server_ready', 'failed', 'cancelled']:
            if time.time() - start_time > timeout:
                print(f"⏰ Timeout waiting for server to be ready")
                training_job.status = 'failed'
                training_job.logs = "Timeout waiting for server to start"
                training_job.save()
                return
                
            time.sleep(1)
            training_job.refresh_from_db()
            print(f"🔍 Checking status: {training_job.status}")
        
        # Check if server failed to start
        if training_job.status in ['failed', 'cancelled']:
            print(f"❌ Server failed to start. Status: {training_job.status}")
            return
        
        # Give additional time for server to fully start listening
        print(f"🚀 Server ready, waiting additional 5 seconds for full startup...")
        time.sleep(5)
        
        print(f"🚀 [SECURE] activate_clients_for_training - using new authenticated API")
        print(f"📋 training_job: {training_job}")
        
        if not training_job.dataset_ids:
            print(f"⚠️ No dataset_ids found in training job")
            return
            
        # Parse selected datasets
        if isinstance(training_job.dataset_ids, str):
            selected_datasets = json.loads(training_job.dataset_ids)
        else:
            selected_datasets = training_job.dataset_ids
            
        print(f"📋 selected_datasets: {selected_datasets}")
        
        unique_connections = {}
        
        # Extract unique connections
        for dataset in selected_datasets:
            conn = dataset['connection']
            conn_key = f"{conn['ip']}:{conn['port']}"
            if conn_key not in unique_connections:
                unique_connections[conn_key] = conn
                
        print(f"📋 unique_connections: {unique_connections}")
        print(f"🔐 [FEDERATED] Will activate {len(unique_connections)} centers with credential isolation")
        
        # Track client activation results
        activated_clients = []
        failed_clients = []
        
        # Get server address for clients (they need to connect to localhost, not 0.0.0.0)
        client_server_address = "localhost:8080"  # Clients connect to localhost
        if 'server' in training_job.config_json:
            server_config = training_job.config_json['server']
            # For clients, use localhost instead of 0.0.0.0
            host = server_config.get('host', 'localhost')
            if host == '0.0.0.0':
                host = 'localhost'  # Convert bind address to connect address
            client_server_address = f"{host}:{server_config.get('port', 8080)}"
        
        # Activate each client with secure, center-specific configuration
        for conn_key, conn in unique_connections.items():
            print(f"🔧 [SECURE] Activating client: {conn['name']} ({conn_key})")
            
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

@login_required
def switch_project_api(request):
    """
    API endpoint to switch the selected project
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            project_id = data.get('project_id')
            
            if project_id is not None and project_id != 'null':
                # Verify the project belongs to the user
                try:
                    project = Project.objects.get(id=project_id, user=request.user)
                    request.session['selected_project_id'] = project.id
                    return JsonResponse({
                        'success': True,
                        'message': f'Switched to project "{project.name}"',
                        'project': {
                            'id': project.id,
                            'name': project.name,
                            'color': project.color
                        }
                    })
                except Project.DoesNotExist:
                    return JsonResponse({
                        'success': False,
                        'error': 'Project not found or access denied'
                    })
            else:
                # Clear project selection (show connections without project)
                if 'selected_project_id' in request.session:
                    del request.session['selected_project_id']
                return JsonResponse({
                    'success': True,
                    'message': 'Switched to "No Project" mode',
                    'project': None
                })
                
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data'
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    })

@login_required
def manage_job_artifacts(request, job_id):
    """
    Pending to implement...
    Vista para gestionar, descargar y borrar artefactos de un TrainingJob.
    Usa datos dummy para la maquetación.
    """
    job = get_object_or_404(TrainingJob, pk=job_id, user=request.user)
    
    # --- Datos Dummy para la Maqueta ---
    dummy_artifacts = [
        {
            'id': 1,
            'name': 'architecture.json',
            'type': 'Architecture',
            'round': None,
            'size': 12345, # 12 KB
            'created_at': datetime.now() - timedelta(hours=2)
        },
        {
            'id': 2,
            'name': 'final_model.pth',
            'type': 'Final Model',
            'round': 100,
            'size': 28400000, # 28.4 MB
            'created_at': datetime.now()
        },
        {
            'id': 3,
            'name': 'checkpoint_round_90.pth',
            'type': 'Checkpoint',
            'round': 90,
            'size': 28400000, # 28.4 MB
            'created_at': datetime.now() - timedelta(minutes=10)
        },
        {
            'id': 4,
            'name': 'checkpoint_round_80.pth',
            'type': 'Checkpoint',
            'round': 80,
            'size': 28400000, # 28.4 MB
            'created_at': datetime.now() - timedelta(minutes=20)
        }
    ]
    
    total_size = sum(art['size'] for art in dummy_artifacts)
    # --- Fin de Datos Dummy ---

    context = {
        'job': job,
        'artifacts': dummy_artifacts,
        'total_size': total_size
    }
    
    return render(request, 'webapp/manage_job_artifacts.html', context)

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
    """API para obtener datos de clientes en tiempo real"""
    
    try:
        job = TrainingJob.objects.get(id=job_id, user=request.user)
        clients_status = job.clients_status or {}
        
        print(f"🌐 API_CLIENTS_DATA: Job {job_id} has {len(clients_status)} clients in status")
        print(f"📋 Available clients: {list(clients_status.keys())}")
        
        clients_data = []
        for client_id, client_info in clients_status.items():
            client_data = {
                'id': client_info['client_id'],
                'name': client_info['connection_name'],
                'ip': client_info['connection_ip'],
                'port': client_info['connection_port'],
                'dataset_name': client_info['dataset_name'],
                'status': client_info['status'],
                'current_round': client_info['current_round'],
                'accuracy': client_info['accuracy'],
                'loss': client_info['loss'],
                'precision': client_info.get('precision', 0),
                'recall': client_info.get('recall', 0),
                'f1': client_info.get('f1', 0),
                'train_samples': client_info['train_samples'],
                'test_samples': client_info.get('test_samples', 0),
                'last_seen': client_info['last_seen'],
            }
            clients_data.append(client_data)
            print(f"📊 Client: {client_info['connection_name']} | Status: {client_info['status']} | Round: {client_info['current_round']} | Acc: {client_info['accuracy']}")
        
        # Calcular estadísticas generales
        overview_stats = {
            'total_clients': len(clients_data),
            'active_clients': len([c for c in clients_data if c['status'] in ['connected', 'training']]),
            'warning_clients': len([c for c in clients_data if c['status'] in ['error', 'offline']]),
            'avg_accuracy': sum(c['accuracy'] for c in clients_data if c['accuracy']) / len([c for c in clients_data if c['accuracy']]) if any(c['accuracy'] for c in clients_data) else 0
        }
        
        print(f"📈 Overview: {overview_stats}")
        
        return JsonResponse({
            'success': True,
            'clients': clients_data,
            'overview_stats': overview_stats
        })
        
    except TrainingJob.DoesNotExist:
        print(f"❌ API_ERROR: Job {job_id} not found")
        return JsonResponse({'success': False, 'error': 'Job not found'}, status=404)

@login_required
def get_client_performance_data(request, job_id, client_id):
    """API para obtener datos de rendimiento histórico de un cliente"""
    
    try:
        job = TrainingJob.objects.get(id=job_id, user=request.user)
        clients_status = job.clients_status or {}
        
        if client_id not in clients_status:
            return JsonResponse({'success': False, 'error': 'Client not found'}, status=404)
        
        client_info = clients_status[client_id]
        rounds_history = client_info.get('rounds_history', {})
        
        # Extraer datos históricos para gráficos
        performance_data = {
            'labels': [f'Round {r}' for r in sorted(rounds_history.keys(), key=int)],
            'accuracy': [rounds_history[r]['accuracy'] for r in sorted(rounds_history.keys(), key=int)],
            'loss': [rounds_history[r]['loss'] for r in sorted(rounds_history.keys(), key=int)],
            'precision': [rounds_history[r].get('precision', 0) for r in sorted(rounds_history.keys(), key=int)],
            'recall': [rounds_history[r].get('recall', 0) for r in sorted(rounds_history.keys(), key=int)],
            'f1': [rounds_history[r].get('f1', 0) for r in sorted(rounds_history.keys(), key=int)]
        }
        
        return JsonResponse({
            'success': True,
            'performance_data': performance_data,
            'client_info': client_info
        })
        
    except TrainingJob.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Job not found'}, status=404)


@login_required
def dataset_detail_view(request, dataset_id):
    """
    Vista polimórfica para mostrar detalles detallados de un dataset desde session data
    dataset_id format: "ds-connection_id-dataset_name" (e.g., "ds-123-hr-failure")
    """
    from .utils.dataset_metrics import DatasetMetricsCalculator
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        print(f"🔍 [DEBUG-DETAIL] Received dataset_id: {dataset_id}")
        
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
        print(f"🔍 [DEBUG-DETAIL] Parsed conn_id: {conn_id_str}, safe_dataset_name: {safe_dataset_name}")
        
        # We need to find the actual dataset name from session data
        # since we converted it to safe format
        
        # Get session data
        session_datasets = request.session.get('datasets', {})
        selected_datasets = request.session.get('selected_datasets', [])
        
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
            print(f"🔍 [DEBUG-DETAIL] Available datasets: {dataset_names}")
            
            # Find the dataset by converting each name to safe format and comparing
            import re
            for i, name in enumerate(dataset_names):
                # Apply same sanitization as in datasets view
                safe_name = name.replace(' ', '-').replace('_', '-').lower()
                safe_name = re.sub(r'[^a-z0-9-]', '', safe_name)
                
                print(f"🔍 [DEBUG-DETAIL] Comparing '{safe_name}' with '{safe_dataset_name}'")
                
                if safe_name == safe_dataset_name:
                    dataset_index = i
                    actual_dataset_name = name
                    print(f"✅ [DEBUG-DETAIL] Found match: {actual_dataset_name} at index {i}")
                    break
            
            if dataset_index is None:
                print(f"❌ [DEBUG-DETAIL] No match found for safe_dataset_name: {safe_dataset_name}")
                messages.error(request, f"Dataset with identifier '{safe_dataset_name}' not found in connection data")
                return redirect('datasets')
                
        except (KeyError, ValueError) as e:
            print(f"❌ [DEBUG-DETAIL] Error finding dataset: {e}")
            messages.error(request, "Invalid dataset data structure")
            return redirect('datasets')
        
        # Build dataset object from session data (adapted for new API structure)
        try:
            print(f"🔍 [DEBUG] Session data structure: {list(conn_data.keys())}")
            print(f"🔍 [DEBUG] Dataset index: {dataset_index}")
            
            # Extract data using the new API structure
            dataset_id_val = conn_data.get('dataset_id', [None])[dataset_index] if isinstance(conn_data.get('dataset_id', []), list) else conn_data.get('dataset_id')
            medical_domain = conn_data.get('medical_domain', [None])[dataset_index] if isinstance(conn_data.get('medical_domain', []), list) else conn_data.get('medical_domain', 'General')
            data_type = conn_data.get('data_type', ['Tabular Data'])[dataset_index] if isinstance(conn_data.get('data_type', []), list) else conn_data.get('data_type', 'Tabular Data')
            description = conn_data.get('description', [None])[dataset_index] if isinstance(conn_data.get('description', []), list) else conn_data.get('description', '')
            target_column = conn_data.get('target_column', [None])[dataset_index] if isinstance(conn_data.get('target_column', []), list) else conn_data.get('target_column', 'unknown')
            num_columns = conn_data.get('num_columns', [0])[dataset_index] if isinstance(conn_data.get('num_columns', []), list) else conn_data.get('num_columns', 0)
            patient_count = conn_data.get('patient_count', [0])[dataset_index] if isinstance(conn_data.get('patient_count', []), list) else conn_data.get('patient_count', 0)
            file_size = conn_data.get('file_size', [0])[dataset_index] if isinstance(conn_data.get('file_size', []), list) else conn_data.get('file_size', 0)
            metadata = conn_data.get('metadata', [{}])[dataset_index] if isinstance(conn_data.get('metadata', []), list) else conn_data.get('metadata', {})
            
            print(f"📊 [DEBUG] Extracted values: patient_count={patient_count}, num_columns={num_columns}, file_size={file_size}")
            
            # Create basic features and target info
            features_info = {
                'input_features': max(0, num_columns - 1),
                'feature_types': {'numeric': max(0, num_columns - 1), 'categorical': 0}
            }
            target_info = {
                'name': target_column,
                'type': 'binary_classification',
                'num_classes': 2
            }
            
            # Create dataset-like object from session data
            dataset_info = {
                'id': dataset_id,
                'dataset_id': dataset_id_val,
                'connection': connection,
                'dataset_name': actual_dataset_name,
                'medical_domain': medical_domain,
                'data_type': data_type,
                'description': description,
                'target_column': target_column,
                'num_columns': num_columns,
                'num_rows': patient_count,
                'patient_count': patient_count,
                'size': file_size,
                'file_size': file_size,
                'metadata': metadata,
                'features_info': features_info,
                'target_info': target_info,
                'dataset_type': 'tabular',  # Default for now, can be extended
                'class_label': target_info.get('name', 'unknown')
            }
            
            print(f"✅ [DEBUG] Dataset info created successfully with metadata: {'Yes' if metadata else 'No'}")
            
            # Check if dataset is selected
            dataset_info['is_selected'] = any(
                sd['connection']['ip'] == connection.ip and 
                sd['dataset_name'] == actual_dataset_name
                for sd in selected_datasets
            )
            
        except (IndexError, KeyError, TypeError) as e:
            logger.error(f"Error parsing dataset data: {str(e)}")
            print(f"❌ [DEBUG] Error details: {e}")
            print(f"❌ [DEBUG] Available keys: {list(conn_data.keys()) if conn_data else 'No data'}")
            messages.error(request, f"Error parsing dataset information: {str(e)}")
            return redirect('datasets')
        
        # Calculate metrics using factory pattern
        calculator = DatasetMetricsCalculator.get_calculator(dataset_info['dataset_type'])
        detailed_metrics = calculator.calculate_metrics_from_session(dataset_info)
        
        # Log for debugging
        print(f"🔍 Dataset details requested for: {actual_dataset_name} from {connection.name}")
        print(f"📊 Dataset info: {dataset_info['num_rows']} rows, {dataset_info['num_columns']} columns")
        if detailed_metrics.get('is_dummy'):
            print(f"⚠️ Using dummy data for dataset type: {dataset_info['dataset_type']}")
        
        context = {
            'dataset': dataset_info,  # Session-based dataset object
            'metrics': detailed_metrics,
            'dataset_type': dataset_info['dataset_type'],
            'is_dummy_data': detailed_metrics.get('is_dummy', False),
            'back_url': request.META.get('HTTP_REFERER', '/datasets/'),
            'dataset_id': dataset_id  # Pass for potential future use
        }
        
        return render(request, 'webapp/dataset_details.html', context)
        
    except Exception as e:
        logger.error(f"Error loading dataset details {dataset_id}: {str(e)}")
        messages.error(request, f"Error loading dataset details: {str(e)}")
        return redirect('datasets')


