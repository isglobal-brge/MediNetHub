from __future__ import annotations
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, authenticate, logout, update_session_auth_hash
from django.contrib import messages
from django.http import JsonResponse
from .models import UserProfile, ModelConfig, TrainingJob, Connection, Dataset, Model, Notification, Project
from .forms import (
    UserUpdateForm,
    ProfileUpdateForm,
    CustomPasswordChangeForm,
)
import json

from django.utils import timezone
from django.views.decorators.http import require_http_methods
from .decorators import ip_rate_limit, user_rate_limit



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



def create_notification(user, title, message, link=None):
    """Create a notification for a user"""
    notification = Notification.objects.create(
        user=user,
        title=title,
        message=message,
        link=link
    )
    return notification


def home(request):
    """
    Landing page view
    """
    return render(request, 'webapp/home.html')


def register(request):
    """
    User registration view
    """
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}!')
            
            # Create UserProfile
            UserProfile.objects.get_or_create(user=user)
            
            # Auto-login after registration
            user = authenticate(username=user.username, password=request.POST['password1'])
            if user is not None:
                login(request, user)
                return redirect('user_dashboard')
            else:
                return redirect('login')
        else:
            # Add form errors to messages
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field.title()}: {error}")
    else:
        form = UserCreationForm()
    
    return render(request, 'webapp/register.html', {'form': form})


@login_required
def profile(request):
    # Ensure a UserProfile object exists for the user.
    user_profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        if 'update_profile' in request.POST:
            user_form = UserUpdateForm(request.POST, instance=request.user)
            profile_form = ProfileUpdateForm(request.POST, instance=user_profile)
            
            if user_form.is_valid() and profile_form.is_valid():
                user_form.save()
                profile_form.save()
                messages.success(request, 'Your profile has been updated!')
                return redirect('profile')
        
        elif 'change_password' in request.POST:
            password_form = CustomPasswordChangeForm(request.user, request.POST)
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)
                messages.success(request, 'Your password has been changed!')
                return redirect('profile')
            else:
                for error in password_form.errors.values():
                    messages.error(request, error)
    else:
        user_form = UserUpdateForm(instance=request.user)
        profile_form = ProfileUpdateForm(instance=user_profile)
        password_form = CustomPasswordChangeForm(request.user)
    
    return render(request, 'webapp/profile.html', {
        'user_form': user_form,
        'profile_form': profile_form,
        'password_form': password_form
    })


def logout_view(request):
    """
    Custom logout view with message
    """
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('home')


@login_required  
def user_dashboard(request):
    """
    Main dashboard for authenticated users showing recent activity and quick stats
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
    
    # Get recent training jobs for the selected project or user
    if selected_project:
        recent_jobs = TrainingJob.objects.filter(user=request.user, project=selected_project).order_by('-created_at')[:5]
    else:
        recent_jobs = TrainingJob.objects.filter(user=request.user).order_by('-created_at')[:5]
    
    # Get recent notifications
    recent_notifications = Notification.objects.filter(user=request.user).order_by('-created_at')[:5]
    
    # Get models count for the selected project or user
    if selected_project:
        models_count = ModelConfig.objects.filter(user=request.user, project=selected_project).count()
    else:
        models_count = ModelConfig.objects.filter(user=request.user).count()

    # Get connections count for the selected project or user
    if selected_project:
        connections_count = Connection.objects.filter(user=request.user, project=selected_project).count()
    else:
        connections_count = Connection.objects.filter(user=request.user).count()

    # Get total jobs count
    if selected_project:
        total_jobs = TrainingJob.objects.filter(user=request.user, project=selected_project).count()
    else:
        total_jobs = TrainingJob.objects.filter(user=request.user).count()

    # Calculate success rate
    if total_jobs > 0:
        if selected_project:
            completed_jobs = TrainingJob.objects.filter(user=request.user, project=selected_project, status='completed').count()
        else:
            completed_jobs = TrainingJob.objects.filter(user=request.user, status='completed').count()
        success_rate = round((completed_jobs / total_jobs) * 100) if total_jobs > 0 else 0
    else:
        success_rate = 0

    # Get active jobs (running or pending)
    if selected_project:
        active_jobs = TrainingJob.objects.filter(
            user=request.user,
            project=selected_project,
            status__in=['running', 'pending', 'server_ready']
        ).order_by('-created_at')[:3]
    else:
        active_jobs = TrainingJob.objects.filter(
            user=request.user,
            status__in=['running', 'pending', 'server_ready']
        ).order_by('-created_at')[:3]

    # Create stats dictionary
    stats = {
        'total_models': models_count,
        'total_jobs': total_jobs,
        'active_connections': connections_count,
        'success_rate': success_rate
    }

    context = {
        'projects': projects,
        'selected_project': selected_project,
        'recent_jobs': recent_jobs,
        'active_jobs': active_jobs,
        'recent_notifications': recent_notifications,
        'stats': stats
    }

    return render(request, 'webapp/dashboard_home.html', context)


@login_required
def notifications(request):
    """
    View to display all notifications for the current user
    """
    # Mark all as read when viewing
    Notification.objects.filter(user=request.user, is_read=False).update(is_read=True)
    
    notifications_list = Notification.objects.filter(user=request.user).order_by('-created_at')
    
    return render(request, 'webapp/notifications.html', {
        'notifications': notifications_list
    })


def notifications_processor(request):
    """
    Context processor for notifications
    """
    if request.user.is_authenticated:
        unread_count = Notification.objects.filter(user=request.user, is_read=False).count()
        return {'unread_notifications_count': unread_count}
    return {'unread_notifications_count': 0}


@login_required
def get_notifications_count(request):
    """API endpoint to get the number of unread notifications"""
    count = Notification.objects.filter(user=request.user, is_read=False).count()
    return JsonResponse({'count': count})


@login_required
def get_recent_notifications(request):
    """API endpoint to get the most recent notifications"""
    notifications_list = Notification.objects.filter(
        user=request.user
    ).order_by('-created_at')[:10]
    
    data = [{
        'id': notification.id,
        'title': notification.title,
        'message': notification.message,
        'link': notification.link,
        'created_at': notification.created_at.strftime('%Y-%m-%d %H:%M'),
        'is_read': notification.is_read
    } for notification in notifications_list]
    
    return JsonResponse({'notifications': data})


@login_required
@require_http_methods(["POST"])
def switch_project_api(request):
    """
    API endpoint to switch the selected project
    """
    try:
        data = json.loads(request.body)
        project_id = data.get('project_id')
        
        if project_id == 'none':
            # Clear project selection
            request.session.pop('selected_project_id', None)
            return JsonResponse({
                'status': 'success',
                'message': 'Project selection cleared',
                'project_name': 'No project selected'
            })
        else:
            # Validate project belongs to user
            try:
                project = Project.objects.get(id=project_id, user=request.user)
                request.session['selected_project_id'] = project.id
                return JsonResponse({
                    'status': 'success',
                    'message': f'Switched to project: {project.name}',
                    'project_name': project.name
                })
            except Project.DoesNotExist:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Project not found or access denied'
                }, status=403)
                
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)