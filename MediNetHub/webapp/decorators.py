from functools import wraps
from django.http import JsonResponse
from django.shortcuts import render
from django.core.cache import cache
from django.utils import timezone
from datetime import timedelta
import hashlib


def simple_rate_limit(key_func=None, rate='5/m', method='POST', block=True):
    """
    Simple rate limiting decorator that works with Django's default cache.
    
    Args:
        key_func: Function to generate cache key (default: uses IP)
        rate: Rate limit in format 'requests/period' (e.g., '5/m', '10/h')
        method: HTTP method to limit (default: 'POST')
        block: Whether to block or just warn (default: True)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(request, *args, **kwargs):
            # Skip rate limiting if not the specified method
            if request.method != method:
                return func(request, *args, **kwargs)
            
            # Parse rate limit
            try:
                requests, period = rate.split('/')
                requests = int(requests)
                
                if period == 'm':
                    period_seconds = 60
                elif period == 'h':
                    period_seconds = 3600
                elif period == 'd':
                    period_seconds = 86400
                else:
                    period_seconds = 60  # Default to minute
                    
            except ValueError:
                # Invalid rate format, skip rate limiting
                return func(request, *args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(request)
            else:
                # Default: use IP address
                ip = request.META.get('REMOTE_ADDR', 'unknown')
                cache_key = f"rate_limit_{func.__name__}_{ip}"
            
            # Hash the key to ensure it's a valid cache key
            cache_key = hashlib.md5(cache_key.encode()).hexdigest()
            
            # Get current count
            current_count = cache.get(cache_key, 0)
            
            # Check if rate limit exceeded
            if current_count >= requests:
                if block:
                    # Check if it's an AJAX request
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return JsonResponse({
                            'error': 'Rate limit exceeded. Please try again later.',
                            'rate_limited': True
                        }, status=429)
                    else:
                        return render(request, 'webapp/rate_limit_exceeded.html', {
                            'message': f'Too many requests. Maximum {requests} requests per {period_seconds//60} minutes allowed.'
                        }, status=429)
                else:
                    # Just log the rate limit violation
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f'Rate limit exceeded for {cache_key}')
            
            # Increment counter
            cache.set(cache_key, current_count + 1, period_seconds)
            
            return func(request, *args, **kwargs)
        
        return wrapper
    return decorator


def ip_rate_limit(rate='10/m', method='POST', block=True):
    """Rate limit by IP address"""
    def key_func(request):
        ip = request.META.get('REMOTE_ADDR', 'unknown')
        return f"ip_rate_limit_{ip}"
    
    return simple_rate_limit(key_func=key_func, rate=rate, method=method, block=block)


def user_rate_limit(rate='20/m', method='POST', block=True):
    """Rate limit by authenticated user"""
    def key_func(request):
        if request.user.is_authenticated:
            return f"user_rate_limit_{request.user.id}"
        else:
            ip = request.META.get('REMOTE_ADDR', 'unknown')
            return f"anon_rate_limit_{ip}"
    
    return simple_rate_limit(key_func=key_func, rate=rate, method=method, block=block) 