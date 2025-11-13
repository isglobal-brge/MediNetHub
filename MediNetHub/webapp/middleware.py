from django.http import JsonResponse
from django.shortcuts import render
from django_ratelimit.exceptions import Ratelimited


class RateLimitMiddleware:
    """
    Middleware to handle rate limiting exceptions and provide user-friendly responses
    """
    
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_exception(self, request, exception):
        if isinstance(exception, Ratelimited):
            # Check if it's an AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'error': 'Rate limit exceeded. Please try again later.',
                    'rate_limited': True
                }, status=429)
            else:
                # For regular HTTP requests, render a template
                return render(request, 'webapp/rate_limit_exceeded.html', {
                    'message': 'Too many requests. Please try again later.'
                }, status=429)
        
        return None 