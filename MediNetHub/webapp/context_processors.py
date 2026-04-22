from .models import Connection


def connections_status(request):
    if not request.user.is_authenticated:
        return {'global_connections_count': 0}
    count = Connection.objects.filter(user=request.user).count()
    return {'global_connections_count': count}
