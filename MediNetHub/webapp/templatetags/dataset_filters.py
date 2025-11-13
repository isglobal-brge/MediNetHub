from django import template

register = template.Library()

@register.filter
def lookup(d, key):
    """
    Template filter to look up a key in a dictionary.
    Usage: {{ dict|lookup:key }}
    """
    if hasattr(d, '__getitem__') and key:
        try:
            return d.get(key, '')
        except (AttributeError, KeyError, TypeError):
            return ''
    return ''

@register.filter  
def dict_keys(d):
    """
    Template filter to get dictionary keys.
    Usage: {{ dict|dict_keys }}
    """
    try:
        return d.keys() if d else []
    except (AttributeError, TypeError):
        return []

@register.filter
def dict_items(d):
    """
    Template filter to get dictionary items.
    Usage: {% for key, value in dict|dict_items %}
    """
    try:
        return d.items() if d else []
    except (AttributeError, TypeError):
        return []