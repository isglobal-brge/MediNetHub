from django import template
from django.utils.text import capfirst
import json
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter(name='replace_underscore')
def replace_underscore(value):
    """Replaces underscores with spaces and capitalizes the first letter."""
    if not isinstance(value, str):
        return value
    return capfirst(value.replace('_', ' '))

@register.filter
def replace(value, arg):
    """
    Replaces all instances of 'arg' in 'value' with a space
    """
    return value.replace(arg, " ") 

@register.filter
def json_string(value):
    """
    Safely converts a Python object to a JSON string for use in JavaScript or HTML attributes.
    This prevents XSS attacks by properly escaping the JSON output.
    """
    if value is None:
        return 'null'

    try:
        json_str = json.dumps(value)
        # Escape characters that could break out of JavaScript context
        json_str = json_str.replace('<', '\\u003c')
        json_str = json_str.replace('>', '\\u003e')
        json_str = json_str.replace('&', '\\u0026')
        json_str = json_str.replace("'", '\\u0027')
        json_str = json_str.replace('"', '&quot;')
        return mark_safe(json_str)
    except (TypeError, ValueError):
        return 'null' 

@register.filter
def to_json(value):
    """
    Converts a Python object to JSON for use in <script> tags.
    Escapes only </script> to prevent script injection while keeping valid JSON.
    Use this filter instead of json_string for JavaScript contexts.
    """
    if value is None:
        return 'null'

    try:
        # Convert to JSON with ASCII encoding for safety
        json_str = json.dumps(value, ensure_ascii=True)
        # Only escape the closing script tag to prevent injection
        json_str = json_str.replace('</', '<\\/')
        return mark_safe(json_str)
    except (TypeError, ValueError):
        return 'null'

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

@register.filter
def normalize_loss(loss_value):
    """
    Normalizes a loss value to be on a 0-1 scale where higher is better.
    Uses the formula: 1 / (1 + loss).
    """
    print(f"[DEBUG] normalize_loss received: {loss_value} (type: {type(loss_value)})")
    if loss_value is None:
        print("[DEBUG] normalize_loss returning 0 for None input")
        return 0
    try:
        result = 1 / (1 + float(loss_value))
        print(f"[DEBUG] normalize_loss calculated: {result}")
        return result
    except (ValueError, TypeError):
        print("[DEBUG] normalize_loss returning 0 due to exception")
        return 0 