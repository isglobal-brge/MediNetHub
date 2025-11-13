#!/bin/sh
# ==============================================================================
# MediNet Docker Entrypoint Script
# ==============================================================================
# This script runs before the main application starts and handles:
# - Auto-generation of Django SECRET_KEY (first run only)
# - Auto-generation of Fernet encryption keys (first run only)
# - Persistence of secrets across container restarts
# ==============================================================================

set -e

# Configuration directory inside the container
CONFIG_DIR="/usr/src/app/config"
SECRET_KEY_FILE="${CONFIG_DIR}/secret_key.txt"
FERNET_KEYS_FILE="${CONFIG_DIR}/fernet_keys.txt"

echo "[MEDINET ENTRYPOINT] Starting initialization..."

# Create config directory if it doesn't exist
mkdir -p "${CONFIG_DIR}"

# ==============================================================================
# Generate Django SECRET_KEY if it doesn't exist
# ==============================================================================
if [ ! -f "${SECRET_KEY_FILE}" ]; then
    echo "[MEDINET ENTRYPOINT] Generating new Django SECRET_KEY..."

    # Generate a secure random SECRET_KEY (50 characters)
    python3 -c "
import secrets
import string

# Generate a 50-character random string with alphanumeric + special chars
alphabet = string.ascii_letters + string.digits + '!@#$%^&*(-_=+)'
secret_key = ''.join(secrets.choice(alphabet) for i in range(50))
print(secret_key)
" > "${SECRET_KEY_FILE}"

    echo "[MEDINET ENTRYPOINT] SECRET_KEY generated and saved to ${SECRET_KEY_FILE}"
else
    echo "[MEDINET ENTRYPOINT] Using existing SECRET_KEY from ${SECRET_KEY_FILE}"
fi

# ==============================================================================
# Generate Fernet encryption keys if they don't exist
# ==============================================================================
if [ ! -f "${FERNET_KEYS_FILE}" ]; then
    echo "[MEDINET ENTRYPOINT] Generating new Fernet encryption keys..."

    # Generate Fernet key using cryptography library
    python3 -c "
from cryptography.fernet import Fernet
import json

# Generate a new Fernet key
fernet_key = Fernet.generate_key()

# Save as JSON array (Django expects a list)
keys = [fernet_key.decode('utf-8')]
print(json.dumps(keys))
" > "${FERNET_KEYS_FILE}"

    echo "[MEDINET ENTRYPOINT] Fernet keys generated and saved to ${FERNET_KEYS_FILE}"
else
    echo "[MEDINET ENTRYPOINT] Using existing Fernet keys from ${FERNET_KEYS_FILE}"
fi

# ==============================================================================
# Set file permissions (read-only for security)
# ==============================================================================
chmod 400 "${SECRET_KEY_FILE}"
chmod 400 "${FERNET_KEYS_FILE}"

echo "[MEDINET ENTRYPOINT] Configuration files secured (read-only)"

# ==============================================================================
# Run database migrations
# ==============================================================================
echo "[MEDINET ENTRYPOINT] Running database migrations..."
python manage.py migrate

# ==============================================================================
# Collect static files
# ==============================================================================
echo "[MEDINET ENTRYPOINT] Collecting static files..."
python manage.py collectstatic --noinput

# ==============================================================================
# Start Django development server
# ==============================================================================
echo "[MEDINET ENTRYPOINT] Starting Django server on port 5000..."
exec python manage.py runserver 0.0.0.0:5000
