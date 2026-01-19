#!/bin/sh
# ==============================================================================
# MediNet Docker Entrypoint Script
# ==============================================================================
# This script runs before the main application starts and handles:
# - Auto-generation of Django SECRET_KEY (first run only)
# - Auto-generation of Fernet encryption keys (first run only)
# - Auto-generation of SSL/TLS certificates for Flower server (first run only)
# - Persistence of secrets across container restarts
# ==============================================================================

set -e

# Configuration directory inside the container
CONFIG_DIR="/usr/src/app/config"
SECRET_KEY_FILE="${CONFIG_DIR}/secret_key.txt"
FERNET_KEYS_FILE="${CONFIG_DIR}/fernet_keys.txt"
CERTS_DIR="${CONFIG_DIR}/certs"
CA_CERT_FILE="${CERTS_DIR}/ca.crt"
CA_KEY_FILE="${CERTS_DIR}/ca.key"
SERVER_CERT_FILE="${CERTS_DIR}/server.crt"
SERVER_KEY_FILE="${CERTS_DIR}/server.key"

echo "[MEDINET ENTRYPOINT] Starting initialization..."

# Create config directory if it doesn't exist
mkdir -p "${CONFIG_DIR}"
mkdir -p "${CERTS_DIR}"

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
# Generate SSL/TLS certificates for Flower federated learning if they don't exist
# ==============================================================================
if [ ! -f "${CA_CERT_FILE}" ] || [ ! -f "${CA_KEY_FILE}" ] || \
   [ ! -f "${SERVER_CERT_FILE}" ] || [ ! -f "${SERVER_KEY_FILE}" ]; then
    echo "[MEDINET ENTRYPOINT] Generating SSL/TLS certificates for Flower..."

    # Generate CA private key (4096 bits for security)
    openssl genrsa -out "${CA_KEY_FILE}" 4096

    # Generate CA certificate (valid for 10 years)
    openssl req -x509 -new -nodes \
        -key "${CA_KEY_FILE}" \
        -sha256 \
        -days 3650 \
        -out "${CA_CERT_FILE}" \
        -subj "/C=ES/ST=Barcelona/L=Barcelona/O=MediNetHub/OU=Federated Learning/CN=MediNetHub CA"

    echo "[MEDINET ENTRYPOINT] CA certificate generated"

    # Generate server private key
    openssl genrsa -out "${SERVER_KEY_FILE}" 4096

    # Generate server certificate signing request (CSR)
    openssl req -new \
        -key "${SERVER_KEY_FILE}" \
        -out "${CERTS_DIR}/server.csr" \
        -subj "/C=ES/ST=Barcelona/L=Barcelona/O=MediNetHub/OU=Flower Server/CN=flower-server"

    # Get host IP for SAN (Subject Alternative Names)
    # This allows clients to connect using the real IP address
    HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "")

    # Also check for FLOWER_SERVER_IP environment variable (set in docker-compose)
    if [ -n "${FLOWER_SERVER_IP}" ]; then
        HOST_IP="${FLOWER_SERVER_IP}"
        echo "[MEDINET ENTRYPOINT] Using FLOWER_SERVER_IP from environment: ${HOST_IP}"
    elif [ -z "${HOST_IP}" ]; then
        HOST_IP="127.0.0.1"
        echo "[MEDINET ENTRYPOINT] WARNING: Could not detect host IP, using localhost"
    else
        echo "[MEDINET ENTRYPOINT] Detected host IP: ${HOST_IP}"
    fi

    # Create extensions file for SAN (Subject Alternative Names)
    # This allows the certificate to work with IP addresses and hostnames
    cat > "${CERTS_DIR}/server_ext.cnf" << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = flower-server
DNS.3 = medinet-hub
IP.1 = 127.0.0.1
IP.2 = ${HOST_IP}
EOF

    # Sign server certificate with CA (valid for 1 year)
    openssl x509 -req \
        -in "${CERTS_DIR}/server.csr" \
        -CA "${CA_CERT_FILE}" \
        -CAkey "${CA_KEY_FILE}" \
        -CAcreateserial \
        -out "${SERVER_CERT_FILE}" \
        -days 365 \
        -sha256 \
        -extfile "${CERTS_DIR}/server_ext.cnf"

    # Clean up temporary files
    rm -f "${CERTS_DIR}/server.csr" "${CERTS_DIR}/server_ext.cnf" "${CERTS_DIR}/ca.srl"

    echo "[MEDINET ENTRYPOINT] Server certificate generated and signed by CA"
    echo "[MEDINET ENTRYPOINT] SSL/TLS certificates ready for Flower server"
else
    echo "[MEDINET ENTRYPOINT] Using existing SSL/TLS certificates from ${CERTS_DIR}"
fi

# ==============================================================================
# Set file permissions (read-only for security)
# ==============================================================================
chmod 400 "${SECRET_KEY_FILE}"
chmod 400 "${FERNET_KEYS_FILE}"
chmod 400 "${CA_KEY_FILE}" 2>/dev/null || true
chmod 400 "${SERVER_KEY_FILE}" 2>/dev/null || true
chmod 444 "${CA_CERT_FILE}" 2>/dev/null || true
chmod 444 "${SERVER_CERT_FILE}" 2>/dev/null || true

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
python manage.py createcachetable


# ==============================================================================
# Start Django development server
# ==============================================================================
echo "[MEDINET ENTRYPOINT] Starting Django server on port 5000..."
exec python manage.py runserver 0.0.0.0:5000
