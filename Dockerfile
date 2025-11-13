FROM python:3.11-slim

# Variables de entorno para optimización y configuración
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DJANGO_SETTINGS_MODULE=medinet.settings \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Directorio de trabajo
WORKDIR /usr/src/app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar código fuente de la aplicación
COPY MediNetHub/ .

# Copiar entrypoint script
COPY entrypoint.sh /usr/src/app/entrypoint.sh

# Convertir line endings a LF (Unix) y dar permisos de ejecución
# Esto asegura compatibilidad incluso si el archivo tiene CRLF (Windows)
RUN sed -i 's/\r$//' /usr/src/app/entrypoint.sh && \
    chmod +x /usr/src/app/entrypoint.sh

# Crear directorios para volúmenes si no existen
RUN mkdir -p /usr/src/app/config \
    /usr/src/app/media \
    /usr/src/app/staticfiles

# Exponer puerto 5000
EXPOSE 5000

# Healthcheck para verificar que la aplicación está funcionando
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000').read()" || exit 1

# Usar entrypoint script que genera secrets, migra y ejecuta servidor
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]