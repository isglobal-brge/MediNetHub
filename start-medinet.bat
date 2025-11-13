@echo off
REM ============================================================================
REM MediNet - Federated Learning Platform
REM Docker Startup Script (Simplified Version)
REM ============================================================================
REM
REM This simplified script starts MediNet using Docker without docker-compose.
REM Includes data persistence through Docker volumes.
REM
REM Requirements:
REM   - Docker Desktop installed and running
REM   - Port 5000 available
REM
REM ============================================================================

echo.
echo ══════════════════════════════════════════════════════════════════
echo           MediNet - Federated Learning Platform
echo                    Docker Startup Script
echo ══════════════════════════════════════════════════════════════════
echo.

REM ============================================================================
REM 1. Verify Docker is running
REM ============================================================================
echo [1/5] Checking Docker...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Docker is not running.
    echo Solution: Start Docker Desktop and wait until it's ready.
    echo.
    pause
    exit /b 1
)
echo       Docker is active
echo.

REM ============================================================================
REM 2. Stop and remove existing container if exists
REM ============================================================================
echo [2/5] Cleaning up previous containers...
docker stop medinet-app 2>nul
docker rm medinet-app 2>nul
echo       Cleanup completed
echo.

REM ============================================================================
REM 3. Build Docker image
REM ============================================================================
echo [3/5] Building Docker image...
echo       This may take a few minutes the first time...
docker build -t medinet:latest .
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Docker image build failed.
    echo Review the logs above for more details.
    echo.
    pause
    exit /b 1
)
echo       Image built successfully
echo.

REM ============================================================================
REM 4. Create volumes for data persistence
REM ============================================================================
echo [4/5] Configuring persistence volumes...
docker volume create medinet-db 2>nul
docker volume create medinet-config 2>nul
docker volume create medinet-media 2>nul
docker volume create medinet-static 2>nul
echo       Volumes created/verified:
echo          - medinet-db (SQLite database)
echo          - medinet-config (Auto-generated secrets: SECRET_KEY, FERNET_KEYS)
echo          - medinet-media (Media files)
echo          - medinet-static (Static files)
echo.

REM ============================================================================
REM 5. Start container with mounted volumes
REM ============================================================================
echo [5/5] Starting MediNet...
docker run -d --name medinet-app -p 5000:5000 -v medinet-db:/usr/src/app -v medinet-config:/usr/src/app/config -v medinet-media:/usr/src/app/media -v medinet-static:/usr/src/app/staticfiles --restart unless-stopped medinet:latest

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Container startup failed.
    echo Review logs with: docker logs medinet-app
    echo.
    pause
    exit /b 1
)

echo       Container started
echo.
echo Waiting for server to be ready (40 seconds)...
timeout /t 40 /nobreak >nul

REM ============================================================================
REM Verify service status
REM ============================================================================
echo.
echo ═══════════════════════════════════════════════════════════════════
echo                       SERVICE STATUS
echo ═══════════════════════════════════════════════════════════════════
echo.

REM Verify container is running
docker ps --filter "name=medinet-app" --format "{{.Status}}" | findstr /C:"Up" >nul
if %errorlevel% equ 0 (
    echo Container: ACTIVE
) else (
    echo Container: INACTIVE or WITH ERRORS
    echo View logs: docker logs medinet-app
)

REM Verify HTTP connectivity
curl -s http://localhost:5000 >nul 2>&1
if %errorlevel% equ 0 (
    echo Web server: RESPONDING
) else (
    echo Web server: NOT RESPONDING YET
    echo It may need more time. View logs: docker logs -f medinet-app
)

echo.
echo ═══════════════════════════════════════════════════════════════════
echo                      MEDINET IS READY
echo ═══════════════════════════════════════════════════════════════════
echo.
echo Access the application at:
echo    http://localhost:5000
echo.
echo Useful commands:
echo    - View logs in real-time:  docker logs -f medinet-app
echo    - Stop application:        docker stop medinet-app
echo    - Restart application:     docker restart medinet-app
echo    - Remove container:        docker rm -f medinet-app
echo    - View volumes:            docker volume ls
echo    - Inspect volume:          docker volume inspect medinet-db
echo.
echo Your data persists in Docker volumes even if you stop the container.
echo.
echo SECURITY NOTE: Auto-generated secrets are stored in medinet-config volume.
echo To view secrets (for backup purposes):
echo    docker volume inspect medinet-config
echo.
echo To delete ALL data (including secrets - BE CAREFUL):
echo    docker volume rm medinet-db medinet-config medinet-media medinet-static
echo.
echo ═══════════════════════════════════════════════════════════════════
echo.
pause
