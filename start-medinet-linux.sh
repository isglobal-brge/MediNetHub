#!/bin/bash
# ============================================================================
# MediNet - Federated Learning Platform
# Docker Startup Script (Linux/macOS Version)
# ============================================================================
#
# This script starts MediNet using Docker without docker-compose.
# Includes data persistence through Docker volumes.
#
# Requirements:
#   - Docker installed and running
#   - Port 5000 available
#
# Usage:
#   chmod +x start-medinet-linux.sh
#   ./start-medinet-linux.sh
#
# ============================================================================

# Color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}           MediNet - Federated Learning Platform${NC}"
    echo -e "${BLUE}                    Docker Startup Script${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}\n"
}

print_step() {
    echo -e "${YELLOW}[$1] $2${NC}"
}

print_success() {
    echo -e "       ${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "\n${RED}ERROR: $1${NC}"
    echo -e "${RED}$2${NC}\n"
}

print_info() {
    echo -e "${BLUE}$1${NC}"
}

# ============================================================================
# Main execution
# ============================================================================

print_header

# ============================================================================
# 1. Verify Docker is running
# ============================================================================
print_step "1/5" "Checking Docker..."

if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running." "Solution: Start Docker and wait until it's ready."
    exit 1
fi

print_success "Docker is active"
echo

# ============================================================================
# 2. Stop and remove existing container if exists
# ============================================================================
print_step "2/5" "Cleaning up previous containers..."

docker stop medinet-app > /dev/null 2>&1
docker rm medinet-app > /dev/null 2>&1

print_success "Cleanup completed"
echo

# ============================================================================
# 3. Build Docker image
# ============================================================================
print_step "3/5" "Building Docker image..."
echo "       This may take a few minutes the first time..."

if ! docker build -t medinet:latest .; then
    print_error "Docker image build failed." "Review the logs above for more details."
    exit 1
fi

print_success "Image built successfully"
echo

# ============================================================================
# 4. Create volumes for data persistence
# ============================================================================
print_step "4/5" "Configuring persistence volumes..."

docker volume create medinet-db > /dev/null 2>&1
docker volume create medinet-config > /dev/null 2>&1
docker volume create medinet-media > /dev/null 2>&1
docker volume create medinet-static > /dev/null 2>&1

print_success "Volumes created/verified:"
echo "          - medinet-db (SQLite database)"
echo "          - medinet-config (Auto-generated secrets: SECRET_KEY, FERNET_KEYS)"
echo "          - medinet-media (Media files)"
echo "          - medinet-static (Static files)"
echo

# ============================================================================
# 5. Start container with mounted volumes
# ============================================================================
print_step "5/5" "Starting MediNet..."

if ! docker run -d \
    --name medinet-app \
    -p 5000:5000 \
    -v medinet-db:/usr/src/app \
    -v medinet-config:/usr/src/app/config \
    -v medinet-media:/usr/src/app/media \
    -v medinet-static:/usr/src/app/staticfiles \
    --restart unless-stopped \
    medinet:latest; then
    print_error "Container startup failed." "Review logs with: docker logs medinet-app"
    exit 1
fi

print_success "Container started"
echo

echo "Waiting for server to be ready (40 seconds)..."
sleep 40

# ============================================================================
# Verify service status
# ============================================================================
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                       SERVICE STATUS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo

# Verify container is running
if docker ps --filter "name=medinet-app" --format "{{.Status}}" | grep -q "Up"; then
    echo -e "Container: ${GREEN}ACTIVE${NC}"
else
    echo -e "Container: ${RED}INACTIVE or WITH ERRORS${NC}"
    echo "View logs: docker logs medinet-app"
fi

# Verify HTTP connectivity
if command -v curl > /dev/null 2>&1; then
    if curl -s http://localhost:5000 > /dev/null 2>&1; then
        echo -e "Web server: ${GREEN}RESPONDING${NC}"
    else
        echo -e "Web server: ${YELLOW}NOT RESPONDING YET${NC}"
        echo "It may need more time. View logs: docker logs -f medinet-app"
    fi
else
    echo -e "Web server: ${YELLOW}CANNOT VERIFY (curl not installed)${NC}"
    echo "Install curl or check manually: http://localhost:5000"
fi

# ============================================================================
# Final information
# ============================================================================
echo
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                      MEDINET IS READY${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo
echo "Access the application at:"
echo -e "    ${BLUE}http://localhost:5000${NC}"
echo
echo "Useful commands:"
echo "    - View logs in real-time:  docker logs -f medinet-app"
echo "    - Stop application:        docker stop medinet-app"
echo "    - Restart application:     docker restart medinet-app"
echo "    - Remove container:        docker rm -f medinet-app"
echo "    - View volumes:            docker volume ls"
echo "    - Inspect volume:          docker volume inspect medinet-db"
echo
echo "Your data persists in Docker volumes even if you stop the container."
echo
echo -e "${YELLOW}SECURITY NOTE:${NC} Auto-generated secrets are stored in medinet-config volume."
echo "To view secrets (for backup purposes):"
echo "    docker volume inspect medinet-config"
echo
echo -e "${RED}To delete ALL data (including secrets - BE CAREFUL):${NC}"
echo "    docker volume rm medinet-db medinet-config medinet-media medinet-static"
echo
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo
