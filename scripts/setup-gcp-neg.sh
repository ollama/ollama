#!/bin/bash
#===============================================================================
# GCP Internet NEG Setup for External Backend
# Creates a Network Endpoint Group to connect external backend to GCP LB
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "${PROJECT_ROOT}/scripts/host-profile.sh" ]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/scripts/host-profile.sh"
    load_host_profile "${PROJECT_ROOT}"
fi

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ID="$(gcloud config get-value project)"
NEG_NAME="ollama-internet-neg"
BACKEND_SERVICE_NAME="ollama-backend"
BACKEND_FQDN="elevatediq.ai"  # Use public domain that resolves to your server
BACKEND_HOST="${BACKEND_HOST:-localhost}"
BACKEND_PORT="11000"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Setting up Internet NEG for External Backend${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${YELLOW}⚠ IMPORTANT:${NC}"
echo -e "  For GCP Load Balancer to reach your backend at ${BACKEND_HOST}:${BACKEND_PORT},"
echo -e "  you need a publicly accessible endpoint."
echo -e ""
echo -e "  ${YELLOW}OPTIONS:${NC}"
echo -e "  1. Use Cloud VPN/Interconnect to connect GCP to your network"
echo -e "  2. Use a public IP with firewall rules"
echo -e "  3. Deploy the backend on a GCE instance"
echo -e "  4. Use Cloud Run or other GCP services"
echo -e ""
echo -e "${RED}Current limitation:${NC} ${BACKEND_HOST} is not publicly reachable"
echo -e "${RED}GCP Load Balancer cannot route to private IPs directly${NC}"
echo -e ""

read -p "Do you have a public IP or VPN connection? (y/N): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Skipping NEG setup.${NC}"
    echo ""
    echo -e "${BLUE}Alternative approaches:${NC}"
    echo ""
    echo -e "${GREEN}1. Reverse Proxy Setup (Recommended):${NC}"
    echo -e "   - Deploy nginx on a GCE instance"
    echo -e "   - Configure Cloud VPN to connect to your network"
    echo -e "   - Proxy requests to ${BACKEND_HOST}:${BACKEND_PORT}"
    echo ""
    echo -e "${GREEN}2. SSH Tunnel:${NC}"
    echo -e "   - Set up persistent SSH tunnel from GCE to your server"
    echo -e "   - Requires bastion host in GCP"
    echo ""
    echo -e "${GREEN}3. Direct Migration:${NC}"
    echo -e "   - Deploy your Docker stack directly on GCE"
    echo -e "   - Use the automation scripts you already have"
    echo ""
    exit 0
fi

# Get public endpoint
echo ""
read -p "Enter your public FQDN (e.g., api.yourdomain.com): " PUBLIC_FQDN
read -p "Enter port (default: 443): " PUBLIC_PORT
PUBLIC_PORT="${PUBLIC_PORT:-443}"

echo ""
echo -e "${BLUE}Creating Internet NEG...${NC}"

# Create Internet NEG
gcloud compute network-endpoint-groups create "$NEG_NAME" \
    --network-endpoint-type=INTERNET_FQDN_PORT \
    --global

echo -e "${GREEN}✓ Internet NEG created${NC}"

# Add network endpoint
echo -e "${BLUE}Adding network endpoint: ${PUBLIC_FQDN}:${PUBLIC_PORT}${NC}"
gcloud compute network-endpoint-groups update "$NEG_NAME" \
    --add-endpoint="fqdn=${PUBLIC_FQDN},port=${PUBLIC_PORT}" \
    --global

echo -e "${GREEN}✓ Network endpoint added${NC}"

# Add NEG to backend service
echo -e "${BLUE}Adding NEG to backend service...${NC}"
gcloud compute backend-services add-backend "$BACKEND_SERVICE_NAME" \
    --network-endpoint-group="$NEG_NAME" \
    --balancing-mode=RATE \
    --max-rate-per-endpoint=100 \
    --global

echo -e "${GREEN}✓ NEG added to backend service${NC}"

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo -e "${BLUE}Load Balancer will now route traffic through:${NC}"
echo -e "  Internet → GCP LB (136.110.229.243) → ${PUBLIC_FQDN}:${PUBLIC_PORT}"
