#!/bin/bash
#===============================================================================
# Setup Internet NEG with Firewalla DDNS
#===============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

NEG_NAME="ollama-internet-neg"
BACKEND_SERVICE_NAME="ollama-backend"
FIREWALLA_FQDN="d8r978f08m4.d.firewalla.org"
BACKEND_PORT="11000"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Configuring Internet NEG with Firewalla DDNS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${GREEN}Using Firewalla DDNS: ${FIREWALLA_FQDN}:${BACKEND_PORT}${NC}"
echo ""

# Verify DDNS resolves
echo -e "${BLUE}Verifying DDNS resolution...${NC}"
if host "$FIREWALLA_FQDN" >/dev/null 2>&1; then
    RESOLVED_IP=$(host "$FIREWALLA_FQDN" | grep "has address" | awk '{print $4}' | head -1)
    echo -e "${GREEN}✓ DDNS resolves to: ${RESOLVED_IP}${NC}"
else
    echo -e "${YELLOW}⚠ Could not resolve DDNS (may still work)${NC}"
fi

# Create Internet NEG
echo -e "\n${BLUE}Creating Internet NEG...${NC}"
if gcloud compute network-endpoint-groups describe "$NEG_NAME" --global >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ NEG already exists, deleting and recreating...${NC}"
    gcloud compute network-endpoint-groups delete "$NEG_NAME" --global --quiet
fi

gcloud compute network-endpoint-groups create "$NEG_NAME" \
    --network-endpoint-type=INTERNET_FQDN_PORT \
    --global

echo -e "${GREEN}✓ Internet NEG created${NC}"

# Add Firewalla endpoint
echo -e "${BLUE}Adding Firewalla endpoint...${NC}"
gcloud compute network-endpoint-groups update "$NEG_NAME" \
    --add-endpoint="fqdn=${FIREWALLA_FQDN},port=${BACKEND_PORT}" \
    --global

echo -e "${GREEN}✓ Network endpoint added: ${FIREWALLA_FQDN}:${BACKEND_PORT}${NC}"

# Add NEG to backend service
echo -e "${BLUE}Adding NEG to backend service...${NC}"
gcloud compute backend-services add-backend "$BACKEND_SERVICE_NAME" \
    --network-endpoint-group="$NEG_NAME" \
    --balancing-mode=RATE \
    --max-rate-per-endpoint=100 \
    --global

echo -e "${GREEN}✓ NEG added to backend service${NC}"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ SETUP COMPLETE${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}Traffic flow:${NC}"
echo -e "  Internet → ollama.elevatediq.ai (136.110.229.243)"
echo -e "  → GCP Load Balancer"
echo -e "  → ${FIREWALLA_FQDN}:${BACKEND_PORT}"
echo -e "  → Your Ollama API"

echo ""
echo -e "${YELLOW}⚠ Make sure:${NC}"
echo -e "  1. Firewalla firewall allows inbound port ${BACKEND_PORT}"
echo -e "  2. API is listening on 0.0.0.0:${BACKEND_PORT} (not just localhost)"
echo -e "  3. Wait for SSL cert provisioning (10-20 min)"

echo ""
echo -e "${BLUE}Test when ready:${NC}"
echo -e "  curl https://ollama.elevatediq.ai/health"

