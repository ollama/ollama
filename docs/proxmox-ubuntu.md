# Multi-GPU Setup Guide: Tesla P40 with Ollama on Ubuntu 22.04 VM (Proxmox)

This guide details the configuration steps required to properly set up multiple Tesla P40 GPUs in passthrough mode for Ollama on an Ubuntu 22.04 VM running on a Proxmox host.

## Prerequisites

- Proxmox VE host system
- Ubuntu 22.04 VM
- 2x NVIDIA Tesla P40 GPUs
- Sufficient system memory (recommended: 64GB+)
- CUDA drivers installed
- Docker (if using containerized Ollama)

## Proxmox VM Configuration

Edit your VM configuration file (`/etc/pve/qemu-server/YOUR_VM_ID.conf`) to include:

```conf
cores: 10
sockets: 2
numa: 1
cpu: host,hidden=1
memory: 65535
machine: q35,viommu=intel
hostpci0: YOUR_GPU_ID_1,pcie=1,x-vga=1
hostpci1: YOUR_GPU_ID_2,pcie=1,x-vga=1
```

## Ubuntu VM Configuration

### 1. Kernel Parameters

Edit `/etc/default/grub` and modify `GRUB_CMDLINE_LINUX_DEFAULT`:

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt intel_iommu=on pcie_acs_override=downstream,multifunction default_hugepagesz=1G hugepagesz=1G"
```

Update GRUB after making changes:
```bash
sudo update-grub
```

### 2. Verify GPU Configuration

After setting up and rebooting, verify the GPU configuration:

```bash
# Check PCIe topology
lspci -tv

# Verify GPU topology
nvidia-smi topo -m
```

## Docker Configuration

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    restart: unless-stopped
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_API_BASE_URL=http://host.docker.internal:11434/api
      - WEBUI_AUTH=false
    depends_on:
      - ollama
    volumes:
      - open-webui:/app/backend/data
    networks:
      - ollama-network
    extra_hosts:
      - "host.docker.internal:host-gateway"
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - OLLAMA_NUM_PARALLEL=2
      - OLLAMA_MAX_LOADED_MODELS=1
      - OLLAMA_KEEP_ALIVE=-1
      - OLLAMA_MAX_QUEUE=1
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_DEBUG=true
      - OLLAMA_NOPRUNE=true
    runtime: nvidia
    volumes:
      - ollama:/root/.ollama
    networks:
      - ollama-network

networks:
  ollama-network:
    driver: bridge
volumes:
  open-webui:
  ollama:
```

## Important Environment Variables

- `OLLAMA_NUM_PARALLEL`: Sets number of parallel model requests (default: 4 or 1 based on memory)
- `OLLAMA_MAX_LOADED_MODELS`: Maximum number of models loaded concurrently (default: 3 * number of GPUs)
- `OLLAMA_MAX_QUEUE`: Maximum number of queued requests (default: 512)
- `OLLAMA_KEEP_ALIVE`: Duration models stay loaded in memory (-1 for infinite)
- `OLLAMA_FLASH_ATTENTION`: Enable flash attention feature
- `OLLAMA_KV_CACHE_TYPE`: Quantization type for K/V cache (options: f16, q8_0, q4_0)

## GPU Management

### Monitoring GPUs

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check GPU topology
nvidia-smi topo -m

# View Ollama process status
ollama ps
```

### Troubleshooting

1. Check Ollama logs:

```bash
# For systemd installations
journalctl -u ollama -f

# For Docker installations
docker logs ollama
```

2. Enable debug logging:

```bash
export OLLAMA_DEBUG=1
```

3. Common issues:

- If GPUs aren't detected, verify IOMMU groups and PCIe passthrough
- Check GPU interconnect type with `nvidia-smi topo -m`
- Verify CUDA installation with `nvidia-smi`

## Performance Optimization

1. K/V Cache Settings:

- Use `OLLAMA_KV_CACHE_TYPE`:
  - `f16`: High precision (default)
  - `q8_0`: 8-bit quantization (1/2 memory of f16)
  - `q4_0`: 4-bit quantization (1/4 memory of f16)

2. Model Loading:

- Enable Flash Attention with `OLLAMA_FLASH_ATTENTION=1`
- Adjust `OLLAMA_MAX_LOADED_MODELS` based on VRAM
- Use `OLLAMA_KEEP_ALIVE=-1` for frequently used models

## Accessing the Web UI

After starting the services:
```
http://your-server-ip:3000
```

## Limitations

- Performance varies based on model size and VRAM availability
- Some models may not fully utilize multiple GPUs
- PCIe bandwidth can affect multi-GPU performance
- GPU scheduling depends on available VRAM and model size

## Managing Docker Containers

### Basic Container Management

Start all services:
```bash
docker compose up -d
```

Stop all services:
```bash
docker compose down
```

Restart specific service:
```bash
docker compose restart ollama
# or
docker compose restart open-webui
```

### Maintenance Commands

View logs:
```bash
# View logs for all services
docker compose logs

# Follow logs in real-time
docker compose logs -f

# View logs for specific service
docker compose logs ollama
docker compose logs open-webui
```

Check container status:
```bash
# List all containers
docker compose ps

# Show container resources
docker stats
```

### Updating Containers

Update to latest images:
```bash
# Pull new images
docker compose pull

# Pull and restart services with new images
docker compose pull && docker compose up -d
```

### Complete Cleanup

Remove everything (including volumes):
```bash
# Stop containers and remove volumes
docker compose down -v

# Remove all stopped containers and unused volumes
docker system prune -v
```

I'll provide a corrected version of just the final section since everything else remains unchanged:

### Troubleshooting Container Issues

If containers fail to start:

```bash
# Check container logs
docker compose logs --tail=100 ollama

# Inspect container
docker inspect ollama
```

If GPU access fails:

```bash
# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

Reset individual container:

```bash
# Stop, remove, and recreate container
docker compose stop ollama
docker compose rm -f ollama
docker compose up -d ollama
```
