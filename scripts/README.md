# Release

## Build Docker Images

```bash
# By default using FLAVORS=musa
PLATFORM=linux/amd64 DOCKER_ORG=mthreads ./scripts/build_docker.sh
PLATFORM=linux/arm64 DOCKER_ORG=mthreads ./scripts/build_docker.sh
# Using FLAVORS=vulkan
PLATFORM=linux/amd64 DOCKER_ORG=mthreads FLAVORS=vulkan ./scripts/build_docker.sh
PLATFORM=linux/arm64 DOCKER_ORG=mthreads FLAVORS=vulkan ./scripts/build_docker.sh
```

## Push Docker Images

```bash
PLATFORM=linux/amd64 DOCKER_ORG=mthreads ./scripts/push_docker_mthreads.sh
PLATFORM=linux/arm64 DOCKER_ORG=mthreads ./scripts/push_docker_mthreads.sh
# Push multi-arch
DOCKER_ORG=mthreads ./scripts/push_docker_mthreads.sh
```

## Build Artifacts

```bash
./scripts/build_linux_musa.sh linux amd64 archive
./scripts/build_linux_vulkan.sh linux arm64 archive
```
