# Base Image
FROM --platform=linux/amd64 library/ubuntu:noble AS builder

# Set Environment Variables
ENV DEBIAN_FRONTEND="noninteractive"
ENV VULKAN_VER_BASE="1.3.296"
ENV VULKAN_VER="${VULKAN_VER_BASE}.0"
ENV UBUNTU_VERSION="noble"
ENV GOLANG_VERSION="1.22.8"
ENV GOARCH="amd64"
ENV CGO_ENABLED=1
ENV LDFLAGS=-s

# Set up faster package mirrors
RUN sed -i 's/archive.ubuntu.com/gb.archive.ubuntu.com/g' /etc/apt/sources.list.d/ubuntu.sources

# Install Required Dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates build-essential ccache cmake wget git curl rsync xz-utils libcap-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Go
RUN mkdir -p /usr/local && \
    curl -s -L https://dl.google.com/go/go${GOLANG_VERSION}.linux-${GOARCH}.tar.gz | tar -xz -C /usr/local && \
    ln -s /usr/local/go/bin/go /usr/local/bin/go && \
    ln -s /usr/local/go/bin/gofmt /usr/local/bin/gofmt

# Install Vulkan SDK
RUN wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | gpg --dearmor -o /etc/apt/trusted.gpg.d/lunarg-signing-key-pub.gpg && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-${UBUNTU_VERSION}.list https://packages.lunarg.com/vulkan/${VULKAN_VER_BASE}/lunarg-vulkan-${VULKAN_VER_BASE}-${UBUNTU_VERSION}.list && \
    apt update && apt install -y vulkan-sdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install AMDVLK (Optional: If you want to use AMDVLK instead of RADV)
RUN wget -qO - http://repo.radeon.com/amdvlk/apt/debian/amdvlk.gpg.key | apt-key add && \
    echo "deb [arch=amd64,i386] http://repo.radeon.com/amdvlk/apt/debian/ bionic main" > /etc/apt/sources.list.d/amdvlk.list && \
    apt update && apt install -y amdvlk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set AMDVLK as the default Vulkan driver
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/amd_icd64.json

# Clone Ollama Vulkan Fork
WORKDIR /opt
RUN git clone https://github.com/pufferffish/ollama-vulkan.git ollama-vulkan

# Download and Apply Patches Automatically
WORKDIR /opt/ollama-vulkan
RUN mkdir -p patches && \
    wget -O patches/00-fix-vulkan-building.patch https://github.com/user-attachments/files/18783263/0002-fix-fix-vulkan-building.patch && \
    git checkout 2d443b3dd660a1fd2760d64538512df93648b4bb && git checkout -b ollama_vulkan_stable && \
    git config user.name "Builder" && git config user.email "builder@local" && \
    git remote add ollama_vanilla https://github.com/ollama/ollama.git && \
    git fetch ollama_vanilla --tags && git checkout v0.5.13 && git checkout -b ollama_vanilla_stable && \
    git checkout ollama_vulkan_stable && git merge ollama_vanilla_stable --allow-unrelated-histories --no-edit && \
    for p in patches/*.patch; do patch -p1 < $p; done

# Build Shared Libraries (CPU & Vulkan)
WORKDIR /opt/ollama-vulkan
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --parallel
RUN cmake --install build --component CPU --strip
RUN cmake --install build --component Vulkan --strip

# Install rocm
RUN apt update
RUN apt install -y wget "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
RUN apt install -y python3-setuptools python3-wheel
RUN wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb -O /tmp/amdgpu-install_6.3.60303-1_all.deb
RUN apt install -y /tmp/amdgpu-install_6.3.60303-1_all.deb
RUN apt update && apt install -y rocm

# Build Final Binary
RUN cd /opt/ollama-vulkan && \
    . scripts/env.sh && \
    mkdir -p dist/bin && \
    go build -trimpath -buildmode=pie -o dist/bin/ollama .

# Final Image
FROM --platform=linux/amd64 library/ubuntu:noble
RUN apt-get update && apt-get install -y ca-certificates libcap2 libvulkan1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy Built Components
COPY --from=builder /opt/ollama-vulkan/dist/bin/ollama /bin/ollama

# Expose Ollama Server Port
EXPOSE 11434
ENV OLLAMA_HOST 0.0.0.0

# Run Ollama Server
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
