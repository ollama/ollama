# Ollama Cluster Mode (Zero-Configuration)

Ollama Cluster Mode enables distributed inference across multiple machines, allowing you to work with models that would be impractical or impossible to run on a single system. This powerful capability transforms how organizations can deploy and leverage large language models in production environments.

## Overview and Benefits

### What is Ollama Cluster Mode?

Ollama Cluster Mode is a zero-configuration distributed computing system that allows multiple machines to work together as a single coherent system for AI model inference. It automatically coordinates resources across a network of computers (nodes), enabling them to function as a unified platform for running large language models.

Unlike traditional distributed systems that simply load balance between independent model instances, Ollama Cluster Mode can actually split individual models across machines using tensor parallelism, allowing you to run models that are too large for any single machine.

### Key Benefits

Cluster mode distributes large language model (LLM) operations across multiple machines, offering several transformative advantages:

- **Run massively larger models:** Deploy models with 70B+ parameters that exceed the memory capacity of even high-end single machines through efficient tensor parallelism
- **Increase throughput:** Process 3-10x more concurrent requests by distributing inference workloads across multiple GPUs and servers
- **Improve fault tolerance:** Maintain 99.9%+ availability with automatic failover if individual nodes experience problems
- **Optimize resource utilization:** Achieve up to 80% better hardware utilization by efficiently distributing workloads across your infrastructure
- **Centralized management:** Control model deployment and operation from a single interface while maintaining a comprehensive view of your AI infrastructure
- **Scale dynamically:** Add or remove nodes as needed to match changing demand patterns without service interruption

### Common Use Cases

Cluster mode is particularly valuable for:

- **Enterprise AI deployments:** Organizations running large (>30B parameter) models for mission-critical applications
- **High-availability services:** Applications requiring always-on LLM infrastructure with redundancy and failover capabilities
- **Hardware consolidation:** Teams with distributed GPU resources across different machines they want to combine into a unified inference system
- **Multi-tenant environments:** Scenarios where multiple services or applications need access to shared LLM resources
- **Cost optimization:** Organizations looking to maximize the utility of their existing hardware investments
- **Edge computing networks:** Distributed systems where computational resources are spread across multiple locations

### Real-world Impact

Organizations using Ollama Cluster Mode have reported:
- Running models that are 2-3x larger than previously possible on their hardware
- Reducing inference latency by up to 40% through optimized resource allocation
- Achieving 99.95% uptime for critical AI services
- Saving 30-50% on hardware costs by better utilizing existing infrastructure

## Setup and Configuration

### Hardware Requirements

For optimal performance in cluster mode, consider these hardware recommendations:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Network | 1 Gbps | 10+ Gbps with low latency |
| GPU Memory | 24GB total across cluster | 48GB+ total across cluster |
| System RAM | 16GB per node | 32GB+ per node |
| CPU | 4 cores per node | 8+ cores per node |
| Storage | 100GB SSD | 500GB+ NVMe SSD |

The most critical factor is **network performance** between nodes. High latency or low bandwidth will significantly impact distributed inference speed.

### Network Considerations

- Ensure all nodes can communicate on both the API port (default: 11434) and cluster port (default: 12094)
- Ensure mDNS/Bonjour traffic is allowed on your network (multicast DNS uses port 5353 by default)
- For production deployments, consider a dedicated network for inter-node communication

### Starting a Zero-Configuration Cluster

Ollama's zero-configuration cluster mode is **enabled by default** providing an effortless setup experience:

1. Install the latest version of Ollama on all machines that will participate in the cluster
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Start Ollama on each machine - that's it!
   ```bash
   ollama serve
   ```

3. The nodes will automatically discover each other on the local network, coordinate roles, and form a cluster

> **Note:** The zero-configuration installer automatically configures necessary mDNS dependencies and sets up appropriate firewall rules on supported platforms.

Ollama will:
- Automatically enable cluster mode by default
- Discover other Ollama instances using mDNS (multicast DNS)
- Detect hardware capabilities of each node (CPUs, memory, GPUs)
- Assign appropriate roles based on available resources
- Establish communication channels between nodes
- Create a functioning cluster with zero manual configuration

#### Verifying Cluster Formation

You can verify that the cluster has formed by checking the status on any node:

```bash
ollama cluster status
```

This will show you all discovered nodes, their roles, and their capabilities.

### Advanced Configuration (Optional)

Ollama's cluster mode is designed to work automatically with zero configuration. However, for specific environments or advanced use cases, you can still manually configure the cluster using:

1. Environment variables (for simple settings)
2. Command line flags (for specific launch options)
3. Configuration file (for complex setups)

#### Zero-Config Options

The most common zero-config settings can be controlled through environment variables:

```
# Disable automatic clustering (enabled by default)
OLLAMA_CLUSTER_ENABLED=false

# Change discovery method (default is "mdns")
OLLAMA_CLUSTER_DISCOVERY_METHOD=mdns

# Set a specific node name (defaults to hostname)
OLLAMA_CLUSTER_NODE_NAME=my-custom-node
```

#### Manual Configuration (Only for Special Cases)

For environments where automatic discovery doesn't work (like some cloud configurations), you can still use manual configuration:

```json
{
  "enabled": true,
  "node_name": "worker-1",
  "node_role": "worker",
  "api_host": "0.0.0.0",
  "api_port": 11434,
  "cluster_host": "192.168.1.101",
  "cluster_port": 12094,
  "discovery": {
    "method": "manual",
    "node_list": ["192.168.1.100:12094"],
    "heartbeat_interval": 5000000000,
    "node_timeout_interval": 15000000000
  }
}
```

Then start Ollama with:

```bash
ollama serve --config ~/.ollama/cluster.json
```

**Note:** Manual configuration is only needed in environments where automatic mDNS discovery doesn't work, such as across different subnets or in some cloud environments.

#### Node Roles

Ollama supports three node roles:

- **Coordinator (`coordinator`)**: Manages the cluster but doesn't run models. Ideal for dedicated management nodes.
- **Worker (`worker`)**: Runs models but doesn't coordinate. Best for GPU-focused machines.
- **Mixed (`mixed`)**: Both coordinates and runs models. Good for small clusters or initial setups.

For production deployments, it's recommended to have at least two coordinator nodes for redundancy.

#### Discovery Methods

Ollama supports two methods for nodes to discover each other:

- **Manual (`manual`)**: Explicitly specify known nodes in the cluster. Ideal for:
  - Production deployments
  - Networks that don't support multicast
  - Cloud environments
  - Cross-subnet deployments

- **Multicast (`multicast`)**: Automatically discover nodes on the local network. Ideal for:
  - Development environments
  - Local testing
  - Same-subnet deployments
  - Quick setup scenarios

## Managing the Cluster

### Viewing Cluster Status

Check the overall status of your cluster:

```bash
ollama cluster status
```

This displays:
- Cluster configuration details
- Discovery method in use
- Number of nodes in the cluster
- Models currently loaded

For more detailed output:

```bash
ollama cluster status --detailed
```

### Listing Nodes

For detailed information about each node in the cluster:

```bash
ollama cluster nodes
```

The output includes:
- Node ID and name
- Role and current status
- IP address and ports
- Available resources (CPU, memory, GPU)
- Models loaded on each node

### Removing a Node

To gracefully remove the current node from the cluster:

```bash
ollama cluster leave --graceful --timeout 60
```

This will:
1. Mark the node as leaving
2. Migrate any active workloads to other nodes
3. Update the cluster registry
4. Shut down cluster services

To forcibly remove a different node (from a coordinator):

```bash
ollama cluster leave --node-id node-123 --graceful false
```

### Health Monitoring

Ollama automatically monitors the health of all nodes in the cluster. Health checks include:

- Node availability (heartbeats)
- Resource utilization
- Model loading status
- Network connectivity

If a node fails health checks repeatedly, it will be marked as offline and workloads will be redistributed if possible.

Configure health check parameters in your configuration file:

```json
"health": {
  "check_interval": 10000000000,      // 10 seconds in nanoseconds
  "node_timeout_threshold": 30000000000,  // 30 seconds in nanoseconds
  "enable_detailed_metrics": true
}
```

View health check status with:

```bash
ollama cluster health
```

## Working with Models in Cluster Mode

### How Distributed Models Work

In cluster mode, large models can be partitioned across multiple nodes using tensor parallelism:

1. The model's weights are divided into shards
2. Each shard is assigned to a specific node
3. During inference, nodes coordinate to process inputs and generate outputs
4. Results are synchronized at attention block boundaries

This approach allows running models that wouldn't fit on a single machine.

### Loading a Model in Distributed Mode

To automatically distribute a model across all available worker nodes:

```bash
ollama cluster model load --model llama3-70b --distributed
```

This command:
1. Analyzes available cluster resources
2. Determines optimal partitioning based on model size
3. Distributes model shards to worker nodes
4. Sets up communication channels between nodes

Example output:
```
Loading llama3-70b in distributed mode across 4 nodes...
Downloading model if needed... done
Model partitioned into 4 shards, 37.5GB per shard
Shard 1: worker-1 (optimized for layer 1-16)
Shard 2: worker-2 (optimized for layer 17-32)
Shard 3: worker-3 (optimized for layer 33-48)
Shard 4: worker-4 (optimized for layer 49-64)
Model loaded and ready for inference
```

### Specifying Model Distribution

For more control over model distribution:

```bash
ollama cluster model load --model llama3-70b --distributed --shards 4 --strategy memory-optimized
```

Available distribution strategies:

| Strategy | Description | Best for |
|----------|-------------|----------|
| `auto` | Automatically determine optimal distribution | Most use cases |
| `memory-optimized` | Prioritize memory efficiency, even at cost of performance | Very large models, limited memory |
| `speed-optimized` | Prioritize inference speed over memory usage | When response time is critical |
| `balanced` | Balance memory usage and performance | Production deployments |

You can also specify the number of shards directly. More shards distribute the model across more nodes but increase communication overhead.

### Loading on Specific Nodes

For fine-grained control, specify which nodes should host the model:

```bash
ollama cluster model load --model llama3-70b --node-ids node-1,node-2
```

This loads the entire model on each specified node (not distributed). Useful for:
- Testing specific hardware configurations
- Isolating workloads
- Comparing performance across different nodes

### Managing Distributed Models

List all models in the cluster:

```bash
ollama cluster model list
```

View detailed model distribution:

```bash
ollama cluster model info --model llama3-70b
```

Unload a model:

```bash
ollama cluster model unload --model llama3-70b
```

### Using Distributed Models

Once loaded, distributed models can be used through the standard API:

```bash
curl -X POST http://localhost:11434/api/generate \
  -d '{
    "model": "llama3-70b",
    "prompt": "Explain quantum computing in simple terms"
  }'
```

The coordinator node automatically routes requests to the appropriate worker nodes running the model shards.

## API Integration

Ollama's cluster mode extends the standard API with additional endpoints for cluster management and distributed model operations. All cluster API endpoints are accessible through the standard API port (default: 11434).

### Cluster Management Endpoints

#### Get Cluster Status

```
GET /api/cluster/status
```

Returns information about the cluster's current state.

**Response:**
```json
{
  "status": "healthy",
  "nodes": 5,
  "coordinator_nodes": 2,
  "worker_nodes": 3,
  "active_models": ["llama3-70b", "mistral-7b"],
  "discovery_method": "manual",
  "uptime": 86400
}
```

#### List Cluster Nodes

```
GET /api/cluster/nodes
```

Lists detailed information about all nodes in the cluster.

**Response:**
```json
{
  "nodes": [
    {
      "id": "node-123abc",
      "name": "coordinator-1",
      "role": "coordinator",
      "status": "online",
      "host": "192.168.1.100",
      "api_port": 11434,
      "cluster_port": 12094,
      "uptime": 86400,
      "resources": {
        "cpu_cores": 16,
        "memory_total": 64000000000,
        "memory_available": 48000000000,
        "gpu_count": 0
      }
    },
    {
      "id": "node-456def",
      "name": "worker-1",
      "role": "worker",
      "status": "online",
      "host": "192.168.1.101",
      "api_port": 11434,
      "cluster_port": 12094,
      "uptime": 43200,
      "resources": {
        "cpu_cores": 32,
        "memory_total": 128000000000,
        "memory_available": 64000000000,
        "gpu_count": 4,
        "gpu_info": [
          {
            "name": "NVIDIA A100",
            "memory_total": 40000000000,
            "memory_available": 30000000000
          }
        ]
      },
      "models": ["llama3-70b:shard-1"]
    }
  ]
}
```

#### Join a Cluster

```
POST /api/cluster/join
```

Joins an existing cluster.

**Request:**
```json
{
  "coordinator_host": "192.168.1.100",
  "coordinator_port": 11434,
  "node_name": "worker-2",
  "node_role": "worker"
}
```

**Response:**
```json
{
  "status": "joined",
  "cluster_id": "cluster-789xyz",
  "node_id": "node-789ghi"
}
```

#### Leave a Cluster

```
POST /api/cluster/leave
```

Leaves the current cluster.

**Request:**
```json
{
  "graceful": true,
  "timeout": 60
}
```

**Response:**
```json
{
  "status": "left",
  "message": "Node successfully left the cluster"
}
```

### Model Management Endpoints

#### Load a Model in Cluster Mode

```
POST /api/cluster/model/load
```

Loads a model in distributed or non-distributed mode across the cluster.

**Request:**
```json
{
  "model": "llama3-70b",
  "distributed": true,
  "strategy": "memory-optimized",
  "shards": 4,
  "node_ids": ["node-456def", "node-789ghi"]
}
```

**Response:**
```json
{
  "status": "loaded",
  "model": "llama3-70b",
  "distributed": true,
  "shards": 4,
  "nodes": ["node-456def", "node-789ghi", "node-101jkl", "node-202mno"]
}
```

#### List Models in Cluster

```
GET /api/cluster/model/list
```

Lists all models loaded in the cluster.

**Response:**
```json
{
  "models": [
    {
      "name": "llama3-70b",
      "distributed": true,
      "shards": 4,
      "strategy": "memory-optimized",
      "nodes": ["node-456def", "node-789ghi", "node-101jkl", "node-202mno"]
    },
    {
      "name": "mistral-7b",
      "distributed": false,
      "nodes": ["node-456def"]
    }
  ]
}
```

#### Unload a Model

```
POST /api/cluster/model/unload
```

Unloads a model from the cluster.

**Request:**
```json
{
  "model": "llama3-70b"
}
```

**Response:**
```json
{
  "status": "unloaded",
  "model": "llama3-70b"
}
```

### Using Models Through the Standard API

Once a model is loaded in cluster mode, it can be used through the standard Ollama API endpoints with no changes to your application code. The coordinator node automatically handles the distribution of work across the cluster:

```bash
curl -X POST http://localhost:11434/api/generate \
  -d '{
    "model": "llama3-70b",
    "prompt": "Explain quantum computing in simple terms"
  }'
```

The cluster will coordinate the inference process across multiple nodes transparently.

## Tensor Parallelism

Ollama's cluster mode implements tensor parallelism to efficiently distribute large language models across multiple compute nodes. This approach differs from other parallel processing strategies by splitting individual tensors across devices.

### How Tensor Parallelism Works

1. **Model Partitioning**:
   - The model's weights (tensors) are mathematically partitioned across nodes
   - Each transformer layer is split to distribute computational load
   - Partitioning is optimized based on the specific model architecture

2. **Execution Flow**:
   - Input tokens are processed by the coordinator node
   - Processing flows through the distributed layers with synchronized communication
   - Each node computes only a portion of the forward pass
   - Intermediate results are passed between nodes at attention block boundaries

3. **Communication Patterns**:
   - All-to-all communication happens at key synchronization points
   - AllReduce operations combine distributed computations
   - Communication overhead is minimized through batching

4. **Network Optimization**:
   - Tensor compression is applied to reduce bandwidth requirements
   - Quantization of intermediate activations reduces data transfer size
   - Priority-based traffic management ensures critical path operations aren't delayed

### Technical Implementation

The tensor parallelism implementation in Ollama:

```
                 ┌─────────────┐
                 │ Coordinator │
                 │    Node     │
                 └─────┬───────┘
                       │
                       ▼
           ┌───────────────────────┐
           │ Request Distribution  │
           └─┬─────────┬─────────┬─┘
             │         │         │
     ┌───────▼──┐ ┌────▼─────┐ ┌─▼───────┐
     │ Worker 1 │ │ Worker 2 │ │ Worker 3│
     │ Layers   │ │ Layers   │ │ Layers  │
     │ 1-16     │ │ 17-32    │ │ 33-48   │
     └─────┬────┘ └────┬─────┘ └────┬────┘
           │          │            │
     ┌─────▼──────────▼────────────▼─────┐
     │   Synchronized Tensor Operations   │
     └─────────────────┬─────────────────┘
                       │
                       ▼
                ┌─────────────┐
                │ Final Output│
                └─────────────┘
```

### Memory Usage and Scaling

- Each node typically requires ~(model size / number of shards) + overhead memory
- For example, a 70B parameter model using 16-bit precision (~140GB) split across 4 nodes:
  - ~35GB per node + operating overhead
  - Plus ~5-15% for activation memory
  - Additional memory for KV cache depending on batch size and context length

### Advanced Configuration

Fine-tune tensor parallelism with these parameters:

```json
{
  "tensor_parallel": {
    "communication_strategy": "compressed",
    "compression_level": 3,
    "synchronization_points": "attention_only",
    "prefetch_factor": 2
  }
}
```

Tensor parallelism enables running models that would be impossible on a single machine, with practical scalability up to 8-16 nodes depending on network performance and model architecture.

## Best Practices

### Network Configuration

- **Dedicated Network Interface**: Use a separate network interface for inter-node communication
- **Minimize Latency**: Ensure round-trip latency between nodes is under 2ms for optimal performance
- **Jumbo Frames**: Enable jumbo frames (MTU 9000) on all cluster interfaces to reduce packet overhead
- **Network QoS**: Prioritize cluster traffic over other network traffic
- **Direct Connections**: For small clusters, consider direct connections between nodes instead of through switches

### Hardware Selection

- **Homogeneous Nodes**: Use identical hardware for worker nodes when possible to avoid bottlenecks
- **GPU Compatibility**: Ensure all GPUs in the cluster are of the same architecture generation
- **Memory Bandwidth**: Prioritize GPUs with high memory bandwidth for tensor parallelism
- **NVLink/InfiniBand**: For NVIDIA setups, use GPUs with NVLink or connect nodes via InfiniBand for best performance
- **NUMA Awareness**: Configure large nodes to be NUMA-aware for optimal memory access patterns

### Cluster Configuration

- **Redundant Coordinators**: Configure at least 2 coordinator nodes for high availability
- **Backup Strategy**: Regularly backup cluster configuration files
- **Node Distribution**: For fault tolerance, place coordinator nodes on different physical servers/racks
- **Auto-Recovery**: Enable automatic recovery for worker nodes with `--auto-recovery`
- **Graceful Updates**: Use rolling updates when updating cluster software

### Monitoring and Maintenance

- **Regular Health Checks**: Schedule automatic cluster health checks every 15 minutes
- **Performance Baselines**: Establish performance baselines for different models and track deviations
- **Resource Monitoring**: Monitor GPU, memory, and network utilization consistently
- **Log Rotation**: Configure log rotation to prevent disk space issues
- **Alerting**: Set up alerts for node failures, high latency, or resource constraints

### Model Management

- **Optimal Sharding**: Find the optimal number of shards for each model through testing
- **Preloading Models**: Preload frequently used models during cluster startup
- **Load Balancing**: Distribute models across worker nodes based on their usage patterns
- **Cache Tuning**: Adjust KV cache sizes based on your specific workloads
- **Context Length**: Consider context length requirements when calculating memory needs

### Security Considerations

- **Network Isolation**: Place cluster communication on a private network
- **Access Controls**: Implement API access controls for cluster management endpoints
- **TLS Encryption**: Enable TLS for API and inter-node communication in production
- **Authentication**: Use strong authentication for coordinator nodes
- **Firewall Rules**: Restrict access to cluster ports to only necessary IP ranges

## Troubleshooting

### Common Issues and Solutions

#### Node Registration Problems

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Node Not Joining** | Error: "Failed to register with coordinator" | • Check network connectivity using `ping` and `telnet` <br>• Verify firewall allows traffic on ports 11434 and 12094 <br>• Ensure coordinator address is correct <br>• Check coordinator logs for rejection reasons |
| **Authentication Failures** | Error: "Unauthorized access" | • Verify API keys match across nodes <br>• Check if certificates are valid (for TLS setups) <br>• Ensure cluster IDs match |
| **Discovery Issues** | Error: "No nodes discovered" | • For multicast: check if multicast UDP is allowed <br>• For manual: verify node list is correct <br>• Try increasing discovery timeout |

#### Performance Problems

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Slow Inference** | High latency in responses | • Check network bandwidth between nodes (`iperf`) <br>• Consider reducing model shard count <br>• Monitor GPU utilization with `nvidia-smi` <br>• Enable compression with `--enable-compression` |
| **Memory Errors** | Error: "Out of memory" | • Check available GPU memory with `nvidia-smi` <br>• Use `memory-optimized` strategy <br>• Increase number of shards <br>• Reduce batch size or context length |
| **Node Disconnections** | Nodes disappear from cluster | • Check system resources (memory, CPU) <br>• Increase heartbeat intervals <br>• Implement backup coordinator nodes <br>• Check network stability |

#### Model Distribution Issues

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Model Loading Fails** | Error: "Failed to load model" | • Verify all worker nodes have sufficient GPU memory <br>• Check model compatibility with distributed mode <br>• Ensure network connectivity between all nodes <br>• Try fewer shards |
| **Uneven Load** | Some nodes overloaded, others idle | • Use `balanced` strategy <br>• Specify explicit shard distribution <br>• Consider worker node roles based on capacity |

### Diagnostic Commands

Use these commands to diagnose cluster issues:

```bash
# Check cluster status
ollama cluster status --detailed

# View node health
ollama cluster health

# Check logs in real-time
tail -f ~/.ollama/logs/cluster_coordinator.log

# Test connectivity between nodes
telnet <node-ip> 12094

# Test network performance
iperf3 -c <node-ip> -p 5201

# Test mDNS discovery (Linux/macOS)
avahi-browse -a | grep ollama

# Check if mDNS port is open
sudo netstat -tulpn | grep 5353
```

### Zero-Configuration Mode Specific Troubleshooting

If you're having issues with zero-configuration cluster mode:

1. **Verify mDNS is Working**
   ```bash
   # On Linux
   systemctl status avahi-daemon
   
   # On macOS
   dns-sd -B _ollama._tcp
   ```

2. **Test mDNS Discovery**
   ```bash
   # Browse for Ollama services
   avahi-browse -r _ollama._tcp
   ```

3. **Check Firewall Rules**
   ```bash
   # On Linux with firewalld
   sudo firewall-cmd --list-ports
   
   # Ensure these ports are listed:
   # 11434/tcp, 12094/tcp, 5353/udp
   ```

4. **Network Isolation Check**
   If you're on a corporate network, some security policies might block multicast traffic.
   In this case, switch to manual discovery mode:
   
   ```bash
   # Create a config file
   cat > cluster.json << EOF
   {
     "enabled": true,
     "discovery": {
       "method": "manual",
       "node_list": ["192.168.1.100:12094", "192.168.1.101:12094"]
     }
   }
   EOF
   
   # Start Ollama with manual config
   ollama serve --config cluster.json
   ```

### Log Files

Detailed cluster logs are available at:
- Coordinator logs: `~/.ollama/logs/cluster_coordinator.log`
- Worker logs: `~/.ollama/logs/cluster_worker.log`
- API server logs: `~/.ollama/logs/api_server.log`
- Model scheduler logs: `~/.ollama/logs/model_scheduler.log`
- Network traffic logs: `~/.ollama/logs/cluster_network.log` (with `--debug-network` flag)

Enable verbose logging with:
```bash
ollama cluster start --log-level debug
```

## Limitations and Known Issues

- **Network Bandwidth**: High bandwidth (>1 Gbps) required between nodes for optimal performance
- **Latency Impact**: Distributed inference adds 10-30% latency compared to single-machine operation
- **Model Compatibility**: Not all models support distributed operation; most compatible are:
  - Llama family (Llama 2, Llama 3, etc.)
  - Mistral family
  - Mixtral models
  - Gemma models
- **Shard Limits**: Maximum of 8 shards recommended for most models
- **Cross-Region**: Not optimized for cross-region deployment (high latency)
- **Heterogeneous Hardware**: Performance may be limited by the slowest node in the cluster