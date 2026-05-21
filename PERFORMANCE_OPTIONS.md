# Ollama 性能优化选项

本文档说明新增的快速性能优化选项。

## 新增配置参数

### 1. NumUBatch - 微批大小

```json
{
  "num_ctx": 4096,
  "num_batch": 512,
  "num_ubatch": 32  // 新增：微批大小
}
```

**作用**：控制每批处理的 token 数量。较小的 `num_ubatch` 可以减少内存峰值使用，适合高并发场景。

**建议**：
- GPU 内存充足：`num_ubatch = num_batch`
- GPU 内存有限：`num_ubatch = num_batch / 4`
- 高并发场景：`num_ubatch = 16-32`

### 2. FlashAttention - Flash Attention 控制

```json
{
  "flash_attention": "enabled"  // 新增：enabled, disabled, auto
}
```

**作用**：启用 Flash Attention 2，减少 HBM 访问，提升 Attention 计算速度。

**选项**：
- `"enabled"` - 强制启用
- `"disabled"` - 强制禁用
- `"auto"` - 自动检测（默认）

**收益**：2-4x Attention 加速

### 3. KVCacheType - KV Cache 量化

```json
{
  "kv_cache_type": "q8_0"  // 新增：f16, q8_0, q4_0
}
```

**作用**：量化 KV Cache，减少显存占用。

**选项**：
- `"f16"` - 16 位浮点（默认，最佳精度）
- `"q8_0"` - 8 位量化（减少 50% 显存）
- `"q4_0"` - 4 位量化（减少 75% 显存，需要 Flash Attention）

**收益**：可支持更大上下文或更多并发请求

## 使用示例

### API 调用

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5",
  "prompt": "你好",
  "options": {
    "num_ctx": 8192,
    "num_ubatch": 64,
    "flash_attention": "enabled",
    "kv_cache_type": "q8_0"
  }
}'
```

### Modelfile

```dockerfile
FROM qwen2.5

# 参数设置
PARAMETER num_ctx 8192
PARAMETER num_ubatch 64
PARAMETER flash_attention enabled
PARAMETER kv_cache_type q8_0
```

### 环境变量（向后兼容）

```bash
# 仍然支持环境变量配置
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0

# API 选项会覆盖环境变量
```

## 性能调优指南

### 场景 1：单用户，最大化速度

```json
{
  "num_ctx": 4096,
  "num_batch": 512,
  "num_ubatch": 512,
  "flash_attention": "enabled",
  "num_gpu": -1
}
```

### 场景 2：多用户，平衡吞吐

```json
{
  "num_ctx": 4096,
  "num_batch": 512,
  "num_ubatch": 32,
  "flash_attention": "auto",
  "kv_cache_type": "f16"
}
```

### 场景 3：大上下文，节省显存

```json
{
  "num_ctx": 32768,
  "num_batch": 512,
  "num_ubatch": 16,
  "flash_attention": "enabled",
  "kv_cache_type": "q8_0"
}
```

## 验证效果

### 检查配置是否生效

```bash
# 启动时查看日志
ollama serve

# 应该看到类似输出：
# using micro batch size num_ubatch=32
# flash attention enabled via API option
```

### 性能基准测试

```bash
# 测试生成速度
time ollama run qwen2.5 "写一首诗"

# 测试并发性能
for i in {1..10}; do
  ollama run qwen2.5 "hello" &
done
wait
```

## 注意事项

1. **Flash Attention 要求**：
   - CUDA GPU: Compute Capability 7.5+ (Turing+)
   - 金属 (Metal): M1/M2/M3 芯片
   - ROCm: 部分支持

2. **KV Cache 量化**：
   - `q4_0` 必须配合 Flash Attention
   - 可能轻微影响输出质量
   - 建议先测试 `q8_0`

3. **NumUBatch 限制**：
   - 不能超过 `num_batch`
   - 过小可能影响吞吐量
   - 建议值：16-128

### 5. ContinuousBatch - 连续批处理

```json
{
  "continuous_batch": true  // 新增：启用连续批处理 (默认: parallel > 1 时自动启用)
}
```

**作用**：动态合并多个序列到同一个 batch，优先处理 decode 序列（低延迟），公平分配 prefill 容量。

**策略**：
- Decode 序列（生成 token）优先，每个占 1 个 slot
- Prefill 序列（处理 prompt）公平分享剩余容量
- 单序列时自动回退到全 batch（零开销）

**建议**：
- 单用户：自动启用（parallel=1 时无影响）
- 多用户：结合 `OLLAMA_NUM_PARALLEL=4` 或更高效果最佳
- 需关闭：`"continuous_batch": false`

**收益**：多用户场景下 2-5x 吞吐量提升，降低 time-to-first-token

### 场景 4：多用户，连续批处理

```json
{
  "num_ctx": 4096,
  "num_batch": 512,
  "num_ubatch": 64,
  "flash_attention": "enabled",
  "kv_cache_type": "q8_0",
  "continuous_batch": true
}
```

配合环境变量：
```bash
export OLLAMA_NUM_PARALLEL=8
```

## 下一步优化

完成这些快速优化后，可以考虑：

1. **PagedAttention** - vLLM 风格的内存管理
2. **Prefill/Decode 分离** - 避免长请求阻塞短请求
3. **Speculative Decoding** - 使用 draft model 加速生成
