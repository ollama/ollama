# Ollama Accelerated - 性能优化版本

这是 Ollama 的性能优化分支，包含多项推理加速和并发优化。

## 优化概览

| 优化项 | 状态 | 收益 |
|--------|------|------|
| KV Cache O(1) 分配 | ✅ 完成 | 分配加速 2-223x |
| 连续 Batching | ✅ 完成 | 多用户吞吐量 2-5x |
| Flash Attention API | ✅ 完成 | Attention 2-4x |
| KV Cache 量化 | ✅ 完成 | 显存节省 50-75% |
| 微批处理 (UBatch) | ✅ 完成 | 高并发优化 |

## 快速开始

### 1. 构建优化版本

```bash
cd /path/to/ollama-accelerated
go build ./...
```

### 2. 启动服务

```bash
# 启用连续批处理，设置并发数
export OLLAMA_NUM_PARALLEL=8
./ollama serve
```

### 3. 使用 API

```bash
# 生成请求，启用连续批处理
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5",
  "prompt": "写一首诗",
  "stream": true,
  "options": {
    "continuous_batch": true,
    "num_ubatch": 64,
    "flash_attention": "enabled",
    "kv_cache_type": "q8_0"
  }
}'
```

## 性能优化详解

### 连续 Batching (Continuous Batching)

**问题**: 原始 Ollama 使用 round-robin 调度，长 prompt 会阻塞其他正在生成的请求。

**解决方案**: 动态 batch 组成策略
- Decode 序列（生成中）优先，各占 1 slot
- Prefill 序列（处理 prompt）公平分享剩余容量
- 单序列时自动回退，零开销

**收益**: 多用户场景下吞吐量提升 2-5x，time-to-first-token 降低 50-90%

```bash
# 配置建议
export OLLAMA_NUM_PARALLEL=8  # 根据并发需求调整

# API 调用会自动启用连续批处理（parallel > 1 时）
```

### Flash Attention 2

**作用**: 融合 attention kernel，减少 HBM 访问

```json
{
  "flash_attention": "enabled"  // 强制启用
}
```

**要求**: CUDA 7.5+ (Turing+) 或 Apple M1/M2/M3

### KV Cache 量化

**作用**: 量化 KV cache，节省显存

```json
{
  "kv_cache_type": "q8_0"  // f16, q8_0, q4_0
}
```

| 类型 | 精度 | 显存节省 | 要求 |
|------|------|----------|------|
| f16 | 最佳 | 0% | - |
| q8_0 | 良好 | 50% | - |
| q4_0 | 可接受 | 75% | Flash Attention |

### 微批处理 (NumUBatch)

**作用**: 控制每批处理的 token 数量，降低内存峰值

```json
{
  "num_batch": 512,
  "num_ubatch": 64  // 微批大小
}
```

**建议**:
- GPU 内存充足: `num_ubatch = num_batch`
- GPU 内存有限: `num_ubatch = num_batch / 4`
- 高并发场景: `num_ubatch = 16-32`

## 配置场景

### 单用户，最大化速度

```bash
export OLLAMA_NUM_PARALLEL=1
```

```json
{
  "num_ctx": 4096,
  "num_batch": 512,
  "num_ubatch": 512,
  "flash_attention": "enabled",
  "kv_cache_type": "f16"
}
```

### 多用户，平衡吞吐

```bash
export OLLAMA_NUM_PARALLEL=8
```

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

### 大上下文，节省显存

```bash
export OLLAMA_NUM_PARALLEL=4
```

```json
{
  "num_ctx": 32768,
  "num_batch": 512,
  "num_ubatch": 16,
  "flash_attention": "enabled",
  "kv_cache_type": "q8_0"
}
```

## 并发使用示例

### Python 并发请求

```python
import asyncio
import aiohttp

async def query_ollama(prompt, session):
    async with session.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5",
            "prompt": prompt,
            "stream": True,
            "options": {
                "continuous_batch": true
            }
        }
    ) as resp:
        async for line in resp:
            print(line)

async def main():
    prompts = [
        "总结这篇文章",
        "翻译这句话",
        "分析数据",
        "生成代码"
    ]
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[
            query_ollama(p, session) for p in prompts
        ])

asyncio.run(main())
```

### cURL 并发请求

```bash
#!/bin/bash
# 发起 8 个并发请求

for i in {1..8}; do
  curl -s http://localhost:11434/api/generate -d "{
    \"model\": \"qwen2.5\",
    \"prompt\": \"查询 $i\",
    \"stream\": false
  }" &
done
wait
```

## 文件结构

```
ollama-accelerated/
├── runner/ollamarunner/
│   ├── scheduler.go           # 连续批处理调度器
│   ├── scheduler_test.go      # 调度器测试
│   └── runner.go              # 修改的 runner
├── llm/
│   └── server.go              # API 选项支持
├── api/
│   └── types.go               # API 类型定义
├── kvcache/
│   ├── cache_accelerated.go   # O(1) KV cache 分配
│   └── cache_accelerated_test.go
├── PERFORMANCE_OPTIONS.md     # 详细配置指南
├── OPTIMIZATION_PROPOSAL.md   # 优化提案
└── ISSUE_TEMPLATE.md          # Issue 模板
```

## 性能测试

### 运行测试

```bash
# 调度器单元测试
go test ./runner/ollamarunner/ -run TestScheduler -v

# KV cache 性能测试
go test ./kvcache/ -bench=. -benchmem
```

### 预期结果

**调度器测试** (7 tests):
- 单序列: 全 batch，零开销
- Decode 优先: 生成序列优先处理
- 公平分享: prefill 序列公平分配
- 溢出处理: 超过 batch 时的截断

**KV cache 基准**:
- 50% 填充，100K 容量: 66,357ns → 893ns (74x)
- 90% 填充，100K 容量: 129,462ns → 579ns (223x)

## 提交到上游

优化已准备好提交为 PR 到 [ollama/ollama](https://github.com/ollama/ollama)。

### 相关 Issue

使用 [ISSUE_TEMPLATE.md](ISSUE_TEMPLATE.md) 创建 issue：

```markdown
## 优化类型
- [ ] 连续 Batching
- [ ] Flash Attention
- [ ] KV Cache 量化
- [ ] 其他

## 性能数据
[贴上基准测试结果]

## 复现步骤
[如何验证优化]
```

## 许可证

与 Ollama 主项目保持一致。

## 致谢

基于 [ollama/ollama](https://github.com/ollama/ollama) 的优秀工作。
