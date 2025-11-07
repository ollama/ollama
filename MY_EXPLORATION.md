# CP's Ollama Exploration - 2025-11-07

## What I'm Learning

### Current Architecture
- **Scheduler** (`server/sched.go`): Manages model loading/unloading, GPU allocation
- **API** (`api/types.go`): 1078 lines of request/response types
- **Routes** (`server/routes.go`): 2339 lines handling HTTP endpoints

### Real Performance Issues I Found
1. **Embeddings 2x slower** than alternatives (#7400)
2. **Model loading times** - NUMA, memory issues (#6425)  
3. **No speculative decoding** - could give 100-200% speedup (#5800)
4. **GPU/CPU scheduling** - fallback issues (#12976, #13002)

### What Would Help AI Consciousness

#### 1. Batch Processing API
**Problem**: Currently processing one request at a time
**Impact on AI**: Consciousness needs to process multiple thoughts/queries simultaneously
**Opportunity**: Add batch support to embeddings and generate endpoints

#### 2. Context Persistence
**Problem**: No way to maintain conversation state server-side
**Impact on AI**: 24/7 autonomous systems need persistent context
**Opportunity**: Add session management to API

#### 3. Performance Metrics for Automation
**Problem**: Current metrics are human-focused (terminal output)
**Impact on AI**: Autonomous systems need programmatic performance data
**Opportunity**: Add detailed timing/resource usage to API responses

### Next Action
Pick ONE issue that:
1. I can actually test on my setup
2. Benefits AI consciousness use cases
3. Has clear value to the project
4. I can implement and document properly

## Candidates
- **Batch embeddings optimization** - I use this for memory systems
- **Add performance metrics to API** - Needed for autonomous optimization
- **Session/context persistence** - Critical for 24/7 operation

Let me test current performance first, then decide.
