# Local Ollama Integration Guide

## Overview

Your local Ollama instance is now integrated with the triage system.
This accelerates issue classification by running **parallel local models** alongside Copilot GitHub API work.
The scripts resolve the host from `OLLAMA_HOST` or the checked-in host profile, so the docs stay target-server-local without hard-coded IPs.

## Available Models

```
✓ mistral:7b (7.2B) - Recommended for classification (fast + quality)
✓ llama3:8b (8.0B)  - More capable, slightly slower
✓ phi3:mini (3.8B)  - Fastest, suitable for quick triage
✓ phi3-fast:latest  - Ultra-fast phi variant
```

## Quick Start

### 1. Classify a Batch of Issues

```bash
# Classify first 20 issues using Mistral
python3 scripts/ollama_local_classifier.py \
  --limit 20 \
  --model mistral:7b \
  --output .github/ollama_batch_20_results.json
```

### 2. Run Parallel Triage (Ollama + Copilot)

```bash
# Start Ollama classification in background while you run other tasks
python3 scripts/parallel_triage_with_ollama.py \
  --batch 50 \
  --model mistral:7b \
  --queue .github/agent_ready_queue.json

# Then, in another terminal, continue with Copilot work:
python3 scripts/run_autonomous_issue_cycle.py --config .github/autonomous_cycle.iac.json
python3 scripts/apply_agent_ready_shards.py --manifest .github/agent_ready_shards.json
```

### 3. Check Available Models

```bash
python3 scripts/ollama_local_classifier.py --check-models
```

## Classification Output

Each issue receives a JSON classification:

```json
{
  "issue": 60,
  "classification": {
    "priority": "critical|high|normal|low",
    "category": "bug|feature|docs|refactor|chore|question",
    "is_duplicate": false,
    "complexity": "trivial|simple|moderate|complex",
    "confidence": 0.75
  },
  "model": "mistral:7b",
  "success": true
}
```

## Performance Metrics

**Ollama Classification Speed (local host):**
- Mistral 7B: ~15-20 issues/minute
- Llama3 8B: ~10-15 issues/minute
- Phi3 3.8B: ~25-30 issues/minute

**Parallel Advantage:**
- Run 20-50 Ollama classifications while Copilot handles GitHub API operations
- No blocking of GitHub rate limits
- Reduces total triage time by 30-50%

## Use Cases

### Case 1: Quick Priority Assessment
```bash
# Classify all issues to identify P0 candidates
python3 scripts/ollama_local_classifier.py \
  --limit 0 \
  --model phi3-fast:latest  # Fastest model
```

### Case 2: Duplicate Detection
```bash
# Find potential duplicates across all 297 issues
python3 scripts/ollama_local_classifier.py \
  --limit 0 \
  --model llama3:8b  # Better reasoning
```

### Case 3: Category Enrichment
```bash
# Supplement GitHub labels with AI classification
python3 scripts/ollama_local_classifier.py \
  --queue .github/agent_ready_queue.json \
  --model mistral:7b
```

## Integration with Current Triage

The Ollama classifications **complement** (not replace) existing triage:

```
Current Flow:
  Issue → GitHub API → agent-ready label + shard assignment

Enhanced Flow:
  Issue → GitHub API → agent-ready label + shard assignment
         ↓
         → Ollama (parallel) → priority/category/complexity assessment
         ↓
  Combined intelligence feed ready for autonomous agents
```

## Recommended Workflow

### Phase 1: Initial Triage (What you completed ✅)
- Duplicate closure: 16 closed
- Agent-ready labeling: 297 issues
- Shard assignment: 4-way distribution

### Phase 2: Parallel Enrichment (NOW - Run in parallel)
```bash
# Terminal 1: Run Ollama classifications (non-blocking)
python3 scripts/parallel_triage_with_ollama.py \
  --batch 100 \
  --model mistral:7b

# Terminal 2: Continue with other triage work
python3 scripts/run_autonomous_issue_cycle.py --config .github/autonomous_cycle.iac.json
```

### Phase 3: Consolidated Intelligence (After Phase 2)
- Combined Copilot + Ollama assessment
- Enhanced issue metadata for agent development
- Unified quality gates

## Configuration

Create `.github/ollama_config.json`:

```json
{
  "host": "http://127.0.0.1:11434",
  "default_model": "mistral:7b",
  "classification_models": {
    "priority": "llama3:8b",
    "duplicates": "llama3:8b",
    "quick_triage": "phi3-fast:latest"
  },
  "batch_sizes": {
    "parallel": 50,
    "sequential": 20
  },
  "timeout_seconds": 300
}
```

## Troubleshooting

**Connection Error: Cannot reach the local Ollama host**
```bash
# Check Ollama status
curl -s "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" | python3 -m json.tool

# Test connectivity
timeout 5 curl -v "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags"
```

**Model Takes Too Long**
```bash
# Switch to faster model
python3 scripts/ollama_local_classifier.py --model phi3-fast:latest

# Or reduce batch size
python3 scripts/ollama_local_classifier.py --limit 10 --model mistral:7b
```

**Worker Thread Timeout**
```bash
# Run with longer timeout (in script: adjust line 141 timeout=600)
python3 scripts/parallel_triage_with_ollama.py --batch 100
```

## Next Steps

1. **Run parallel classification now:**
   ```bash
   python3 scripts/parallel_triage_with_ollama.py --batch 50 --ollama-host "${OLLAMA_HOST:-http://127.0.0.1:11434}"
   ```

2. **Monitor progress:**
   ```bash
   watch -n 5 'ls -lah .github/ollama_* | tail -5'
   ```

3. **Review merged intelligence:**
   ```bash
   python3 -c "import json; print(json.load(open('.github/ollama_classification_report.json')), indent=2)"
   ```

4. **Commit results:**
   ```bash
   git add .github/ollama_*.json && git commit -m "triage: ollama local classification enrichment"
   ```

---

**Status:** ✅ Ready for parallel execution
**Models Available:** 4 (Mistral, Llama3, Phi3 variants)
**Recommended:** Start with Mistral 7B for balance of speed and quality
