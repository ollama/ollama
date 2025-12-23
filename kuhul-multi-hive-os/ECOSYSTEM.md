# 🌐 K'uhul Ecosystem Integration

K'uhul Multi Hive OS is the **execution core** of the ASX Language Framework ecosystem, integrating seamlessly with all ASX components.

---

## 🏗️ ASX Ecosystem Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ASX ECOSYSTEM                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │            K'UHUL ENGINE (This Project)              │      │
│  │         Multi-Agent Execution Core                   │      │
│  └───────────────────┬──────────────────────────────────┘      │
│                      │                                          │
│         ┌────────────┼────────────┬────────────┐               │
│         │            │            │            │               │
│    ┌────▼────┐  ┌───▼────┐  ┌───▼────┐  ┌───▼────┐           │
│    │ XJSON   │  │  KLH   │  │ SCXQ2  │  │  TAPE  │           │
│    │ Runtime │  │  Orch. │  │ Engine │  │Runtime │           │
│    └─────────┘  └────────┘  └────────┘  └────────┘           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              NPM Package Ecosystem                    │      │
│  │  • @xjson/klh-orchestrator                           │      │
│  │  • @xjson/xjson-server                               │      │
│  │  • @xjson/asx-blocks-core                            │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Integration with ASX Packages

### 1. **@xjson/klh-orchestrator**

K'uhul implements the KLH (K'uhul Hive) orchestration protocol:

```python
from integration.asx_bridge import KLHOrchestrator

# Connect to multiple K'uhul hive nodes
orchestrator = KLHOrchestrator(hive_nodes=[
    "http://localhost:8000",      # Primary hive
    "http://hive2.local:8000",    # Secondary hive
    "http://hive3.local:8000"     # Tertiary hive
])

# Distribute task across all hives
results = await orchestrator.distribute_task({
    "message": "Analyze this dataset across all hives",
    "data": large_dataset,
    "strategy": "parallel"
})
```

**NPM Package**: [@xjson/klh-orchestrator](https://www.npmjs.com/package/@xjson/klh-orchestrator)

---

### 2. **@xjson/xjson-server**

K'uhul executes XJSON workflows using the ASX bridge:

```python
from integration.asx_bridge import run_xjson_async

# Execute XJSON workflow
xjson_workflow = {
    "@hive.query": {
        "agent": "coder",
        "message": "Generate a REST API in Python",
        "capture": "ctx.generated_code"
    }
}

result = await run_xjson_async(xjson_workflow, hive_client)
print(result['ctx']['generated_code'])
```

**NPM Package**: [@xjson/xjson-server](https://www.npmjs.com/package/@xjson/xjson-server)

---

### 3. **@xjson/asx-blocks-core**

K'uhul uses ASX Blocks for UI and logic components:

```python
# Import ASX blocks
from integration.asx_bridge import ASXTape

# Create a tape with ASX blocks
tape = ASXTape({
    "tape_id": "data-analysis-tape",
    "blocks": [
        {
            "xjson": {
                "@hive.query": {
                    "agent": "analyst",
                    "message": "Analyze sales data for Q4"
                }
            }
        }
    ]
})

# Execute the tape
result = await tape.execute({"quarter": "Q4", "year": 2024})
```

**NPM Package**: [@xjson/asx-blocks-core](https://www.npmjs.com/package/@xjson/asx-blocks-core)

---

## 🔌 Integration Points

### REST API Integration

K'uhul exposes a RESTful API that can be consumed by any ASX application:

```javascript
// Node.js example using fetch
const response = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    message: "Analyze user behavior patterns"
  })
});

const result = await response.json();
console.log(result.response);
```

### WebSocket Streaming (Planned)

```javascript
// Future WebSocket support
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Agent ${data.agent}: ${data.response}`);
};

ws.send(JSON.stringify({
  message: "Generate a complex algorithm",
  stream: true
}));
```

---

## 🌟 Use Cases

### 1. **XJSON Application Server**

Use K'uhul as the backend for XJSON applications:

```json
{
  "@tape": {
    "tape_id": "user-onboarding",
    "@blocks": [
      {
        "@hive.query": {
          "agent": "memory",
          "message": "Check if user $input.email exists"
        }
      },
      {
        "@if": {
          "cond": {"left": "$ctx.user_exists", "right": false, "op": "=="},
          "@then": [
            {
              "@hive.query": {
                "agent": "creative",
                "message": "Generate personalized welcome message"
              }
            }
          ]
        }
      }
    ]
  }
}
```

---

### 2. **Multi-Hive Distributed Computing**

Scale across multiple K'uhul instances:

```python
from integration.asx_bridge import KLHOrchestrator

# Create a distributed hive cluster
cluster = KLHOrchestrator([
    "http://hive-1:8000",
    "http://hive-2:8000",
    "http://hive-3:8000",
    "http://hive-4:8000"
])

# Process large dataset in parallel
shards = split_data(large_dataset, num_shards=4)

tasks = [
    {"node": f"hive-{i+1}", "data": shard}
    for i, shard in enumerate(shards)
]

results = await cluster.parallel_process(tasks)
final_result = merge_results(results)
```

---

### 3. **SCXQ2 Data Compression**

Use K'uhul's SCXQ2 engine for symbolic compression:

```python
from integration.asx_bridge import XJSONEngine

engine = XJSONEngine()

# Compress large data
compressed = engine.op_scxq2_compress({
    "data": {
        "user_data": large_user_dataset,
        "analytics": analytics_data
    }
})

# Result is symbolically compressed
print(compressed)  # {"_symbols": {...}, "data": {...}}
```

---

## 📡 Network Protocols

### K'uhul Protocol (KP)

K'uhul implements the KP protocol for inter-hive communication:

**Message Format:**
```json
{
  "protocol": "KP/1.0",
  "hive_id": "kuhul-primary",
  "message_type": "query",
  "payload": {
    "agent": "queen",
    "message": "Coordinate task across hives",
    "priority": 8
  },
  "timestamp": "2024-11-19T12:00:00Z"
}
```

**Response Format:**
```json
{
  "protocol": "KP/1.0",
  "hive_id": "kuhul-primary",
  "message_type": "response",
  "payload": {
    "agent": "queen",
    "response": "Task coordinated successfully",
    "agents_activated": ["queen", "coder", "analyst"]
  },
  "timestamp": "2024-11-19T12:00:05Z"
}
```

---

## 🔗 External Integrations

### Docker Compose Setup

Deploy K'uhul with the full ASX stack:

```yaml
version: '3.8'

services:
  kuhul-hive:
    image: kuhul/multi-hive-os:latest
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - HIVE_ID=kuhul-primary
    depends_on:
      - ollama
      - xjson-server

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  xjson-server:
    image: xjson/xjson-server:latest
    ports:
      - "3000:3000"
    environment:
      - KUHUL_API=http://kuhul-hive:8000

volumes:
  ollama_data:
```

---

## 🚀 Getting Started with Ecosystem

### 1. Install K'uhul

```bash
git clone https://github.com/cannaseedus-bot/KUHUL.git
cd KUHUL
pip install -r backend/requirements.txt
./start_hive.sh
```

### 2. Install ASX Packages

```bash
npm install @xjson/klh-orchestrator
npm install @xjson/xjson-server
npm install @xjson/asx-blocks-core
```

### 3. Create Your First Integration

```javascript
// app.js - Node.js integration
const KLH = require('@xjson/klh-orchestrator');

const hive = new KLH.HiveClient({
  host: 'http://localhost:8000',
  hiveId: 'kuhul-primary'
});

async function queryHive() {
  const result = await hive.query({
    agent: 'queen',
    message: 'Analyze system performance'
  });

  console.log(`Response: ${result.response}`);
  console.log(`Agents used: ${result.agents_used.join(', ')}`);
}

queryHive();
```

---

## 📚 Learn More

- **ASX Framework**: [github.com/cannaseedus-bot/asx-language-framework](https://github.com/cannaseedus-bot/asx-language-framework)
- **KLH Orchestrator**: [@xjson/klh-orchestrator](https://www.npmjs.com/package/@xjson/klh-orchestrator)
- **XJSON Server**: [@xjson/xjson-server](https://www.npmjs.com/package/@xjson/xjson-server)
- **ASX Blocks**: [@xjson/asx-blocks-core](https://www.npmjs.com/package/@xjson/asx-blocks-core)
- **K'uhul Repository**: [github.com/cannaseedus-bot/KUHUL](https://github.com/cannaseedus-bot/KUHUL)

---

**🛸 K'uhul - The Heart of the ASX Ecosystem 🐝**
