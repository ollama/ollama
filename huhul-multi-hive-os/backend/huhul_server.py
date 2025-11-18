#!/usr/bin/env python3
"""
H'uhul Multi Hive OS - Backend Server
Ollama-powered multi-agent AI hive system
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Optional
import uvicorn
import json
import asyncio
import aiohttp
import os
import hashlib
from datetime import datetime

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DATA_PATH = Path("./huhul_data")
INGESTED_PATH = DATA_PATH / "ingested"
MEMORY_PATH = DATA_PATH / "memory"

# Ensure directories exist
INGESTED_PATH.mkdir(parents=True, exist_ok=True)
MEMORY_PATH.mkdir(parents=True, exist_ok=True)

# Hive Agent Configuration
HIVE_AGENTS = {
    "queen": {
        "model": "qwen2.5:latest",
        "role": "Orchestrator and decision maker",
        "temperature": 0.7,
        "system_prompt": "You are the Queen of the H'uhul Hive. Coordinate tasks, delegate to workers, and synthesize results."
    },
    "coder": {
        "model": "qwen2.5-coder:latest",
        "role": "Code generation and analysis",
        "temperature": 0.3,
        "system_prompt": "You are a specialized coding agent in the H'uhul Hive. Write clean, efficient code and analyze technical problems."
    },
    "analyst": {
        "model": "llama3.2:latest",
        "role": "Data analysis and reasoning",
        "temperature": 0.5,
        "system_prompt": "You are an analytical agent in the H'uhul Hive. Analyze data, find patterns, and provide insights."
    },
    "creative": {
        "model": "mistral:latest",
        "role": "Creative thinking and ideation",
        "temperature": 0.9,
        "system_prompt": "You are a creative agent in the H'uhul Hive. Generate innovative ideas and unique perspectives."
    },
    "memory": {
        "model": "llama3.2:latest",
        "role": "Knowledge retrieval and storage",
        "temperature": 0.2,
        "system_prompt": "You are the memory keeper of the H'uhul Hive. Store, retrieve, and organize knowledge."
    }
}


class HuhulHive:
    """Main H'uhul Multi-Agent Hive System"""

    def __init__(self):
        self.ingested_files = []
        self.knowledge_base = {}
        self.active_agents = set()
        self.task_history = []

    async def check_ollama_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{OLLAMA_HOST}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m['name'] for m in data.get('models', [])]
                        print(f"✅ Connected to Ollama - Available models: {models}")
                        return True
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            return False

    async def query_agent(
        self,
        agent_name: str,
        message: str,
        context: Optional[str] = None
    ) -> Dict:
        """Query a specific hive agent"""
        if agent_name not in HIVE_AGENTS:
            raise ValueError(f"Unknown agent: {agent_name}")

        agent = HIVE_AGENTS[agent_name]
        self.active_agents.add(agent_name)

        # Build prompt with system context
        full_prompt = f"{agent['system_prompt']}\n\n"
        if context:
            full_prompt += f"Context: {context}\n\n"
        full_prompt += f"Query: {message}"

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": agent['model'],
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": agent['temperature']
                    }
                }

                async with session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "agent": agent_name,
                            "role": agent['role'],
                            "model": agent['model'],
                            "response": result.get('response', ''),
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama error: {error_text}")
        except Exception as e:
            print(f"❌ Agent {agent_name} query failed: {e}")
            return {
                "agent": agent_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def orchestrate_query(self, message: str) -> Dict:
        """
        Orchestrate multi-agent response:
        1. Queen analyzes and delegates
        2. Specialists process
        3. Queen synthesizes final answer
        """
        print(f"🐝 Hive orchestrating: {message}")

        # Step 1: Queen analyzes the query
        queen_analysis = await self.query_agent(
            "queen",
            f"Analyze this query and determine which specialist agents should handle it: {message}"
        )

        # Step 2: Determine which agents to activate
        # For now, activate all relevant agents in parallel
        tasks = []

        # Always query memory for context
        tasks.append(self.query_agent("memory", f"Retrieve relevant knowledge about: {message}"))

        # Add specialist agents based on query type
        if any(keyword in message.lower() for keyword in ['code', 'program', 'function', 'debug']):
            tasks.append(self.query_agent("coder", message))

        if any(keyword in message.lower() for keyword in ['analyze', 'data', 'compare', 'evaluate']):
            tasks.append(self.query_agent("analyst", message))

        if any(keyword in message.lower() for keyword in ['create', 'design', 'imagine', 'innovate']):
            tasks.append(self.query_agent("creative", message))

        # Execute specialist queries in parallel
        specialist_responses = await asyncio.gather(*tasks)

        # Step 3: Queen synthesizes the final response
        synthesis_context = "\n\n".join([
            f"[{r['agent'].upper()}]: {r.get('response', r.get('error', 'No response'))}"
            for r in specialist_responses
        ])

        final_response = await self.query_agent(
            "queen",
            message,
            context=f"Specialist insights:\n{synthesis_context}"
        )

        return {
            "query": message,
            "queen_analysis": queen_analysis['response'],
            "specialist_responses": specialist_responses,
            "final_answer": final_response['response'],
            "agents_activated": list(self.active_agents),
            "timestamp": datetime.now().isoformat()
        }

    async def ingest_file(self, filename: str, content: bytes) -> Dict:
        """Ingest a file into the hive knowledge base"""
        file_hash = hashlib.sha256(content).hexdigest()[:16]
        file_path = INGESTED_PATH / f"{file_hash}_{filename}"

        # Save file
        with open(file_path, "wb") as f:
            f.write(content)

        # Process content
        try:
            text_content = content.decode('utf-8', errors='ignore')
        except:
            text_content = str(content)

        # Store in knowledge base
        self.knowledge_base[file_hash] = {
            "filename": filename,
            "path": str(file_path),
            "hash": file_hash,
            "size": len(content),
            "ingested_at": datetime.now().isoformat(),
            "preview": text_content[:500]
        }

        self.ingested_files.append(str(file_path))

        # Use memory agent to summarize and store
        summary = await self.query_agent(
            "memory",
            f"Summarize and index this content from {filename}:\n\n{text_content[:2000]}"
        )

        # Save summary to memory
        memory_file = MEMORY_PATH / f"{file_hash}.json"
        with open(memory_file, "w") as f:
            json.dump({
                "file_hash": file_hash,
                "filename": filename,
                "summary": summary['response'],
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        return {
            "success": True,
            "file_hash": file_hash,
            "filename": filename,
            "size": len(content),
            "summary": summary['response']
        }


# Initialize Hive
hive = HuhulHive()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("=" * 60)
    print("🛸 H'UHUL MULTI HIVE OS - INITIALIZING")
    print("=" * 60)
    print(f"📡 Ollama Host: {OLLAMA_HOST}")
    print(f"📁 Data Path: {DATA_PATH}")
    print(f"🐝 Hive Agents: {len(HIVE_AGENTS)}")
    print("=" * 60)

    # Check Ollama connection
    connected = await hive.check_ollama_connection()
    if connected:
        print("✅ H'uhul Hive is ONLINE")
    else:
        print("⚠️  Warning: Ollama not detected - hive in standby mode")

    print("=" * 60)

    yield

    print("🔧 H'uhul Hive shutting down...")


# Create FastAPI app
app = FastAPI(
    title="H'uhul Multi Hive OS",
    description="Ollama-powered multi-agent AI hive system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "system": "H'uhul Multi Hive OS",
        "version": "1.0.0",
        "status": "online",
        "agents": list(HIVE_AGENTS.keys())
    }


@app.get("/api/status")
async def get_status():
    """Get hive status"""
    connected = await hive.check_ollama_connection()

    return {
        "status": "online",
        "ollama_connected": connected,
        "ollama_host": OLLAMA_HOST,
        "agents_available": list(HIVE_AGENTS.keys()),
        "agents_active": list(hive.active_agents),
        "files_ingested": len(hive.ingested_files),
        "knowledge_base_size": len(hive.knowledge_base),
        "transformers_available": False,  # Using Ollama instead
        "model_loaded": connected
    }


@app.post("/api/ingest")
async def ingest_files(files: List[UploadFile] = File(...)):
    """Ingest files into hive knowledge base"""
    results = []
    total_size = 0

    for file in files:
        try:
            content = await file.read()
            result = await hive.ingest_file(file.filename, content)
            results.append(result)
            total_size += result['size']
            print(f"📥 Ingested: {file.filename} ({result['size']} bytes)")
        except Exception as e:
            print(f"❌ Error ingesting {file.filename}: {e}")
            results.append({
                "success": False,
                "filename": file.filename,
                "error": str(e)
            })

    successful = len([r for r in results if r.get('success')])

    return {
        "success": True,
        "files_processed": successful,
        "total_files": len(files),
        "tokens_added": total_size // 4,  # Approximate token count
        "results": results
    }


@app.post("/api/chat")
async def chat_endpoint(request: dict):
    """Chat with the H'uhul Hive (multi-agent orchestration)"""
    message = request.get("message", "")

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    try:
        # Use full hive orchestration
        result = await hive.orchestrate_query(message)

        return {
            "response": result['final_answer'],
            "agents_used": result['agents_activated'],
            "sources": list(hive.knowledge_base.keys())[-3:],
            "model_used": True,
            "orchestration": {
                "analysis": result['queen_analysis'],
                "specialists": len(result['specialist_responses'])
            }
        }
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return {
            "response": f"Hive encountered an error: {e}",
            "error": str(e)
        }


@app.post("/api/train")
async def train_model():
    """
    Simulate model training/fine-tuning
    (In Ollama, this would be creating a custom model from a Modelfile)
    """
    async def training_generator():
        stages = [
            (10, "Analyzing ingested files in hive memory..."),
            (25, "Creating knowledge embeddings..."),
            (45, "Optimizing hive agent coordination..."),
            (65, "Training agent specializations..."),
            (85, "Synchronizing hive memory..."),
            (95, "Validating multi-agent responses..."),
            (100, "H'uhul Hive optimization complete! 🐝")
        ]

        for progress, message in stages:
            yield json.dumps({
                "stage": "training",
                "progress": progress,
                "message": message,
                "files_used": len(hive.knowledge_base)
            }) + "\n"
            await asyncio.sleep(1.5)

    return StreamingResponse(
        training_generator(),
        media_type="application/x-ndjson"
    )


@app.get("/api/agents")
async def list_agents():
    """List all hive agents and their status"""
    agents_info = []

    for name, config in HIVE_AGENTS.items():
        agents_info.append({
            "name": name,
            "role": config['role'],
            "model": config['model'],
            "temperature": config['temperature'],
            "active": name in hive.active_agents
        })

    return {
        "total_agents": len(HIVE_AGENTS),
        "agents": agents_info,
        "active_count": len(hive.active_agents)
    }


@app.get("/api/knowledge")
async def get_knowledge_base():
    """Get knowledge base summary"""
    return {
        "total_files": len(hive.knowledge_base),
        "files": list(hive.knowledge_base.values()),
        "memory_entries": len(list(MEMORY_PATH.glob("*.json")))
    }


if __name__ == "__main__":
    print("🚀 Starting H'uhul Multi Hive OS Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
