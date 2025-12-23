#!/usr/bin/env python3
"""
ASX Integration Bridge for K'uhul Multi Hive OS
Implements XJSON execution and KLH orchestration patterns
"""

import json
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp


class XJSONEngine:
    """
    XJSON (Executable JSON) Engine
    Executes XJSON blocks within the K'uhul Hive context
    """

    def __init__(self, ctx: Optional[Dict] = None, hive_client: Optional[Any] = None):
        self.ctx = ctx or {"input": {}, "ctx": {}, "hive": {}}
        self.hive_client = hive_client

    def _resolve(self, value: Any) -> Any:
        """Resolve $ variables from context"""
        if isinstance(value, str) and value.startswith("$"):
            path = value[1:].split(".")
            cur = self.ctx
            for p in path:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
            return cur
        return value

    def _format(self, value: Any) -> Any:
        """Format strings with context substitution"""
        if isinstance(value, str) and "$" in value:
            # Simple variable replacement
            for key in ["name", "description", "query", "response"]:
                value = value.replace(f"$input.{key}", str(self.ctx["input"].get(key, "")))
                value = value.replace(f"$ctx.{key}", str(self.ctx["ctx"].get(key, "")))
                value = value.replace(f"$hive.{key}", str(self.ctx["hive"].get(key, "")))
        return value

    # ========== XJSON Opcodes ==========

    def op_log(self, block: Dict) -> None:
        """@log - Log message"""
        msg = self._format(block.get("@log", ""))
        print(f"[XJSON LOG] {msg}")

    def op_py_jar(self, spec: Dict) -> tuple:
        """@py.jar - Execute JAR file"""
        jar = self._format(spec["jar"])
        args = [self._format(a) for a in spec.get("args", [])]
        cmd = ["java", "-jar", jar] + args

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, err = proc.communicate()

        # Capture output to context
        capture = spec.get("capture")
        if capture:
            path = capture.split(".")
            cur = self.ctx
            for p in path[:-1]:
                cur = cur.setdefault(p, {})
            cur[path[-1]] = {"stdout": out, "stderr": err}

        return out, err

    def op_py_file_write(self, spec: Dict) -> None:
        """@py.file.write - Write file"""
        path = self._format(spec["path"])
        data = self._resolve(spec["data"])
        pretty = spec.get("pretty", False)

        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, indent=2 if pretty else None)
            else:
                f.write(str(data))

    def op_py_file_read(self, spec: Dict) -> Any:
        """@py.file.read - Read file"""
        path = self._format(spec["path"])
        as_json = spec.get("json", False)

        with open(path, "r", encoding="utf-8") as f:
            if as_json:
                return json.load(f)
            else:
                return f.read()

    async def op_hive_query(self, spec: Dict) -> Dict:
        """@hive.query - Query hive agent"""
        if not self.hive_client:
            return {"error": "Hive client not available"}

        agent = spec.get("agent", "queen")
        message = self._format(spec["message"])
        context = self._format(spec.get("context", ""))

        # This would call the hive's query_agent method
        result = await self.hive_client.query_agent(agent, message, context)

        # Capture to context
        capture = spec.get("capture")
        if capture:
            path = capture.split(".")
            cur = self.ctx
            for p in path[:-1]:
                cur = cur.setdefault(p, {})
            cur[path[-1]] = result

        return result

    def op_scxq2_compress(self, spec: Dict) -> str:
        """@scxq2.compress - Symbolic compression"""
        data = self._resolve(spec["data"])
        # Simplified SCXQ2 compression (symbolic representation)
        # In production, this would use the full SCXQ2 engine
        compressed = self._scxq2_simple_compress(data)
        return compressed

    def _scxq2_simple_compress(self, data: Any) -> str:
        """Simple symbolic compression algorithm"""
        if isinstance(data, dict):
            # Create symbolic representation
            symbols = {}
            compressed = {}
            symbol_counter = 0

            for key, value in data.items():
                if isinstance(value, str) and len(value) > 10:
                    symbol = f"S{symbol_counter}"
                    symbols[symbol] = value
                    compressed[key] = f"${symbol}"
                    symbol_counter += 1
                else:
                    compressed[key] = value

            return json.dumps({"_symbols": symbols, "data": compressed})
        return json.dumps(data)

    # ========== Evaluator ==========

    def run_if_block(self, xj: Dict) -> Dict:
        """Execute @if conditional block"""
        root = xj.get("@if")
        if not root:
            return self.ctx

        cond = root.get("cond")
        left = self._resolve(cond["left"])
        right = self._resolve(cond["right"])
        op = cond.get("op", "==")

        result = False
        if op == "!=":
            result = left != right
        elif op == "==":
            result = left == right
        elif op == ">":
            result = float(left) > float(right)
        elif op == "<":
            result = float(left) < float(right)

        blocks = root["@then"] if result else root.get("@else", [])

        # Execute blocks
        for b in blocks:
            if "@log" in b:
                self.op_log(b)
            if "@py.jar" in b:
                self.op_py_jar(b["@py.jar"])
            if "@py.file.write" in b:
                self.op_py_file_write(b["@py.file.write"])
            if "@py.file.read" in b:
                data = self.op_py_file_read(b["@py.file.read"])
                self.ctx["ctx"]["lastRead"] = data

        return self.ctx

    async def run_async(self, xj: Dict) -> Dict:
        """Execute XJSON with async support"""
        if "@if" in xj:
            return self.run_if_block(xj)

        # Process async operations
        if "@hive.query" in xj:
            return await self.op_hive_query(xj["@hive.query"])

        return self.ctx


class KLHOrchestrator:
    """
    KLH (K'uhul Hive) Orchestrator
    Manages multi-hive coordination and task distribution
    """

    def __init__(self, hive_nodes: List[str] = None):
        self.hive_nodes = hive_nodes or ["http://localhost:8000"]
        self.task_queue = []
        self.results = {}

    async def distribute_task(self, task: Dict) -> List[Dict]:
        """Distribute task across multiple hive nodes"""
        results = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for node_url in self.hive_nodes:
                tasks.append(self._send_to_node(session, node_url, task))

            results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def _send_to_node(
        self,
        session: aiohttp.ClientSession,
        node_url: str,
        task: Dict
    ) -> Dict:
        """Send task to a specific hive node"""
        try:
            async with session.post(
                f"{node_url}/api/chat",
                json={"message": task.get("message", "")}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Node {node_url} returned {response.status}"}
        except Exception as e:
            return {"error": f"Node {node_url} failed: {e}"}

    def create_shard(self, shard_config: Dict) -> Dict:
        """Create a new hive shard (isolated execution context)"""
        shard_id = f"shard_{len(self.results)}"
        self.results[shard_id] = {
            "config": shard_config,
            "status": "initialized",
            "created_at": None  # Would use datetime
        }
        return {"shard_id": shard_id, "status": "created"}


class ASXTape:
    """
    ASX Tape Runtime
    Modular execution container for ASX blocks
    """

    def __init__(self, tape_config: Dict):
        self.tape_id = tape_config.get("tape_id", "tape_0")
        self.blocks = tape_config.get("blocks", [])
        self.xjson_engine = XJSONEngine()

    async def execute(self, input_data: Dict) -> Dict:
        """Execute the tape with input data"""
        self.xjson_engine.ctx["input"] = input_data
        results = []

        for block in self.blocks:
            if "xjson" in block:
                result = await self.xjson_engine.run_async(block["xjson"])
                results.append(result)

        return {
            "tape_id": self.tape_id,
            "results": results,
            "final_context": self.xjson_engine.ctx
        }


# ========== Helper Functions ==========

def run_xjson_job(path: str, input_data: Dict) -> Dict:
    """Execute XJSON from file"""
    with open(path, "r", encoding="utf-8") as f:
        xj = json.load(f)

    eng = XJSONEngine(ctx={"input": input_data, "ctx": {}})
    return eng.run_if_block(xj)


async def run_xjson_async(xjson_code: Dict, hive_client: Any = None) -> Dict:
    """Execute XJSON asynchronously"""
    eng = XJSONEngine(hive_client=hive_client)
    return await eng.run_async(xjson_code)


# ========== Example Usage ==========

if __name__ == "__main__":
    # Example XJSON
    example_xjson = {
        "@if": {
            "cond": {
                "left": "$input.type",
                "right": "test",
                "op": "=="
            },
            "@then": [
                {"@log": "Running test mode: $input.name"},
                {
                    "@py.file.write": {
                        "path": "./output/test.json",
                        "data": "$input",
                        "pretty": True
                    }
                }
            ],
            "@else": [
                {"@log": "Production mode"}
            ]
        }
    }

    # Run XJSON
    result = run_xjson_job(
        "example.json",  # Would load from file
        {"type": "test", "name": "K'uhul Test"}
    )

    print("XJSON Result:", json.dumps(result, indent=2))
