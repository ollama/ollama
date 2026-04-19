#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "transformers",
#   "jinja2",
#   "mcp",
# ]
# ///
"""
HuggingFace Prompt Renderer MCP Server

Model Context Protocol (MCP) server for rendering conversation messages into
model-specific prompt strings using HuggingFace tokenizer chat templates.

Usage:
    # Run MCP server over stdio
    uv run cmd/prompt-rendering/server.py --mcp

    # Start FastAPI server for manual testing
    uv run cmd/prompt-rendering/server.py --host 0.0.0.0 --port 8000

    # Test with curl
    curl -X POST http://localhost:8000/generate-prompt \\
      -H "Content-Type: application/json" \\
      -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
"""

from typing import Any, Dict, List, Optional

import argparse
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer

try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    FastMCP = None

# Cache for tokenizers to avoid reloading
_tokenizer_cache: Dict[str, Any] = {}


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    functions: Optional[str] = None  # For OLMo-style function passing
    function_calls: Optional[str] = None  # For OLMo-style function call results


class GeneratePromptRequest(BaseModel):
    messages: List[Message]
    model: str
    tools: Optional[List[Dict[str, Any]]] = None
    # Whether to inject tools into system message as 'functions' key (for OLMo-style templates)
    inject_tools_as_functions: Optional[bool] = True


class GeneratePromptResponse(BaseModel):
    prompt: str
    model: str


# FastAPI app
app = FastAPI(title="HuggingFace Prompt Generator", version="1.0.0")


def get_tokenizer(model_name: str) -> Any:
    """Get or create tokenizer for the given model."""
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    return _tokenizer_cache[model_name]


def is_deepseek_model(model_name: str) -> bool:
    """Check if this is a DeepSeek model."""
    return "deepseek" in model_name.lower()


def normalize_messages(
    raw_messages: List[Any],
    tools: Optional[List[Dict[str, Any]]],
    inject_tools_as_functions: bool,
    model: str,
) -> List[Dict[str, Any]]:
    """Normalize messages for different chat template formats."""
    messages: List[Dict[str, Any]] = []
    tools_json = json.dumps(tools) if tools else None
    is_deepseek = is_deepseek_model(model)

    for msg in raw_messages:
        message = msg if isinstance(msg, Message) else Message(**msg)
        message_dict: Dict[str, Any] = {"role": message.role, "content": None}

        if message.content is not None:
            message_dict["content"] = message.content

        # Handle explicit functions field (OLMo-style)
        if message.functions is not None:
            message_dict["functions"] = message.functions
        # Inject tools into system message as 'functions' (for OLMo templates)
        elif inject_tools_as_functions and message.role == "system" and tools_json:
            message_dict["functions"] = tools_json

        # Handle explicit function_calls field (OLMo-style)
        if message.function_calls is not None:
            message_dict["function_calls"] = message.function_calls
        # Convert tool_calls for templates
        elif message.tool_calls is not None:
            if is_deepseek:
                # DeepSeek format: arguments must be a JSON string
                tool_calls = []
                for tool_call in message.tool_calls:
                    tc = {
                        "type": "function",
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": json.dumps(tool_call["function"]["arguments"])
                            if isinstance(tool_call["function"]["arguments"], dict)
                            else tool_call["function"]["arguments"],
                        },
                    }
                    tool_calls.append(tc)
                message_dict["tool_calls"] = tool_calls
            elif inject_tools_as_functions:
                # Convert to OLMo function_calls format
                message_dict["function_calls"] = json.dumps(message.tool_calls)
            else:
                # Standard transformers format
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_call_copy = tool_call.copy()
                    if (
                        "function" in tool_call_copy
                        and "arguments" in tool_call_copy["function"]
                    ):
                        try:
                            tool_call_copy["function"]["arguments"] = json.loads(
                                tool_call_copy["function"]["arguments"]
                            )
                        except (json.JSONDecodeError, TypeError):
                            pass
                    tool_calls.append(tool_call_copy)
                message_dict["tool_calls"] = tool_calls

        if message.tool_call_id is not None:
            message_dict["tool_call_id"] = message.tool_call_id

        messages.append(message_dict)

    return messages


def build_prompt(
    raw_messages: List[Any],
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    inject_tools_as_functions: bool,
) -> str:
    """Build prompt from messages using the model's chat template."""
    messages = normalize_messages(
        raw_messages=raw_messages,
        tools=tools,
        inject_tools_as_functions=inject_tools_as_functions,
        model=model,
    )

    tokenizer = get_tokenizer(model)

    # For OLMo-style templates, don't pass tools separately (they're in messages)
    if tools and not inject_tools_as_functions:
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


@app.post("/generate-prompt", response_model=GeneratePromptResponse)
async def generate_prompt(request: GeneratePromptRequest):
    """
    Generate a prompt from messages using the specified model's chat template.
    Optionally includes tool definitions if provided.
    """
    try:
        prompt = build_prompt(
            raw_messages=request.messages,
            model=request.model,
            tools=request.tools,
            inject_tools_as_functions=request.inject_tools_as_functions,
        )
        return GeneratePromptResponse(prompt=prompt, model=request.model)

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate prompt: {str(e)}",
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if FastMCP is not None:
    mcp = FastMCP("huggingface-prompt-renderer")

    @mcp.tool()
    def generate_prompt_tool(
        messages: List[Dict[str, Any]],
        model: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        tools: Optional[List[Dict[str, Any]]] = None,
        inject_tools_as_functions: bool = True,
    ) -> Dict[str, str]:
        """
        Render conversation messages into a model-specific prompt string using HuggingFace tokenizer chat templates.

        This tool takes a list of message objects and applies the target model's chat template to produce
        the exact prompt string that would be fed to the model. It handles various message formats including
        standard OpenAI-style, OLMo-style (functions/function_calls), and DeepSeek-specific formatting.

        Use this tool to:
        - Verify that a model's chat template correctly formats your conversation
        - Test edge cases: tool calling, tool responses, interleaved thinking and tool calls, multiple tools in single response
        - Compare prompt output across different models to understand template differences
        - Debug issues with message formatting that cause unexpected model behavior

        Message format supports:
        - role: "user", "assistant", "system", "tool"
        - content: string content of the message
        - tool_calls: list of tool call objects (OpenAI format: {type, function: {name, arguments}})
        - tool_call_id: for tool role messages, references the call being responded to
        - functions: optional field for OLMo-style tool definitions
        - function_calls: optional field for OLMo-style tool call results

        Parameters:
        - messages: List of message dictionaries forming the conversation
        - model: HuggingFace model identifier (default: Qwen/Qwen3-Coder-480B-A35B-Instruct)
        - tools: Optional list of tool/function definitions for function calling models
        - inject_tools_as_functions: If True, injects tools into system message as 'functions' key (OLMo-style). If False, passes tools separately to apply_chat_template.

        Returns: Dictionary with 'prompt' (rendered string) and 'model' keys.

        Recommended test cases:
        1. Simple conversation: user -> assistant
        2. Tool calling: user -> assistant with tool_call -> tool response -> assistant
        3. Multiple tool calls in one assistant message
        4. Multiple tool responses interleaved with assistant reasoning
        5. Nested tool calls (assistant calls tool, uses result to call another)
        6. System message with tool definitions
        7. Empty or None content in messages
        8. Very long messages to test truncation handling
        """
        prompt = build_prompt(
            raw_messages=messages,
            model=model,
            tools=tools,
            inject_tools_as_functions=inject_tools_as_functions,
        )
        return {"prompt": prompt, "model": model}
else:
    mcp = None


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace Prompt Renderer MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mcp", action="store_true", help="Run MCP server over stdio"
    )
    parser.add_argument("--host", default="0.0.0.0", help="FastAPI host")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port")
    args = parser.parse_args()

    if args.mcp:
        if mcp is None:
            raise RuntimeError("MCP server requested but mcp is not installed.")
        mcp.run()
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
