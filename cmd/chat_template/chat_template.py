#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers>=4.57.0",
#   "jinja2",
#   "fastapi",
#   "uvicorn",
#   "pydantic",
#   "requests",
# ]
# ///
"""
Chat Template Testing Tool

Test HuggingFace chat templates against Ollama renderers.

Usage:
    # Run predefined test cases against a HuggingFace model
    uv run cmd/chat_template/chat_template.py --model PrimeIntellect/INTELLECT-3

    # Compare HuggingFace output with Ollama renderer
    uv run cmd/chat_template/chat_template.py --model PrimeIntellect/INTELLECT-3 --ollama-model intellect3

    # Start server for manual curl testing
    uv run cmd/chat_template/chat_template.py --serve

    # Show chat template for a model
    uv run cmd/chat_template/chat_template.py --model PrimeIntellect/INTELLECT-3 --show-template
"""

import argparse
import json
import sys
from typing import Any

from transformers import AutoTokenizer


TEST_CASES = [
    {
        "name": "basic_user_message",
        "messages": [{"role": "user", "content": "Hello!"}],
        "tools": None,
    },
    {
        "name": "with_system_message",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        "tools": None,
    },
    {
        "name": "multi_turn_conversation",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ],
        "tools": None,
    },
    {
        "name": "with_tools",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "required": ["location"],
                        "properties": {
                            "location": {"type": "string", "description": "The city"}
                        },
                    },
                },
            }
        ],
    },
    {
        "name": "tool_call_and_response",
        "messages": [
            {"role": "user", "content": "What is the weather in SF?"},
            {
                "role": "assistant",
                "content": "Let me check the weather.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "San Francisco"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temperature": 68}', "tool_call_id": "call_1"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "required": ["location"],
                        "properties": {
                            "location": {"type": "string", "description": "The city"}
                        },
                    },
                },
            }
        ],
    },
    {
        "name": "parallel_tool_calls",
        "messages": [
            {"role": "user", "content": "Get weather in SF and NYC"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "San Francisco"},
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "New York"},
                        },
                    },
                ],
            },
            {"role": "tool", "content": '{"temperature": 68}', "tool_call_id": "call_1"},
            {"role": "tool", "content": '{"temperature": 55}', "tool_call_id": "call_2"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ],
    },
    # Thinking tests
    {
        "name": "assistant_with_thinking",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "thinking": "Let me calculate: 2 + 2 = 4. This is basic arithmetic.",
            },
            {"role": "user", "content": "And 3+3?"},
        ],
        "tools": None,
    },
    {
        "name": "thinking_with_tool_call",
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": "I'll check the weather for you.",
                "thinking": "The user wants to know the weather in Paris. I should call the get_weather function.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "Paris"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temperature": 18, "condition": "cloudy"}', "tool_call_id": "call_1"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ],
    },
    {
        "name": "thinking_only_no_content",
        "messages": [
            {"role": "user", "content": "Think about this silently."},
            {
                "role": "assistant",
                "content": "",  # HuggingFace requires content field
                "thinking": "I'm thinking about this but won't respond with visible content.",
            },
            {"role": "user", "content": "What did you think?"},
        ],
        "tools": None,
    },
]

# Cache for tokenizers
_tokenizer_cache: dict[str, Any] = {}


def get_tokenizer(model_name: str):
    """Get or create tokenizer for the given model."""
    if model_name not in _tokenizer_cache:
        print(f"Loading tokenizer for {model_name}...", file=sys.stderr)
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[model_name]


def apply_template(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> str:
    """Apply HuggingFace chat template to messages."""
    tokenizer = get_tokenizer(model)

    if tools:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def get_ollama_prompt(
    ollama_model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    ollama_host: str = "http://localhost:11434",
) -> str | None:
    """Get rendered prompt from Ollama using debug_render_only."""
    import requests

    # Convert messages to Ollama format
    ollama_messages = []
    for msg in messages:
        ollama_msg = {"role": msg["role"]}
        if "content" in msg:
            ollama_msg["content"] = msg["content"]
        if "thinking" in msg:
            ollama_msg["thinking"] = msg["thinking"]
        if "tool_calls" in msg:
            # Convert tool_calls to Ollama format
            tool_calls = []
            for tc in msg["tool_calls"]:
                tool_call = {
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                }
                if "id" in tc:
                    tool_call["id"] = tc["id"]
                tool_calls.append(tool_call)
            ollama_msg["tool_calls"] = tool_calls
        if "tool_call_id" in msg:
            ollama_msg["tool_call_id"] = msg["tool_call_id"]
        ollama_messages.append(ollama_msg)

    payload = {
        "model": ollama_model,
        "messages": ollama_messages,
        "stream": False,
        "_debug_render_only": True,
    }

    if tools:
        payload["tools"] = tools

    try:
        resp = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Field name is _debug_info with underscore prefix
        if "_debug_info" in data and "rendered_template" in data["_debug_info"]:
            return data["_debug_info"]["rendered_template"]
        return None
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] Cannot connect to Ollama at {ollama_host}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [ERROR] Ollama request failed: {e}", file=sys.stderr)
        return None


def compute_diff(hf_prompt: str, ollama_prompt: str) -> str:
    """Compute a unified diff between HuggingFace and Ollama prompts."""
    import difflib

    hf_lines = hf_prompt.splitlines(keepends=True)
    ollama_lines = ollama_prompt.splitlines(keepends=True)

    diff = difflib.unified_diff(
        ollama_lines,
        hf_lines,
        fromfile="Ollama",
        tofile="HuggingFace",
        lineterm="",
    )
    return "".join(diff)


def print_test_output(
    name: str,
    messages: list[dict],
    tools: list[dict] | None,
    hf_prompt: str,
    ollama_prompt: str | None = None,
    as_repr: bool = False,
):
    """Print test output in a format suitable for Go test creation and LLM diffing."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print("=" * 60)
    print("\n--- Input Messages ---")
    print(json.dumps(messages, indent=2))
    if tools:
        print("\n--- Tools ---")
        print(json.dumps(tools, indent=2))

    if ollama_prompt is not None:
        # Comparison mode
        if hf_prompt == ollama_prompt:
            print("\n--- Result: MATCH ---")
            print("\n--- Prompt (both identical) ---")
            if as_repr:
                print(repr(hf_prompt))
            else:
                print(hf_prompt)
        else:
            print("\n--- Result: MISMATCH ---")
            print("\n--- HuggingFace Prompt ---")
            if as_repr:
                print(repr(hf_prompt))
            else:
                print(hf_prompt)
            print("\n--- Ollama Prompt ---")
            if as_repr:
                print(repr(ollama_prompt))
            else:
                print(ollama_prompt)
            print("\n--- Diff (Ollama -> HuggingFace) ---")
            diff = compute_diff(hf_prompt, ollama_prompt)
            if diff:
                print(diff)
            else:
                print("(no line-level diff, check whitespace)")
    else:
        # HuggingFace only mode
        print("\n--- HuggingFace Prompt ---")
        if as_repr:
            print(repr(hf_prompt))
        else:
            print(hf_prompt)

    print("=" * 60)


def run_tests(
    model: str,
    as_repr: bool = False,
    test_filter: str | None = None,
    ollama_model: str | None = None,
    ollama_host: str = "http://localhost:11434",
):
    """Run all predefined test cases against a model."""
    if ollama_model:
        print(f"\nComparing HuggingFace ({model}) vs Ollama ({ollama_model})\n")
    else:
        print(f"\nRunning tests against: {model}\n")

    matches = 0
    mismatches = 0
    errors = 0

    for test_case in TEST_CASES:
        name = test_case["name"]
        messages = test_case["messages"]
        tools = test_case["tools"]

        # Filter tests if specified
        if test_filter and test_filter.lower() not in name.lower():
            continue

        try:
            hf_prompt = apply_template(model, messages, tools)

            ollama_prompt = None
            if ollama_model:
                ollama_prompt = get_ollama_prompt(
                    ollama_model, messages, tools, ollama_host
                )
                if ollama_prompt is None:
                    errors += 1
                elif hf_prompt == ollama_prompt:
                    matches += 1
                else:
                    mismatches += 1

            print_test_output(
                name, messages, tools, hf_prompt, ollama_prompt, as_repr=as_repr
            )
        except Exception as e:
            errors += 1
            print(f"\n{'='*60}")
            print(f"Test: {name} - FAILED")
            print(f"--- Input Messages ---")
            print(json.dumps(messages, indent=2))
            if tools:
                print(f"--- Tools ---")
                print(json.dumps(tools, indent=2))
            print(f"--- Error ---")
            print(f"{e}")
            print("=" * 60)

    # Print summary if comparing
    if ollama_model:
        total = matches + mismatches + errors
        print(f"\n{'='*60}")
        print("SUMMARY")
        print("=" * 60)
        print(f"  Total:      {total}")
        print(f"  Matches:    {matches}")
        print(f"  Mismatches: {mismatches}")
        print(f"  Errors:     {errors}")
        print("=" * 60)


def show_template(model: str):
    """Show the chat template for a model."""
    tokenizer = get_tokenizer(model)
    print(f"\nChat template for {model}:\n")
    print("-" * 60)
    print(tokenizer.chat_template)
    print("-" * 60)


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server for manual testing."""
    from typing import Optional, List, Dict, Any as TypingAny

    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    class Message(BaseModel):
        role: str
        content: Optional[str] = None
        tool_calls: Optional[List[Dict[str, TypingAny]]] = None
        tool_call_id: Optional[str] = None

    class GeneratePromptRequest(BaseModel):
        messages: List[Message]
        model: str = "PrimeIntellect/INTELLECT-3"
        tools: Optional[List[Dict[str, TypingAny]]] = None
        inject_tools_as_functions: bool = False

    class GeneratePromptResponse(BaseModel):
        prompt: str
        model: str

    app = FastAPI(title="HuggingFace Prompt Generator", version="1.0.0")

    @app.post("/generate-prompt", response_model=GeneratePromptResponse)
    async def generate_prompt(request: GeneratePromptRequest):
        try:
            messages = []
            for msg in request.messages:
                message_dict = {"role": msg.role}
                if msg.content is not None:
                    message_dict["content"] = msg.content
                if msg.tool_calls is not None:
                    tool_calls = []
                    for tc in msg.tool_calls:
                        tc_copy = tc.copy()
                        if "function" in tc_copy and "arguments" in tc_copy["function"]:
                            args = tc_copy["function"]["arguments"]
                            if isinstance(args, str):
                                try:
                                    tc_copy["function"]["arguments"] = json.loads(args)
                                except json.JSONDecodeError:
                                    pass
                        tool_calls.append(tc_copy)
                    message_dict["tool_calls"] = tool_calls
                if msg.tool_call_id is not None:
                    message_dict["tool_call_id"] = msg.tool_call_id
                messages.append(message_dict)

            prompt = apply_template(request.model, messages, request.tools)
            return GeneratePromptResponse(prompt=prompt, model=request.model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    print(f"Starting server on http://{host}:{port}")
    print("Endpoints:")
    print("  POST /generate-prompt - Generate prompt from messages")
    print("  GET  /health          - Health check")
    uvicorn.run(app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace Prompt Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="HuggingFace model name (e.g., PrimeIntellect/INTELLECT-3)",
    )
    parser.add_argument(
        "--ollama-model",
        "-o",
        type=str,
        help="Ollama model name to compare against (e.g., qwen3-coder)",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--serve",
        "-s",
        action="store_true",
        help="Start FastAPI server for manual curl testing",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--show-template",
        "-t",
        action="store_true",
        help="Show the chat template for the model",
    )
    parser.add_argument(
        "--repr",
        "-r",
        action="store_true",
        help="Output prompts as Python repr (shows escape sequences)",
    )
    parser.add_argument(
        "--filter",
        "-f",
        type=str,
        help="Filter tests by name (substring match)",
    )

    args = parser.parse_args()

    if args.serve:
        start_server(port=args.port)
    elif args.model:
        if args.show_template:
            show_template(args.model)
        else:
            run_tests(
                args.model,
                as_repr=args.repr,
                test_filter=args.filter,
                ollama_model=args.ollama_model,
                ollama_host=args.ollama_host,
            )
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  uv run cmd/chat_template/chat_template.py --model PrimeIntellect/INTELLECT-3")
        print("  uv run cmd/chat_template/chat_template.py --model Qwen/Qwen3-Coder-480B-A35B-Instruct --ollama-model qwen3-coder")
        print("  uv run cmd/chat_template/chat_template.py --serve")
        sys.exit(1)


if __name__ == "__main__":
    main()

