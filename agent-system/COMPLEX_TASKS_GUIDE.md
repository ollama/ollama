# Complex Task Capabilities Guide

This guide explains how to use the enhanced agent system for handling complex, multi-step tasks.

## Overview

Your agent system has been upgraded with several powerful capabilities:

1. **Task Planning & Decomposition** - Break complex tasks into manageable subtasks
2. **Tool System** - Structured tool calling for file operations, code execution, and more
3. **Workflow Orchestration** - Manage complex multi-step workflows with state and error handling
4. **Enhanced Memory** - Better long-term memory and context management

## 1. Task Planning System

### What It Does

The Task Planner breaks down complex tasks into smaller, actionable subtasks with dependencies, then executes them step by step.

### How to Use

#### Via API

```bash
# Plan a complex task
curl -X POST http://localhost:3000/api/tasks/plan \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Build a web scraper that extracts product information from an e-commerce site",
    "agentKey": "planner"
  }'

# Execute the planned task
curl -X POST http://localhost:3000/api/tasks/{taskId}/execute \
  -H "Content-Type: application/json" \
  -d '{
    "agentKey": "planner",
    "options": {"stopOnError": false}
  }'

# Check task status
curl http://localhost:3000/api/tasks/{taskId}
```

#### Example Tasks

- "Create a data analysis pipeline that processes CSV files and generates visualizations"
- "Build a REST API with authentication and CRUD operations"
- "Develop a machine learning model to predict house prices"
- "Set up a CI/CD pipeline for a Node.js application"

### Features

- **Automatic Decomposition**: Breaks tasks into logical subtasks
- **Dependency Management**: Handles task dependencies automatically
- **Error Handling**: Continues or stops based on configuration
- **Result Synthesis**: Combines all subtask results into a final summary

## 2. Tool System

### What It Does

Provides structured tool calling for agents to perform operations like file I/O, code execution, HTTP requests, and data processing.

### Available Tools

#### File Operations

```javascript
// Read a file
[TOOL: read_file {"filepath": "config.json"}]

// Write to a file
[TOOL: write_file {"filepath": "output.txt", "content": "Hello World"}]

// List directory
[TOOL: list_directory {"dirpath": "data"}]
```

#### Code Execution

```javascript
// Execute Python code
[TOOL: execute_code {"code": "print('Hello from Python')", "language": "python"}]

// Execute JavaScript
[TOOL: execute_code {"code": "console.log('Hello from Node')", "language": "javascript"}]

// Execute shell commands
[TOOL: execute_code {"code": "ls -la", "language": "shell"}]
```

#### Data Processing

```javascript
// Process JSON
[TOOL: process_json {"json_string": "{\"key\":\"value\"}", "operation": "keys"}]
```

#### HTTP Requests

```javascript
// Make HTTP request
[TOOL: http_request {"url": "https://api.example.com/data", "method": "GET"}]
```

### Security

- File operations are restricted to the working directory and data directory
- Code execution runs in isolated temporary files
- Command execution blocks dangerous operations (rm -rf, format, etc.)
- Timeouts prevent hanging operations

### Using Tools in Conversations

Agents automatically detect and execute tool calls in their responses. Just ask naturally:

- "Read the config.json file"
- "Run this Python code: print('hello')"
- "List all files in the data directory"
- "Make an HTTP request to https://api.example.com"

## 3. Workflow Orchestration

### What It Does

Manages complex multi-step workflows with state management, conditional execution, parallel processing, and error handling.

### Workflow Step Types

#### Agent Call

Execute an agent with a message:

```json
{
  "type": "agent_call",
  "name": "Research step",
  "agent": "researcher",
  "message": "Research {{topic}}",
  "systemPrompt": "Focus on recent developments",
  "outputKey": "researchResults"
}
```

#### Tool Call

Execute a tool:

```json
{
  "type": "tool_call",
  "name": "Read config",
  "tool": "read_file",
  "params": {
    "filepath": "{{configPath}}"
  },
  "outputKey": "config"
}
```

#### Task Plan

Plan and execute a complex task:

```json
{
  "type": "task_plan",
  "name": "Build feature",
  "task": "Build user authentication system",
  "agent": "planner"
}
```

#### Parallel Execution

Run multiple steps in parallel:

```json
{
  "type": "parallel",
  "steps": [
    {"type": "agent_call", "agent": "researcher", "message": "Research A"},
    {"type": "agent_call", "agent": "coder", "message": "Code B"}
  ]
}
```

#### Conditional Execution

Execute based on conditions:

```json
{
  "type": "condition",
  "condition": {
    "type": "exists",
    "key": "previousResult"
  },
  "then": {
    "type": "agent_call",
    "message": "Process {{previousResult}}"
  },
  "else": {
    "type": "agent_call",
    "message": "Initialize process"
  }
}
```

#### Loop

Iterate over items:

```json
{
  "type": "loop",
  "items": "{{fileList}}",
  "itemKey": "file",
  "step": {
    "type": "tool_call",
    "tool": "read_file",
    "params": {
      "filepath": "{{file}}"
    }
  }
}
```

### Example Workflow

```json
{
  "name": "Data Analysis Pipeline",
  "description": "Process data files and generate report",
  "steps": [
    {
      "type": "tool_call",
      "name": "List data files",
      "tool": "list_directory",
      "params": {"dirpath": "data"},
      "outputKey": "files"
    },
    {
      "type": "loop",
      "items": "{{files}}",
      "itemKey": "file",
      "step": {
        "type": "tool_call",
        "tool": "read_file",
        "params": {"filepath": "{{file.path}}"}
      }
    },
    {
      "type": "agent_call",
      "name": "Analyze data",
      "agent": "researcher",
      "message": "Analyze the data from {{files}} and create a summary"
    }
  ]
}
```

### Using Workflows

```bash
# Start a workflow
curl -X POST http://localhost:3000/api/workflows/start \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Workflow",
    "description": "Process data",
    "steps": [...],
    "initialContext": {"key": "value"}
  }'

# Check workflow status
curl http://localhost:3000/api/workflows/{workflowId}

# Cancel workflow
curl -X POST http://localhost:3000/api/workflows/{workflowId}/cancel
```

## 4. Enhanced Memory System

### Features

- **Conversation Summaries**: Automatic summarization of past conversations
- **Permanent Learnings**: Key insights that persist across sessions
- **User Profile**: Comprehensive user understanding
- **Context Awareness**: Agents remember previous interactions

### Memory Management

The system automatically:
- Summarizes long conversations
- Extracts key learnings
- Tracks user preferences
- Maintains context across sessions

## Best Practices

### For Complex Tasks

1. **Start with Planning**: Use the task planner for multi-step tasks
2. **Break It Down**: Complex tasks should be decomposed into subtasks
3. **Use Tools**: Leverage the tool system for file operations and code execution
4. **Orchestrate Workflows**: Use workflows for repeatable processes
5. **Handle Errors**: Configure error handling strategies in workflows

### For Simple Tasks

- Direct agent conversations work best for simple questions
- Use tool calls for specific operations (read file, run code)
- Use [SEARCH:] and [BROWSER:] for web access

### Security Considerations

- File operations are restricted to safe directories
- Code execution runs in isolated environments
- Dangerous commands are blocked
- All operations have timeouts

## Examples

### Example 1: Data Processing Pipeline

```bash
# Plan the task
curl -X POST http://localhost:3000/api/tasks/plan \
  -d '{"task": "Process CSV files, clean data, and generate visualizations"}'

# Execute
curl -X POST http://localhost:3000/api/tasks/{taskId}/execute
```

### Example 2: Code Review Workflow

```json
{
  "name": "Code Review",
  "steps": [
    {
      "type": "tool_call",
      "tool": "read_file",
      "params": {"filepath": "src/main.js"},
      "outputKey": "code"
    },
    {
      "type": "agent_call",
      "agent": "coder",
      "message": "Review this code: {{code}}"
    },
    {
      "type": "agent_call",
      "agent": "critic",
      "message": "Critique the code review: {{previousResult}}"
    }
  ]
}
```

### Example 3: Automated Testing

```json
{
  "name": "Run Tests",
  "steps": [
    {
      "type": "tool_call",
      "tool": "execute_code",
      "params": {
        "code": "npm test",
        "language": "shell"
      }
    },
    {
      "type": "condition",
      "condition": {"type": "equals", "key": "testResult", "value": "passed"},
      "then": {
        "type": "agent_call",
        "agent": "planner",
        "message": "Tests passed! Plan next steps."
      },
      "else": {
        "type": "agent_call",
        "agent": "coder",
        "message": "Tests failed. Analyze and fix issues."
      }
    }
  ]
}
```

## API Reference

### Task Planning

- `POST /api/tasks/plan` - Plan a complex task
- `POST /api/tasks/:taskId/execute` - Execute a planned task
- `GET /api/tasks/:taskId` - Get task status
- `GET /api/tasks` - List all tasks

### Tools

- `GET /api/tools` - List available tools
- `POST /api/tools/execute` - Execute a tool directly

### Workflows

- `POST /api/workflows/start` - Start a workflow
- `GET /api/workflows/:workflowId` - Get workflow status
- `GET /api/workflows` - List all workflows
- `POST /api/workflows/:workflowId/cancel` - Cancel a workflow

## Troubleshooting

### Task Planning Fails

- Ensure Ollama is running and models are available
- Check that the task description is clear and specific
- Try a different agent (planner, researcher, etc.)

### Tool Execution Errors

- Verify file paths are within allowed directories
- Check that code syntax is correct for the language
- Ensure commands are not blocked by security rules

### Workflow Issues

- Check workflow step syntax
- Verify context variables are properly set
- Review error handling configuration

## Next Steps

1. Try planning a complex task you're working on
2. Experiment with tool calls in agent conversations
3. Create a workflow for a repetitive task
4. Explore the API endpoints

For more information, see:
- `task-planner.js` - Task planning implementation
- `tool-system.js` - Tool system implementation
- `workflow-orchestrator.js` - Workflow orchestration implementation

