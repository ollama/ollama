# Agent Collaboration System Guide

## Overview

The Agent Collaboration System allows multiple AI agents to work together on complex tasks through a structured multi-round workflow. Instead of single-agent conversations, you can now orchestrate collaborative discussions where agents build on each other's insights.

## How It Works

### Workflow Structure

1. **Round 1**: Primary agent provides initial response
2. **Round 2-N**: Other agents review, critique, and expand
3. **Final Round**: Synthesizer creates comprehensive answer

### Key Features

- **Dynamic Task Assignment**: Define any task for collaboration
- **Flexible Agent Selection**: Choose 2-4 agents from your team
- **Template Presets**: Pre-configured workflows for common scenarios
- **Real-time Progress**: Watch the collaboration unfold round-by-round
- **Collapsible History**: Expand/collapse each round for easy review

## Using the System

### 1. Open Collaboration Panel

Click the **[COLLABORATE]** button in the control panel.

### 2. Configure Your Collaboration

**Task Description**:
- Enter a clear description of what you want the agents to solve
- Example: "Review this Python function for security vulnerabilities and performance issues"

**Select Template**:
- **Code Review**: Coder analyzes → Critic/Researcher review → Planner synthesizes
- **Research & Analysis**: Researcher investigates → Others analyze → Synthesis
- **Brainstorming**: Planner generates ideas → All agents expand → Final plan
- **Custom**: Define your own agent lineup

**Select Agents**:
- Choose at least 2 agents (maximum 4)
- Templates auto-select recommended agents
- Mix specialists for diverse perspectives

**Number of Rounds**:
- Set 2-10 rounds (default: 3)
- More rounds = deeper analysis but longer processing

### 3. Monitor Progress

Once started:
- **Session ID**: Unique identifier for this collaboration
- **Status**: Running, Completed, or Error
- **Current Round**: Live progress indicator
- **Participants**: List of active agents

Each round displays:
- Agent name and role (primary/review)
- Timestamp
- Full response (expandable)

### 4. Review Results

When complete:
- **Final Synthesis**: Comprehensive answer synthesizing all perspectives
- **Collaboration History**: Collapsible view of all rounds
- **Copy to Chat**: Add synthesis to main terminal
- **New Collaboration**: Start another session

## Available Templates

### Code Review
**Purpose**: Multi-agent code analysis
**Agents**: Coder (primary) → Critic + Researcher (review) → Planner (synthesis)
**Rounds**: 3
**Best For**: Security audits, performance reviews, architecture feedback

### Research & Analysis
**Purpose**: Deep research with multiple perspectives
**Agents**: Researcher (primary) → Critic + Planner (review) → Coder (synthesis)
**Rounds**: 4
**Best For**: Topic exploration, fact-checking, comprehensive reports

### Brainstorming
**Purpose**: Creative ideation session
**Agents**: Planner (primary) → All agents contribute → Planner (synthesis)
**Rounds**: 2
**Best For**: Feature ideas, problem-solving, strategic planning

### Custom
**Purpose**: User-defined workflow
**Agents**: You choose
**Rounds**: Configurable
**Best For**: Unique scenarios requiring specific agent combinations

## Example Use Cases

### Security Audit
```
Template: Code Review
Task: "Audit this authentication function for security vulnerabilities"
Agents: Coder, Critic, Researcher
Rounds: 3
```

### Architecture Decision
```
Template: Research & Analysis
Task: "Should we use microservices or monolith for this project?"
Agents: Researcher, Planner, Coder
Rounds: 4
```

### Feature Planning
```
Template: Brainstorming
Task: "Generate ideas for improving user onboarding flow"
Agents: All 4 agents
Rounds: 2
```

## Tips for Best Results

1. **Clear Task Descriptions**: Be specific about what you want analyzed
2. **Right Agent Mix**: Choose agents whose expertise matches your task
3. **Appropriate Rounds**: Simple tasks need 2-3 rounds, complex ones 4-5
4. **Template Selection**: Use presets when they fit, custom for unique needs
5. **Monitor Progress**: Watch the real-time updates to understand the discussion flow

## Technical Details

### API Endpoints

- `GET /api/collaboration/templates` - Get available templates
- `POST /api/collaboration/start` - Start new session
- `GET /api/collaboration/:sessionId` - Get session status
- `GET /api/collaboration` - List all sessions
- `POST /api/collaboration/:sessionId/cancel` - Cancel running session

### Polling Mechanism

The frontend polls the backend every 2 seconds to update progress. Polling stops automatically when:
- Collaboration completes successfully
- An error occurs
- User closes the modal

### Session Storage

Active sessions are stored in-memory on the server. Historical sessions remain available until server restart.

## Comparison: Single Agent vs Collaboration

### Single Agent Mode
- **Speed**: Fast (one response)
- **Perspective**: Single viewpoint
- **Best For**: Simple queries, quick answers
- **Example**: "What is dependency injection?"

### Collaboration Mode
- **Speed**: Slower (multiple rounds)
- **Perspective**: Multi-agent consensus
- **Best For**: Complex analysis, diverse viewpoints
- **Example**: "Design a scalable authentication system"

## Keyboard Shortcuts

- `Ctrl+C` - Open collaboration panel (coming soon)
- `Esc` - Close collaboration modal

## Future Enhancements

- [ ] Streaming responses in real-time
- [ ] Session persistence and history
- [ ] Agent voting/consensus mechanisms
- [ ] Custom agent role definitions
- [ ] Export collaboration as markdown/PDF
- [ ] Agent debate mode (opposing viewpoints)

---

**Version**: 1.0
**Last Updated**: 2025-11-08
**Status**: Production Ready
