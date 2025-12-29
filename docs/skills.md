# Ollama Skills

Skills are reusable capability packages that extend what agents can do. They bundle instructions, scripts, and data that teach an agent how to perform specific tasks.

## Quick Start

### Creating a Skill

Create a directory with a `SKILL.md` file:

```
my-skill/
├── SKILL.md          # Required: Instructions for the agent
└── scripts/          # Optional: Executable scripts
    └── run.py
```

The `SKILL.md` file must have YAML frontmatter:

```markdown
---
name: my-skill
description: A brief description of what this skill does
---

# My Skill

## Purpose
Explain what this skill does and when to use it.

## Instructions
Step-by-step instructions for the agent on how to use this skill.

## Examples
Show example inputs and expected outputs.
```

### Using Skills in an Agent

Reference skills in your Agentfile:

```dockerfile
FROM llama3.2:3b
AGENT_TYPE conversational

# Local skill (bundled with agent)
SKILL ./path/to/my-skill

# Registry skill (pulled from ollama.com)
SKILL library/skill/calculator:1.0.0

# User skill from registry
SKILL myname/skill/calculator:1.0.0

SYSTEM You are a helpful assistant.
```

### Managing Skills

```bash
# Push a skill to the registry (uses your namespace)
ollama skill push myname/skill/calculator:1.0.0 ./my-skill

# Pull a skill from the official library
ollama skill pull skill/calculator:1.0.0

# Pull a skill from a user's namespace
ollama skill pull myname/skill/calculator:1.0.0

# List installed skills
ollama skill list

# Show skill details
ollama skill show skill/calculator:1.0.0

# Remove a skill
ollama skill rm skill/calculator:1.0.0
```

### Dynamic Skills in Chat

You can add and remove skills dynamically during an interactive chat session:

```
>>> /skills
Available Skills:
  calculator (sha256:abc123def456...)

>>> /skill add ./my-local-skill
Added skill 'my-skill' from ./my-local-skill

>>> /skill list
Skills loaded in this session:
  my-skill (local: /path/to/my-local-skill)

>>> /skill remove my-skill
Removed skill 'my-skill'
```

| Command | Description |
|---------|-------------|
| `/skills` | Show all available skills (model + session) |
| `/skill add <path>` | Add a skill from a local path |
| `/skill remove <name>` | Remove a skill by name |
| `/skill list` | List skills loaded in this session |

Dynamic skills take effect on the next message. This is useful for:
- Testing skills during development
- Temporarily adding capabilities to a model
- Experimenting with skill combinations

## Skill Reference Formats

Skills use a 5-part name structure: `host/namespace/kind/model:tag`

| Format | Example | Description |
|--------|---------|-------------|
| Local path | `./skills/calc` | Bundled with agent at create time |
| Library skill | `skill/calculator:1.0.0` | From the official skill library (library/skill/calculator) |
| User skill | `alice/skill/calc:1.0.0` | From a user's namespace |
| Full path | `registry.ollama.ai/alice/skill/calc:1.0.0` | Fully qualified with host |

The `kind` field distinguishes skills from models:
- `skill` - Skill packages
- `agent` - Agent packages (future)
- (empty) - Regular models

## SKILL.md Structure

### Required Frontmatter

```yaml
---
name: skill-name        # Must match directory name
description: Brief description of the skill
---
```

### Recommended Sections

1. **Purpose**: What the skill does and when to use it
2. **When to use**: Trigger conditions for the agent
3. **Instructions**: Step-by-step usage guide
4. **Examples**: Input/output examples
5. **Scripts**: Documentation for any bundled scripts

### Example: Calculator Skill

```markdown
---
name: calculator
description: Performs mathematical calculations using Python
---

# Calculator Skill

## Purpose
This skill performs mathematical calculations using a bundled Python script.

## When to use
- User asks to calculate something
- User wants to do math operations
- Any arithmetic is needed

## Instructions
1. When calculation is needed, use the `run_skill_script` tool
2. Call: `python3 scripts/calculate.py "<expression>"`
3. Return the result to the user

## Examples

**Input**: "What is 25 * 4?"
**Action**: `run_skill_script` with command `python3 scripts/calculate.py '25 * 4'`
**Output**: "25 * 4 = 100"
```

## Storage Layout

```
~/.ollama/models/
├── blobs/
│   └── sha256-<digest>           # Skill tar.gz blob
├── manifests/
│   └── registry.ollama.ai/
│       └── skill/                # Library skills
│           └── calculator/
│               └── 1.0.0
│       └── skill-username/       # User skills
│           └── my-skill/
│               └── latest
└── skills/
    └── sha256-<digest>/          # Extracted skill cache
        ├── SKILL.md
        └── scripts/
```

---

# Security Considerations

## Current State (Development)

The current implementation has several security considerations that need to be addressed before production use.

### 1. Script Execution

**Risk**: Skills can bundle arbitrary scripts that execute on the host system.

**Current behavior**:
- Scripts run with the same permissions as the Ollama process
- No sandboxing or isolation
- Full filesystem access

**Mitigations needed**:
- [ ] Sandbox script execution (containers, seccomp, etc.)
- [ ] Resource limits (CPU, memory, time)
- [ ] Filesystem isolation (read-only mounts, restricted paths)
- [ ] Network policy controls
- [ ] Capability dropping

### 2. Skill Provenance

**Risk**: Malicious skills could be pushed to the registry.

**Current behavior**:
- No code signing or verification
- No malware scanning
- Trust based on namespace ownership

**Mitigations needed**:
- [ ] Skill signing with author keys
- [ ] Registry-side malware scanning
- [ ] Content policy enforcement
- [ ] Reputation system for skill authors

### 3. Namespace Squatting

**Risk**: Malicious actors could register skill names that impersonate official tools.

**Current behavior**:
- First-come-first-served namespace registration
- No verification of skill names

**Mitigations needed**:
- [ ] Reserved namespace list (official tools, common names)
- [ ] Trademark/name verification for popular skills
- [ ] Clear namespacing conventions

### 4. Supply Chain Attacks

**Risk**: Compromised skills could inject malicious code into agents.

**Current behavior**:
- Skills pulled without integrity verification beyond digest
- No dependency tracking

**Mitigations needed**:
- [ ] SBOM (Software Bill of Materials) for skills
- [ ] Dependency vulnerability scanning
- [ ] Pinned versions in Agentfiles
- [ ] Audit logging of skill usage

### 5. Data Exfiltration

**Risk**: Skills could exfiltrate sensitive data from conversations or the host.

**Current behavior**:
- Skills have access to conversation context
- Scripts can make network requests

**Mitigations needed**:
- [ ] Network egress controls
- [ ] Sensitive data detection/masking
- [ ] Audit logging of script network activity
- [ ] User consent for data access

### 6. Privilege Escalation

**Risk**: Skills could escalate privileges through script execution.

**Current behavior**:
- Scripts inherit Ollama process privileges
- No capability restrictions

**Mitigations needed**:
- [ ] Run scripts as unprivileged user
- [ ] Drop all capabilities
- [ ] Mandatory access controls (SELinux/AppArmor)

## Recommended Security Model

### Skill Trust Levels

```
┌─────────────────────────────────────────────────────────────┐
│ Level 0: Untrusted (default)                                │
│ - No script execution                                       │
│ - Instructions only                                         │
│ - Safe for any skill                                        │
├─────────────────────────────────────────────────────────────┤
│ Level 1: Sandboxed                                          │
│ - Scripts run in isolated container                         │
│ - No network access                                         │
│ - Read-only filesystem                                      │
│ - Resource limits enforced                                  │
├─────────────────────────────────────────────────────────────┤
│ Level 2: Trusted                                            │
│ - Scripts run with network access                           │
│ - Can write to designated directories                       │
│ - Requires explicit user approval                           │
├─────────────────────────────────────────────────────────────┤
│ Level 3: Privileged (admin only)                            │
│ - Full host access                                          │
│ - System administration skills                              │
│ - Requires admin approval                                   │
└─────────────────────────────────────────────────────────────┘
```

### Skill Manifest Security Fields (Future)

```yaml
---
name: my-skill
description: A skill description
security:
  trust_level: sandboxed
  permissions:
    - network:read          # Can make HTTP GET requests
    - filesystem:read:/data # Can read from /data
  resource_limits:
    max_memory: 256MB
    max_cpu_time: 30s
    max_disk: 100MB
  signature: sha256:abc...  # Author signature
---
```

---

# Future Considerations

## Feature Roadmap

### Phase 1: Foundation (Current)
- [x] Skill bundling with agents
- [x] Local skill development
- [x] Basic CLI commands (push, pull, list, rm, show)
- [x] Registry blob storage
- [ ] Registry namespace configuration

### Phase 2: Security
- [ ] Script sandboxing
- [ ] Permission model
- [ ] Skill signing
- [ ] Audit logging

### Phase 3: Discovery
- [ ] Skill search on ollama.com
- [ ] Skill ratings and reviews
- [ ] Usage analytics
- [ ] Featured/trending skills

### Phase 4: Advanced Features
- [ ] Skill dependencies
- [ ] Skill versioning constraints
- [ ] Skill composition (skills using skills)
- [ ] Skill testing framework

## Open Questions

### 1. Skill Execution Model

**Question**: How should skills execute scripts?

Options:
- **A) In-process**: Fast but unsafe
- **B) Subprocess**: Current approach, moderate isolation
- **C) Container**: Good isolation, requires container runtime
- **D) WASM**: Portable and safe, limited capabilities
- **E) Remote execution**: Offload to secure service

### 2. Skill Versioning

**Question**: How strict should version pinning be?

Options:
- **A) Always latest**: Simple but risky
- **B) Semantic versioning**: `^1.0.0` allows minor updates
- **C) Exact pinning**: `=1.0.0` requires explicit updates
- **D) Digest pinning**: `@sha256:abc` immutable reference

### 3. Skill Permissions

**Question**: How should users grant permissions to skills?

Options:
- **A) All or nothing**: Accept all permissions or don't use
- **B) Granular consent**: Approve each permission individually
- **C) Trust levels**: Pre-defined permission bundles
- **D) Runtime prompts**: Ask when permission is first used

### 4. Skill Discovery

**Question**: How should users find skills?

Options:
- **A) Central registry only**: ollama.com/skills
- **B) Federated registries**: Multiple skill sources
- **C) Git repositories**: Pull from GitHub, etc.
- **D) All of the above**: Multiple discovery mechanisms

### 5. Skill Monetization

**Question**: Should skill authors be able to monetize?

Options:
- **A) Free only**: All skills are free and open
- **B) Paid skills**: Authors can charge for skills
- **C) Freemium**: Free tier with paid features
- **D) Donations**: Voluntary support for authors

### 6. Skill Updates

**Question**: How should skill updates be handled?

Options:
- **A) Manual**: User explicitly updates
- **B) Auto-update**: Always use latest
- **C) Notify**: Alert user to available updates
- **D) Policy-based**: Organization controls update policy

## API Considerations

### Skill Metadata API

```
GET /api/skills
GET /api/skills/:namespace/:name
GET /api/skills/:namespace/:name/versions
GET /api/skills/:namespace/:name/readme
```

### Skill Execution API

```
POST /api/skills/:namespace/:name/execute
{
  "command": "python3 scripts/run.py",
  "args": ["--input", "data"],
  "timeout": 30
}
```

### Skill Permissions API

```
GET /api/skills/:namespace/:name/permissions
POST /api/skills/:namespace/:name/permissions/grant
DELETE /api/skills/:namespace/:name/permissions/revoke
```

## Testing Considerations

### Skill Testing Framework

```bash
# Run skill tests
ollama skill test ./my-skill

# Test with specific model
ollama skill test ./my-skill --model llama3.2:3b

# Generate test report
ollama skill test ./my-skill --report
```

### Test File Format

```yaml
# my-skill/tests/test.yaml
tests:
  - name: "basic calculation"
    input: "What is 2 + 2?"
    expect:
      contains: "4"
      tool_called: "run_skill_script"

  - name: "complex expression"
    input: "Calculate 15% of 200"
    expect:
      contains: "30"
```

## Compatibility Considerations

### Minimum Ollama Version

Skills should declare minimum Ollama version:

```yaml
---
name: my-skill
requires:
  ollama: ">=0.4.0"
---
```

### Model Compatibility

Skills may require specific model capabilities:

```yaml
---
name: vision-skill
requires:
  capabilities:
    - vision
    - tools
---
```

## Migration Path

### From Local to Registry

```bash
# Develop locally
SKILL ./my-skill

# Push when ready
ollama skill push myname/my-skill:1.0.0 ./my-skill

# Update Agentfile
SKILL skill/myname/my-skill:1.0.0
```

### Version Upgrades

```bash
# Check for updates
ollama skill outdated

# Update specific skill
ollama skill update calculator:1.0.0

# Update all skills
ollama skill update --all
```
