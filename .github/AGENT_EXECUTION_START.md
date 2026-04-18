# Autonomous Agent Execution Start

**Status**: ✅ READY TO START  
**Generated**: 2026-04-18  
**Dispatch Manifest**: `.github/agent_autonomous_dispatch.json`  
**Workpack Location**: `.github/lane_workpacks/shard_*.json`

## Quick Start for Agents

You are an autonomous development agent. Your job is to claim a batch of GitHub issues and implement them end-to-end following the [Autonomous Developer Agent Instructions](instructions/autonomous-dev.instructions.md).

### Step 1: Claim Your Batch

Run the claim script to atomically claim a batch from your assigned lane:

```bash
# Check available work
python3 scripts/agent_claim_work.py --status

# Claim a batch from your assigned shard (replace with your shard number)
python3 scripts/agent_claim_work.py --claim --shard 1

# View your claimed batch
python3 scripts/agent_claim_work.py --status
```

### Step 2: Understand Your Assignment

Each batch contains ~12 GitHub issues with:
- **Issue Number**: GitHub issue ID (e.g., #123)
- **Title**: Human-readable issue title
- **Priority**: P0 (critical) → P3 (low)
- **AI Classification**: Ollama-generated estimate of issue type and complexity
- **Recommended Model**: For local LLM analysis (e.g., llama3:8b)

Example from shard_1_workpack.json:
```json
{
  "issue_number": 42,
  "title": "Add Kubernetes deployment support",
  "priority": "P0",
  "ai_classification": {
    "category": "feature",
    "complexity": "high",
    "confidence": 0.94
  },
  "batch": 1,
  "recommended_model": "llama3:8b"
}
```

### Step 3: Execute the Autonomous Workflow

For EACH issue in your batch, follow **[Phase 1: Issue Analysis through Phase 8: Completion & Closure](instructions/autonomous-dev.instructions.md#workflow)** from the Autonomous Developer Instructions:

1. **Phase 1: Issue Analysis** (Review GitHub issue, extract requirements)
2. **Phase 2: Design & Planning** (Plan implementation)
3. **Phase 3: Branch Creation** (Create feature branch)
4. **Phase 4: Implementation** (Write code, tests, docs)
5. **Phase 5: Local Validation** (Run tests, linting, coverage)
6. **Phase 6: Pull Request Creation** (Post PR to GitHub)
7. **Phase 7: Code Review Response** (Address feedback)
8. **Phase 8: Completion & Closure** (Merge and close issue)

### Step 4: Report Progress

After completing each issue, update the progress tracking file:

```bash
# Mark issue #42 as claimed by you
python3 scripts/agent_claim_work.py --mark-issue 42 --status in-progress

# Mark issue #42 as PR submitted (#1234)
python3 scripts/agent_claim_work.py --mark-issue 42 --status pr-submitted --pr 1234

# Mark issue #42 as merged/closed
python3 scripts/agent_claim_work.py --mark-issue 42 --status completed --pr 1234
```

### Reference Files

- **Workpack Assignment**: See `.github/agent_autonomous_dispatch.json` for lane/batch mapping
- **Your Lane**: Check `.github/lane_workpacks/shard_N_workpack.json` for your specific issues
- **Live Issue State**: Cross-reference with https://github.com/kushin77/ollama/issues
- **Execution Rules**: Review execution guardrails in `.github/agent_execution_control.json`

## Execution Rules (Non-Negotiable)

All agents MUST follow these guarantees:

### Idempotence
- Multiple agents working simultaneously will NOT conflict (deterministic lane/batch assignments)
- Safe to retry: if agent crashes, another can claim same batch and resume

### Verifiable Closure
- Issues closed only after PR is merged to main
- Commit evidence required: link merged PR in all closure comments
- No issues closed without evidence

### Code Quality Gates
Before posting any PR:
- ✅ All tests pass locally
- ✅ Code coverage ≥ 95%
- ✅ Linting: 0 errors, 0 new warnings
- ✅ Type checking passes (if applicable)
- ✅ Security audit: 0 critical, 0 high issues
- ✅ No breaking changes to API/CLI

### Safe Shutdown
If you encounter blockers:
1. Post a comment on the issue explaining the blocker
2. Tag `@maintainer` with clarification request
3. Move to next issue in batch (don't force solutions)
4. Return to blocked issue after maintainer response

## Lane Assignments

| Lane | Shard | Issues | P0 Count | Recommended Model | Status |
|------|-------|--------|----------|-------------------|--------|
| lane/1 | shard_1 | 74 | 9 | llama3:8b | 🟢 Ready |
| lane/2 | shard_2 | 74 | 7 | llama3:8b | 🟢 Ready |
| lane/3 | shard_3 | 73 | 8 | llama3:8b | 🟢 Ready |
| lane/4 | shard_4 | 73 | 6 | llama3:8b | 🟢 Ready |

**Total**: 294 issues, 30 P0s, 28 batches × ~12 issues/batch

## Monitoring Commands

```bash
# Check overall progress
cat .github/agent_execution_progress.json | jq '.summary'

# See which agents are working on what
cat .github/agent_execution_progress.json | jq '.in_progress[]'

# List completed issues
cat .github/agent_execution_progress.json | jq '.completed[]'
```

## What Happens After

1. **Autonomous Execution**: Agents claim and execute batches independently
2. **Concurrent PRs**: Multiple agents may submit PRs simultaneously (safe: different issues)
3. **Progress Tracking**: Real-time tracking in `.github/agent_execution_progress.json`
4. **Completion**: Issues auto-closed after PR merge validation
5. **Audit Trail**: GitHub comments link all evidence (PRs, commits, test results)

## Questions?

If you have questions about:
- **What to implement**: Read the GitHub issue carefully; post clarification request if ambiguous
- **How to structure**: See `.github/instructions/cmd-cli.instructions.md` for specific domains
- **Code style**: See `.github/copilot-instructions.md` for Go conventions
- **Architecture**: See `CONTRIBUTING.md` and relevant domain instruction files

---

**Ready to start?** Run: `python3 scripts/agent_claim_work.py --claim --shard 1`

**Need help?** Check the [Autonomous Developer Agent Instructions](instructions/autonomous-dev.instructions.md)

