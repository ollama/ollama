#!/usr/bin/env python3
"""
Agent Work Claim Management

Agents use this script to:
1. Claim a batch from their assigned shard
2. Mark issues as in-progress, pr-submitted, or completed
3. Check their assignment and progress

Usage:
    # Check current execution status (default action)
    python3 scripts/agent_claim_work.py

    # Claim a batch from shard_1
    python3 scripts/agent_claim_work.py --claim --shard 1

    # Mark issue as in-progress
    python3 scripts/agent_claim_work.py --mark-issue 42 --issue-status in-progress

    # Mark issue as PR submitted
    python3 scripts/agent_claim_work.py --mark-issue 42 --issue-status pr-submitted --pr 1234

    # Mark issue as completed
    python3 scripts/agent_claim_work.py --mark-issue 42 --issue-status completed --pr 1234
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import hashlib
import fcntl
import tempfile

# Get repo root
REPO_ROOT = Path(__file__).parent.parent
GITHUB_DIR = REPO_ROOT / ".github"
WORKPACKS_DIR = GITHUB_DIR / "lane_workpacks"
PROGRESS_FILE = GITHUB_DIR / "agent_execution_progress.json"


def load_workpack(shard: int) -> dict:
    """Load a specific shard workpack."""
    workpack_path = WORKPACKS_DIR / f"shard_{shard}_workpack.json"
    if not workpack_path.exists():
        raise FileNotFoundError(f"Workpack not found: {workpack_path}")
    
    with open(workpack_path, 'r') as f:
        return json.load(f)


def load_progress() -> dict:
    """Load execution progress tracking with retry on lock."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.1 * (attempt + 1))
            else:
                raise


def save_progress(progress: dict) -> None:
    """Save execution progress tracking with atomic write and file locking."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Write to temporary file first
            temp_fd, temp_path = tempfile.mkstemp(dir=PROGRESS_FILE.parent, text=True)
            try:
                with os.fdopen(temp_fd, 'w') as temp_file:
                    json.dump(progress, temp_file, indent=2)
                # Atomic move
                import shutil
                shutil.move(temp_path, str(PROGRESS_FILE))
            except Exception:
                try:
                    os.close(temp_fd)
                except:
                    pass
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise
            break
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.1 * (attempt + 1))
            else:
                raise


def generate_agent_id():
    """Generate a unique agent ID based on hostname and time."""
    import socket
    hostname = socket.gethostname()
    timestamp = datetime.now().isoformat()
    return f"{hostname}-{timestamp[:10]}"


def claim_batch(shard: int, agent_id: str = None) -> dict:
    """
    Agent claims the next unclaimed batch from their shard.
    Returns the batch details.
    """
    if agent_id is None:
        agent_id = generate_agent_id()
    
    # Load workpack for this shard
    workpack = load_workpack(shard)
    progress = load_progress()
    
    # Find first unclaimed batch in this shard
    unclaimed_batch = None
    for batch in workpack.get('batches', []):
        batch_id = batch.get('batch_id', '')
        
        # Check if already claimed
        already_claimed = False
        for agent_batch in progress.get('active_agents', []):
            if agent_batch.get('claimed_batch_id') == batch_id:
                already_claimed = True
                break
        
        if not already_claimed:
            unclaimed_batch = batch
            break
    
    if unclaimed_batch is None:
        print(f"❌ No unclaimed batches available in shard_{shard}")
        print(f"   Check .github/agent_execution_progress.json for status")
        return None
    
    # Register this agent as claiming the batch
    batch_id = unclaimed_batch.get('batch_id', '')
    issue_count = unclaimed_batch.get('issue_count', 0)
    
    agent_record = {
        "agent_id": agent_id,
        "shard": shard,
        "claimed_batch_id": batch_id,
        "claim_timestamp": datetime.now().isoformat(),
        "issues_count": issue_count
    }
    
    progress['active_agents'].append(agent_record)
    progress['progress']['claimed'] += 1
    save_progress(progress)
    
    print(f"✅ Claimed batch {batch_id}")
    print(f"   Issues: {issue_count}")
    print(f"   Agent ID: {agent_id}")
    print(f"\n   Your issue numbers:")
    issue_numbers = unclaimed_batch.get('issue_numbers', [])
    for issue_num in issue_numbers[:5]:
        print(f"     - #{issue_num}")
    if len(issue_numbers) > 5:
        print(f"     ... and {len(issue_numbers) - 5} more")
    
    return {
        "agent_id": agent_id,
        "batch": unclaimed_batch,
        "batch_id": batch_id
    }


def mark_issue(issue_number: int, status: str, pr_number: int = None) -> None:
    """
    Mark an issue as in-progress, pr-submitted, or completed.
    """
    progress = load_progress()
    
    # Find and update the issue in progress lists
    for category in ['in_progress', 'pr_submitted', 'completed']:
        progress[category] = [i for i in progress.get(category, []) if i.get('issue_number') != issue_number]
    
    timestamp = datetime.now().isoformat()
    
    issue_record = {
        "issue_number": issue_number,
        "timestamp": timestamp
    }
    
    if pr_number:
        issue_record['pr_number'] = pr_number
    
    if status == 'in-progress':
        progress['in_progress'].append(issue_record)
        progress['progress']['in_progress'] = len(progress['in_progress'])
        print(f"✅ Issue #{issue_number} marked as IN-PROGRESS")
    elif status == 'pr-submitted':
        issue_record['pr_number'] = pr_number
        progress['pr_submitted'].append(issue_record)
        progress['progress']['pr_submitted'] = len(progress['pr_submitted'])
        print(f"✅ Issue #{issue_number} marked as PR-SUBMITTED (PR #{pr_number})")
    elif status == 'completed':
        issue_record['pr_number'] = pr_number
        progress['completed'].append(issue_record)
        progress['progress']['completed'] = len(progress['completed'])
        print(f"✅ Issue #{issue_number} marked as COMPLETED (PR #{pr_number})")
    else:
        print(f"❌ Unknown status: {status}")
        return
    
    save_progress(progress)


def show_status() -> None:
    """Display current execution status."""
    progress = load_progress()
    
    print("\n" + "="*70)
    print("AGENT EXECUTION PROGRESS")
    print("="*70)
    
    print(f"\nOverall Progress:")
    print(f"  Claimed:       {progress['progress']['claimed']}/28 batches")
    print(f"  In Progress:   {progress['progress']['in_progress']} issues")
    print(f"  PR Submitted:  {progress['progress']['pr_submitted']} issues")
    print(f"  Completed:     {progress['progress']['completed']} issues")
    
    if progress['active_agents']:
        print(f"\nActive Agents:")
        for agent in progress['active_agents']:
            print(f"  - {agent['agent_id']}")
            print(f"    Batch: {agent['claimed_batch_id']} ({agent['issues_count']} issues)")
            print(f"    Claimed: {agent['claim_timestamp'][:10]}")
    
    if progress['in_progress']:
        print(f"\nIn Progress Issues:")
        for issue in progress['in_progress'][:10]:
            print(f"  - #{issue['issue_number']}")
        if len(progress['in_progress']) > 10:
            print(f"  ... and {len(progress['in_progress']) - 10} more")
    
    if progress['pr_submitted']:
        print(f"\nPR Submitted Issues:")
        for issue in progress['pr_submitted'][:10]:
            pr_text = f" (PR #{issue['pr_number']})" if 'pr_number' in issue else ""
            print(f"  - #{issue['issue_number']}{pr_text}")
        if len(progress['pr_submitted']) > 10:
            print(f"  ... and {len(progress['pr_submitted']) - 10} more")
    
    if progress['completed']:
        print(f"\nCompleted Issues:")
        for issue in progress['completed'][:10]:
            pr_text = f" (PR #{issue['pr_number']})" if 'pr_number' in issue else ""
            print(f"  - #{issue['issue_number']}{pr_text}")
        if len(progress['completed']) > 10:
            print(f"  ... and {len(progress['completed']) - 10} more")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Agent Work Claim Management")
    
    parser.add_argument('--claim', action='store_true', help='Claim a batch from your shard')
    parser.add_argument('--shard', type=int, choices=[1, 2, 3, 4], help='Shard number (1-4)')
    parser.add_argument('--agent-id', type=str, help='Custom agent ID (auto-generated if not provided)')
    
    parser.add_argument('--mark-issue', type=int, help='Mark issue with given number')
    parser.add_argument('--issue-status', type=str, choices=['in-progress', 'pr-submitted', 'completed'],
                        help='Status to mark issue as')
    parser.add_argument('--pr', type=int, help='PR number (required for pr-submitted and completed)')
    
    args = parser.parse_args()
    
    # Handle claim operation
    if args.claim:
        if not args.shard:
            parser.error('--shard is required with --claim')
        result = claim_batch(args.shard, args.agent_id)
        if result:
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Handle mark-issue operation
    if args.mark_issue:
        if not args.issue_status:
            parser.error('--issue-status is required with --mark-issue')
        if args.issue_status in ['pr-submitted', 'completed'] and not args.pr:
            parser.error('--pr is required for pr-submitted and completed status')
        mark_issue(args.mark_issue, args.issue_status, args.pr)
        sys.exit(0)
    
    # Default: show status
    show_status()


if __name__ == '__main__':
    main()
