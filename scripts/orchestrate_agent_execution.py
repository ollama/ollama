#!/usr/bin/env python3
"""
Autonomous Execution Orchestrator

Coordinates autonomous agents across all 4 lanes to execute the full 
triage workflow. Can be used to:

1. Start agents in parallel across lanes
2. Monitor progress in real-time
3. Handle failures and recovery
4. Report completion status

Usage:
    # Start all 4 agents (one per lane)
    python3 scripts/orchestrate_agent_execution.py --start-all

    # Monitor progress
    python3 scripts/orchestrate_agent_execution.py --monitor --interval 10

    # Run with test limit (5 issues per batch)
    python3 scripts/orchestrate_agent_execution.py --start-all --test --limit 5

    # Show current status
    python3 scripts/orchestrate_agent_execution.py --status
"""

import sys
import os
import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add scripts directory to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agent_claim_work import load_progress, save_progress, show_status


def run_agent(shard: int, agent_name: str, limit: int = None, test_mode: bool = False) -> dict:
    """
    Run an autonomous agent for a specific shard.
    
    Returns:
        dict with execution results
    """
    
    print(f"\n[{agent_name}] Starting agent for shard_{shard}...")
    
    cmd = [
        "python3",
        "scripts/run_autonomous_agent.py",
        "--shard", str(shard)
    ]
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per shard
        )
        
        success = result.returncode == 0
        
        return {
            "shard": shard,
            "agent_name": agent_name,
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "timestamp": datetime.now().isoformat()
        }
    
    except subprocess.TimeoutExpired:
        return {
            "shard": shard,
            "agent_name": agent_name,
            "success": False,
            "error": f"Agent execution timed out after 1 hour",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "shard": shard,
            "agent_name": agent_name,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def start_all_agents(limit: int = None, test_mode: bool = False) -> dict:
    """
    Start agents in parallel for all 4 shards.
    
    Returns:
        dict with results from all agents
    """
    
    print(f"\n{'='*70}")
    print(f"ORCHESTRATOR: Starting autonomous agents")
    print(f"{'='*70}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Shards: 4 (parallel)")
    if limit:
        print(f"Limit per batch: {limit} issues")
    if test_mode:
        print(f"Mode: TEST (first batch only)")
    print(f"{'='*70}\n")
    
    results = {
        "start_timestamp": datetime.now().isoformat(),
        "test_mode": test_mode,
        "limit": limit,
        "agents": {}
    }
    
    # Run all 4 agents in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_agent, shard, f"Agent-{shard}", limit, test_mode): shard
            for shard in [1, 2, 3, 4]
        }
        
        for future in as_completed(futures):
            shard = futures[future]
            try:
                result = future.result()
                results['agents'][f'shard_{shard}'] = result
                
                if result['success']:
                    print(f"\n✅ Agent-{shard} completed successfully")
                else:
                    print(f"\n❌ Agent-{shard} encountered errors:")
                    if 'error' in result:
                        print(f"   {result['error']}")
                    else:
                        # Show last few lines of stderr
                        if result['stderr']:
                            lines = result['stderr'].strip().split('\n')[-5:]
                            for line in lines:
                                print(f"   {line}")
            
            except Exception as e:
                results['agents'][f'shard_{shard}'] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"\n❌ Agent-{shard} failed with exception: {e}")
    
    results['end_timestamp'] = datetime.now().isoformat()
    
    # Calculate summary
    successful = sum(1 for agent in results['agents'].values() if agent.get('success', False))
    results['summary'] = {
        "total_agents": 4,
        "successful": successful,
        "failed": 4 - successful
    }
    
    return results


def monitor_progress(interval: int = 10, duration: int = None):
    """
    Monitor execution progress in real-time.
    
    Args:
        interval: Seconds between status updates
        duration: Maximum seconds to monitor (None for infinite)
    """
    
    print(f"\n{'='*70}")
    print(f"ORCHESTRATOR: Monitoring progress")
    print(f"{'='*70}")
    print(f"Refresh interval: {interval}s")
    if duration:
        print(f"Monitor duration: {duration}s")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    
    while True:
        # Show current progress
        show_status()
        
        # Check if we should stop monitoring
        if duration:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= duration:
                print(f"\nMonitoring duration reached ({duration}s)")
                break
        
        # Wait for next update
        try:
            print(f"\nNext update in {interval}s (Ctrl+C to stop)...\n")
            time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n\nMonitoring stopped by user")
            break


def show_execution_report():
    """Generate and display execution report."""
    
    progress = load_progress()
    
    print(f"\n{'='*70}")
    print(f"EXECUTION REPORT - {datetime.now().isoformat()}")
    print(f"{'='*70}\n")
    
    print(f"Overall Progress:")
    print(f"  Total Issues: {progress['total_issues']}")
    print(f"  Total Batches: {progress['total_batches']}")
    print(f"  Claimed Batches: {progress['progress']['claimed']}/28")
    print(f"  In Progress: {progress['progress']['in_progress']}")
    print(f"  PR Submitted: {progress['progress']['pr_submitted']}")
    print(f"  Completed: {progress['progress']['completed']}")
    
    completion_pct = (progress['progress']['completed'] / progress['total_issues']) * 100 if progress['total_issues'] > 0 else 0
    print(f"\n  Completion: {completion_pct:.1f}% ({progress['progress']['completed']}/{progress['total_issues']} issues)")
    
    # Show lane breakdown
    print(f"\nLane Breakdown:")
    for lane_key, lane_data in progress['lanes'].items():
        shard = lane_data['shard']
        assigned = lane_data['issues_assigned']
        # Count completed issues in this lane
        completed_in_lane = sum(1 for issue in progress['completed'] 
                               if issue.get('shard') == lane_key.replace('lane_', ''))
        print(f"  {lane_key}: {completed_in_lane}/{assigned} issues completed")
    
    # Show active agents
    if progress['active_agents']:
        print(f"\nActive Agents ({len(progress['active_agents'])}):")
        for agent in progress['active_agents']:
            print(f"  - {agent['agent_id']}")
            print(f"    Batch: {agent['claimed_batch_id']}")
            print(f"    Claimed: {agent['claim_timestamp'][:10]}")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrator for autonomous agent execution",
        epilog="Example: python3 scripts/orchestrate_agent_execution.py --start-all"
    )
    
    parser.add_argument('--start-all', action='store_true',
                        help='Start all 4 agents in parallel')
    parser.add_argument('--monitor', action='store_true',
                        help='Monitor execution progress')
    parser.add_argument('--interval', type=int, default=10,
                        help='Monitor refresh interval in seconds (default: 10)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Monitor for N seconds (default: infinite)')
    parser.add_argument('--status', action='store_true',
                        help='Show current execution status')
    parser.add_argument('--report', action='store_true',
                        help='Show detailed execution report')
    parser.add_argument('--test', action='store_true',
                        help='Test mode (limit issues per batch)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit issues per batch (for testing)')
    
    args = parser.parse_args()
    
    # If no action specified, show status
    if not any([args.start_all, args.monitor, args.status, args.report]):
        args.status = True
    
    # Execute requested actions
    if args.start_all:
        results = start_all_agents(limit=args.limit, test_mode=args.test)
        print(f"\n{'='*70}")
        print(f"EXECUTION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Agents: {results['summary']['total_agents']}")
        print(f"Successful: {results['summary']['successful']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Duration: {results['start_timestamp']} → {results['end_timestamp']}")
        print(f"{'='*70}\n")
    
    if args.monitor:
        monitor_progress(interval=args.interval, duration=args.duration)
    
    if args.status:
        show_status()
    
    if args.report:
        show_execution_report()


if __name__ == '__main__':
    main()
