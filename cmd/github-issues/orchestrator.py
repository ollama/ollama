#!/usr/bin/env python3
"""
GitHub Issues Orchestrator
Unified system for triage, execution, and management

Integrates:
- Issue triage and classification
- AI-agnostic execution (Claude, Grok, Gemini)
- Immutable state management
- IaC-based workflows

Usage:
  python orchestrator.py --action triage --limit 10
  python orchestrator.py --action execute --issue 15649 --ai claude,grok
  python orchestrator.py --action status --issue 15649
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Import local modules
sys.path.insert(0, str(Path(__file__).parent))

def print_banner():
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║  GitHub Issues Orchestrator - IaC, Immutable, Independent             ║
║  Multi-AI Compatible Execution Framework                              ║
╚════════════════════════════════════════════════════════════════════════╝
    """)

def cmd_triage(args):
    """Triage issues"""
    print(f"\n🔍 Triaging issues (limit: {args.limit})...")
    
    # Load triage rules
    with open('.github/triage.rules.json') as f:
        rules = json.load(f)
    
    print(f"✓ Loaded {len(rules['rules'])} triage rules")
    print(f"  Priority scoring system: 0-100")
    print(f"  Severity levels: Critical, High, Medium, Low")
    print(f"  Auto-actions: Labels, routing, notifications")
    
    # Would import and run triage.py here
    print(f"\n✅ To start triaging:")
    print(f"   python cmd/github-issues/triage.py")

def cmd_execute(args):
    """Execute fixes using AI systems"""
    print(f"\n🤖 Executing fix for issue #{args.issue}")
    
    ai_systems = args.ai.split(',') if args.ai else ['claude']
    print(f"   Using AI systems: {', '.join(ai_systems)}")
    print(f"   Phase: Analysis -> Planning -> Implementation -> Validation")
    
    # Would import and run ai_executor.py here
    print(f"\n✅ To start execution:")
    print(f"   python cmd/github-issues/ai_executor.py")

def cmd_status(args):
    """Show issue status"""
    print(f"\n📊 Status of issue #{args.issue}")
    
    # Load state
    state_file = Path(f'.github/state/issue_{args.issue}.jsonl')
    if state_file.exists():
        print(f"\n📜 State transitions:")
        with open(state_file) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    print(f"  • {record['timestamp']}: {record['content'].get('state', '?')}")
    else:
        print(f"  No state history found")
    
    # Load execution log
    exec_file = Path(f'.github/ai_execution_log.json')
    if exec_file.exists():
        with open(exec_file) as f:
            log = json.load(f)
            if log.get('issue_num') == args.issue:
                print(f"\n🤖 AI Execution Results:")
                print(f"  Status: {log['results'].get('status', '?')}")
                print(f"  Consensus: {log['results'].get('consensus', {}).get('consensus_status', '?')}")

def cmd_report(args):
    """Generate comprehensive report"""
    print(f"\n📈 Generating report...")
    
    # Load triage snapshot
    snapshot_file = Path('.github/triage_snapshot.json')
    if snapshot_file.exists():
        with open(snapshot_file) as f:
            snapshot = json.load(f)
            print(f"\n📊 Triage Summary:")
            print(f"  Total issues analyzed: {snapshot['total_issues_analyzed']}")
            print(f"  Successfully triaged: {snapshot['triaged_count']}")
            
            # Group by severity
            severity_map = {}
            for result in snapshot.get('triage_results', []):
                issue = result.get('issue', {})
                severity = issue.get('severity', 'unknown')
                severity_map[severity] = severity_map.get(severity, 0) + 1
            
            print(f"\n  By Severity:")
            for severity in ['critical', 'high', 'medium', 'low']:
                count = severity_map.get(severity, 0)
                if count > 0:
                    print(f"    - {severity.capitalize()}: {count}")
    
    print(f"\n✅ Report generation complete")

def cmd_plan(args):
    """Generate implementation plan"""
    print(f"\n📋 Generating implementation plan...")
    
    plan = {
        'generated_at': datetime.now().isoformat(),
        'version': '1.0',
        'phases': [
            {
                'phase': 1,
                'name': 'Issue Triage',
                'description': 'Classify and prioritize all GitHub issues',
                'tools': ['triage.py'],
                'output': '.github/triage_snapshot.json',
                'timeline': '1 hour'
            },
            {
                'phase': 2,
                'name': 'AI Analysis',
                'description': 'Run issues through multiple AI systems',
                'tools': ['ai_executor.py'],
                'output': '.github/ai_execution_log.json',
                'timeline': '2-4 hours',
                'ai_systems': ['claude', 'grok', 'gemini']
            },
            {
                'phase': 3,
                'name': 'Consensus Building',
                'description': 'Reach consensus across AI systems',
                'consensus_mechanism': 'majority_vote',
                'output': 'consensus_recommendations.json',
                'timeline': '1 hour'
            },
            {
                'phase': 4,
                'name': 'Implementation',
                'description': 'Implement fixes based on consensus',
                'review_required': True,
                'testing_required': True,
                'timeline': 'variable'
            },
            {
                'phase': 5,
                'name': 'Validation',
                'description': 'Validate fixes and track outcomes',
                'regression_testing': True,
                'metrics_collection': True,
                'timeline': 'variable'
            }
        ],
        'success_criteria': [
            'All critical issues triaged',
            'AI consensus reached on fixes',
            'Implementations pass tests',
            'No regressions detected',
            'Documentation updated'
        ]
    }
    
    # Save plan
    with open('.github/implementation_plan.json', 'w') as f:
        json.dump(plan, f, indent=2)
    
    print(f"\n📋 Implementation Plan:")
    for phase in plan['phases']:
        print(f"\n  Phase {phase['phase']}: {phase['name']}")
        print(f"    {phase['description']}")
        print(f"    Timeline: {phase['timeline']}")
    
    print(f"\n✅ Plan saved to: .github/implementation_plan.json")

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='GitHub Issues Orchestrator - Unified Management System'
    )
    
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Triage command
    triage_parser = subparsers.add_parser('triage', help='Triage GitHub issues')
    triage_parser.add_argument('--limit', type=int, default=10,
                              help='Number of issues to triage (default: 10)')
    triage_parser.set_defaults(func=cmd_triage)
    
    # Execute command
    exec_parser = subparsers.add_parser('execute', help='Execute fixes using AI')
    exec_parser.add_argument('--issue', type=int, required=True,
                            help='Issue number to fix')
    exec_parser.add_argument('--ai', default='claude',
                            help='AI systems to use (comma-separated: claude,grok,gemini)')
    exec_parser.set_defaults(func=cmd_execute)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check issue status')
    status_parser.add_argument('--issue', type=int, required=True,
                              help='Issue number')
    status_parser.set_defaults(func=cmd_status)
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate report')
    report_parser.set_defaults(func=cmd_report)
    
    # Plan command
    plan_parser = subparsers.add_parser('plan', help='Generate implementation plan')
    plan_parser.set_defaults(func=cmd_plan)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1
    else:
        parser.print_help()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
