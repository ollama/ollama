#!/usr/bin/env python3
"""
Cleanup stale branches according to governance rules.
Maintains audit trail and prevents accidental deletions.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional


class BranchCleanupManager:
    def __init__(self, age_days: int = 180, exclude_patterns: List[str] = None, dry_run: bool = True):
        self.age_days = age_days
        self.exclude_patterns = exclude_patterns or ['main', 'release']
        self.dry_run = dry_run
        self.now = datetime.utcnow()
        self.audit_log = []
        self.cleanup_config = self._load_governance_config()

    def _load_governance_config(self) -> Dict:
        """Load governance rules from IaC config."""
        config_path = Path('.github/branch-governance.iac.json')
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load governance config: {e}")
        return {'governance': {'enabled': True}}

    def _get_branch_info(self, branch: str) -> Dict:
        """Get branch metadata: age, last commit, activity."""
        try:
            # Get last commit timestamp
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%cI', f'origin/{branch}'],
                capture_output=True, text=True, check=True
            )
            if not result.stdout.strip():
                return None

            commit_date = datetime.fromisoformat(result.stdout.strip().replace('Z', '+00:00'))
            age_days = (self.now - commit_date).days

            # Get commit hash
            result_hash = subprocess.run(
                ['git', 'log', '-1', '--format=%H', f'origin/{branch}'],
                capture_output=True, text=True, check=True
            )
            commit_hash = result_hash.stdout.strip()

            return {
                'branch': branch,
                'last_commit_date': commit_date.isoformat(),
                'age_days': age_days,
                'commit_hash': commit_hash,
                'is_merged': self._is_merged(branch)
            }
        except Exception as e:
            return None

    def _is_merged(self, branch: str) -> bool:
        """Check if branch is merged to main."""
        try:
            result = subprocess.run(
                ['git', 'merge-base', '--is-ancestor', f'origin/{branch}', 'origin/main'],
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False

    def _should_exclude(self, branch: str) -> bool:
        """Check if branch matches exclusion patterns."""
        for pattern in self.exclude_patterns:
            if branch.startswith(pattern) or branch == pattern:
                return True
        return False

    def _should_delete(self, branch_info: Dict) -> Tuple[bool, str]:
        """Determine if a branch should be deleted."""
        if not branch_info:
            return False, "no_info"

        branch = branch_info['branch']
        age = branch_info['age_days']
        is_merged = branch_info['is_merged']

        # Don't delete protected branches
        if self._should_exclude(branch):
            return False, "excluded_pattern"

        # Don't delete recent branches
        if age < self.age_days:
            return False, f"too_new_{age}_days"

        # Only delete if stale AND unmerged (careful!)
        # OR if merged to main (safe to delete)
        if is_merged:
            return True, "merged_to_main"
        elif age > self.age_days * 2:
            return True, "stale_unmerged"

        return False, "not_ready"

    def _delete_branch(self, branch: str) -> Tuple[bool, str]:
        """Delete a branch from remote."""
        if self.dry_run:
            return True, "dry_run"

        try:
            result = subprocess.run(
                ['git', 'push', 'origin', '--delete', branch],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return True, "deleted"
            else:
                return False, result.stderr or "unknown_error"
        except Exception as e:
            return False, str(e)

    def cleanup(self) -> Dict:
        """Execute cleanup according to policy."""
        # Get all remote branches
        result = subprocess.run(
            ['git', 'branch', '-r', '--format=%(refname:short)'],
            capture_output=True, text=True, check=True
        )

        branches = [b.strip().replace('origin/', '') for b in result.stdout.strip().split('\n')
                   if b.strip() and 'HEAD' not in b]

        report = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'dry_run': self.dry_run,
            'policy': {
                'max_age_days': self.age_days,
                'excluded_patterns': self.exclude_patterns
            },
            'results': {
                'total_branches': len(branches),
                'deleted': [],
                'protected': [],
                'too_new': [],
                'errors': []
            }
        }

        for branch in branches:
            info = self._get_branch_info(branch)
            should_delete, reason = self._should_delete(info)

            if should_delete:
                success, msg = self._delete_branch(branch)
                if success:
                    report['results']['deleted'].append({
                        'branch': branch,
                        'reason': reason,
                        'age_days': info['age_days'],
                        'commit': info['commit_hash']
                    })
                    self._log_audit(branch, reason, info, 'deleted')
                else:
                    report['results']['errors'].append({
                        'branch': branch,
                        'error': msg
                    })
                    self._log_audit(branch, reason, info, 'failed', msg)
            else:
                if reason == 'excluded_pattern':
                    report['results']['protected'].append({
                        'branch': branch,
                        'reason': reason
                    })
                else:
                    report['results']['too_new'].append({
                        'branch': branch,
                        'reason': reason
                    })

        return report

    def _log_audit(self, branch: str, reason: str, info: Dict, action: str, error: str = None):
        """Log action to audit trail."""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'branch': branch,
            'action': action,
            'reason': reason,
            'age_days': info.get('age_days', -1),
            'commit': info.get('commit_hash', 'unknown'),
            'last_commit_date': info.get('last_commit_date', 'unknown'),
            'is_merged': info.get('is_merged', False)
        }
        if error:
            audit_entry['error'] = error

        # Append to immutable log
        audit_path = Path('.github/branch_cleanup_audit.jsonl')
        with open(audit_path, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')

    def save_report(self, report: Dict, filename: str = '.github/branch_cleanup_report.json'):
        """Save cleanup report to file."""
        Path('.github').mkdir(exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Cleanup stale branches')
    parser.add_argument('--age-days', type=int, default=180, help='Max age in days')
    parser.add_argument('--exclude-patterns', type=str, default='main,release,automation',
                       help='Comma-separated patterns to exclude')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Dry run mode')
    parser.add_argument('--execute', action='store_true', help='Actually delete branches')
    parser.add_argument('--report-file', default='.github/branch_cleanup_report.json')

    args = parser.parse_args()

    # Fetch latest branch info
    subprocess.run(['git', 'fetch', 'origin', '-p'], check=True)  # -p: prune

    manager = BranchCleanupManager(
        age_days=args.age_days,
        exclude_patterns=args.exclude_patterns.split(','),
        dry_run=not args.execute
    )

    report = manager.cleanup()
    manager.save_report(report, args.report_file)

    # Print summary
    deleted_count = len(report['results']['deleted'])
    protected_count = len(report['results']['protected'])
    error_count = len(report['results']['errors'])

    print(f"\n{'='*70}")
    print(f"Branch Cleanup Report")
    print(f"{'='*70}")
    print(f"Dry Run: {report['dry_run']}")
    print(f"Branches deleted:  {deleted_count}")
    print(f"Branches protected: {protected_count}")
    print(f"Branches errors:   {error_count}")
    print(f"{'='*70}\n")

    if deleted_count > 0:
        print("Deleted branches:")
        for item in report['results']['deleted']:
            print(f"  ✓ {item['branch']} ({item['age_days']} days old)")

    if error_count > 0:
        print("\nErrors:")
        for item in report['results']['errors']:
            print(f"  ✗ {item['branch']}: {item['error']}")

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
