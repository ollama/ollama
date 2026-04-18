#!/usr/bin/env python3
"""
Batch issue processor for autonomous triage and management.
Processes multiple issues in a single execution.
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

class BatchIssueProcessor:
    """Process multiple issues in batch."""

    def __init__(self, repo: str, token: str, dry_run: bool = True):
        self.repo = repo
        self.token = token
        self.dry_run = dry_run
        self.owner, self.repo_name = repo.split('/')
        self.base_url = f"https://api.github.com/repos/{repo}"
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github+json',
            'User-Agent': 'batch-issue-processor'
        }
        self.results = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'repository': repo,
            'dry_run': dry_run,
            'processed': 0,
            'updated': 0,
            'closed': 0,
            'issues': []
        }

    def _api_call(self, method: str, url: str, data: Optional[Dict] = None) -> Tuple[int, Dict]:
        """Make GitHub API call."""
        headers = dict(self.headers)
        body = None

        if data is not None:
            headers['Content-Type'] = 'application/json'
            body = json.dumps(data).encode()

        req = urllib.request.Request(url, method=method, headers=headers, data=body)

        try:
            with urllib.request.urlopen(req) as resp:
                response_body = resp.read().decode()
                return resp.status, json.loads(response_body) if response_body else {}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            try:
                error_data = json.loads(error_body) if error_body else {}
            except:
                error_data = {'raw': error_body}
            return e.code, error_data

    def fetch_issues(self, state: str = 'open', limit: int = 50) -> List[Dict]:
        """Fetch issues from repository."""
        issues = []
        page = 1

        while len(issues) < limit or limit == 0:
            per_page = min(100, limit - len(issues)) if limit > 0 else 100
            status, data = self._api_call(
                'GET',
                f"{self.base_url}/issues?state={state}&per_page={per_page}&page={page}"
            )

            if status != 200 or not data:
                break

            # Filter out pull requests
            page_issues = [i for i in data if 'pull_request' not in i]
            issues.extend(page_issues)

            if len(data) < per_page:
                break

            page += 1

        return issues[:limit] if limit > 0 else issues

    def update_issue_labels(self, issue_number: int, labels: List[str]) -> bool:
        """Update issue labels."""
        if self.dry_run:
            return True

        status, _ = self._api_call(
            'POST',
            f"{self.base_url}/issues/{issue_number}/labels",
            {'labels': labels}
        )
        return status == 200

    def update_issue_milestone(self, issue_number: int, milestone_number: int) -> bool:
        """Update issue milestone."""
        if self.dry_run:
            return True

        status, _ = self._api_call(
            'PATCH',
            f"{self.base_url}/issues/{issue_number}",
            {'milestone': milestone_number}
        )
        return status == 200

    def process_stale_issues(self, days_threshold: int = 60) -> List[int]:
        """Process stale issues (no activity)."""
        cutoff_date = (datetime.utcnow() - timedelta(days=days_threshold)).isoformat()
        issues = self.fetch_issues(state='open', limit=0)

        stale = []
        for issue in issues:
            updated_at = issue.get('updated_at', '')
            if updated_at < cutoff_date and not any(l['name'] == 'pinned' for l in issue.get('labels', [])):
                stale.append(issue['number'])

        return stale

    def process_issues(self, issues: List[Dict]) -> Dict:
        """Process all issues."""
        for issue in issues:
            issue_num = issue['number']
            self.results['processed'] += 1

            issue_result = {
                'number': issue_num,
                'title': issue.get('title'),
                'status': 'processed',
                'changes': []
            }

            # Check for missing acceptance criteria
            body = (issue.get('body') or '').lower()
            if 'acceptance criteria' not in body and '- [ ]' not in body:
                if self.update_issue_labels(issue_num, ['needs-acceptance-criteria']):
                    issue_result['changes'].append('added:needs-acceptance-criteria')
                    self.results['updated'] += 1

            # Check for stale issues
            updated_at = datetime.fromisoformat(
                issue['updated_at'].replace('Z', '+00:00')
            )
            days_since_update = (datetime.utcnow().replace(tzinfo=updated_at.tzinfo) - updated_at).days

            if days_since_update > 60:
                if self.update_issue_labels(issue_num, ['stale']):
                    issue_result['changes'].append('added:stale')
                    self.results['updated'] += 1

            self.results['issues'].append(issue_result)

        return self.results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Batch process GitHub issues')
    parser.add_argument('--limit', type=int, default=50, help='Max issues to process (0=all)')
    parser.add_argument('--state', choices=['open', 'closed', 'all'], default='open')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Dry run mode')

    args = parser.parse_args()

    repo = os.environ.get('GITHUB_REPOSITORY', 'kushin77/ollama')
    token = os.environ.get('GITHUB_TOKEN', '')

    if not token:
        print("ERROR: GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    processor = BatchIssueProcessor(repo, token, dry_run=args.dry_run)

    issues = processor.fetch_issues(state=args.state, limit=args.limit)
    results = processor.process_issues(issues)

    # Save results
    output_file = Path('.github') / f"issue_batch_report_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\n✅ Report saved to {output_file}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
