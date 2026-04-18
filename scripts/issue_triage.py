#!/usr/bin/env python3
"""
Autonomous issue triage system.
Classifies, categorizes, and routes issues according to governance rules.
"""

import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class IssueTriageAgent:
    """Autonomous agent for issue triage and categorization."""

    def __init__(self, repo: str, token: str):
        self.repo = repo
        self.token = token
        self.owner, self.repo_name = repo.split('/')
        self.base_url = f"https://api.github.com/repos/{repo}"
        self.governance = self._load_governance_config()
        self.triage_rules = self._load_triage_rules()
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github+json',
            'User-Agent': 'issue-triage-agent'
        }

    def _load_governance_config(self) -> Dict:
        """Load issue governance configuration."""
        config_file = Path('.github/issue-governance.iac.json')
        if config_file.exists():
            with open(config_file) as f:
                return json.load(f)
        return {}

    def _load_triage_rules(self) -> Dict:
        """Load issue triage rules."""
        rules_file = Path('.github/issue-triage.iac.json')
        if rules_file.exists():
            with open(rules_file) as f:
                return json.load(f)
        return {}

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

    def classify_issue(self, issue: Dict) -> Dict:
        """Classify issue based on content."""
        title = issue.get('title', '').lower()
        body = issue.get('body', '').lower()
        content = f"{title} {body}"

        classification = {
            'category': 'other',
            'priority': 'medium',
            'confidence': 0.0,
            'suggested_labels': []
        }

        # Check auto-classification rules
        auto_rules = self.triage_rules.get('auto_classification', {}).get('rules', [])

        for rule in auto_rules:
            keywords = rule.get('detect', '').split(':')[1] if ':' in rule.get('detect', '') else ''
            matches = any(kw.strip() in content for kw in keywords.split('|'))

            if matches:
                confidence = rule.get('confidence', 0.7)
                if confidence > classification['confidence']:
                    classification['category'] = rule.get('classify_as', 'other')
                    classification['confidence'] = confidence

        return classification

    def extract_requirements(self, issue: Dict) -> Dict:
        """Extract requirements from issue."""
        body = issue.get('body', '')

        requirements = {
            'has_description': len(body) > 50,
            'has_acceptance_criteria': 'acceptance criteria' in body.lower() or '- [ ]' in body,
            'has_reproduction_steps': 'steps to reproduce' in body.lower(),
            'has_expected_behavior': 'expected' in body.lower(),
            'has_actual_behavior': 'actual' in body.lower(),
            'issues_referenced': self._extract_issue_references(body),
            'todos': self._extract_todos(body)
        }

        return requirements

    def _extract_issue_references(self, text: str) -> List[int]:
        """Extract issue numbers referenced in text."""
        import re
        matches = re.findall(r'#(\d+)', text)
        return [int(m) for m in matches]

    def _extract_todos(self, text: str) -> List[str]:
        """Extract todo items from issue."""
        import re
        todos = re.findall(r'- \[ \] (.+)', text)
        return todos

    def get_suggested_labels(self, issue: Dict, classification: Dict) -> List[str]:
        """Get suggested labels for issue."""
        labels = [f"type:{classification['category']}"]
        labels.append(f"priority:{classification['priority']}")
        labels.append("needs-triage")

        requirements = self.extract_requirements(issue)
        if not requirements['has_acceptance_criteria']:
            labels.append("needs-acceptance-criteria")
        if not requirements['has_description']:
            labels.append("needs-details")

        return labels

    def apply_labels(self, issue_number: int, labels: List[str]) -> bool:
        """Apply labels to issue."""
        status, _ = self._api_call(
            'POST',
            f"{self.base_url}/issues/{issue_number}/labels",
            {'labels': labels}
        )
        return status == 200

    def create_triage_comment(self, issue_number: int, issue: Dict, classification: Dict, requirements: Dict) -> bool:
        """Create triage comment on issue."""
        comment = f"""## 🤖 Automated Triage

**Category:** {classification['category']}
**Priority:** {classification['priority']}
**Confidence:** {classification['confidence']*100:.0f}%

### Requirements Check
- Description: {'✅' if requirements['has_description'] else '❌'}
- Acceptance Criteria: {'✅' if requirements['has_acceptance_criteria'] else '❌ Please add acceptance criteria'}
- Reproduction Steps: {'✅' if requirements['has_reproduction_steps'] else '⚠️ (if bug)'}
- Expected Behavior: {'✅' if requirements['has_expected_behavior'] else '⚠️ (if bug)'}

### Next Steps
1. Review proposed category and priority
2. Add any missing information
3. Assign to developer when ready
4. Create linked branch when starting

---
*This bot helps triage issues. Please ensure all acceptance criteria are included before starting implementation.*
"""

        status, _ = self._api_call(
            'POST',
            f"{self.base_url}/issues/{issue_number}/comments",
            {'body': comment}
        )
        return status == 201

    def triage_issue(self, issue_number: int) -> Dict:
        """Triage a single issue."""
        result = {
            'issue_number': issue_number,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'status': 'failed',
            'details': {}
        }

        # Fetch issue
        status, issue = self._api_call('GET', f"{self.base_url}/issues/{issue_number}")
        if status != 200:
            result['error'] = f"Failed to fetch issue: {status}"
            return result

        # Skip if pull request
        if 'pull_request' in issue:
            result['status'] = 'skipped'
            result['details']['reason'] = 'is_pull_request'
            return result

        # Skip if already processed (has triage comment)
        comments_status, comments = self._api_call(
            'GET',
            f"{self.base_url}/issues/{issue_number}/comments?per_page=100"
        )
        if comments_status == 200 and isinstance(comments, list):
            has_triage = any('🤖 Automated Triage' in (c.get('body') or '') for c in comments)
            if has_triage:
                result['status'] = 'skipped'
                result['details']['reason'] = 'already_triaged'
                return result

        # Classify issue
        classification = self.classify_issue(issue)
        requirements = self.extract_requirements(issue)
        suggested_labels = self.get_suggested_labels(issue, classification)

        result['details'] = {
            'classification': classification,
            'requirements': requirements,
            'suggested_labels': suggested_labels,
            'existing_labels': [l['name'] for l in issue.get('labels', [])]
        }

        # Apply labels
        labels_ok = self.apply_labels(issue_number, suggested_labels)
        result['details']['applied_labels'] = labels_ok

        # Create comment
        comment_ok = self.create_triage_comment(
            issue_number, issue, classification, requirements
        )
        result['details']['comment_created'] = comment_ok

        result['status'] = 'success' if labels_ok and comment_ok else 'partial'
        return result

def main():
    """Main entry point."""
    repo = os.environ.get('GITHUB_REPOSITORY', 'kushin77/ollama')
    token = os.environ.get('GITHUB_TOKEN', '')

    if not token:
        print("ERROR: GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    issue_number = int(os.environ.get('GITHUB_ISSUE_NUMBER', '0'))

    if issue_number == 0:
        print("ERROR: GITHUB_ISSUE_NUMBER not set", file=sys.stderr)
        sys.exit(1)

    # Create agent and triage
    agent = IssueTriageAgent(repo, token)
    result = agent.triage_issue(issue_number)

    # Save result
    audit_file = Path('.github/issue_audit_trail.jsonl')
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_file, 'a') as f:
        f.write(json.dumps(result) + '\n')

    # Print summary
    print(json.dumps(result, indent=2))

    return 0 if result['status'] != 'failed' else 1

if __name__ == '__main__':
    sys.exit(main())
