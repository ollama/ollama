#!/usr/bin/env python3
"""
GitHub API Handler for Issue Updates
Manages: label updates, issue closure, comments, state transitions
"""

import json
import os
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re


class GitHubAPIHandler:
    """Handles GitHub API operations for issue management"""

    def __init__(self, repo: str = "kushin77/ollama", token: str = None):
        """
        Initialize GitHub API handler

        Args:
            repo: Repository in format "owner/repo"
            token: GitHub Personal Access Token (defaults to GITHUB_TOKEN env var)
        """
        self.repo = repo
        self.token = (
            token
            or os.getenv('OLLAMA_GITHUB_TOKEN')
            or os.getenv('GITHUB_TOKEN')
            or os.getenv('GH_TOKEN')
        )
        self.base_url = f"https://api.github.com/repos/{repo}"
        self.rate_limit_remaining = None
        self.rate_limit_reset = None

    def _make_request(self, method: str, url: str, data: Optional[Dict] = None) -> Tuple[int, Dict]:
        """Make HTTP request using urllib"""
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Ollama-Issue-Executor',
            'Content-Type': 'application/json'
        }

        if self.token:
            headers['Authorization'] = f'token {self.token}'

        request_data = None
        if data:
            request_data = json.dumps(data).encode('utf-8')

        req = urllib.request.Request(url, data=request_data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req) as response:
                status = response.status
                resp_data = json.loads(response.read().decode('utf-8'))

                # Update rate limit
                if 'X-RateLimit-Remaining' in response.headers:
                    self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
                if 'X-RateLimit-Reset' in response.headers:
                    self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])

                return status, resp_data
        except urllib.error.HTTPError as e:
            status = e.code
            if 'X-RateLimit-Remaining' in e.headers:
                try:
                    self.rate_limit_remaining = int(e.headers['X-RateLimit-Remaining'])
                except Exception:
                    pass
            if 'X-RateLimit-Reset' in e.headers:
                try:
                    self.rate_limit_reset = int(e.headers['X-RateLimit-Reset'])
                except Exception:
                    pass
            try:
                resp_data = json.loads(e.read().decode('utf-8'))
            except:
                resp_data = {'error': str(e)}
            return status, resp_data
        except Exception as e:
            return 500, {'error': str(e)}

    def check_repo_access(self) -> Tuple[bool, str]:
        """Check if repository is accessible"""
        try:
            status, resp = self._make_request('GET', f"{self.base_url}")
            if status == 401:
                return False, "Authentication failed (invalid token)"
            elif status == 403:
                msg = resp.get('message', '') if isinstance(resp, dict) else ''
                if 'rate limit exceeded' in msg.lower():
                    reset_info = f"; reset={self.rate_limit_reset}" if self.rate_limit_reset else ""
                    return False, f"Access forbidden: GitHub API rate limit exceeded (remaining={self.rate_limit_remaining}{reset_info})"
                return False, "Access forbidden"
            elif status == 404:
                return False, f"Repository not found: {self.repo}"
            elif status == 200:
                return True, "Repository accessible"
            else:
                return False, f"Unexpected status: {status}"
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    def fetch_issues(self, state: str = "open", per_page: int = 50, max_issues: Optional[int] = None) -> List[Dict]:
        """
        Fetch issues from repository

        Args:
            state: "open", "closed", or "all"
            per_page: Number of issues per page (max 100)
            max_issues: Optional hard cap on total issues returned

        Returns:
            List of issue objects
        """
        page_size = max(1, min(per_page, 100))
        page = 1
        collected: List[Dict] = []

        try:
            while True:
                url = (
                    f"{self.base_url}/issues?state={state}&per_page={page_size}"
                    f"&page={page}&sort=created&direction=desc"
                )
                status, resp = self._make_request('GET', url)

                if status != 200:
                    print(f"⚠️  Failed to fetch issues page {page}: {status}")
                    break

                page_items = resp if isinstance(resp, list) else []
                if not page_items:
                    break

                # Filter out pull requests and append issues.
                for item in page_items:
                    if 'pull_request' not in item:
                        collected.append(item)
                        if max_issues and len(collected) >= max_issues:
                            return collected[:max_issues]

                if len(page_items) < page_size:
                    break
                page += 1

            return collected
        except Exception as e:
            print(f"❌ Error fetching issues: {e}")
            return []

    def get_issue(self, issue_num: int) -> Optional[Dict]:
        """Fetch a single issue object."""
        url = f"{self.base_url}/issues/{issue_num}"
        try:
            status, resp = self._make_request('GET', url)
            if status == 200 and isinstance(resp, dict):
                return resp
            return None
        except Exception:
            return None

    def list_comments(self, issue_num: int, per_page: int = 100) -> List[Dict]:
        """Fetch comments for an issue."""
        url = f"{self.base_url}/issues/{issue_num}/comments?per_page={per_page}"
        try:
            status, resp = self._make_request('GET', url)
            if status == 200 and isinstance(resp, list):
                return resp
            return []
        except Exception:
            return []

    def issue_has_comment_marker(self, issue_num: int, marker: str) -> bool:
        """Return True if a comment body already contains marker text."""
        if not marker:
            return False
        comments = self.list_comments(issue_num)
        for comment in comments:
            body = comment.get('body', '')
            if isinstance(body, str) and marker in body:
                return True
        return False

    def add_label(self, issue_num: int, labels: List[str]) -> Tuple[bool, str]:
        """
        Add labels to an issue

        Args:
            issue_num: Issue number
            labels: List of label names to add

        Returns:
            (success, message)
        """
        url = f"{self.base_url}/issues/{issue_num}/labels"

        try:
            status, resp = self._make_request('POST', url, labels)

            if status == 200:
                return True, f"✅ Added labels: {', '.join(labels)}"
            else:
                return False, f"Failed to add labels: {status} - {str(resp)}"
        except Exception as e:
            return False, f"Error adding labels: {str(e)}"

    def remove_label(self, issue_num: int, label: str) -> Tuple[bool, str]:
        """Remove a label from an issue"""
        url = f"{self.base_url}/issues/{issue_num}/labels/{label}"

        try:
            status, resp = self._make_request('DELETE', url)

            if status == 204 or status == 200:
                return True, f"✅ Removed label: {label}"
            else:
                return False, f"Failed to remove label: {status}"
        except Exception as e:
            return False, f"Error removing label: {str(e)}"

    def add_comment(self, issue_num: int, body: str) -> Tuple[bool, str]:
        """
        Add a comment to an issue

        Args:
            issue_num: Issue number
            body: Comment body (markdown supported)

        Returns:
            (success, message)
        """
        url = f"{self.base_url}/issues/{issue_num}/comments"

        try:
            status, resp = self._make_request('POST', url, {'body': body})

            if status == 201:
                comment_id = resp.get('id', 'unknown')
                return True, f"✅ Comment added (ID: {comment_id})"
            else:
                return False, f"Failed to add comment: {status}"
        except Exception as e:
            return False, f"Error adding comment: {str(e)}"

    def update_issue_state(self, issue_num: int, state: str) -> Tuple[bool, str]:
        """
        Update issue state (open/closed)

        Args:
            issue_num: Issue number
            state: "open" or "closed"

        Returns:
            (success, message)
        """
        url = f"{self.base_url}/issues/{issue_num}"

        if state not in ['open', 'closed']:
            return False, f"Invalid state: {state}. Must be 'open' or 'closed'"

        try:
            current = self.get_issue(issue_num)
            if current and current.get('state') == state:
                return True, f"ℹ️ Issue #{issue_num} already in state: {state}"

            status, resp = self._make_request('PATCH', url, {'state': state})

            if status == 200:
                return True, f"✅ Issue #{issue_num} state changed to: {state}"
            else:
                return False, f"Failed to update state: {status}"
        except Exception as e:
            return False, f"Error updating state: {str(e)}"

    def close_issue_with_comment(self, issue_num: int, comment: str, labels: List[str] = None) -> Dict:
        """
        Close an issue with a comment and optional labels

        Args:
            issue_num: Issue number
            comment: Comment to add before closing
            labels: Optional labels to add

        Returns:
            Dict with results of each operation
        """
        results = {'issue': issue_num, 'operations': {}}

        # Add labels if provided
        if labels:
            success, msg = self.add_label(issue_num, labels)
            results['operations']['add_labels'] = {'success': success, 'message': msg}

        # Add comment
        if comment:
            success, msg = self.add_comment(issue_num, comment)
            results['operations']['add_comment'] = {'success': success, 'message': msg}

        # Close issue
        success, msg = self.update_issue_state(issue_num, 'closed')
        results['operations']['close_issue'] = {'success': success, 'message': msg}

        results['success'] = all(op.get('success', False) for op in results['operations'].values())
        return results

    def create_issue_summary(self, issue_num: int, analysis: Dict, plan: Dict) -> str:
        """
        Create a comprehensive summary comment for an issue based on AI analysis

        Args:
            issue_num: Issue number
            analysis: Analysis results from AI
            plan: Implementation plan from AI

        Returns:
            Formatted markdown comment
        """
        summary = f"""## 🤖 Automated Triage & Analysis Report

**Issue**: #{issue_num}
**Analyzed**: {datetime.now().isoformat()}

### Analysis
- **Type**: {analysis.get('analysis_type', 'unknown')}
- **Severity**: {analysis.get('severity_assessment', 'medium')}
- **Components**: {', '.join(analysis.get('affected_components', ['unknown']))}

### Implementation Plan
- **Effort**: {plan.get('estimated_effort', 'medium')}
- **Steps**:
"""
        for step in plan.get('steps', []):
            step_num = step.get('step', '?')
            action = step.get('action', 'unknown')
            summary += f"  {step_num}. {action}\n"

        summary += "\n*This issue has been triaged and marked for implementation.*"
        return summary

    def get_rate_limit_status(self) -> Dict:
        """Get current GitHub API rate limit status"""
        url = "https://api.github.com/rate_limit"

        try:
            status, resp = self._make_request('GET', url)
            if status == 200:
                data = resp.get('resources', {})
                core = data.get('core', {})
                return {
                    'remaining': core.get('remaining', 'unknown'),
                    'limit': core.get('limit', 'unknown'),
                    'reset': core.get('reset', 'unknown')
                }
        except Exception as e:
            print(f"⚠️  Could not fetch rate limit: {e}")

        return {
            'remaining': self.rate_limit_remaining or 'unknown',
            'limit': 'unknown',
            'reset': self.rate_limit_reset or 'unknown'
        }

class IssueExecutionHandler:
    """High-level handler for executing issue resolution"""

    def __init__(self, repo: str = "kushin77/ollama", token: str = None):
        self.gh = GitHubAPIHandler(repo, token)
        self.execution_log = []

    def execute_on_issue(self, issue_num: int, analysis: Dict, plan: Dict,
                        labels_to_add: List[str] = None,
                        close_issue: bool = False,
                        comment_marker: str = "") -> Dict:
        """
        Execute full resolution workflow on an issue

        Args:
            issue_num: Issue number
            analysis: AI analysis results
            plan: AI implementation plan
            labels_to_add: Labels to add to issue
            close_issue: Whether to close the issue

        Returns:
            Execution result
        """
        result = {
            'issue': issue_num,
            'timestamp': datetime.now().isoformat(),
            'actions': []
        }

        try:
            # Add labels
            if labels_to_add:
                success, msg = self.gh.add_label(issue_num, labels_to_add)
                result['actions'].append({
                    'action': 'add_labels',
                    'success': success,
                    'message': msg
                })

            # Add summary comment only if marker is not already present
            should_add_comment = True
            if comment_marker and self.gh.issue_has_comment_marker(issue_num, comment_marker):
                should_add_comment = False
                result['actions'].append({
                    'action': 'add_summary_comment',
                    'success': True,
                    'message': f"ℹ️ Comment marker already exists: {comment_marker}"
                })

            if should_add_comment:
                summary = self.gh.create_issue_summary(issue_num, analysis, plan)
                if comment_marker:
                    summary = f"{comment_marker}\n\n{summary}"
                success, msg = self.gh.add_comment(issue_num, summary)
                result['actions'].append({
                    'action': 'add_summary_comment',
                    'success': success,
                    'message': msg
                })

            # Close issue if requested
            if close_issue:
                success, msg = self.gh.update_issue_state(issue_num, 'closed')
                result['actions'].append({
                    'action': 'close_issue',
                    'success': success,
                    'message': msg
                })

            result['success'] = all(a.get('success', False) for a in result['actions'])

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)

        self.execution_log.append(result)
        return result

    def get_execution_log(self) -> List[Dict]:
        """Get log of all executions"""
        return self.execution_log
