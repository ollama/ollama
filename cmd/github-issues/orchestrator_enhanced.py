#!/usr/bin/env python3
"""
Complete Issue Orchestrator - Triage, Analyze, Execute, Update
Integrates: Triage System -> AI Analysis -> GitHub API Updates
"""

import json
import sys
import os
import hashlib
import subprocess
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
from pathlib import Path


def _get_token_from_gsm() -> Optional[str]:
    """Retrieve GitHub token from Google Secret Manager via gcloud CLI or REST API.

        Matches the git-credential-gsm helper exactly:
            GCP project:  GSM_PROJECT env var  (default: gcp-eiq)
            Secret name:  GSM_SECRET_NAME env var (default: prod-github-token)

    gcloud must be authenticated (gcloud auth login / ADC / service account).
    OLLAMA_GSM_ENABLED=true must be set to activate this path.
    """
    if not os.getenv('OLLAMA_GSM_ENABLED', '').lower() in ('true', '1', 'yes'):
        return None

    # Match git-credential-gsm defaults exactly
    project_id = (
        os.getenv('GSM_PROJECT')
        or os.getenv('OLLAMA_GSM_PROJECT_ID')
        or 'gcp-eiq'
    )
    secret_name = (
        os.getenv('GSM_SECRET_NAME')
        or os.getenv('OLLAMA_GSM_SECRET_NAME')
        or 'prod-github-token'
    )

    # Try gcloud CLI first
    try:
        result = subprocess.run(
            ['gcloud', 'secrets', 'versions', 'access', 'latest',
             f'--secret={secret_name}', f'--project={project_id}'],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and result.stdout.strip():
            token = result.stdout.strip()
            print(f"🔑 Auth mode: token loaded from GSM (gcloud) secret '{secret_name}'")
            return token
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try Google metadata server / REST API with Application Default Credentials
    try:
        # Get access token from metadata server (works on GCE/GKE/Cloud Run)
        meta_req = urllib.request.Request(
            'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
            headers={'Metadata-Flavor': 'Google'}
        )
        with urllib.request.urlopen(meta_req, timeout=3) as resp:
            access_token = json.loads(resp.read())['access_token']

        secret_url = (
            f'https://secretmanager.googleapis.com/v1/projects/{project_id}'
            f'/secrets/{secret_name}/versions/latest:access'
        )
        sm_req = urllib.request.Request(
            secret_url,
            headers={'Authorization': f'Bearer {access_token}'}
        )
        with urllib.request.urlopen(sm_req, timeout=10) as resp:
            import base64
            payload = json.loads(resp.read())
            token = base64.b64decode(payload['payload']['data']).decode('utf-8').strip()
            if token:
                print(f"🔑 Auth mode: token loaded from GSM (REST) secret '{secret_name}'")
                return token
    except Exception:
        pass

    print(f"⚠️  GSM enabled but could not retrieve secret '{secret_name}' from project '{project_id}'")
    return None


def _load_token_from_git_credentials(repo: str) -> Optional[str]:
    """Retrieve a GitHub token through git credential helpers.

    Uses the repository's configured credential.helper (git-credential-gsm by default
    in this workspace, backed by GCP project gcp-eiq secret prod-github-token).
    Requires gcloud to have an active authenticated account.
    """
    credential_path = f"{repo}.git" if repo else ""
    request = "protocol=https\nhost=github.com\n"
    if credential_path:
        request += f"path={credential_path}\n"
    request += "\n"

    try:
        result = subprocess.run(
            ["git", "credential", "fill"],
            input=request,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0 or not result.stdout:
        return None

    for line in result.stdout.splitlines():
        if line.startswith("password="):
            token = line.split("=", 1)[1].strip()
            return token or None

    return None


def _resolve_github_token(repo: str, explicit_token: Optional[str], token_file: Optional[str]) -> Tuple[Optional[str], str]:
    """Resolve GitHub token from declarative sources in priority order."""
    if explicit_token:
        return explicit_token, 'cli'

    token = _load_token_from_file(token_file)
    if token:
        return token, 'token_file'

    token = _load_token_from_git_credentials(repo)
    if token:
        return token, 'git_credentials'

    token = _get_token_from_gsm()
    if token:
        return token, 'gsm'

    token = (
        os.getenv('OLLAMA_GITHUB_TOKEN')
        or os.getenv('GITHUB_TOKEN')
        or os.getenv('GH_TOKEN')
    )
    if token:
        return token, 'environment'

    return None, 'none'


def _hash_config(config: Dict) -> str:
    payload = json.dumps(config, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]

# Import our modules
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from github_api_handler import GitHubAPIHandler, IssueExecutionHandler
from triage import TriageSystem, IssueSeverity, TriageIssue
from ai_executor import ClaudeExecutor, GrokExecutor, MultiAIExecutor, AIProvider
from immutable_state import ImmutableState, IssueStateChart


class ImmutableOpsLedger:
    """Append-only operations ledger with hash chaining for immutability."""

    def __init__(self, ledger_path: str = ".github/issue_ops_ledger.jsonl"):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_last_hash(self) -> str:
        if not self.ledger_path.exists():
            return "GENESIS"
        last_line = ""
        with open(self.ledger_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    last_line = line
        if not last_line:
            return "GENESIS"
        try:
            return json.loads(last_line).get('entry_hash', 'GENESIS')
        except Exception:
            return "GENESIS"

    def append_entry(self, entry: Dict) -> Dict:
        prev_hash = self._get_last_hash()
        material = {
            'prev_hash': prev_hash,
            'entry': entry,
        }
        payload = json.dumps(material, sort_keys=True)
        entry_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
        immutable_entry = {
            'prev_hash': prev_hash,
            'entry_hash': entry_hash,
            'recorded_at': datetime.now().isoformat(),
            **entry,
        }
        with open(self.ledger_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(immutable_entry) + '\n')
        return immutable_entry


def _detect_repo_from_git_remote(default_repo: str) -> str:
    """Infer owner/repo from git remote origin URL."""
    try:
        proc = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return default_repo
        remote = proc.stdout.strip()
        if not remote:
            return default_repo

        # Handles https://github.com/owner/repo.git and git@github.com:owner/repo.git
        if remote.endswith('.git'):
            remote = remote[:-4]
        if 'github.com/' in remote:
            return remote.split('github.com/', 1)[1]
        if 'github.com:' in remote:
            return remote.split('github.com:', 1)[1]
    except Exception:
        pass
    return default_repo


def _load_token_from_file(token_file: Optional[str]) -> Optional[str]:
    if not token_file:
        return None
    try:
        with open(token_file, 'r', encoding='utf-8') as f:
            token = f.read().strip()
            return token or None
    except Exception:
        return None

class IssueOrchestrator:
    """Unified orchestrator for the complete issue management pipeline"""

    def __init__(self, repo: str = "kushin77/ollama", token: str = None,
                 fallback_repo: str = "ollama/ollama", allow_fallback: bool = False,
                 auto_close_labels: Optional[List[str]] = None,
                 auth_source: str = 'none'):
        """
        Initialize orchestrator

        Args:
            repo: Primary repository (owner/repo)
            token: GitHub token
            fallback_repo: Fallback repo if primary is not accessible
        """
        self.primary_repo = repo
        self.fallback_repo = fallback_repo
        self.allow_fallback = allow_fallback
        self.token = token
        self.gh = None
        self.executor = None
        self.triage_system = TriageSystem()
        self.state_charts = {}  # Map of issue_num -> IssueStateChart
        self.execution_results = []
        self.last_summary = None
        self.selected_repo = None
        self.auth_source = auth_source
        self.run_id = hashlib.sha256(datetime.now().isoformat().encode('utf-8')).hexdigest()[:12]
        self.ops_ledger = ImmutableOpsLedger()
        self.auto_close_labels = {l.strip().lower() for l in (auto_close_labels or []) if l.strip()}

        # Try primary repo, fall back if needed
        self._initialize_github_handler()

    def _initialize_github_handler(self):
        """Initialize GitHub handler, trying primary then fallback repo"""
        print(f"🔍 Checking repository access...")

        # Try primary repo
        gh_primary = GitHubAPIHandler(self.primary_repo, self.token)
        accessible, msg = gh_primary.check_repo_access()

        if accessible:
            print(f"✅ Using primary repo: {self.primary_repo}")
            self.gh = gh_primary
            self.executor = IssueExecutionHandler(self.primary_repo, self.token)
            self.selected_repo = self.primary_repo
        else:
            print(f"⚠️  Primary repo not accessible: {msg}")
            if self.allow_fallback:
                print(f"📦 Trying fallback repo: {self.fallback_repo}")

                gh_fallback = GitHubAPIHandler(self.fallback_repo, self.token)
                accessible, msg = gh_fallback.check_repo_access()

                if accessible:
                    print(f"✅ Using fallback repo: {self.fallback_repo}")
                    self.gh = gh_fallback
                    self.executor = IssueExecutionHandler(self.fallback_repo, self.token)
                    self.selected_repo = self.fallback_repo
                    return

                print(f"❌ Fallback repo also not accessible: {msg}")

            print(f"⚠️  Running in offline mode for target repo {self.primary_repo} (no GitHub updates)")
            self.gh = None
            self.executor = None

    def triage_and_execute(self, max_issues: int = 20,
                          target_severity: str = "high",
                          dry_run: bool = True) -> Dict:
        """
        Complete pipeline: fetch -> triage -> analyze -> execute

        Args:
            max_issues: Maximum issues to process
            target_severity: Severity level to target (high/medium/low)
            dry_run: If True, don't actually update issues on GitHub

        Returns:
            Execution summary
        """

        can_mutate = bool(self.executor and getattr(self.executor.gh, 'token', None))
        effective_dry_run = dry_run or not can_mutate

        summary = {
            'timestamp': datetime.now().isoformat(),
            'repository': self.selected_repo or 'offline',
            'requested_repository': self.primary_repo,
            'run_id': self.run_id,
            'auth_source': self.auth_source,
            'ops_ledger_file': str(self.ops_ledger.ledger_path),
            'pipeline': {
                'fetch': {},
                'triage': {},
                'analysis': {},
                'execution': {
                    'status': 'pending',
                    'dry_run': effective_dry_run,
                    'attempted': 0,
                    'succeeded': 0,
                    'failed': 0
                }
            },
            'issues_processed': []
        }

        # Phase 1: Fetch Issues
        print("\n" + "="*70)
        print("PHASE 1: FETCH ISSUES")
        print("="*70)

        if not self.gh:
            print("❌ No GitHub access for target repository. Skipping fetch and execution.")
            issues = []
        else:
            print(f"📥 Fetching issues from {self.selected_repo}...")
            issues = self.gh.fetch_issues(
                state='open',
                per_page=min(max_issues, 100),
                max_issues=max_issues,
            )

        summary['pipeline']['fetch'] = {
            'status': 'success',
            'issues_fetched': len(issues)
        }
        print(f"✅ Fetched {len(issues)} issues")

        if len(issues) == 0:
            summary['pipeline']['triage'] = {
                'status': 'skipped',
                'reason': 'no_issues_or_no_access',
                'total_triaged': 0,
                'target_severity': target_severity,
                'target_count': 0
            }
            summary['pipeline']['analysis'] = {
                'status': 'skipped',
                'reason': 'no_issues'
            }
            summary['pipeline']['execution']['status'] = 'skipped'
            self.last_summary = summary
            return summary

        # Phase 2: Triage
        print("\n" + "="*70)
        print("PHASE 2: TRIAGE ISSUES")
        print("="*70)

        triaged_issues = []
        for issue in issues:
            # Convert GitHub API issue to TriageIssue
            labels = [l['name'] for l in issue.get('labels', [])] if isinstance(issue.get('labels', []), list) and issue['labels'] and isinstance(issue['labels'][0], dict) else issue.get('labels', [])

            triage_issue = TriageIssue(
                issue_num=issue['number'],
                title=issue.get('title', ''),
                body=issue.get('body', ''),
                labels=labels,
                author=issue.get('user', {}).get('login', 'unknown') if isinstance(issue.get('user'), dict) else 'unknown',
                created=issue.get('created_at', ''),
                comments=issue.get('comments', 0)
            )

            # Triage the issue
            triage_result = self.triage_system.triage(triage_issue)

            # Store with severity for filtering
            triage_object = type('TriageResult', (), {
                'issue_number': issue['number'],
                'severity': IssueSeverity(triage_result['severity']),
                'issue_type': type('IssueType', (), {'value': triage_result['type']})(),
                'triage_result': triage_result
            })()

            triaged_issues.append(triage_object)
            severity_label = "🔴" if triage_object.severity == IssueSeverity.HIGH else \
                            "🟠" if triage_object.severity == IssueSeverity.MEDIUM else "🟡"
            print(f"{severity_label} #{issue['number']:5} | {triage_result['type']:15} | {triage_result['severity']}")

        # Filter by target severity
        target_issues = [i for i in triaged_issues
                        if i.severity.value == target_severity]

        summary['pipeline']['triage'] = {
            'status': 'success',
            'total_triaged': len(triaged_issues),
            'target_severity': target_severity,
            'target_count': len(target_issues)
        }
        print(f"✅ Triaged {len(triaged_issues)} issues")
        print(f"✅ Found {len(target_issues)} {target_severity.upper()} priority issues")

        # Phase 3: AI Analysis & Execution
        print("\n" + "="*70)
        print(f"PHASE 3: AI ANALYSIS ({target_severity.upper()} PRIORITY)")
        print("="*70)

        multi_ai = MultiAIExecutor([AIProvider.CLAUDE, AIProvider.GROK])

        for triaged in target_issues:
            issue_num = triaged.issue_number
            issue = next((i for i in issues if i['number'] == issue_num), None)
            if not issue:
                continue

            print(f"\n📋 Issue #{issue_num}: {issue.get('title', 'N/A')[:50]}")

            # Run AI analysis
            ai_result = multi_ai.execute_parallel(issue)

            # Extract consensus
            consensus = ai_result.get('consensus', {})
            print(f"   Consensus: {consensus.get('consensus_status', 'unknown')} ({consensus.get('agreement_count', 0)}/{consensus.get('total_systems', 0)} AI systems)")

            # Phase 4: Execute on Issue (update GitHub)
            if not effective_dry_run and self.executor:
                print(f"   ⚙️  Executing updates...")
                summary['pipeline']['execution']['attempted'] += 1

                analysis = ai_result['ai_results'].get('claude', {})
                plan = analysis.get('phases', {}).get('planning', {})

                labels = self._get_labels_for_issue(triaged)
                close_issue = self._should_auto_close(issue)
                comment_marker = f"<!-- ollama-issue-orchestrator:v1 issue:{issue_num} -->"

                exec_result = self.executor.execute_on_issue(
                    issue_num,
                    analysis=analysis,
                    plan=plan,
                    labels_to_add=labels,
                    close_issue=close_issue,
                    comment_marker=comment_marker,
                )

                # Record result
                self.execution_results.append(exec_result)
                success_marker = "✅" if exec_result.get('success') else "❌"
                print(f"   {success_marker} Execution complete")
                if exec_result.get('success'):
                    summary['pipeline']['execution']['succeeded'] += 1
                else:
                    summary['pipeline']['execution']['failed'] += 1

                self.ops_ledger.append_entry({
                    'run_id': self.run_id,
                    'repository': self.selected_repo or 'offline',
                    'issue': issue_num,
                    'severity': triaged.severity.value,
                    'mode': 'execute',
                    'success': bool(exec_result.get('success')),
                    'actions': exec_result.get('actions', []),
                    'close_blocked': any(
                        a.get('blocked') for a in exec_result.get('actions', [])
                        if a.get('action') == 'close_issue'
                    ),
                })
            else:
                print(f"   🔒 DRY-RUN mode (not updating GitHub)")
                summary['pipeline']['execution']['attempted'] += 1
                summary['pipeline']['execution']['succeeded'] += 1
                self.execution_results.append({
                    'issue': issue_num,
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'dry_run': True,
                    'actions': []
                })

                self.ops_ledger.append_entry({
                    'run_id': self.run_id,
                    'repository': self.selected_repo or 'offline',
                    'issue': issue_num,
                    'severity': triaged.severity.value,
                    'mode': 'dry_run',
                    'success': False,
                    'note': 'dry-run: no GitHub mutations performed',
                    'actions': [],
                })

            # Update state tracking
            if issue_num not in self.state_charts:
                self.state_charts[issue_num] = IssueStateChart(issue_num)
            self.state_charts[issue_num].transition('triaged')

            # Add to summary
            summary['issues_processed'].append({
                'number': issue_num,
                'severity': triaged.severity.value,
                'type': triaged.triage_result['type'],
                'title': issue.get('title', ''),
                'status': 'analyzed'
            })

        summary['pipeline']['analysis'] = {
            'status': 'success',
            'analyzed_count': len(target_issues)
        }

        if summary['pipeline']['execution']['failed'] == 0:
            summary['pipeline']['execution']['status'] = 'success'
        elif summary['pipeline']['execution']['succeeded'] > 0:
            summary['pipeline']['execution']['status'] = 'partial'
        else:
            summary['pipeline']['execution']['status'] = 'failed'

        # Phase 5: Report
        print("\n" + "="*70)
        print("PHASE 5: EXECUTION SUMMARY")
        print("="*70)

        self._print_summary(summary)
        self.last_summary = summary

        return summary

    def _should_auto_close(self, issue: Dict) -> bool:
        """Return True when issue matches configured auto-close labels."""
        if not self.auto_close_labels:
            return False
        raw_labels = issue.get('labels', [])
        issue_labels = set()
        if isinstance(raw_labels, list):
            for label in raw_labels:
                if isinstance(label, dict):
                    name = label.get('name', '')
                else:
                    name = str(label)
                if name:
                    issue_labels.add(name.lower())
        return bool(issue_labels.intersection(self.auto_close_labels))

    def _get_labels_for_issue(self, triaged) -> List[str]:
        """Determine appropriate labels for issue"""
        labels = []

        # Add severity label
        if triaged.severity == IssueSeverity.HIGH:
            labels.append('priority/high')
        elif triaged.severity == IssueSeverity.MEDIUM:
            labels.append('priority/medium')
        else:
            labels.append('priority/low')

        # Add type labels based on triage result
        issue_type = triaged.triage_result['type']
        type_to_label = {
            'security': 'security',
            'bug': 'bug',
            'feature': 'enhancement',
            'documentation': 'documentation',
            'performance': 'performance'
        }

        label = type_to_label.get(issue_type)
        if label:
            labels.append(label)

        # Add triaged label
        labels.append('status/triaged')

        return labels

    def _get_demo_issues(self) -> List[Dict]:
        """Return demo issues for offline testing"""
        return [
            {
                'number': 15649,
                'title': 'Ollama startup issue on macOS',
                'body': 'Application fails to start on macOS with error...',
                'labels': [{'name': 'bug'}]
            },
            {
                'number': 15648,
                'title': 'MLX models fail to load',
                'body': 'MLX backend is missing libmlx dependency...',
                'labels': [{'name': 'bug'}, {'name': 'gpu'}]
            },
            {
                'number': 15647,
                'title': 'API compatibility with OpenAI format',
                'body': 'Some OpenAI endpoints not supported...',
                'labels': [{'name': 'enhancement'}]
            },
            {
                'number': 15646,
                'title': 'Docker build failing',
                'body': 'Docker build process fails on Linux...',
                'labels': [{'name': 'bug'}]
            },
            {
                'number': 15645,
                'title': 'Memory leak in model inference',
                'body': 'Memory usage increases over time...',
                'labels': [{'name': 'bug'}]
            }
        ]

    def _print_summary(self, summary: Dict):
        """Print execution summary"""
        print(f"\n📊 EXECUTION REPORT")
        print(f"   Repository: {summary['repository']}")
        print(f"   Timestamp: {summary['timestamp']}")
        print(f"\n🔍 Pipeline Status:")
        for phase, result in summary['pipeline'].items():
            status = result.get('status', 'unknown')
            marker = "✅" if status == 'success' else "❌"
            print(f"   {marker} {phase.upper()}: {status}")

        print(f"\n📋 Issues Processed: {len(summary['issues_processed'])}")
        for issue in summary['issues_processed']:
            severity_emoji = "🔴" if issue['severity'] == 'high' else \
                            "🟠" if issue['severity'] == 'medium' else "🟡"
            print(f"   {severity_emoji} #{issue['number']} | {issue['title'][:40]}")

    def _aggregate_pipeline(self, orchestration_summary: Optional[Dict]) -> Dict:
        """Aggregate per-severity pipeline results into a single run summary."""
        if not orchestration_summary or orchestration_summary.get('mode') != 'all-severities':
            return self.last_summary.get('pipeline', {}) if self.last_summary else {}

        summaries = orchestration_summary.get('summaries', [])
        target_counts = {}
        fetch_counts = []
        triage_counts = []
        analyzed_count = 0
        execution_attempted = 0
        execution_succeeded = 0
        execution_failed = 0
        execution_dry_run = True

        for summary in summaries:
            pipeline = summary.get('pipeline', {})
            fetch = pipeline.get('fetch', {})
            triage = pipeline.get('triage', {})
            analysis = pipeline.get('analysis', {})
            execution = pipeline.get('execution', {})

            if 'issues_fetched' in fetch:
                fetch_counts.append(fetch.get('issues_fetched', 0))
            if 'total_triaged' in triage:
                triage_counts.append(triage.get('total_triaged', 0))
            if triage.get('target_severity'):
                target_counts[triage['target_severity']] = triage.get('target_count', 0)
            analyzed_count += analysis.get('analyzed_count', 0)
            execution_attempted += execution.get('attempted', 0)
            execution_succeeded += execution.get('succeeded', 0)
            execution_failed += execution.get('failed', 0)
            execution_dry_run = execution_dry_run and execution.get('dry_run', True)

        execution_status = 'success'
        if execution_failed and execution_succeeded:
            execution_status = 'partial'
        elif execution_failed and not execution_succeeded:
            execution_status = 'failed'

        return {
            'fetch': {
                'status': 'success',
                'issues_fetched': max(fetch_counts) if fetch_counts else 0,
                'passes': len(summaries),
            },
            'triage': {
                'status': 'success',
                'total_triaged': max(triage_counts) if triage_counts else 0,
                'target_counts': target_counts,
            },
            'analysis': {
                'status': 'success',
                'analyzed_count': analyzed_count,
            },
            'execution': {
                'status': execution_status,
                'dry_run': execution_dry_run,
                'attempted': execution_attempted,
                'succeeded': execution_succeeded,
                'failed': execution_failed,
            },
        }

    def generate_full_report(self, orchestration_summary: Optional[Dict] = None) -> Dict:
        """Generate comprehensive execution report"""
        # Collect all state transitions
        all_transitions = {}
        for issue_num, state_chart in self.state_charts.items():
            all_transitions[issue_num] = state_chart.transitions

        pipeline = self._aggregate_pipeline(orchestration_summary)
        if orchestration_summary and orchestration_summary.get('mode') == 'all-severities':
            total_issues_processed = len(self.execution_results)
        else:
            total_issues_processed = len(self.last_summary.get('issues_processed', [])) if self.last_summary else len(self.execution_results)

        return {
            'timestamp': datetime.now().isoformat(),
            'repository': self.selected_repo or 'offline',
            'requested_repository': self.primary_repo,
            'auth_source': self.auth_source,
            'total_issues_processed': total_issues_processed,
            'pipeline_stages': ['fetch', 'triage', 'analysis', 'github_updates'],
            'pipeline': pipeline,
            'execution_results': self.execution_results,
            'state_transitions': all_transitions
        }

def main():
    detected_repo = _detect_repo_from_git_remote('kushin77/ollama')
    parser = argparse.ArgumentParser(description='GitHub Issue Orchestrator')
    parser.add_argument('--repo', default=detected_repo,
                       help='Primary repository (owner/repo)')
    parser.add_argument('--fallback', default='ollama/ollama',
                       help='Fallback repository')
    parser.add_argument('--allow-fallback', action='store_true',
                       help='Allow fallback repo if primary is inaccessible')
    parser.add_argument('--token', default=None,
                       help='GitHub API token')
    parser.add_argument('--token-file', default=os.getenv('GITHUB_TOKEN_FILE'),
                       help='Path to a file containing GitHub API token')
    parser.add_argument('--max-issues', type=int, default=20,
                       help='Maximum issues to process')
    parser.add_argument('--severity', default='high',
                       choices=['high', 'medium', 'low'],
                       help='Target severity level')
    parser.add_argument('--all-severities', action='store_true',
                       help='Run high, medium, and low processing sequentially')
    parser.add_argument('--execute', action='store_true',
                       help='Actually update issues (default is dry-run)')
    parser.add_argument('--output', default='.github/orchestrator_report.json',
                       help='Output report file')
    parser.add_argument('--auto-close-labels', default='duplicate,invalid,wontfix',
                       help='Comma-separated labels that trigger auto-close in execute mode')

    args = parser.parse_args()

    args.token, auth_source = _resolve_github_token(args.repo, args.token, args.token_file)

    print("\n" + "="*70)
    print("🤖 GITHUB ISSUE ORCHESTRATOR")
    print("="*70)
    if args.token:
        print(f"🔐 Auth mode: token available via {auth_source} (authenticated GitHub API)")
    else:
        print("🔓 Auth mode: no token found (subject to public API rate limits)")

    # Create orchestrator
    orchestrator = IssueOrchestrator(
        repo=args.repo,
        token=args.token,
        fallback_repo=args.fallback,
        allow_fallback=args.allow_fallback,
        auto_close_labels=[x for x in args.auto_close_labels.split(',') if x.strip()],
        auth_source=auth_source,
    )

    run_config = {
        'run_id': orchestrator.run_id,
        'requested_repository': args.repo,
        'selected_repository': orchestrator.selected_repo or 'offline',
        'fallback_repository': args.fallback,
        'allow_fallback': args.allow_fallback,
        'all_severities': args.all_severities,
        'severity': args.severity,
        'max_issues': args.max_issues,
        'execute': args.execute,
        'auto_close_labels': [x for x in args.auto_close_labels.split(',') if x.strip()],
        'auth_source': auth_source,
        'token_present': bool(args.token),
    }
    run_config['config_hash'] = _hash_config(run_config)
    orchestrator.ops_ledger.append_entry({
        'run_id': orchestrator.run_id,
        'repository': orchestrator.selected_repo or 'offline',
        'mode': 'run_start',
        'success': True,
        'config': run_config,
    })

    # Run complete pipeline
    if args.all_severities:
        summaries = []
        for severity in ['high', 'medium', 'low']:
            print(f"\n▶ Running severity pass: {severity}")
            summary = orchestrator.triage_and_execute(
                max_issues=args.max_issues,
                target_severity=severity,
                dry_run=not args.execute
            )
            summaries.append(summary)
        summary = {
            'run_id': orchestrator.run_id,
            'mode': 'all-severities',
            'summaries': summaries,
            'timestamp': datetime.now().isoformat(),
        }
    else:
        summary = orchestrator.triage_and_execute(
            max_issues=args.max_issues,
            target_severity=args.severity,
            dry_run=not args.execute
        )

    # Save report
    report = orchestrator.generate_full_report(summary)
    report['run_config'] = run_config
    report['orchestration'] = summary
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    orchestrator.ops_ledger.append_entry({
        'run_id': orchestrator.run_id,
        'repository': orchestrator.selected_repo or 'offline',
        'mode': 'run_complete',
        'success': True,
        'report_file': args.output,
        'issues_processed': report['total_issues_processed'],
        'config_hash': run_config['config_hash'],
    })

    print(f"\n✅ Report saved to: {args.output}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
