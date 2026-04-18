#!/usr/bin/env python3
"""
GitHub Issues Triage & Execution System
IaC-based, immutable, independent, multi-AI compatible

Principles:
- IaC: All configurations declared, versioned, reproducible
- Immutable: State snapshots, content-addressable, append-only audit logs
- Independent: Decoupled components, async-safe, AI-agnostic
"""

import json
import hashlib
import subprocess
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import os
import re

class IssueSeverity(Enum):
    CRITICAL = "critical"  # Breaks functionality, data loss risk
    HIGH = "high"          # Significant impact, needs fix soon
    MEDIUM = "medium"      # Should be addressed
    LOW = "low"            # Nice to have, enhancement
    DOCUMENTATION = "documentation"
    
class IssueType(Enum):
    BUG = "bug"
    FEATURE = "feature"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    DEPENDENCY = "dependency"
    PERFORMANCE = "performance"
    SECURITY = "security"

class IssueStatus(Enum):
    NEW = "new"
    TRIAGED = "triaged"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    TESTING = "testing"
    CLOSED = "closed"
    WONTFIX = "wontfix"

class TriageIssue:
    """Immutable issue record with content-addressable ID"""
    
    def __init__(self, issue_num: int, title: str, body: str, labels: List[str],
                 author: str, created: str, comments: int):
        self.issue_num = issue_num
        self.title = title
        self.body = body
        self.labels = labels
        self.author = author
        self.created = created
        self.comments = comments
        self.timestamp = datetime.now().isoformat()
        
    def get_content_hash(self) -> str:
        """Content-addressable ID for immutability"""
        content = f"{self.issue_num}:{self.title}:{self.body}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _extract_structured_tags(self) -> List[str]:
        return re.findall(r'\[([^\]]+)\]', self.title)

    def _get_text_blob(self) -> str:
        return f"{self.title} {self.body}".lower()

    def get_workflow(self, issue_type: "IssueType") -> str:
        if issue_type == IssueType.FEATURE:
            return 'feature_implementation'
        if issue_type == IssueType.PERFORMANCE:
            return 'performance_optimization'
        return 'bug_investigation'
    
    def classify(self) -> tuple[IssueSeverity, IssueType]:
        """Classify issue by severity and type"""
        label_set = {l.lower() for l in self.labels}
        text = self._get_text_blob()
        tags = {tag.upper() for tag in self._extract_structured_tags()}
        
        # Determine severity
        severity = IssueSeverity.MEDIUM
        if 'P0' in tags:
            severity = IssueSeverity.HIGH
        elif 'P1' in tags:
            severity = IssueSeverity.MEDIUM
        elif 'P2' in tags:
            severity = IssueSeverity.LOW
        elif any(tag.startswith('ENHANCEMENT-') for tag in tags):
            severity = IssueSeverity.LOW
        elif any(x in label_set for x in ['critical', 'emergency', 'data-loss']):
            severity = IssueSeverity.CRITICAL
        elif any(x in label_set for x in ['bug', 'regression']):
            severity = IssueSeverity.HIGH
        elif any(x in label_set for x in ['security']):
            severity = IssueSeverity.CRITICAL
        elif any(x in label_set for x in ['feature request', 'enhancement']):
            severity = IssueSeverity.LOW
        elif any(x in label_set for x in ['documentation']):
            severity = IssueSeverity.DOCUMENTATION
        elif any(x in text for x in ['terraform', 'lockfile', 'provider version', 'folder-structure', 'validation failure']):
            severity = IssueSeverity.HIGH
        elif any(x in text for x in ['enhancement', 'assistant', 'workflow', 'generator', 'mode']):
            severity = IssueSeverity.LOW
            
        # Determine type
        issue_type = IssueType.BUG
        if any(tag.startswith('ENHANCEMENT-') for tag in tags) or 'feature request' in label_set or 'enhancement' in label_set:
            issue_type = IssueType.FEATURE
        elif 'documentation' in label_set:
            issue_type = IssueType.DOCUMENTATION
        elif 'performance' in label_set:
            issue_type = IssueType.PERFORMANCE
        elif 'security' in label_set:
            issue_type = IssueType.SECURITY
        elif 'refactor' in label_set or 'cleanup' in label_set:
            issue_type = IssueType.REFACTOR
        elif any(x in label_set for x in ['dependencies', 'dependency']):
            issue_type = IssueType.DEPENDENCY
        elif any(x in text for x in ['terraform', 'provider version', 'lockfile', 'dependency']):
            issue_type = IssueType.DEPENDENCY
        elif any(x in text for x in ['documentation', 'readme', 'docs']):
            issue_type = IssueType.DOCUMENTATION
        elif any(x in text for x in ['performance', 'latency', 'benchmark']):
            issue_type = IssueType.PERFORMANCE
            
        return severity, issue_type
    
    def to_dict(self) -> Dict:
        """Immutable snapshot"""
        severity, issue_type = self.classify()
        return {
            'content_hash': self.get_content_hash(),
            'issue_num': self.issue_num,
            'title': self.title,
            'labels': self.labels,
            'structured_tags': self._extract_structured_tags(),
            'author': self.author,
            'created': self.created,
            'comments': self.comments,
            'severity': severity.value,
            'type': issue_type.value,
            'workflow': self.get_workflow(issue_type),
            'timestamp': self.timestamp,
            'status': IssueStatus.TRIAGED.value
        }

class TriageSystem:
    """Immutable triage system with declarative rules"""
    
    def __init__(self, rules_file: str = ".github/triage.rules.json"):
        self.rules_file = rules_file
        self.rules = self._load_rules()
        self.audit_log = []
        
    def _load_rules(self) -> Dict:
        """Load IaC-declared triage rules"""
        if os.path.exists(self.rules_file):
            with open(self.rules_file) as f:
                return json.load(f)
        return self._default_rules()
    
    def _default_rules(self) -> Dict:
        """Default IaC rules"""
        return {
            "version": "1.0",
            "rules": [
                {
                    "name": "security_critical",
                    "labels": ["security"],
                    "severity": "critical",
                    "auto_action": "assign:security-team"
                },
                {
                    "name": "mlx_bugs",
                    "patterns": ["MLX", "metal", "gpu"],
                    "severity": "high",
                    "labels_to_add": ["macos", "gpu"]
                },
                {
                    "name": "startup_issues",
                    "patterns": ["startup", "launch", "initialization"],
                    "severity": "high",
                    "investigation_needed": True
                }
            ]
        }
    
    def triage(self, issue: TriageIssue) -> Dict:
        """Triage issue and apply rules"""
        severity, issue_type = issue.classify()
        actions = []
        matched_rules = []
        
        # Apply rules
        for rule in self.rules.get('rules', []):
            if self._matches_rule(issue, rule):
                matched_rules.append(rule.get('id') or rule.get('name', 'unknown'))
                actions.extend(rule.get('auto_actions', []))

        priority_score_map = {
            IssueSeverity.CRITICAL: 100,
            IssueSeverity.HIGH: 90,
            IssueSeverity.MEDIUM: 60,
            IssueSeverity.LOW: 25,
            IssueSeverity.DOCUMENTATION: 20,
        }
        
        record = {
            'issue_num': issue.issue_num,
            'severity': severity.value,
            'type': issue_type.value,
            'content_hash': issue.get_content_hash(),
            'structured_tags': issue._extract_structured_tags(),
            'workflow': issue.get_workflow(issue_type),
            'priority_score': priority_score_map[severity],
            'matched_rules': matched_rules,
            'recommended_actions': actions,
            'triaged_at': datetime.now().isoformat()
        }
        
        # Append to immutable audit log
        self.audit_log.append(record)
        
        return record
    
    def _matches_rule(self, issue: TriageIssue, rule: Dict) -> bool:
        """Check if issue matches rule"""
        # Label matching
        if 'labels' in rule:
            rule_labels = set(rule['labels'])
            issue_labels = set(l.lower() for l in issue.labels)
            if rule_labels & issue_labels:
                return True
        
        # Pattern matching in title/body
        if 'patterns' in rule:
            text = f"{issue.title} {issue.body}".lower()
            if any(p.lower() in text for p in rule['patterns']):
                return True
        
        return False

class AIExecutor:
    """AI-agnostic issue executor interface"""
    
    def __init__(self, model_name: str = "claude"):
        self.model_name = model_name
        self.execution_log = []
        
    def generate_implementation_plan(self, issue: TriageIssue) -> Dict:
        """Generate IaC-based implementation plan"""
        return {
            "issue_num": issue.issue_num,
            "title": issue.title,
            "content_hash": issue.get_content_hash(),
            "model": self.model_name,
            "plan": {
                "phase_1_analysis": {
                    "description": "Analyze root cause",
                    "tools": ["grep", "git log", "static analysis"],
                    "expected_output": "Root cause analysis document"
                },
                "phase_2_implementation": {
                    "description": "Implement solution",
                    "iac_template": f".github/templates/fix_{issue.issue_num}.yml",
                    "testing": ["unit", "integration", "e2e"],
                    "expected_output": "PR with tests"
                },
                "phase_3_validation": {
                    "description": "Validate fix",
                    "validation_steps": ["test suite", "regression", "performance"],
                    "expected_output": "Validation report"
                }
            },
            "generated_at": datetime.now().isoformat()
        }

class IACTemplate:
    """Infrastructure as Code template for issue fixes"""
    
    @staticmethod
    def generate_fix_template(issue_num: int, issue_type: str) -> str:
        """Generate declarative fix template"""
        return f"""---
apiVersion: ollama.io/v1
kind: IssueResolution
metadata:
  name: issue-{issue_num}
  created: {datetime.now().isoformat()}
spec:
  issue_number: {issue_num}
  issue_type: {issue_type}
  
  # Immutable snapshot of issue state
  snapshot:
    content_hash: ${{CONTENT_HASH}}
    timestamp: {datetime.now().isoformat()}
  
  # Declarative fix specification
  fix:
    description: "Fix for issue {issue_num}"
    
    # Changes: append-only, content-addressed
    changes:
    - path: "src/path/to/file.go"
      operation: "patch"
      content_hash: "${{PATCH_HASH}}"
      change_type: "fix"
      
    # Testing: declarative
    testing:
      unit:
        - "cmd: go test ./..."
          expected: "PASS"
      integration:
        - "cmd: ./integration-test.sh"
          expected: "All tests passed"
          
    # Validation: immutable records
    validation:
      before: "${{BASELINE_METRICS}}"
      after: "${{NEW_METRICS}}"
      regression_check: "PASS"
      
    # Metadata
    author: "AI-System"
    reviewer: "required"
    signed_by: "${{GPG_KEY}}"
"""

def main():
    print("=" * 70)
    print("GitHub Issues Triage & Execution System (IaC-based)")
    print("=" * 70)
    
    # Load issues
    with open('/tmp/issues_full.json') as f:
        raw_issues = json.load(f)
    
    issues = [i for i in raw_issues if 'pull_request' not in i]
    
    # Initialize systems
    triage_system = TriageSystem()
    executor = AIExecutor(model_name="claude")
    
    print(f"\n📊 Processing {len(issues)} issues...\\n")
    
    triaged = []
    for issue_data in issues[:10]:  # Process top 10
        issue = TriageIssue(
            issue_data['number'],
            issue_data['title'],
            issue_data['body'] or '',
            [l['name'] for l in issue_data.get('labels', [])],
            issue_data['user']['login'],
            issue_data['created_at'],
            issue_data['comments']
        )
        
        # Triage
        triage_result = triage_system.triage(issue)
        
        # Generate plan
        plan = executor.generate_implementation_plan(issue)
        
        triaged.append({
            'issue': issue.to_dict(),
            'triage': triage_result,
            'plan': plan
        })
        
        print(f"✓ #{issue.issue_num:5} | Severity: {triage_result['severity']:10} | "
              f"Actions: {len(triage_result['recommended_actions'])}")
    
    # Save immutable snapshot
    snapshot = {
        'version': '1.0',
        'timestamp': datetime.now().isoformat(),
        'total_issues_analyzed': len(issues),
        'triaged_count': len(triaged),
        'triage_results': triaged,
        'audit_log': triage_system.audit_log
    }
    
    with open('.github/triage_snapshot.json', 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"\n✅ Triaged {len(triaged)} issues")
    print(f"📄 Snapshot saved to: .github/triage_snapshot.json")
    print(f"\n🤖 Ready for multi-AI execution pipeline")

if __name__ == '__main__':
    main()
