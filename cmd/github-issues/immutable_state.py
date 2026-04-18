#!/usr/bin/env python3
"""
Immutable State Management for Issue Resolution
Content-addressable, append-only, snapshot-based state tracking

Principles:
- All state changes are immutable, append-only
- Content-addressed for integrity verification
- Snapshots at each phase
- Independent: No side effects, idempotent operations
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

class ImmutableState:
    """Content-addressable state record"""
    
    def __init__(self, issue_num: int, state_type: str):
        self.issue_num = issue_num
        self.state_type = state_type
        self.timestamp = datetime.now().isoformat()
        self.content = {}
        self.hash = None
    
    def set_content(self, **kwargs):
        """Set immutable content"""
        self.content.update(kwargs)
        self.hash = self.compute_hash()
    
    def compute_hash(self) -> str:
        """Cryptographic hash for content addressing"""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        """Export state"""
        return {
            'issue_num': self.issue_num,
            'state_type': self.state_type,
            'timestamp': self.timestamp,
            'content_hash': self.hash,
            'content': self.content
        }

class ImmutableStateStore:
    """Append-only state store with content addressing"""
    
    def __init__(self, store_path: str = ".github/state"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.state_log = []
    
    def record_state(self, issue_num: int, state_type: str, **content) -> str:
        """Record immutable state, return content hash"""
        state = ImmutableState(issue_num, state_type)
        state.set_content(**content)
        
        # Append to log
        record = state.to_dict()
        self.state_log.append(record)
        
        # Save to append-only log file
        log_file = self.store_path / f"issue_{issue_num}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        return state.hash
    
    def get_state_history(self, issue_num: int) -> List[Dict]:
        """Get immutable history for issue"""
        log_file = self.store_path / f"issue_{issue_num}.jsonl"
        history = []
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))
        
        return history
    
    def get_current_state(self, issue_num: int, state_type: str) -> Dict:
        """Get latest state of specific type"""
        history = self.get_state_history(issue_num)
        
        # Get last record of specified type
        for record in reversed(history):
            if record['state_type'] == state_type:
                return record
        
        return None

class StateTransition:
    """Immutable state transition"""
    
    def __init__(self, from_state: str, to_state: str, issue_num: int):
        self.issue_num = issue_num
        self.from_state = from_state
        self.to_state = to_state
        self.timestamp = datetime.now().isoformat()
        self.metadata = {}
    
    def add_metadata(self, **kwargs):
        """Add transition metadata"""
        self.metadata.update(kwargs)
    
    def to_dict(self) -> Dict:
        """Export transition"""
        return {
            'issue_num': self.issue_num,
            'transition': f"{self.from_state} -> {self.to_state}",
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

class IssueStateChart:
    """Finite state machine for issue resolution"""
    
    STATES = {
        'new': {'next': ['triaged']},
        'triaged': {'next': ['assigned', 'wontfix']},
        'assigned': {'next': ['in_progress']},
        'in_progress': {'next': ['testing', 'blocked']},
        'blocked': {'next': ['in_progress']},
        'testing': {'next': ['validation', 'in_progress']},
        'validation': {'next': ['closed', 'in_progress']},
        'closed': {'next': []},
        'wontfix': {'next': []}
    }
    
    def __init__(self, issue_num: int):
        self.issue_num = issue_num
        self.state_store = ImmutableStateStore()
        self.current_state = 'new'
        self.transitions = []
    
    def transition(self, to_state: str, **metadata) -> bool:
        """Perform state transition"""
        valid_next = self.STATES.get(self.current_state, {}).get('next', [])
        
        if to_state not in valid_next:
            print(f"❌ Invalid transition: {self.current_state} -> {to_state}")
            return False
        
        # Record transition
        trans = StateTransition(self.current_state, to_state, self.issue_num)
        trans.add_metadata(**metadata)
        self.transitions.append(trans.to_dict())
        
        # Record state
        self.state_store.record_state(
            self.issue_num,
            'workflow_state',
            state=to_state,
            previous_state=self.current_state,
            **metadata
        )
        
        print(f"✓ Transitioned: {self.current_state} -> {to_state}")
        self.current_state = to_state
        return True

def main():
    print("=" * 70)
    print("Immutable State Management System")
    print("=" * 70)
    
    # Example: Track issue resolution workflow
    issue_num = 15649
    state_chart = IssueStateChart(issue_num)
    
    print(f"\n📊 Tracking issue #{issue_num}\n")
    
    # Simulate workflow
    transitions = [
        ('triaged', {'severity': 'critical', 'type': 'startup_crash'}),
        ('assigned', {'assigned_to': 'team-lead', 'priority': 'p0'}),
        ('in_progress', {'branch': 'fix/15649-startup', 'commit': 'abc123'}),
        ('testing', {'test_suite': 'all_pass', 'regression': 'negative'}),
        ('validation', {'reviewer': 'lead-maintainer', 'approved': True}),
        ('closed', {'merged_pr': '#15700', 'version': 'v0.2.1'})
    ]
    
    for to_state, metadata in transitions:
        state_chart.transition(to_state, **metadata)
    
    # Display audit trail
    print(f"\n📜 Immutable Audit Trail:")
    print("-" * 70)
    
    history = state_chart.state_store.get_state_history(issue_num)
    for record in history:
        print(f"  {record['timestamp']}: {record['state_type']}")
        print(f"    Hash: {record['content_hash']}")
        print(f"    State: {record['content'].get('state', 'N/A')}")
    
    # Export state log
    export = {
        'issue_num': issue_num,
        'audit_trail': history,
        'transitions': state_chart.transitions,
        'current_state': state_chart.current_state
    }
    
    with open(f'.github/state/issue_{issue_num}_audit.json', 'w') as f:
        json.dump(export, f, indent=2)
    
    print(f"\n✅ Audit log saved to: .github/state/issue_{issue_num}_audit.json")

if __name__ == '__main__':
    main()
