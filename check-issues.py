#!/usr/bin/env python3
"""
GitHub Issues Report Generator for Ollama
Fetches and displays open issues from ollama/ollama repository

Usage:
    python3 check-issues.py ghp_xxxxx
    python3 check-issues.py  # Uses OLLAMA_GITHUB_TOKEN env var
"""

import os
import sys
import json
import urllib.request
import urllib.error
from collections import Counter
from datetime import datetime

def fetch_issues(token):
    """Fetch open issues from ollama/ollama repository"""
    owner = "ollama"
    repo = "ollama"
    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    
    params = "state=open&sort=updated&order=desc&per_page=100"
    url = f"{api_url}?{params}"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data
    except urllib.error.HTTPError as e:
        print(f"❌ GitHub API Error: {e.code}")
        print(f"   {e.reason}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error fetching issues: {e}")
        sys.exit(1)

def print_report(issues):
    """Print formatted issue report"""
    
    print("\n📊 Ollama Repository Issues Report")
    print("===================================\n")
    
    total = len(issues)
    print(f"📈 Summary Statistics")
    print(f"─────────────────────")
    print(f"Total Open Issues: {total}\n")
    
    # Most recent issues
    print(f"⏰ Most Recently Updated Issues (Top 15)")
    print(f"─────────────────────────────────────────")
    for i, issue in enumerate(issues[:15]):
        number = issue['number']
        title = issue['title'][:45].ljust(45)
        author = issue['user']['login'][:12].ljust(12)
        updated = issue['updated_at'][:10]
        print(f"#{number:<6} {title} @{author} {updated}")
    
    print()
    
    # Labels analysis
    all_labels = []
    for issue in issues:
        for label in issue.get('labels', []):
            all_labels.append(label['name'])
    
    if all_labels:
        print(f"🏷️  Top Issue Labels")
        print(f"─────────────────")
        label_counts = Counter(all_labels).most_common(10)
        for label, count in label_counts:
            print(f"  {label:<20} {count} issues")
    
    print()
    
    # Priority labels
    print(f"⚠️  Issues by Priority")
    print(f"────────────────────")
    
    bug_count = sum(1 for issue in issues 
                   if any(label['name'] == 'bug' for label in issue.get('labels', [])))
    feature_count = sum(1 for issue in issues 
                       if any(label['name'] == 'enhancement' for label in issue.get('labels', [])))
    docs_count = sum(1 for issue in issues 
                    if any(label['name'] == 'documentation' for label in issue.get('labels', [])))
    help_wanted = sum(1 for issue in issues 
                     if any(label['name'] == 'help wanted' for label in issue.get('labels', [])))
    
    print(f"  🐛 Bug reports: {bug_count}")
    print(f"  ✨ Feature requests: {feature_count}")
    print(f"  📚 Documentation needed: {docs_count}")
    print(f"  🤝 Help wanted: {help_wanted}")
    
    print()
    print(f"✅ Links")
    print(f"─────")
    print(f"  View all issues: https://github.com/ollama/ollama/issues")
    print(f"  Repository: https://github.com/ollama/ollama")
    
    print()
    print(f"✓ Report generated successfully")

def main():
    # Get token from argument or environment
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = os.environ.get('OLLAMA_GITHUB_TOKEN', '')
    
    if not token:
        print("❌ GitHub token not provided")
        print()
        print("Usage:")
        print("  python3 check-issues.py ghp_xxxxxxxxxxxxx")
        print("  or")
        print("  export OLLAMA_GITHUB_TOKEN=ghp_xxxxxxxxxxxxx")
        print("  python3 check-issues.py")
        print()
        print("Get a token at: https://github.com/settings/tokens")
        sys.exit(1)
    
    print("📊 Ollama Repository Issues Report")
    print("===================================")
    print()
    print("⏳ Fetching open issues from ollama/ollama...")
    
    issues = fetch_issues(token)
    print_report(issues)

if __name__ == "__main__":
    main()
