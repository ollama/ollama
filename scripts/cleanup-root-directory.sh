#!/bin/bash
# Root directory cleanup script
# Organizes 60+ status reports into docs/reports/ to maintain elite filesystem standards
# Usage: bash scripts/cleanup-root-directory.sh

set -e

echo "🧹 Cleaning up root directory..."
echo ""

# Create target directories
mkdir -p docs/reports
mkdir -p docs/archive

# List of files to archive
declare -a files_to_archive=(
    "ALL_OPTIONS_EXECUTION_SUMMARY.txt"
    "CODE_DEVELOPMENT_ROADMAP.md"
    "COMPLETION_REPORT.md"
    "COMPLIANCE_IMPROVEMENTS_SUMMARY.md"
    "COMPLIANCE_STATUS.md"
    "CONTINUATION_PLAN.md"
    "CRITICAL_STATUS_ASSESSMENT.txt"
    "DEEP_SCAN_COMPLETION_SUMMARY.md"
    "DEEP_SCAN_REPORT.md"
    "DELIVERABLES_INDEX.md"
    "DEPLOYMENT_ANALYSIS_COMPLETION_REPORT.md"
    "DEPLOYMENT_COMPLETE.txt"
    "DEPLOYMENT_EXECUTION_GUIDE.md"
    "DEPLOYMENT_EXECUTION_STARTED.md"
    "DEPLOYMENT_FINAL_STATUS.md"
    "DEPLOYMENT_READINESS_REPORT.md"
    "DEPLOYMENT_STATUS.md"
    "DEPLOYMENT_STATUS.txt"
    "DEVELOPMENT_SETUP.md"
    "ELITE_STANDARDS_QUICK_REFERENCE.md"
    "FINAL_DEPLOYMENT_SUMMARY.md"
    "FINAL_INTEGRATION_SUMMARY.md"
    "FINAL_PHASE_4_SUMMARY.txt"
    "FINAL_PROJECT_SUMMARY.md"
    "FINAL_SUMMARY.txt"
    "FINAL_VERIFICATION_REPORT.md"
    "GCP_LB_MANDATE_COMPLETION.md"
    "GCP_OAUTH_CONFIGURATION.md"
    "GOV_AI_SCOUT_OAUTH_IMPLEMENTATION.md"
    "INCOMPLETE_TASKS_CONSOLIDATED.md"
    "INDEX.md"
    "MASTER_INDEX.md"
    "MISSION_COMPLETE.md"
    "NEXT_ACTIONS.md"
    "PERFORMANCE_OPTIMIZATION_ROADMAP.md"
    "PHASE_4_COMPLETE_READY_FOR_DEPLOYMENT.md"
    "PHASE_4_COMPLETION_SUMMARY.md"
    "PHASE_4_DELIVERABLES_INDEX.md"
    "PHASE_4_EXECUTIVE_SUMMARY.md"
    "PHASE_4_FILES_CREATED.txt"
    "PHASE_4_TO_PRODUCTION.md"
    "POST_DEPLOYMENT_COMPLETION.txt"
    "POST_DEPLOYMENT_INDEX.md"
    "POST_DEPLOYMENT_MONITORING_GUIDE.md"
    "PRODUCTION_DEPLOYMENT_VALIDATION.md"
    "PROJECT_STATUS.md"
    "QUICK_REFERENCE_OPERATIONS.txt"
    "QUICK_REFERENCE.md"
    "READY_TO_DEPLOY.md"
    "SCAN_COMPLETION.txt"
    "SERVER_LIVE_STATUS.md"
    "SESSION_COMPLETION.md"
    "TASK_COMPLETION_SUMMARY.md"
    "WEEK_1_OPERATIONS_PLAYBOOK.md"
)

# Move files to docs/reports
echo "📁 Archiving status reports to docs/reports/..."
for file in "${files_to_archive[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" docs/reports/
        echo "  ✓ $file"
    fi
done

echo ""
echo "📝 Creating index of archived reports..."

# Create index file
cat > docs/reports/INDEX.md << 'EOF'
# Archived Status Reports & Documentation

This directory contains historical status reports, completion documents, and project phase summaries.

## Organization

All files are organized by date and category:
- DEPLOYMENT_* files: Deployment-related reports
- PHASE_4_* files: Phase 4 completion documents
- FINAL_* files: Final project summaries
- COMPLIANCE_* files: Compliance audit reports
- Other: General status and task reports

## Accessing Active Documentation

For current documentation, see:
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Contribution guidelines
- [README.md](../../README.md) - Project overview
- [.github/copilot-instructions.md](../../.github/copilot-instructions.md) - Elite standards
- [docs/](../) - Current documentation

## Historical Context

These reports document the project's evolution and major milestones. They are kept for historical reference but should not be treated as current guidance.

For current best practices, refer to the main documentation files.
EOF

echo "  ✓ Created docs/reports/INDEX.md"

echo ""
echo "✅ Root directory cleanup complete!"
echo ""
echo "Summary:"
echo "  📦 Archived: $(ls -1 docs/reports/*.{md,txt} 2>/dev/null | wc -l) files"
echo "  📂 Location: docs/reports/"
echo "  📋 Index: docs/reports/INDEX.md"
echo ""
echo "Root directory now contains only essential files (README.md, CONTRIBUTING.md, etc.)"
echo ""
