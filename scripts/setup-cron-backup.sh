#!/bin/bash
# Setup automated GCS backups via cron

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📅 Setting up automated GCS backups..."
echo ""

# Check if gcs-sync.sh exists
if [ ! -f "$SCRIPT_DIR/gcs-sync.sh" ]; then
    echo "❌ Error: gcs-sync.sh not found at $SCRIPT_DIR/gcs-sync.sh"
    exit 1
fi

# Make gcs-sync.sh executable
chmod +x "$SCRIPT_DIR/gcs-sync.sh"

# Cron job configuration
CRON_SCHEDULE="0 */6 * * *"  # Every 6 hours
CRON_JOB="$CRON_SCHEDULE cd $PROJECT_ROOT && $SCRIPT_DIR/gcs-sync.sh >> /tmp/gcs-backup.log 2>&1"

echo "Cron schedule: Every 6 hours"
echo "Log file: /tmp/gcs-backup.log"
echo ""

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "gcs-sync.sh"; then
    echo "⚠️  Cron job already exists. Removing old entry..."
    crontab -l 2>/dev/null | grep -v "gcs-sync.sh" | crontab -
fi

# Add new cron job
echo "📝 Adding cron job..."
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo ""
echo "✅ Automated backup configured!"
echo ""
echo "📊 Current crontab:"
crontab -l | grep gcs-sync || echo "No backup jobs found"
echo ""
echo "🧪 Test backup manually:"
echo "   $SCRIPT_DIR/gcs-sync.sh"
echo ""
echo "📜 View backup logs:"
echo "   tail -f /tmp/gcs-backup.log"
