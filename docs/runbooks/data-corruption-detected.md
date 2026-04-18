# Runbook: Data Corruption or Loss Detected

**Version**: 1.0 | **Severity**: SEV1 | **Time to Resolution**: 60 min

---

## Detection

- **Alert**: Manual discovery or automated integrity check failure
- **Symptom**: Data inconsistency, missing records, corrupted values in production
- **Dashboard**: [Chronicle Audit Logs](https://chronicle.example.com)

---

## Immediate Actions (0-5 min)

**CRITICAL: Do NOT make changes until root cause identified**

```bash
# Create war room: #incident-data-corruption

# Page IMMEDIATELY: @cto + @database-admin + @security-lead

# Assess scope:
# 1. How much data affected? (rows? tables? databases?)
# 2. When did corruption start? (exact timestamp?)
# 3. Is it still ongoing or stopped?

# Take evidence:
# 1. Screenshot of corrupted data
# 2. Save audit logs: SELECT * FROM chronicle_logs WHERE time > [corruption_time - 1h]
# 3. Database backup: IMMEDIATE BACKUP (if not auto-running)
```

---

## Diagnosis (5-30 min)

```bash
# Step 1: Isolate affected data
# Query corruption pattern
psql $PROD_DB -c "SELECT * FROM [table] WHERE [corrupt_column] = [bad_value];"

# Step 2: Find root cause
# A) Was there a recent deployment?
#    git log --oneline --since="2 hours ago"
#
# B) Was there a database migration?
#    SELECT * FROM schema_migrations WHERE created_at > [2 hours ago]
#
# C) Was there malicious activity?
#    gcloud logging read 'protoPayload.methodName="cloudsql.instances.update"'
#    Look for: DELETE, DROP, UPDATE statements from unauthorized users
#
# D) Hardware failure?
#    gcloud sql instances describe [instance] | grep state
#    Check: Disk health, replication status
```

---

## Remediation (30-60 min)

**CRITICAL: All decisions must be reviewed by CTO and documented**

### Option A: Restore from Backup (< 30 min)

```bash
# IF: Corruption is recent and backup is clean
#
# Step 1: Identify clean backup
gcloud sql backups list --instance=$DB_INSTANCE

# Step 2: Restore (creates new instance)
gcloud sql backups restore [BACKUP_ID] --backup-instance=$DB_INSTANCE

# Step 3: Verify restoration (sample data)
psql [NEW_INSTANCE_CONNECTION] -c "SELECT COUNT(*) FROM [table];"

# Step 4: Cut over traffic
# Update connection string in Cloud Run → Point to new instance

# Step 5: Archive old instance for analysis
```

### Option B: Selective Fix (if restoration not needed)

```bash
# IF: Only small amount of data corrupted
#
# Step 1: Backup corrupted data for forensics
#    SELECT * INTO corrupted_backup FROM [table] WHERE [corrupt_condition]
#
# Step 2: Delete corrupted rows
#    DELETE FROM [table] WHERE [corrupt_condition]
#
# Step 3: If data can be reconstructed:
#    INSERT INTO [table] SELECT [reconstructed_data]
#
# Step 4: Verify referential integrity
#    Manually check dependent tables
```

---

## Escalation

- **If > 1% of data affected**: Immediately escalate to @founders
- **If customer data affected**: Follow GDPR/HIPAA breach protocol
- **If cause is internal bug**: Halt all deployments until fixed
- **If cause is external attack**: Activate security incident response

---

## Post-Incident

1. Complete root cause analysis
2. Forensic analysis of corrupted data
3. Implement preventive measures (validation, auditing)
4. Notify affected customers if required
5. Review database backup strategy

**Created**: 2026-01-26
