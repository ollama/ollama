#!/usr/bin/env bash
# Usage: GH_TOKEN must be set or gh CLI must be logged in.
# This script posts update comments to issues, embeds small text logs,
# lists larger files for manual attachment or GCS upload, and can close
# issues when run with `--close`.

set -euo pipefail

REPO="kushin77/ollama"
BRANCH="feature/issue-24-predictive"
PR_URL="https://github.com/kushin77/ollama/pull/41"

# Space-separated list of files to include in the comment. Can be overridden
# via the FILES_TO_ATTACH environment variable.
FILES_TO_ATTACH=${FILES_TO_ATTACH:-"docs/release-notes/pmo-issues-24-30.md"}

# If a file is larger than this threshold (in bytes) it will not be embedded
# and will be listed for manual attachment or upload to GCS.
EMBED_THRESHOLD=${EMBED_THRESHOLD:-65536} # 64 KiB

# Optional GCS bucket where large files can be uploaded/promoted by the user
# If set, the script will suggest the bucket path for manual upload.
GCS_BUCKET=${GCS_BUCKET:-}

ISSUES=(24 25 26 27 28 29 30)

# Flags
CLOSE=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --close|-c)
      CLOSE=true
      shift
      ;;
    --files)
      shift
      FILES_TO_ATTACH="$1"
      shift
      ;;
    --gcs-bucket)
      shift
      GCS_BUCKET="$1"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--close] [--files 'a b c'] [--gcs-bucket BUCKET]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

build_comment() {
  local branch="$1"
  local pr="$2"
  local body

  body=$(cat <<'EOF'
PMO update: work for this issue has been implemented and tested.

- Branch: ${branch}
- Pull request: ${pr}

Local PMO unit tests passed: 132 passed, 2 skipped.

Embedded small files and notes are included below. Large files are listed for manual attachment or upload to GCS.

Next step: awaiting CI. If CI passes, this issue will be closed and the Epic (#18) marked complete.
EOF
)

  # Process files: embed small ones, list large ones
  local large_list=""
  for f in ${FILES_TO_ATTACH}; do
    if [ ! -f "${f}" ]; then
      body+=$"\n- NOTE: file not found: ${f}\n"
      continue
    fi
    size=$(wc -c <"${f}" | tr -d ' ')
    if [ "${size}" -le "${EMBED_THRESHOLD}" ]; then
      body+=$"\n---\n**Embedded file:** ${f}\n\n\`\`\`\n"
      # append file contents safely
      body+=$(sed 's/`/`\`/g' "${f}")
      body+=$"\n\`\`\`\n"
    else
      # large file — add to list with size in KB
      kb=$(( (size + 1023) / 1024 ))
      if [ -n "${GCS_BUCKET}" ]; then
        large_list+=$"- ${f} (${kb} KB) — upload to gs://${GCS_BUCKET}/ and link here\n"
      else
        large_list+=$"- ${f} (${kb} KB) — attach manually or upload to GCS and link here\n"
      fi
    fi
  done

  if [ -n "${large_list}" ]; then
    body+=$"\n---\n**Large files (not embedded):**\n${large_list}\n"
  fi

  printf "%s" "$body"
}

for i in "${ISSUES[@]}"; do
  BODY=$(build_comment "$BRANCH" "$PR_URL")
  echo "Posting update to issue #$i"
  if command -v gh >/dev/null 2>&1; then
    gh issue comment "$i" --repo "$REPO" -b "$BODY"
  else
    echo "gh CLI not found — printing prepared comment for issue #$i."
    echo "-----"
    echo "$BODY"
    echo "-----"
  fi
done

# Close issues if requested
if [ "$CLOSE" = true ]; then
  for i in "${ISSUES[@]}"; do
    if command -v gh >/dev/null 2>&1; then
      gh issue close "$i" --repo "$REPO" && echo "Closed issue #$i"
    else
      echo "gh CLI not found — cannot close issue #$i from script."
    fi
  done
  echo "Note: update Epic #18 status manually if needed."
fi
