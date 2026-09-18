# Skills Maintainer Plan

This document breaks the current skills work into reviewable slices and records trust/safety semantics that impact compatibility.

## Slice 1: Core Skills Lifecycle

Scope:

- `skills list`, `install`, `enable`, `disable`, `run`
- `skill.toml` core schema (`name`, `command`, `io`, `permissions`)
- interactive visibility (`/skills`, `/skill`, `/skilltrace`)

Out of scope:

- provenance/signatures
- policy enforcement
- audit log

Acceptance:

- local install + run flow works end to end
- permission prompt and grant scopes are covered by tests

## Slice 2: Distribution + Recovery

Scope:

- pinned Git/GitHub install and update
- backup, rollback, uninstall
- catalog search support

Out of scope:

- provenance/signature verification
- policy file

Acceptance:

- update/rollback/uninstall behavior is deterministic
- pinned refs are mandatory for remote install sources

## Slice 3: Trust + Safety Hardening

Scope:

- provenance verification (`sha256`, optional `ed25519` signature/public key)
- policy file (`policy.json`) for source/permissions/sandbox constraints
- audit events (`audit.log`) and rotation

Acceptance:

- install/update fail when declared provenance is invalid
- run/install honor policy constraints
- local `--verified` filters are cryptographic only

## Trust Model Decisions

- Local skill verification is based only on verified provenance metadata.
- Source URL or organization name does not imply verification.
- `provenance.sha256` covers a canonicalized manifest plus skill files.
- Signature verification uses `ed25519` over the computed digest bytes.
- Trusted keys are optional and enforced only when configured in policy.

## Migration Behavior

- Existing installed skills without provenance remain runnable by default.
- These skills appear as unverified in local verified-filtered search results.
- Operators can enforce stricter behavior by setting `policy.json` with:
  - `require_sha256`
  - `require_signature`
  - `trusted_public_keys`

## Follow-up Work

- document key generation/signing helper tooling for skill authors
- add integration test coverage for policy changes across existing installs
- expose audit log tailing in CLI if maintainers want first-class operator UX
