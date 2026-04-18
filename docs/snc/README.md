# Standard Naming Convention (SNC)

This document defines the naming patterns used across repository documentation and related artifacts.

## Documentation Paths

- Use lower-case kebab-case for folder names.
- Use `README.md` as the landing page for a canonical documentation folder.
- Use purpose-specific filenames only when there is a single canonical document for that topic.
- Use compatibility snapshot names only for legacy bridges that point to a canonical folder.

## Script and Artifact Names

- Use lower-case kebab-case for shell scripts and utility commands.
- Use descriptive environment-based names for host inventories, such as `development.env` and `production.env`.
- Use timestamped names for generated evidence and reports.

## Operational Naming

- Use `shared`, `indexed`, `meta`, `deep`, `roadmaps`, `structure`, `repo-rules`, `instructions`, `ssot`, and `snc` as canonical documentation buckets.
- Use `README.md` within each bucket as the human entry point.
- Keep aliases minimal and explicit when a historical path must remain supported.
