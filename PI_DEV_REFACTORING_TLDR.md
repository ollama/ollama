# pi.dev CLI - AI Refactoring TLDR

When using pi.dev CLI for semantic refactoring:
1. **Always use LSP**: Prefer true rename/code-actions over regex/manual edits.
2. **Pre-flight**: Run diagnostics, use definition, references, hover.
3. **Validation**: Run typechecks (`npm run typecheck`/`cargo check`), linting, and formatting (`cargo fmt`) after edits.
4. **Safety**: For large structural changes (e.g. move method, extract class), plan first and apply incrementally. The CLI has IDE-like features, but not full JetBrains parity yet.

