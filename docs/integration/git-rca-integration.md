# Integrating git-rca-workspace

This document describes how to integrate the `git-rca-workspace` helper repository to improve developer tooling, VS Code workspace configuration, and Copilot assistance.

Why integrate

- Provides curated VS Code workspace settings, recommended extensions, and developer helpers used by the Copilot agent and local workflows.
- Improves local parity with CI and streamlines diagnosis of repository issues using RCA tooling.

Recommended integration approaches

1. Submodule (recommended for reproducible installs)

```bash
git submodule add https://github.com/kushin77/git-rca-workspace.git tools/git-rca-workspace
git submodule update --init --recursive
```

2. Shallow clone (if you prefer not to add a submodule)

```bash
mkdir -p tools
git clone --depth=1 https://github.com/kushin77/git-rca-workspace.git tools/git-rca-workspace
```

Post-install steps

- Open the repository in VS Code and add the `tools/git-rca-workspace` folder to the workspace (File → Add Folder to Workspace...).
- Install recommended extensions (see `tools/git-rca-workspace/.vscode/extensions.json`).
- Optionally run the included diagnostics and onboarding scripts from the `tools/git-rca-workspace` directory.

How Copilot/VSCode benefits

- Uses shared `settings.json`, `launch.json`, and workspace-level snippets to give Copilot consistent context across teammates and CI.
- Provides common tasks (linting, formatting, debug) that the Copilot agent can surface as run actions.

Security note

- Treat the external repo as a tooling dependency. Review its scripts and settings before adding as a submodule.
- Prefer to pin a commit for the submodule when creating production branches.

Quick troubleshooting

- If workspace features do not appear, confirm the tools folder is present and restart VS Code.
- If submodule fails to update: run `git submodule sync --recursive && git submodule update --init --recursive`.

Resources

- https://github.com/kushin77/git-rca-workspace
