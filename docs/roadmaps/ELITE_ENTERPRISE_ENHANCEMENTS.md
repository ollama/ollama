# Elite Enterprise Enhancements

This roadmap captures the next high-value improvements for the repository after the current docs and host-profile cleanup.

## Goal

Keep the repository enterprise-grade by making the structure explicit, the configuration immutable, the execution path repo-relative, and the evidence ephemeral unless it is intentionally retained.

## Priority Order

1. Finish collapsing any remaining loose documentation into canonical buckets.
2. Keep target-server-local execution anchored on host inventories and shared loaders.
3. Expand deterministic validation where host-aware scripts change.
4. Keep planning artifacts in `docs/roadmaps/` and avoid duplicating them in reports.
5. Only introduce new documentation buckets when there is real source material to anchor them.

## Already Delivered

- `docs/deep/` now anchors deep-scan and long-form evidence.
- `docs/shared/`, `docs/indexed/`, `docs/meta/`, `docs/structure/`, `docs/repo-rules/`, `docs/instructions/`, `docs/ssot/`, and `docs/snc/` are canonical documentation buckets.
- `docs/operations/ON_PREM_DEPLOYMENT_MODEL.md` defines the target-server-local path.
- `docs/roadmaps/` now exists as the planning home for future enhancements.

## Next Actions

- Review any remaining root-level docs and classify each as canonical, compatibility, roadmap, or archive.
- Extend host-profile validation only when new host-aware scripts are added.
- Keep GitHub issue work tied to concrete roadmap items so the queue stays actionable.
- Maintain the clean-tree rule: one canonical home per document family.

## Coordination Notes

- Other agents should update this roadmap before adding new planning docs.
- If a new bucket is proposed, add the evidence family first, then add the bucket, then wire the navigation.
- Do not create a pnpm bucket unless a real pnpm workspace file is present.
