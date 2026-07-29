# Design notes

Historical design records for the MaxusAI fork. These are **archived reasoning, not
current guidance** — they document approaches that were explored and, in some cases,
abandoned. For how the fork behaves today, see [`docs/maxusai/`](../maxusai/).

- [Gemma 4 vision token budgets](gemma4-vision-token-budgets.md) — **historical.** The
  original `image_min_tokens` / `image_max_tokens` plan, written against the Go-native
  inference runner that upstream has since deleted. Records the Google visual-token ladder,
  the scheduler-reload requirement, and the option-naming rationale. The shipped feature
  works differently and uses different defaults (40 / 1120, not 70 / 560).
- [Gemma 4 vision token budgets — upstream rebase & forward-port notes](gemma4-vision-token-budgets-upstream-rebase.md)
  — **historical.** Why the feature was rebased onto `f63eea3d` (the last upstream commit
  with the Go runner still wired) instead of forward-ported, which of the two upstream PRs
  removed what, and the base-selection trade-off. Its forward-port conclusion was later
  disproved — annotated inline.

Both were rescued from `feat/gemma4-visual-token-budgets-last-go-runner` before that branch
was retired; the code they describe was never merged and is no longer applicable.
