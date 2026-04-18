from execute_local_triage import (
    classify_commit_evidence,
    extract_issue_keywords,
    issue_reference_pattern,
    parse_git_log,
)


def test_issue_reference_pattern_does_not_match_larger_issue_numbers() -> None:
    pattern = issue_reference_pattern(378)

    assert pattern.search("fixes #378")
    assert pattern.search("see https://github.com/kushin77/ollama/issues/378") is None
    assert pattern.search("(#3782)") is None
    assert pattern.search("(#1378)") is None


def test_extract_issue_keywords_filters_generic_noise() -> None:
    keywords = extract_issue_keywords(
        "bug(github): http.Client has no timeout — fix potential goroutine leak"
    )

    assert keywords == ["client", "timeout", "potential", "goroutine", "leak"]


def test_extract_issue_keywords_splits_camel_case_tokens() -> None:
    keywords = extract_issue_keywords("Migrate AuditTrail to pmo-agent-audit repo")

    assert keywords == ["audit", "trail"]


def test_parse_git_log_splits_commit_records() -> None:
    output = (
        "abc123\x1ffix(github): resolve timeout leak for #376\x1fbody text\x1e"
        "def456\x1fdocs: mention #376\x1fmore body\x1e"
    )

    records = parse_git_log(output)

    assert records == [
        {"commit": "abc123", "subject": "fix(github): resolve timeout leak for #376", "body": "body text"},
        {"commit": "def456", "subject": "docs: mention #376", "body": "more body"},
    ]


def test_classify_commit_evidence_rejects_unrelated_merge_pr() -> None:
    evidence = classify_commit_evidence(
        376,
        "bug(github): http.Client has no timeout — fix potential goroutine leak",
        {
            "commit": "c84bbf1",
            "subject": "Merge pull request #376 from jmorganca/mxyng/from-map-ignore-nil",
            "body": "",
        },
    )

    assert evidence is None


def test_classify_commit_evidence_rejects_substring_issue_match() -> None:
    evidence = classify_commit_evidence(
        378,
        "feat(observability): Replace hand-rolled Prometheus text with prometheus/client_golang",
        {
            "commit": "b84aea1",
            "subject": "Critical fix from llama.cpp JSON grammar (#3782)",
            "body": "",
        },
    )

    assert evidence is None


def test_classify_commit_evidence_accepts_exact_reference_with_contextual_match() -> None:
    evidence = classify_commit_evidence(
        376,
        "bug(github): http.Client has no timeout — fix potential goroutine leak",
        {
            "commit": "deadbeef",
            "subject": "fix(github): resolve client timeout leak for #376",
            "body": "Closes #376 after adding an http.Client timeout guard.",
        },
    )

    assert evidence == {
        "commit": "deadbeef",
        "subject": "fix(github): resolve client timeout leak for #376",
        "matched_keywords": ["client", "timeout", "leak"],
        "has_closure_verb": True,
        "confidence": "high",
    }


def test_classify_commit_evidence_rejects_generic_migration_summary() -> None:
    evidence = classify_commit_evidence(
        65,
        "Migrate AuditTrail to pmo-agent-audit repo",
        {
            "commit": "97d63fc",
            "subject": "docs: add comprehensive GitHub issues closure summary",
            "body": "Closes #65 after PMO agent migration work.",
        },
    )

    assert evidence is None
