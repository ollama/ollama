import argparse
import tempfile
import unittest
from pathlib import Path

from client import load_task, run_collaboration_loop


class FakeArchitect:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "initial solution"


class FakeReviewer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def review(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return f"revision {len(self.prompts)}"


class PassingReviewer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def review(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return " PASS "


class CollaborationLoopTests(unittest.TestCase):
    def test_runs_requested_review_rounds(self) -> None:
        architect = FakeArchitect()
        reviewer = FakeReviewer()

        result = run_collaboration_loop(
            "Build a CSV processor", architect, reviewer, rounds=2
        )

        self.assertEqual(result, "revision 2")
        self.assertEqual(len(architect.prompts), 1)
        self.assertEqual(len(reviewer.prompts), 2)
        self.assertIn("initial solution", reviewer.prompts[0])
        self.assertIn("revision 1", reviewer.prompts[1])

    def test_stops_when_reviewer_returns_pass(self) -> None:
        reviewer = PassingReviewer()

        result = run_collaboration_loop(
            "Build a CSV processor", FakeArchitect(), reviewer, rounds=3
        )

        self.assertEqual(result, "initial solution")
        self.assertEqual(len(reviewer.prompts), 1)

    def test_rejects_empty_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            run_collaboration_loop("", FakeArchitect(), FakeReviewer())

    def test_rejects_invalid_round_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            run_collaboration_loop(
                "Build a CSV processor", FakeArchitect(), FakeReviewer(), rounds=0
            )

    def test_loads_utf8_task_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task_file = Path(directory) / "task.txt"
            task_file.write_text("Process new CSV files", encoding="utf-8")
            args = argparse.Namespace(task=None, task_file=task_file)

            self.assertEqual(load_task(args), "Process new CSV files")


if __name__ == "__main__":
    unittest.main()
