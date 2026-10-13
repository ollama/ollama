import argparse
import os
import sys
from pathlib import Path
from typing import Protocol

from anthropic import APIError as AnthropicAPIError
from anthropic import Anthropic
from openai import APIError as OpenAIAPIError
from openai import OpenAI


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"
DEFAULT_OPENAI_MODEL = "gpt-4o"
PERFECT_SIGNAL = "PASS"
DEFAULT_MAX_TOKENS = 4096
RETRY_MAX_TOKENS = 8192

ARCHITECT_SYSTEM_PROMPT = """\
You are an expert software architect and developer with extensive production
experience.

Before answering, privately compare at least two implementation approaches.
Challenge their correctness, performance, security, and edge-case handling,
then choose the strongest approach. Do not reveal private reasoning or drafts.
Return only the complete final solution and concise, useful explanations.
Avoid canned phrases and artificial-sounding prose.
"""

REVIEWER_SYSTEM_PROMPT = """\
You are a rigorous senior code reviewer and quality gate.

Check the proposed solution for correctness defects, security risks,
performance bottlenecks, missing edge cases, and unmet requirements. Rewrite
artificial-sounding explanations in a direct, professional style. If changes
are needed, return the complete improved solution and concise reasons. If the
solution needs no changes, reply with exactly PASS and nothing else.
"""


class Architect(Protocol):
    def generate(self, prompt: str) -> str: ...


class Reviewer(Protocol):
    def review(self, prompt: str) -> str: ...


class ClaudeArchitect:
    def __init__(
        self,
        client: Anthropic,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        retry_max_tokens: int = RETRY_MAX_TOKENS,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.retry_max_tokens = retry_max_tokens

    def generate(self, prompt: str) -> str:
        def create_response(max_tokens: int):
            return self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=ARCHITECT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

        response = create_response(self.max_tokens)
        if response.stop_reason == "max_tokens":
            response = create_response(self.retry_max_tokens)
            if response.stop_reason == "max_tokens":
                raise RuntimeError(
                    "Claude response was truncated twice (stop_reason: max_tokens)."
                )
        if response.stop_reason not in ("stop", "end_turn"):
            raise RuntimeError(
                f"Claude returned an unexpected stop_reason: {response.stop_reason}"
            )
        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not text:
            raise RuntimeError("Claude returned an empty response.")
        return text


class GPTReviewer:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        retry_max_tokens: int = RETRY_MAX_TOKENS,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.retry_max_tokens = retry_max_tokens

    def review(self, prompt: str) -> str:
        def create_response(max_tokens: int):
            return self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": REVIEWER_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
            )

        response = create_response(self.max_tokens)
        if not response.choices:
            raise RuntimeError("OpenAI returned no choices.")

        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            response = create_response(self.retry_max_tokens)
            if not response.choices:
                raise RuntimeError("OpenAI returned no choices.")
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                raise RuntimeError(
                    "OpenAI response was truncated twice (finish_reason: length)."
                )
        if finish_reason not in ("stop", "end_turn"):
            raise RuntimeError(
                f"OpenAI returned an unexpected finish_reason: {finish_reason}"
            )
        text = response.choices[0].message.content
        if not text or not text.strip():
            raise RuntimeError("OpenAI returned an empty response.")
        return text.strip()


def run_collaboration_loop(
    task: str,
    architect: Architect,
    reviewer: Reviewer,
    rounds: int = 2,
) -> str:
    if not task.strip():
        raise ValueError("Task must not be empty.")
    if rounds < 1:
        raise ValueError("Review rounds must be at least 1.")

    print(f"Task: {task}\n{'=' * 50}")
    print("\n[Step 1] Claude is designing and implementing the solution...")
    current_solution = architect.generate(
        "Design a sound architecture and provide a complete implementation for "
        f"this request:\n\n{task}"
    )
    print(f"\n--- Claude solution ---\n{current_solution}\n")

    for index in range(rounds):
        print(f"[Step {index + 2}] GPT review ({index + 1}/{rounds})...")
        reviewed_solution = reviewer.review(
            "Review the solution below for logical errors, security risks, "
            "performance problems, missing edge cases, and incomplete requirements. "
            "Follow your quality-gate instructions exactly."
            f"\n\nTask:\n{task}\n\nCurrent solution:\n{current_solution}"
        )
        if reviewed_solution.strip() == PERFECT_SIGNAL:
            print("\nGPT approved the solution. Stopping review early.\n")
            break

        current_solution = reviewed_solution
        print(f"\n--- GPT revised solution ---\n{current_solution}\n")

    return current_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a solution with Claude and refine it with OpenAI."
    )
    task_source = parser.add_mutually_exclusive_group(required=True)
    task_source.add_argument("--task", help="Task text to send to the models.")
    task_source.add_argument(
        "--task-file", type=Path, help="UTF-8 file containing the task."
    )
    parser.add_argument(
        "--rounds", type=int, default=2, help="Number of GPT review rounds."
    )
    parser.add_argument(
        "--output", type=Path, help="Optional file for the final model response."
    )
    parser.add_argument(
        "--claude-model",
        default=os.getenv("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL),
    )
    parser.add_argument(
        "--openai-model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set.")
    return value


def load_task(args: argparse.Namespace) -> str:
    if args.task_file is not None:
        return args.task_file.read_text(encoding="utf-8")
    if args.task is not None:
        return args.task
    raise ValueError("A task or task file is required.")


def main() -> int:
    args = parse_args()
    try:
        task = load_task(args)
        architect = ClaudeArchitect(
            Anthropic(
                api_key=require_env("ANTHROPIC_API_KEY"),
                max_retries=2,
                timeout=120.0,
            ),
            args.claude_model,
        )
        reviewer = GPTReviewer(
            OpenAI(
                api_key=require_env("OPENAI_API_KEY"),
                max_retries=2,
                timeout=120.0,
            ),
            args.openai_model,
        )
        result = run_collaboration_loop(task, architect, reviewer, args.rounds)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result + "\n", encoding="utf-8")
            print(f"Final response written to {args.output}")
        return 0
    except (
        AnthropicAPIError,
        OpenAIAPIError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
