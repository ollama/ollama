#!/usr/bin/env python3
"""
Validate FAANG standards compliance in the codebase.

This script enforces folder structure, naming conventions, and code quality
standards defined in FAANG-ELITE-STANDARDS.md and FOLDER-STRUCTURE-STANDARDS.md

Usage:
    python scripts/validate-standards.py [--fix] [--verbose]
"""

import re
import sys
from pathlib import Path
from typing import ClassVar


class StandardsValidator:
    """Validates code against FAANG standards."""

    FORBIDDEN_DIRS: ClassVar[set[str]] = {
        "Utils",
        "Utility",
        "utils_old",
        "old_code",
        "backup",
        "temp",
        "tmp",
        "test_",
        "tests_old",
        "__old__",
        "deprecated",
    }

    REQUIRED_DIRS: ClassVar[dict[str, str]] = {
        "ollama/config": "Configuration layer",
        "ollama/api": "HTTP API layer",
        "ollama/api/routes": "Route handlers",
        "ollama/api/schemas": "Pydantic schemas",
        "ollama/services": "Business logic",
        "ollama/repositories": "Data access layer",
        "ollama/models": "SQLAlchemy models",
        "ollama/middleware": "HTTP middleware",
        "ollama/monitoring": "Observability",
        "tests/unit": "Unit tests",
        "tests/integration": "Integration tests",
        "tests/fixtures": "Test fixtures",
    }

    MULTIPLE_CLASSES_ALLOWED: ClassVar[set[str]] = {
        "exceptions.py",
        "types.py",
        "constants.py",
        "enums.py",
        "schemas.py",
    }

    def __init__(self, root: Path | None = None, verbose: bool = False) -> None:
        """Initialize validator."""
        self.root = root or Path(".")
        self.verbose = verbose
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self) -> bool:
        """Run all validation checks."""
        self.validate_required_dirs()
        self.validate_forbidden_dirs()
        self.validate_folder_naming()
        self.validate_python_files()
        return self._report_results()

    def validate_required_dirs(self) -> None:
        """Check that required directories exist."""
        for required_dir, description in self.REQUIRED_DIRS.items():
            if not (self.root / required_dir).exists():
                self.errors.append(f"❌ Missing required directory: {required_dir}")
                if self.verbose:
                    print(f"   Purpose: {description}")

    def validate_forbidden_dirs(self) -> None:
        """Check for forbidden directories."""
        for forbidden in self.FORBIDDEN_DIRS:
            if (self.root / forbidden).exists():
                self.errors.append(f"❌ Forbidden directory found: {forbidden}")

    def validate_folder_naming(self) -> None:
        """Check folder naming conventions."""
        for item in (self.root / "ollama").rglob("*"):
            if not item.is_dir() or item.name.startswith("."):
                continue

            # Should be lowercase
            if item.name != item.name.lower():
                self.warnings.append(f"⚠️  Directory not lowercase: {item.relative_to(self.root)}")

            # Should use underscores, not hyphens
            if "-" in item.name and item.name not in {"api-routes"}:
                self.warnings.append(
                    f"⚠️  Directory uses hyphens (use underscores): "
                    f"{item.relative_to(self.root)}"
                )

    def validate_python_files(self) -> None:
        """Check Python file standards."""
        for py_file in (self.root / "ollama").rglob("*.py"):
            if py_file.name == "__init__.py":
                self._validate_init_file(py_file)
            else:
                self._validate_module_file(py_file)

    def _validate_init_file(self, file: Path) -> None:
        """Validate __init__.py file."""
        content = file.read_text()

        # Check for excessive initialization logic
        lines = content.strip().split("\n")
        if len([line for line in lines if line.strip() and not line.strip().startswith("#")]) > 10:
            self.warnings.append(f"⚠️  __init__.py too complex: {file.relative_to(self.root)}")

    def _validate_module_file(self, file: Path) -> None:
        """Validate regular module file."""
        content = file.read_text()

        # Check for module docstring
        if not (content.strip().startswith('"""') or content.strip().startswith("'''")):
            self.errors.append(f"❌ Missing module docstring: {file.relative_to(self.root)}")

        # Count classes
        class_count = len(re.findall(r"^class\s+\w+", content, re.MULTILINE))
        if class_count > 1 and file.name not in self.MULTIPLE_CLASSES_ALLOWED:
            self.errors.append(
                f"❌ Multiple classes in {file.relative_to(self.root)} "
                f"({class_count} found, max 1 allowed)"
            )

        # Check file size
        lines = len(content.split("\n"))
        if lines > 600:
            self.warnings.append(
                f"⚠️  File too large: {file.relative_to(self.root)} " f"({lines} lines, max 600)"
            )

    def _report_results(self) -> bool:
        """Report validation results."""
        print("\n" + "=" * 70)
        print("FAANG STANDARDS VALIDATION REPORT")
        print("=" * 70 + "\n")

        if self.errors:
            print(f"❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   {error}")
            print()

        if self.warnings:
            print(f"⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   {warning}")
            print()

        if not self.errors and not self.warnings:
            print("✅ All standards validated successfully!\n")
            return True

        if self.errors:
            print(f"Result: ❌ FAILED ({len(self.errors)} errors)\n")
            return False

        print(f"Result: ⚠️  PASSED (with {len(self.warnings)} warnings)\n")
        return True


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate FAANG standards compliance")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    root_path = project_root

    validator = StandardsValidator(root_path, verbose=args.verbose)
    success = validator.validate_all()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
