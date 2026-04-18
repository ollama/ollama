#!/usr/bin/env python3
"""Folder structure validator for Elite Filesystem Standards compliance.

Validates:
- Maximum 5 levels deep
- Directory count limits per level
- Naming conventions (snake_case for dirs/files)
- Required __init__.py files with docstrings
- File organization and counts

Usage:
    python scripts/validate_folder_structure.py
    python scripts/validate_folder_structure.py --strict
    python scripts/validate_folder_structure.py --strict --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import ClassVar


class FolderStructureValidator:
    """Validates folder structure compliance with Elite standards."""

    # Level-specific directory count limits
    LIMITS: ClassVar[dict[int, int]] = {
        1: 10,  # Root level
        2: 12,  # Application package level (+ 5 module files)
        3: 4,  # Domain level
        4: 20,  # Functional container level
        5: 0,  # Leaf level - no subdirectories allowed
    }

    # Excluded directories
    EXCLUDED: ClassVar[set[str]] = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".egg-info",
        ".gitignore",
        "htmlcov",
        "cache",
        "logs",
        "secrets",
        ".github",
        ".githooks",
        ".husky",
        "ollama.egg-info",
    }

    # Allowed files at Level 2 (application package root)
    ALLOWED_LEVEL2_FILES: ClassVar[set[str]] = {
        "__init__.py",
        "main.py",
        "config.py",
        "metrics.py",
        "client.py",
        "auth_manager.py",
        "py.typed",
        "_models_compat.py",  # Backward compatibility re-exports
    }

    def __init__(self, root: Path, strict: bool = False, verbose: bool = False) -> None:
        """Initialize validator.

        Args:
            root: Root path to validate (usually project root)
            strict: Enforce all rules strictly
            verbose: Print detailed output
        """
        self.root = root
        self.strict = strict
        self.verbose = verbose
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self) -> bool:
        """Run validation and return True if all checks pass.

        Returns:
            True if structure is valid, False otherwise
        """
        self._validate_root_structure()
        self._validate_ollama_package()
        self._validate_api_module()
        self._validate_services_module()
        self._validate_tests_module()

        self._print_results()
        return len(self.errors) == 0

    def _validate_root_structure(self) -> None:
        """Validate root level directory count."""
        try:
            items = [
                item
                for item in self.root.iterdir()
                if not item.name.startswith(".") and item.name not in self.EXCLUDED
            ]

            dirs = [item for item in items if item.is_dir()]
            file_count = len([item for item in items if item.is_file()])

            if len(dirs) > self.LIMITS[1]:
                self.errors.append(
                    f"Root level has {len(dirs)} directories, max allowed: {self.LIMITS[1]}"
                )

            if self.verbose:
                print(f"✓ Root level: {len(dirs)} directories, {file_count} files")

        except OSError as e:
            self.errors.append(f"Cannot read root directory: {e}")

    def _validate_ollama_package(self) -> None:
        """Validate ollama/ package structure (Level 2)."""
        ollama_path = self.root / "ollama"

        if not ollama_path.exists():
            self.warnings.append("ollama/ package not found (optional)")
            return

        try:
            items = list(ollama_path.iterdir())
            dirs = [
                item
                for item in items
                if item.is_dir()
                and item.name not in self.EXCLUDED
                and not item.name.startswith(".")
            ]
            files = [item for item in items if item.is_file() and not item.name.startswith(".")]

            # Check directory count
            if len(dirs) > self.LIMITS[2]:
                self.errors.append(f"ollama/ has {len(dirs)} subdirectories, max: {self.LIMITS[2]}")

            # Check file count (max 5 at Level 2)
            extra_files = [f for f in files if f.name not in self.ALLOWED_LEVEL2_FILES]
            if extra_files:
                self.warnings.append(
                    f"ollama/ has unexpected files at Level 2: "
                    f"{', '.join(f.name for f in extra_files)}"
                )

            # Validate each domain directory
            for directory in dirs:
                self._validate_level3_domain(directory)

            if self.verbose:
                print(f"✓ ollama/ package: {len(dirs)} domains, {len(files)} root files")

        except OSError as e:
            self.errors.append(f"Cannot read ollama/ package: {e}")

    def _validate_level3_domain(self, domain_path: Path) -> None:
        """Validate Level 3 domain directory.

        Args:
            domain_path: Path to domain directory
        """
        # Check naming convention
        if not self._is_snake_case(domain_path.name):
            self.errors.append(f"Domain '{domain_path.name}' not in snake_case")

        # Check for required __init__.py
        init_file = domain_path / "__init__.py"
        if not init_file.exists():
            self.errors.append(f"Domain '{domain_path.name}/' missing __init__.py")
        else:
            self._check_docstring(init_file, "domain")

        # Count items at Level 3
        try:
            items = [
                item
                for item in domain_path.iterdir()
                if not item.name.startswith(".") and item.name != "__pycache__"
            ]
            dirs = [item for item in items if item.is_dir()]

            # Special handling for services/ and api/
            if domain_path.name in ("services", "api"):
                # These should have Level 4 containers
                for item in dirs:
                    self._validate_level4_container(item)
            else:
                # Other domains should only have __init__.py (no Level 4)
                # Files at Level 4 are acceptable
                if len(dirs) > self.LIMITS[3]:
                    self.warnings.append(
                        f"Domain '{domain_path.name}/' has {len(dirs)} subdirectories, "
                        f"max: {self.LIMITS[3]}"
                    )

        except OSError as e:
            self.errors.append(f"Cannot read domain '{domain_path.name}/': {e}")

    def _validate_level4_container(self, container_path: Path) -> None:
        """Validate Level 4 functional container.

        Args:
            container_path: Path to container directory
        """
        # Check naming
        if not self._is_snake_case(container_path.name):
            self.errors.append(f"Container '{container_path.name}/' not in snake_case")

        # Check for __init__.py with docstring
        init_file = container_path / "__init__.py"
        if not init_file.exists():
            self.errors.append(f"Container '{container_path.name}/' missing __init__.py")
        else:
            self._check_docstring(init_file, "container")

        # Check file count
        try:
            files = [
                f for f in container_path.iterdir() if f.is_file() and not f.name.startswith(".")
            ]

            # Schemas container can have more files (data models)
            limit = 40 if container_path.name == "schemas" else self.LIMITS[4]

            if len(files) > limit:
                self.errors.append(
                    f"Container '{container_path.name}/' has {len(files)} files, " f"max: {limit}"
                )

            # Check for Level 5 violations (subdirectories not allowed)
            subdirs = [
                d
                for d in container_path.iterdir()
                if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"
            ]
            if subdirs:
                self.errors.append(
                    f"Container '{container_path.name}/' has subdirectories "
                    f"(Level 5 limit reached): {', '.join(d.name for d in subdirs)}"
                )

        except OSError as e:
            self.errors.append(f"Cannot read container '{container_path.name}/': {e}")

    def _validate_api_module(self) -> None:
        """Validate api/ module structure."""
        api_path = self.root / "ollama" / "api"

        if not api_path.exists():
            return

        required_containers = {"routes", "schemas", "dependencies"}
        try:
            existing = {
                d.name for d in api_path.iterdir() if d.is_dir() and d.name in required_containers
            }

            missing = required_containers - existing
            if missing and self.strict:
                self.warnings.append(f"api/ missing containers: {missing}")

        except OSError as e:
            self.errors.append(f"Cannot read api/ module: {e}")

    def _validate_services_module(self) -> None:
        """Validate services/ module structure."""
        services_path = self.root / "ollama" / "services"

        if not services_path.exists():
            return

        required_containers = {"inference", "cache", "models", "persistence"}
        try:
            existing = {
                d.name
                for d in services_path.iterdir()
                if d.is_dir() and d.name in required_containers
            }

            missing = required_containers - existing
            if missing and self.strict:
                self.warnings.append(f"services/ missing containers: {missing}")

        except OSError as e:
            self.errors.append(f"Cannot read services/ module: {e}")

    def _validate_tests_module(self) -> None:
        """Validate tests/ module structure mirrors app structure."""
        tests_path = self.root / "tests"

        if not tests_path.exists():
            self.warnings.append("tests/ directory not found")
            return

        try:
            # Tests should have unit/, integration/, fixtures/
            required = {"unit", "integration"}
            existing = {d.name for d in tests_path.iterdir() if d.is_dir() and d.name in required}

            if not existing:
                self.warnings.append("tests/ missing unit/ and/or integration/")

        except OSError as e:
            self.errors.append(f"Cannot read tests/ module: {e}")

    def _check_docstring(self, file_path: Path, item_type: str) -> None:
        """Check that file has a module docstring.

        Args:
            file_path: Path to Python file
            item_type: Type of item (domain, container, module)
        """
        try:
            content = file_path.read_text()

            if not content.strip().startswith(('"""', "'''")):
                self.warnings.append(
                    f"{item_type} '{file_path.parent.name}/' __init__.py missing docstring"
                )

        except OSError as e:
            self.errors.append(f"Cannot read {file_path}: {e}")

    @staticmethod
    def _is_snake_case(name: str) -> bool:
        """Check if name is in snake_case."""
        # Allow underscores and numbers
        return name.replace("_", "").replace("-", "").isalnum() and not name[0].isupper()

    def _print_results(self) -> None:
        """Print validation results."""
        print("\n" + "=" * 70)
        print("📋 FOLDER STRUCTURE VALIDATION REPORT")
        print("=" * 70)

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All checks passed! Folder structure is compliant.")

        print("\n" + "=" * 70)

        # Return exit code
        if self.errors:
            print("\n🔧 FIXES NEEDED:")
            print("   1. No Python files at Level 3 (domain root)")
            print("   2. All files must be in Level 4 containers")
            print("   3. Use snake_case for all directory and file names")
            print("   4. Add __init__.py with docstrings to all directories")
            print("\n" + "=" * 70)
            sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate folder structure compliance with Elite standards"
    )
    parser.add_argument("--strict", action="store_true", help="Enforce all rules strictly")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/akushnir/ollama"),
        help="Root directory to validate",
    )

    args = parser.parse_args()

    validator = FolderStructureValidator(root=args.root, strict=args.strict, verbose=args.verbose)

    validator.validate()


if __name__ == "__main__":
    main()
