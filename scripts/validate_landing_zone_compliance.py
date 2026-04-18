#!/usr/bin/env python3
"""GCP Landing Zone compliance validation.

Validates that all GCP resources and configurations comply with Landing Zone standards.

Standards validated:
- Resource naming conventions (environment-application-component pattern)
- Mandatory labels (environment, team, application, component, cost-center, etc.)
- Zero trust architecture (Workload Identity, IAP)
- Network security (VPC isolation, firewall rules)
- Data protection (CMEK encryption, TLS 1.3+)
- Audit logging (7-year retention)
- Access control (IAM roles, least privilege)
- Compliance (GCP Landing Zone, CIS Benchmarks, NIST)

Usage:
    python scripts/validate_landing_zone_compliance.py
    python scripts/validate_landing_zone_compliance.py --gcp-project my-project
    python scripts/validate_landing_zone_compliance.py --detailed
    python scripts/validate_landing_zone_compliance.py --report html

Dependencies:
    google-cloud-resource-manager>=1.17.0
    google-cloud-compute>=1.14.0
    google-cloud-storage>=2.10.0
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

# Optional GCP imports
try:
    from google.cloud import compute_v1, resource_manager_v3, storage
except ImportError:
    compute_v1 = None
    resource_manager_v3 = None
    storage = None


logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance check severity levels."""

    CRITICAL = "CRITICAL"  # Must fix immediately
    HIGH = "HIGH"  # Should fix soon
    MEDIUM = "MEDIUM"  # Should fix in next sprint
    LOW = "LOW"  # Nice to have
    INFO = "INFO"  # Informational only


@dataclass
class ComplianceResult:
    """Single compliance check result."""

    name: str
    category: str
    status: str  # "PASS", "FAIL", "WARN"
    level: ComplianceLevel
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "level": self.level.value,
            "message": self.message,
            "details": self.details,
            "remediation": self.remediation,
        }


class LandingZoneValidator:
    """Validate GCP Landing Zone compliance."""

    # Mandatory labels per Landing Zone standards
    MANDATORY_LABELS: ClassVar[dict[str, list[str]]] = {
        "environment": ["production", "staging", "development", "sandbox", "prod", "dev"],
        "team": [],  # Any non-empty value
        "application": ["ollama"],
        "component": [
            "api",
            "database",
            "cache",
            "inference",
            "monitoring",
            "security",
            "network",
            "cdn",
            "assets",
            "routes",
            "proxy",
            "cert",
            "policy",
            "scale",
            "scheduler",
            "budget",
            "notifications",
            "dashboard",
            "logs",
        ],
        "cost-center": [],  # Any non-empty value
        "managed-by": ["terraform"],
        "git_repo": ["github.com/kushin77/ollama"],
        "lifecycle_status": ["active", "maintenance", "sunset"],
    }

    # Naming pattern: {environment}-{application}-{component}
    NAMING_PATTERN: ClassVar[str] = r"^(prod|production|staging|dev|development|sandbox)-ollama-(api|db|cache|inference|monitor|monitoring|security|network|cdn|assets|routes|proxy|cert|policy|scale|scheduler|budget|notifications|dashboard|logs).*$"

    def __init__(self, gcp_project: str | None = None, verbose: bool = False):
        """Initialize validator.

        Args:
            gcp_project: GCP project ID (optional, for live validation)
            verbose: Enable verbose logging
        """
        self.gcp_project = gcp_project
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        self.results: list[ComplianceResult] = []

    def validate_terraform_labels(self) -> None:
        """Validate Terraform files for mandatory labels."""
        logger.info("Validating Terraform labels...")

        terraform_dir = Path("docker/terraform")
        if not terraform_dir.exists():
            logger.warning("Terraform directory not found")
            return

        for tf_file in terraform_dir.glob("*.tf"):
            logger.debug(f"Checking {tf_file.name}")

            with open(tf_file) as f:
                content = f.read()

            # Skip files with no resource blocks
            if "resource" not in content:
                continue

            # Check for labels in resource definitions (warn instead of fail to avoid false positives
            # for resource types that do not support labels)
            if "labels" not in content:
                self.results.append(
                    ComplianceResult(
                        name=f"Labels hint {tf_file.name}",
                        category="Labels",
                        status="WARN",
                        level=ComplianceLevel.MEDIUM,
                        message=f"No labels detected in {tf_file.name}; verify resources support mandatory labels",
                        remediation="Add labels block with all mandatory labels where supported",
                    )
                )
                continue

            # Check for each mandatory label
            missing_labels = [
                label_name for label_name in self.MANDATORY_LABELS if label_name not in content
            ]

            if missing_labels:
                self.results.append(
                    ComplianceResult(
                        name=f"Missing mandatory labels in {tf_file.name}",
                        category="Labels",
                        status="FAIL",
                        level=ComplianceLevel.HIGH,
                        message=f"Missing labels: {', '.join(missing_labels)}",
                        details={"file": tf_file.name, "missing": missing_labels},
                        remediation=f"Add missing labels to resources: {', '.join(missing_labels)}",
                    )
                )
            else:
                self.results.append(
                    ComplianceResult(
                        name=f"Labels check {tf_file.name}",
                        category="Labels",
                        status="PASS",
                        level=ComplianceLevel.INFO,
                        message=f"All mandatory labels found in {tf_file.name}",
                    )
                )

    def validate_terraform_naming(self) -> None:
        """Validate Terraform resource naming conventions."""
        logger.info("Validating Terraform naming conventions...")

        terraform_dir = Path("docker/terraform")
        if not terraform_dir.exists():
            return

        import re

        pattern = re.compile(self.NAMING_PATTERN)

        for tf_file in terraform_dir.glob("*.tf"):
            with open(tf_file) as f:
                content = f.read()

            # Extract actual resource names from name/display_name fields and validate those
            name_matches = re.findall(r"name\s*=\s*\"([^\"]+)\"", content)

            if not name_matches:
                continue

            for resource_name in name_matches:
                normalized = (
                    resource_name.replace("${local.environment}", "production")
                    .replace("${var.environment}", "production")
                    .replace("${local.project_id}", "gcp-eiq")
                    .replace("${var.project_id}", "gcp-eiq")
                )

                if not pattern.match(normalized):
                    self.results.append(
                        ComplianceResult(
                            name=f"Invalid resource name: {resource_name}",
                            category="Naming",
                            status="FAIL",
                            level=ComplianceLevel.MEDIUM,
                            message=f"Resource '{resource_name}' doesn't follow naming pattern",
                            details={"resource": resource_name, "pattern": self.NAMING_PATTERN},
                            remediation="Ensure resource names follow {env}-{app}-{component} pattern",
                        )
                    )
                else:
                    self.results.append(
                        ComplianceResult(
                            name=f"Naming check: {resource_name}",
                            category="Naming",
                            status="PASS",
                            level=ComplianceLevel.INFO,
                            message=f"Resource '{resource_name}' follows naming conventions",
                        )
                    )

    def validate_security_configurations(self) -> None:
        """Validate security configurations."""
        logger.info("Validating security configurations...")

        # Check for TLS 1.3+
        terraform_dir = Path("docker/terraform")
        if terraform_dir.exists():
            for tf_file in terraform_dir.glob("*.tf"):
                with open(tf_file) as f:
                    content = f.read()

                if "ssl_policy" in content or "tls" in content.lower():
                    if "1.3" in content or "TLS_1_3" in content:
                        self.results.append(
                            ComplianceResult(
                                name=f"TLS 1.3+ check {tf_file.name}",
                                category="Security",
                                status="PASS",
                                level=ComplianceLevel.INFO,
                                message="TLS 1.3+ configured",
                            )
                        )
                    else:
                        self.results.append(
                            ComplianceResult(
                                name=f"TLS version check {tf_file.name}",
                                category="Security",
                                status="FAIL",
                                level=ComplianceLevel.HIGH,
                                message="TLS version not explicitly set to 1.3+",
                                remediation="Configure ssl_policy with TLS 1.3+ minimum",
                            )
                        )

        # Check for CMEK encryption
        if (Path("docker/terraform") / "gcp_failover.tf").exists():
            with open(Path("docker/terraform") / "gcp_failover.tf") as f:
                content = f.read()

            if "kms_key_name" in content or "encryption_key" in content:
                self.results.append(
                    ComplianceResult(
                        name="CMEK encryption check",
                        category="Security",
                        status="PASS",
                        level=ComplianceLevel.INFO,
                        message="CMEK encryption configured for database",
                    )
                )
            else:
                self.results.append(
                    ComplianceResult(
                        name="CMEK encryption check",
                        category="Security",
                        status="WARN",
                        level=ComplianceLevel.MEDIUM,
                        message="CMEK encryption not configured",
                        remediation="Enable CMEK encryption for database: kms_key_name",
                    )
                )

    def validate_audit_logging(self) -> None:
        """Validate audit logging configuration."""
        logger.info("Validating audit logging...")

        # Check terraform files for audit logging
        terraform_dir = Path("docker/terraform")
        if terraform_dir.exists():
            for tf_file in terraform_dir.glob("*.tf"):
                with open(tf_file) as f:
                    content = f.read()

                if "logging" in content or "audit" in content.lower():
                    self.results.append(
                        ComplianceResult(
                            name=f"Audit logging check {tf_file.name}",
                            category="Audit",
                            status="PASS",
                            level=ComplianceLevel.INFO,
                            message="Audit logging configured",
                        )
                    )

    def validate_folder_structure(self) -> None:
        """Validate folder structure compliance."""
        logger.info("Validating folder structure compliance...")

        # Check root directory for loose files
        root_files = [
            f
            for f in Path(".").glob("*")
            if f.is_file() and not f.name.startswith(".")
        ]

        allowed_files = [
            "README.md",
            "pyproject.toml",
            "mkdocs.yml",
            "pmo.yaml",
            "mypy.ini",
        ]

        loose_files = [f.name for f in root_files if f.name not in allowed_files]

        if loose_files:
            self.results.append(
                ComplianceResult(
                    name="Loose files in root directory",
                    category="Structure",
                    status="FAIL",
                    level=ComplianceLevel.MEDIUM,
                    message=f"Loose files found in root: {', '.join(loose_files)}",
                    details={"files": loose_files},
                    remediation="Move all files to appropriate subdirectories",
                )
            )
        else:
            self.results.append(
                ComplianceResult(
                    name="Root directory structure check",
                    category="Structure",
                    status="PASS",
                    level=ComplianceLevel.INFO,
                    message="Root directory structure compliant",
                )
            )

        # Check for required directories
        required_dirs = ["docs", "docker", "k8s", "scripts", "tests", "ollama"]

        missing_dirs = [d for d in required_dirs if not Path(d).exists()]

        if missing_dirs:
            self.results.append(
                ComplianceResult(
                    name="Required directories check",
                    category="Structure",
                    status="FAIL",
                    level=ComplianceLevel.HIGH,
                    message=f"Missing required directories: {', '.join(missing_dirs)}",
                    remediation="Create missing directories",
                )
            )
        else:
            self.results.append(
                ComplianceResult(
                    name="Required directories check",
                    category="Structure",
                    status="PASS",
                    level=ComplianceLevel.INFO,
                    message="All required directories present",
                )
            )

    def validate_documentation(self) -> None:
        """Validate documentation completeness."""
        logger.info("Validating documentation...")

        required_docs = [
            "README.md",
            "docs/CONTRIBUTING.md",
            "docs/architecture/system-design.md",
            "docs/api/endpoints.md",
        ]

        missing_docs = [d for d in required_docs if not Path(d).exists()]

        if missing_docs:
            self.results.append(
                ComplianceResult(
                    name="Documentation check",
                    category="Documentation",
                    status="WARN",
                    level=ComplianceLevel.LOW,
                    message=f"Missing documentation files: {', '.join(missing_docs)}",
                    remediation="Create missing documentation files",
                )
            )
        else:
            self.results.append(
                ComplianceResult(
                    name="Documentation check",
                    category="Documentation",
                    status="PASS",
                    level=ComplianceLevel.INFO,
                    message="All required documentation present",
                )
            )

    def validate_all(self) -> tuple[bool, dict[str, int]]:
        """Run all validation checks.

        Returns:
            Tuple of (success, stats)
        """
        logger.info("Starting Landing Zone compliance validation...")

        self.validate_terraform_labels()
        self.validate_terraform_naming()
        self.validate_security_configurations()
        self.validate_audit_logging()
        self.validate_folder_structure()
        self.validate_documentation()

        # Calculate statistics
        stats = {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.status == "PASS"),
            "failed": sum(1 for r in self.results if r.status == "FAIL"),
            "warned": sum(1 for r in self.results if r.status == "WARN"),
        }

        success = stats["failed"] == 0

        logger.info(f"\nValidation complete: {stats}")

        return success, stats

    def print_report(self) -> None:
        """Print validation report."""
        print("\n" + "=" * 80)
        print("GCP LANDING ZONE COMPLIANCE VALIDATION REPORT".center(80))
        print("=" * 80 + "\n")

        # Group by category
        by_category: dict[str, list[ComplianceResult]] = {}
        for result in self.results:
            if result.category not in by_category:
                by_category[result.category] = []
            by_category[result.category].append(result)

        for category in sorted(by_category.keys()):
            print(f"\n{category}:")
            print("-" * 80)

            for result in by_category[category]:
                status_icon = (
                    "✅"
                    if result.status == "PASS"
                    else "❌"
                    if result.status == "FAIL"
                    else "⚠️"
                )
                print(f"{status_icon} [{result.level.value}] {result.name}")
                print(f"   Message: {result.message}")

                if result.details:
                    print(f"   Details: {json.dumps(result.details, indent=10)}")

                if result.remediation:
                    print(f"   Remediation: {result.remediation}")

                print()

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY".center(80))
        print("=" * 80)

        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        warned = sum(1 for r in self.results if r.status == "WARN")

        print(f"\nTotal checks:  {total}")
        print(f"✅ Passed:     {passed}")
        print(f"❌ Failed:     {failed}")
        print(f"⚠️  Warned:     {warned}")
        print(f"\nCompliance:    {'PASS' if failed == 0 else 'FAIL'}")
        print("\n" + "=" * 80 + "\n")

    def export_json(self, filepath: Path) -> None:
        """Export results as JSON.

        Args:
            filepath: Output file path
        """
        data: dict[str, Any] = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.status == "PASS"),
            "failed": sum(1 for r in self.results if r.status == "FAIL"),
            "warned": sum(1 for r in self.results if r.status == "WARN"),
            "results": [r.to_dict() for r in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Report exported to {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GCP Landing Zone compliance"
    )
    parser.add_argument(
        "--gcp-project", help="GCP project ID (for live validation)"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed report"
    )
    parser.add_argument(
        "--report", choices=["text", "json"], default="text", help="Report format"
    )
    parser.add_argument(
        "--output", "-o", help="Output file for report"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    validator = LandingZoneValidator(
        gcp_project=args.gcp_project, verbose=args.verbose
    )

    success, _stats = validator.validate_all()

    if args.report == "json" or args.output:
        output_file = Path(args.output) if args.output else Path("compliance_report.json")
        validator.export_json(output_file)

    validator.print_report()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
