#!/usr/bin/env python3
"""Generate infrastructure diagrams from Terraform.

This script auto-generates architecture diagrams using the Python diagrams library.
Diagrams are updated automatically when Terraform files change.

Diagrams generated:
- Deployment topology (primary + secondary regions)
- Service architecture (API, database, cache, inference)
- Data flow (request/response paths)
- Failover flow (health checks, automatic switchover)

Usage:
    python scripts/generate_architecture_diagrams.py
    python scripts/generate_architecture_diagrams.py --watch
    python scripts/generate_architecture_diagrams.py --watch --verbose

Dependencies:
    diagrams>=0.23.3
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from diagrams import Cluster, Diagram, Edge
    from diagrams.gcp.compute import ComputeEngine, InstanceGroup
    from diagrams.gcp.network import GlobalLoadBalancer, CloudArmor
    from diagrams.gcp.database import CloudSQL
    from diagrams.gcp.storage import GCS
    from diagrams.onprem.database import PostgreSQL, Redis
    from diagrams.onprem.inmemory import Redis as RedisCache
    from diagrams.onprem.container import Docker
    from diagrams.c4 import Person, Container, Database, System
except ImportError:
    print("Error: diagrams library not installed.")
    print("Install with: pip install diagrams")
    sys.exit(1)


logger = logging.getLogger(__name__)


class DiagramGenerator:
    """Generate infrastructure diagrams from Terraform."""

    def __init__(self, output_dir: Path = Path("docs/diagrams"), verbose: bool = False):
        """Initialize diagram generator.

        Args:
            output_dir: Directory to save diagrams
            verbose: Enable verbose logging
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _get_terraform_hash(self) -> str:
        """Get hash of all Terraform files.

        Returns:
            SHA256 hash of combined Terraform file contents
        """
        terraform_dir = Path("docker/terraform")
        if not terraform_dir.exists():
            return ""

        hasher = hashlib.sha256()
        for tf_file in sorted(terraform_dir.glob("*.tf")):
            with open(tf_file, "rb") as f:
                hasher.update(f.read())

        return hasher.hexdigest()

    def _load_state_file(self) -> Optional[Dict]:
        """Load previous diagram state.

        Returns:
            Dictionary with previous Terraform hash, or None if not found
        """
        state_file = self.output_dir / ".diagram_state.json"
        if not state_file.exists():
            return None

        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_state_file(self, terraform_hash: str) -> None:
        """Save current diagram state.

        Args:
            terraform_hash: Current Terraform files hash
        """
        state_file = self.output_dir / ".diagram_state.json"
        state = {"terraform_hash": terraform_hash, "timestamp": time.time()}

        with open(state_file, "w") as f:
            json.dump(state, f)

    def _needs_update(self) -> bool:
        """Check if diagrams need to be regenerated.

        Returns:
            True if Terraform files changed or diagrams don't exist
        """
        current_hash = self._get_terraform_hash()
        if not current_hash:
            return False

        state = self._load_state_file()
        if not state:
            return True

        previous_hash = state.get("terraform_hash")
        return current_hash != previous_hash

    def generate_deployment_topology(self) -> None:
        """Generate deployment topology diagram.

        Shows primary and secondary regions with managed instance groups,
        load balancer, health checks, and service interconnections.
        """
        logger.info("Generating deployment topology diagram...")

        with Diagram(
            "Ollama Deployment Topology",
            show=False,
            filename=str(self.output_dir / "deployment_topology"),
            graph_attr={"rankdir": "TB", "bgcolor": "transparent"},
        ):

            with Cluster("External Clients"):
                clients = Person("Users\n(Internet)")

            with Cluster("GCP Network"):
                with Cluster("Global Load Balancer (https://elevatediq.ai/ollama)"):
                    lb = GlobalLoadBalancer("GCP Global\nLoad Balancer")
                    armor = CloudArmor("Cloud Armor\nDDoS Protection")

                with Cluster("Primary Region (us-central1)"):
                    with Cluster("Managed Instance Group"):
                        primary_mig = InstanceGroup("Primary MIG\n3 instances")

                    with Cluster("Internal Services"):
                        primary_api = Docker("FastAPI\n:8000")
                        primary_db = PostgreSQL("PostgreSQL\n:5432")
                        primary_cache = RedisCache("Redis\n:6379")
                        primary_ollama = Docker("Ollama\n:11434")

                with Cluster("Secondary Region (us-east1)\nFailover"):
                    with Cluster("Managed Instance Group"):
                        secondary_mig = InstanceGroup("Secondary MIG\n3 instances")

                    with Cluster("Internal Services"):
                        secondary_api = Docker("FastAPI\n:8000")
                        secondary_db = PostgreSQL("PostgreSQL\n:5432")
                        secondary_cache = RedisCache("Redis\n:6379")
                        secondary_ollama = Docker("Ollama\n:11434")

            # External connections
            clients >> Edge(label="HTTPS/TLS 1.3+") >> armor
            armor >> Edge(label="Authenticate\nRate Limit") >> lb

            # Primary region
            lb >> Edge(label="Active\n(failover=false)", color="green") >> primary_mig
            primary_mig >> primary_api
            primary_api >> Edge(label="Read/Write") >> primary_db
            primary_api >> Edge(label="Cache") >> primary_cache
            primary_api >> Edge(label="Inference") >> primary_ollama

            # Secondary region
            lb >> Edge(label="Standby\n(failover=true)", color="orange") >> secondary_mig
            secondary_mig >> secondary_api
            secondary_api >> Edge(label="Read/Write") >> secondary_db
            secondary_api >> Edge(label="Cache") >> secondary_cache
            secondary_api >> Edge(label="Inference") >> secondary_ollama

        logger.info("✅ Deployment topology saved")

    def generate_service_architecture(self) -> None:
        """Generate service architecture diagram.

        Shows internal service architecture with API, database, cache,
        and inference engine components.
        """
        logger.info("Generating service architecture diagram...")

        with Diagram(
            "Ollama Service Architecture",
            show=False,
            filename=str(self.output_dir / "service_architecture"),
            graph_attr={"rankdir": "LR", "bgcolor": "transparent"},
        ):

            with Cluster("API Layer"):
                api = Container("FastAPI Server", "Python", "HTTP API")

            with Cluster("Service Layer"):
                auth = Container("Auth Service", "Python", "JWT/API Keys")
                inference = Container("Inference Service", "Python", "Model Inference")
                cache_svc = Container("Cache Service", "Python", "Redis Operations")

            with Cluster("Data Layer"):
                db = Database("PostgreSQL", "Conversations, Users")
                cache_store = Database("Redis", "Sessions, Cache")
                models = GCS("GCS Bucket", "Model Assets")

            with Cluster("AI Layer"):
                ollama = Container("Ollama Engine", "Local", "Model Execution")
                models >> Edge(label="Pull Models") >> ollama

            with Cluster("Observability"):
                metrics = Container("Prometheus", "Monitoring", "Metrics")
                logs = Container("Structlog", "Logging", "Structured Logs")

            # Connections
            api >> Edge(label="Authenticate") >> auth
            api >> Edge(label="Generate Text") >> inference
            api >> Edge(label="Cache Get/Set") >> cache_svc
            inference >> Edge(label="Inference") >> ollama
            cache_svc >> cache_store
            api >> Edge(label="Save/Query") >> db
            api >> metrics
            api >> logs

        logger.info("✅ Service architecture saved")

    def generate_data_flow(self) -> None:
        """Generate data flow diagram.

        Shows request/response data flow through the system.
        """
        logger.info("Generating data flow diagram...")

        with Diagram(
            "Ollama Data Flow",
            show=False,
            filename=str(self.output_dir / "data_flow"),
            graph_attr={"rankdir": "TB", "bgcolor": "transparent"},
        ):

            with Cluster("Client"):
                client = Person("User/App")

            with Cluster("GCP Load Balancer"):
                lb = GlobalLoadBalancer("Global LB\n(https://elevatediq.ai/ollama)")

            with Cluster("API Server"):
                auth_check = Container("Auth", "Python", "Validate API Key")
                rate_limit = Container("Rate Limit", "Python", "Check Limit")
                route = Container("Route", "FastAPI", "Select Handler")

            with Cluster("Handler"):
                validate = Container("Validate", "Python", "Input Validation")
                business_logic = Container("Business Logic", "Python", "Process Request")

            with Cluster("Data Layer"):
                db = Database("PostgreSQL", "Read/Write")
                cache = Database("Redis", "Get/Set")

            with Cluster("Inference"):
                inference = Container("Inference", "Ollama", "Generate Response")

            with Cluster("Response"):
                format_resp = Container("Format", "Python", "JSON Response")
                return_resp = Container("Return", "HTTPS", "Send Client")

            # Flow
            client >> Edge(label="1. POST /api/v1/generate") >> lb
            lb >> Edge(label="2. Route to API") >> auth_check
            auth_check >> Edge(label="3. Check Rate Limit") >> rate_limit
            rate_limit >> Edge(label="4. Route Request") >> route
            route >> Edge(label="5. Validate Input") >> validate
            validate >> Edge(label="6. Process") >> business_logic
            business_logic >> Edge(label="7. Check Cache") >> cache
            business_logic >> Edge(label="8. Save State") >> db
            business_logic >> Edge(label="9. Generate") >> inference
            inference >> Edge(label="10. Get Response") >> format_resp
            format_resp >> Edge(label="11. Send") >> return_resp
            return_resp >> Edge(label="12. Receive") >> client

        logger.info("✅ Data flow saved")

    def generate_failover_flow(self) -> None:
        """Generate failover flow diagram.

        Shows health check monitoring and automatic failover process.
        """
        logger.info("Generating failover flow diagram...")

        with Diagram(
            "Ollama Failover Flow",
            show=False,
            filename=str(self.output_dir / "failover_flow"),
            graph_attr={"rankdir": "TB", "bgcolor": "transparent"},
        ):

            with Cluster("Load Balancer"):
                lb = GlobalLoadBalancer("Global\nLoad Balancer")

            with Cluster("Primary Region\n(us-central1)"):
                primary = Container(
                    "Primary Backend", "GCP", "failover=false (Active)"
                )
                health_primary = Container("Health Check", "HTTP", "GET /health")

            with Cluster("Secondary Region\n(us-east1)"):
                secondary = Container(
                    "Secondary Backend", "GCP", "failover=true (Standby)"
                )
                health_secondary = Container("Health Check", "HTTP", "GET /health")

            with Cluster("Monitoring"):
                metrics = Container("Metrics", "Prometheus", "Latency, Errors")

            # Normal operation
            lb >> Edge(label="Active", color="green") >> primary
            primary >> health_primary
            health_primary >> Edge(label="200 OK ✓", color="green") >> lb
            secondary >> health_secondary

            # Failover scenario
            health_primary >> Edge(
                label="Failure #1\nFailure #2\nFailure #3 → Failover",
                color="red",
            ) >> lb
            lb >> Edge(label="Standby", color="orange") >> secondary
            health_secondary >> Edge(label="200 OK ✓", color="green") >> lb

            # Metrics
            lb >> Edge(label="Track") >> metrics
            health_primary >> Edge(label="Report") >> metrics
            health_secondary >> Edge(label="Report") >> metrics

        logger.info("✅ Failover flow saved")

    def generate_all(self) -> Tuple[bool, int]:
        """Generate all diagrams.

        Returns:
            Tuple of (success, diagram_count)
        """
        if not self._needs_update():
            logger.info("Diagrams up to date (Terraform files unchanged)")
            return True, 0

        logger.info("Terraform files changed, regenerating diagrams...")

        try:
            self.generate_deployment_topology()
            self.generate_service_architecture()
            self.generate_data_flow()
            self.generate_failover_flow()

            # Save state
            self._save_state_file(self._get_terraform_hash())

            logger.info("✅ All diagrams generated successfully")
            return True, 4

        except Exception as e:
            logger.error(f"❌ Error generating diagrams: {e}", exc_info=True)
            return False, 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate infrastructure diagrams from Terraform"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch for Terraform changes and regenerate diagrams",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--output", "-o", default="docs/diagrams", help="Output directory"
    )

    args = parser.parse_args()

    generator = DiagramGenerator(output_dir=Path(args.output), verbose=args.verbose)

    if args.watch:
        logger.info("Watching Terraform files for changes...")
        logger.info("Press Ctrl+C to stop")

        previous_hash = ""
        try:
            while True:
                current_hash = generator._get_terraform_hash()
                if current_hash and current_hash != previous_hash:
                    if previous_hash:  # Skip first iteration
                        success, count = generator.generate_all()
                        if success and count > 0:
                            logger.info(
                                f"Diagrams regenerated ({count} files) at {time.strftime('%H:%M:%S')}"
                            )
                    previous_hash = current_hash

                time.sleep(2)  # Check every 2 seconds

        except KeyboardInterrupt:
            logger.info("\nStopped watching")
            return 0

    else:
        success, count = generator.generate_all()
        if success:
            logger.info(f"Generated {count} diagrams in {args.output}/")
            return 0
        else:
            logger.error("Failed to generate diagrams")
            return 1


if __name__ == "__main__":
    sys.exit(main())
