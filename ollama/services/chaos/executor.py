"""
Chaos Experiment Executor
=========================

Executes chaos injections on actual infrastructure: network chaos,
resource exhaustion, service failures, and cascading fault scenarios.

Provides:
    - Network chaos injection (latency, packet loss, bandwidth throttling)
    - Compute resource chaos (CPU throttling, memory limits)
    - Service failure injection (pod crashes, connection killing)
    - Cascading failure orchestration
    - Graceful failure handling and recovery
    - Structured logging with detailed metrics

Example:
    >>> from ollama.services.chaos import ChaosExecutor
    >>> executor = ChaosExecutor()
    >>> await executor.inject_network_chaos(
    ...     target_pod="inference-pod-123",
    ...     latency_ms=200,
    ...     packet_loss_percent=5
    ... )
    >>> await executor.verify_chaos_active(target_pod="inference-pod-123")
    >>> await executor.cleanup_chaos(target_pod="inference-pod-123")
"""

import subprocess
from dataclasses import dataclass
from typing import Any

import structlog

from ollama.services.chaos.config import ComputeConfig, NetworkConfig

log = structlog.get_logger(__name__)


@dataclass
class ChaosInjectionStatus:
    """Status of chaos injection."""

    target: str
    chaos_type: str
    active: bool
    applied_at: str | None = None
    # Config may be omitted; make it Optional to avoid mypy assignment errors
    config: dict[str, Any] | None = None


class ChaosExecutor:
    """Executes chaos experiments on actual infrastructure."""

    # Kubernetes namespace for ollama
    NAMESPACE = "ollama"

    # Container runtime (docker or containerd)
    CONTAINER_RUNTIME = "docker"

    def __init__(self) -> None:
        """Initialize chaos executor."""
        self.active_injections: dict[str, ChaosInjectionStatus] = {}
        self.namespace = self.NAMESPACE

    async def inject_network_chaos(
        self,
        target_pod: str,
        config: NetworkConfig,
    ) -> ChaosInjectionStatus:
        """Inject network chaos into target pod.

        Args:
            target_pod: Target pod name or container ID
            config: Network configuration with latency, loss, etc.

        Returns:
            Injection status

        Raises:
            RuntimeError: If injection fails
        """
        log.info(
            "network_chaos_injection_start",
            target_pod=target_pod,
            latency_ms=config.latency_ms,
            packet_loss_percent=config.packet_loss_percent,
        )

        try:
            # Get pod container ID
            container_id = await self._get_container_id(target_pod)

            # Use tc (traffic control) to inject network chaos
            # This requires CAP_NET_ADMIN on container

            # Add latency
            if config.latency_ms > 0:
                cmd = (
                    f"docker exec {container_id} tc qdisc add dev eth0 root netem "
                    f"delay {config.latency_ms}ms {config.jitter_ms}ms"
                )
                await self._execute_command(cmd)

            # Add packet loss
            if config.packet_loss_percent > 0:
                cmd = (
                    f"docker exec {container_id} tc qdisc add dev eth0 parent 1:1 netem "
                    f"loss {config.packet_loss_percent}%"
                )
                await self._execute_command(cmd)

            # Add bandwidth throttling
            if (config.bandwidth_limit_mbps or 0) > 0:
                cmd = (
                    f"docker exec {container_id} tc class add dev eth0 parent 1: classid 1:1 "
                    f"htb rate {config.bandwidth_limit_mbps}mbit"
                )
                await self._execute_command(cmd)

            status = ChaosInjectionStatus(
                target=target_pod,
                chaos_type="network",
                active=True,
                config={
                    "latency_ms": config.latency_ms,
                    "jitter_ms": config.jitter_ms,
                    "packet_loss_percent": config.packet_loss_percent,
                    "bandwidth_limit_mbps": config.bandwidth_limit_mbps,
                },
            )

            self.active_injections[target_pod] = status
            log.info(
                "network_chaos_injection_success",
                target_pod=target_pod,
            )

            return status

        except Exception as e:
            log.error(
                "network_chaos_injection_failed",
                target_pod=target_pod,
                error=str(e),
            )
            raise RuntimeError(f"Network chaos injection failed: {e}") from e

    async def inject_compute_chaos(
        self,
        target_pod: str,
        config: ComputeConfig,
    ) -> ChaosInjectionStatus:
        """Inject compute resource chaos into target pod.

        Args:
            target_pod: Target pod name or container ID
            config: Compute configuration with CPU, memory limits

        Returns:
            Injection status

        Raises:
            RuntimeError: If injection fails
        """
        # Normalize ComputeConfig fields with safe fallbacks for backward
        # compatibility: older callers or different field names may exist in
        # the config model. Use getattr with sensible defaults.
        cpu_throttle_percent = int(
            getattr(config, "cpu_throttle_percent", None)
            or getattr(config, "cpu_percent", 0)  # percent as float
            or 0
        )
        memory_limit_mb = getattr(config, "memory_limit_mb", None) or getattr(
            config, "memory_mb", 0
        )

        log.info(
            "compute_chaos_injection_start",
            target_pod=target_pod,
            cpu_throttle_percent=cpu_throttle_percent,
            memory_limit_mb=memory_limit_mb,
        )

        try:
            container_id = await self._get_container_id(target_pod)

            # Update CPU limits using cgroups
            if cpu_throttle_percent > 0:
                cpu_quota = int(100000 * (cpu_throttle_percent / 100))
                cmd = (
                    f"docker exec {container_id} sh -c "
                    f"'echo {cpu_quota} > /sys/fs/cgroup/cpu/cpu.cfs_quota_us'"
                )
                await self._execute_command(cmd)

            # Update memory limits
            if memory_limit_mb and memory_limit_mb > 0:
                memory_bytes = memory_limit_mb * 1024 * 1024
                cmd = (
                    f"docker exec {container_id} sh -c "
                    f"'echo {memory_bytes} > /sys/fs/cgroup/memory/memory.limit_in_bytes'"
                )
                await self._execute_command(cmd)

            # Inject CPU burn if configured (backwards compatible field names)
            cpu_burn_percent = getattr(config, "cpu_burn_percent", 0) or 0
            duration_seconds = getattr(config, "duration_seconds", 0) or 0
            if cpu_burn_percent > 0 and duration_seconds > 0:
                cmd = (
                    f"docker exec -d {container_id} stress --cpu 1 "
                    f"--timeout {duration_seconds}s"
                )
                await self._execute_command(cmd)

            # Inject memory pressure if configured
            memory_pressure_percent = getattr(config, "memory_pressure_percent", 0) or 0
            if memory_pressure_percent > 0 and memory_limit_mb and duration_seconds > 0:
                memory_alloc = int((memory_limit_mb * memory_pressure_percent) / 100)
                cmd = (
                    f"docker exec -d {container_id} stress --vm 1 "
                    f"--vm-bytes {memory_alloc}M --timeout {duration_seconds}s"
                )
                await self._execute_command(cmd)

            status = ChaosInjectionStatus(
                target=target_pod,
                chaos_type="compute",
                active=True,
                config={
                    "cpu_throttle_percent": cpu_throttle_percent,
                    "cpu_burn_percent": cpu_burn_percent,
                    "memory_limit_mb": memory_limit_mb,
                    "memory_pressure_percent": memory_pressure_percent,
                },
            )

            self.active_injections[target_pod] = status
            log.info(
                "compute_chaos_injection_success",
                target_pod=target_pod,
            )

            return status

        except Exception as e:
            log.error(
                "compute_chaos_injection_failed",
                target_pod=target_pod,
                error=str(e),
            )
            raise RuntimeError(f"Compute chaos injection failed: {e}") from e

    async def inject_service_failure(
        self,
        target_pod: str,
        failure_mode: str = "crash",
    ) -> ChaosInjectionStatus:
        """Inject service failure into target pod.

        Args:
            target_pod: Target pod name or container ID
            failure_mode: Type of failure (crash, hang, timeout)

        Returns:
            Injection status

        Raises:
            RuntimeError: If injection fails
        """
        log.info(
            "service_failure_injection_start",
            target_pod=target_pod,
            failure_mode=failure_mode,
        )

        try:
            container_id = await self._get_container_id(target_pod)

            if failure_mode == "crash":
                # Kill container process
                cmd = f"docker kill {container_id}"
                await self._execute_command(cmd)
                log.info("container_killed", target_pod=target_pod)

            elif failure_mode == "hang":
                # Pause container (simulates hang)
                cmd = f"docker pause {container_id}"
                await self._execute_command(cmd)
                log.info("container_paused", target_pod=target_pod)

            elif failure_mode == "timeout":
                # Add long network latency (simulates timeout)
                cmd = (
                    f"docker exec {container_id} tc qdisc add dev eth0 root netem " f"delay 30000ms"
                )
                await self._execute_command(cmd)
                log.info("container_timeout_simulated", target_pod=target_pod)

            status = ChaosInjectionStatus(
                target=target_pod,
                chaos_type="service_failure",
                active=True,
                config={"failure_mode": failure_mode},
            )

            self.active_injections[target_pod] = status
            log.info(
                "service_failure_injection_success",
                target_pod=target_pod,
            )

            return status

        except Exception as e:
            log.error(
                "service_failure_injection_failed",
                target_pod=target_pod,
                error=str(e),
            )
            raise RuntimeError(f"Service failure injection failed: {e}") from e

    async def inject_cascading_failure(
        self,
        primary_pod: str,
        dependent_pods: list[str],
        delay_seconds: int = 5,
    ) -> dict[str, ChaosInjectionStatus]:
        """Inject cascading failure across multiple services.

        Args:
            primary_pod: Primary service to fail first
            dependent_pods: Dependent services that should fail
            delay_seconds: Delay before failing dependent pods

        Returns:
            Dictionary mapping pod names to injection status
        """
        log.info(
            "cascading_failure_injection_start",
            primary_pod=primary_pod,
            dependent_pods=dependent_pods,
            delay_seconds=delay_seconds,
        )

        results = {}

        try:
            # Inject primary failure
            primary_status = await self.inject_service_failure(primary_pod, "crash")
            results[primary_pod] = primary_status

            # Wait before cascading
            import asyncio

            await asyncio.sleep(delay_seconds)

            # Inject dependent failures
            for pod in dependent_pods:
                try:
                    status = await self.inject_service_failure(pod, "hang")
                    results[pod] = status
                except Exception as e:
                    log.warning(
                        "cascading_failure_partial",
                        pod=pod,
                        error=str(e),
                    )

            log.info(
                "cascading_failure_injection_success",
                primary_pod=primary_pod,
                failed_pods=list(results.keys()),
            )

            return results

        except Exception as e:
            log.error(
                "cascading_failure_injection_failed",
                primary_pod=primary_pod,
                error=str(e),
            )
            raise RuntimeError(f"Cascading failure injection failed: {e}") from e

    async def cleanup_chaos(self, target_pod: str) -> bool:
        """Clean up chaos injection from target pod.

        Args:
            target_pod: Target pod to clean up

        Returns:
            True if cleanup successful, False otherwise
        """
        log.info("chaos_cleanup_start", target_pod=target_pod)

        try:
            container_id = await self._get_container_id(target_pod)

            # Remove traffic control rules
            try:
                cmd = f"docker exec {container_id} tc qdisc del dev eth0 root"
                await self._execute_command(cmd)
            except Exception as e:
                log.warning("tc_cleanup_failed", error=str(e))

            # Resume paused containers
            try:
                cmd = f"docker unpause {container_id}"
                await self._execute_command(cmd)
            except Exception:
                pass  # Container might not be paused

            # Remove from active injections
            self.active_injections.pop(target_pod, None)

            log.info("chaos_cleanup_success", target_pod=target_pod)
            return True

        except Exception as e:
            log.error("chaos_cleanup_failed", target_pod=target_pod, error=str(e))
            return False

    async def verify_chaos_active(self, target_pod: str) -> bool:
        """Verify that chaos injection is active on target pod.

        Args:
            target_pod: Target pod to verify

        Returns:
            True if chaos active, False otherwise
        """
        status = self.active_injections.get(target_pod)
        return status is not None and status.active

    async def _get_container_id(self, target_pod: str) -> str:
        """Get container ID for target pod.

        Args:
            target_pod: Pod name or container ID

        Returns:
            Container ID

        Raises:
            RuntimeError: If container not found
        """
        # If already a container ID, return as-is
        if len(target_pod) == 12 or (len(target_pod) == 64):
            return target_pod

        # Query pod for container ID
        cmd = f"kubectl get pod {target_pod} -n {self.namespace} -o jsonpath='{{.status.containerStatuses[0].containerID}}'"

        try:
            output = await self._execute_command(cmd)
            # Extract container ID from containerd:// format
            container_id = output.strip().split("//")[-1] if "//" in output else output
            return container_id
        except Exception:
            # Fallback: assume it's a docker container name
            log.warning("pod_to_container_id_failed", target_pod=target_pod)
            return target_pod

    async def _execute_command(self, command: str) -> str:
        """Execute shell command.

        Args:
            command: Command to execute

        Returns:
            Command output

        Raises:
            RuntimeError: If command fails
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Command failed with code {result.returncode}: {result.stderr}")

            return result.stdout

        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Command execution timed out") from e
        except Exception as e:
            log.error("command_execution_failed", command=command, error=str(e))
            raise RuntimeError(f"Command execution failed: {e}") from e
