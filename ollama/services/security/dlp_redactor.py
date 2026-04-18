"""PII Redaction Service - Google Cloud DLP integration.

Provides ability to redact sensitive info (SSN, credit cards, emails, etc.)
from user prompts and AI responses before they are processed or returned.
"""

from typing import Any

import structlog

try:
    from google.cloud import dlp_v2
except ImportError:
    # Fallback for local development without GCP libraries
    dlp_v2 = None

log = structlog.get_logger(__name__)


class DLPRedactor:
    """Redacts PII using Google Cloud DLP API."""

    def __init__(self, project_id: str | None = None) -> None:
        """Initialize DLP Redactor.

        Args:
            project_id: GCP Project ID. Required for DLP calls.
        """
        self.project_id = project_id
        if dlp_v2:
            self.client = dlp_v2.DlpServiceClient()
        else:
            self.client = None
            log.warning("dlp_not_available", message="google-cloud-dlp not installed")

    async def redact(self, text: str) -> str:
        """Redact PII from text.

        Args:
            text: Input text (prompt or response).

        Returns:
            Redacted text.
        """
        if not self.client or not self.project_id:
            # Fallback for testing or missing config
            return text

        try:
            # Define what to inspect for
            inspect_config = {
                "info_types": [
                    {"name": "PERSON_NAME"},
                    {"name": "EMAIL_ADDRESS"},
                    {"name": "PHONE_NUMBER"},
                    {"name": "CREDIT_CARD_NUMBER"},
                    {"name": "US_SOCIAL_SECURITY_NUMBER"},
                ]
            }

            # Define how to replace matches
            deidentify_config: dict[str, Any] = {
                "info_type_transformations": {
                    "transformations": [
                        {"primitive_transformation": {"replace_with_info_type_config": {}}}
                    ]
                }
            }

            item = {"value": text}
            parent = f"projects/{self.project_id}"

            # API Call (Sync in current google-cloud-dlp version, but we wrap in async)
            # Future: Use threadpool if latency is high
            response = self.client.deidentify_content(
                request={
                    "parent": parent,
                    "deidentify_config": deidentify_config,
                    "inspect_config": inspect_config,
                    "item": item,
                }
            )
            redacted_text = str(response.item.value)
            log.info("pii_redacted", original_len=len(text), redacted_len=len(redacted_text))
            return redacted_text

        except Exception as e:
            log.error("pii_redaction_failed", error=str(e))
            return text  # Fail open for prompt, might need fail closed based on security policy
