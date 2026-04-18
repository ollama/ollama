"""Feature Flags System for Safe Experimentation.

Provides a unified interface for managing feature toggles with support for:
- A/B testing (percentage-based rollouts)
- Gradual rollouts (time-based progressive deployment)
- User targeting (segment-based feature access)
- Metrics collection (feature usage tracking)
- Fallback strategies (graceful degradation)

Example:
    >>> from ollama.services.feature_flags import FeatureFlagManager
    >>> ffm = FeatureFlagManager(config=config)
    >>> if await ffm.is_enabled("new_inference_endpoint", user_id="usr_123"):
    ...     response = await new_inference_engine.generate(prompt)
    ... else:
    ...     response = await legacy_inference_engine.generate(prompt)

Integration:
    Feature flags are configured in YAML, evaluated per-request, and tracked
    in Prometheus for monitoring. All flags support instant on/off toggling
    without redeployment.
"""

from ollama.services.feature_flags.config import FeatureFlag, FeatureFlagConfig
from ollama.services.feature_flags.evaluator import FeatureFlagEvaluator
from ollama.services.feature_flags.manager import FeatureFlagManager

__all__ = [
    "FeatureFlag",
    "FeatureFlagConfig",
    "FeatureFlagEvaluator",
    "FeatureFlagManager",
]
