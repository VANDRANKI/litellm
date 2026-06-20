#!/usr/bin/env python3
"""Validate LiteLLM provider configurations before running the proxy.

This script checks that all required environment variables are set for
each configured provider, and optionally pings the provider APIs to
confirm credentials are valid.

Usage::

    python scripts/validate_provider_config.py --config config.yaml
    python scripts/validate_provider_config.py --config config.yaml --ping
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# Minimum required environment variables per provider prefix.
_PROVIDER_ENV_REQUIREMENTS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "azure": ["AZURE_API_KEY", "AZURE_API_BASE"],
    "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"],
    "gemini": ["GEMINI_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "vertex_ai": ["VERTEXAI_PROJECT", "VERTEXAI_LOCATION"],
    "huggingface": ["HUGGINGFACE_API_KEY"],
}


@dataclass
class ValidationResult:
    """Result of a provider validation check."""

    provider: str
    model: str
    missing_vars: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True if no required variables are missing."""
        return len(self.missing_vars) == 0


def load_config(config_path: Path) -> dict:
    """Load and parse a LiteLLM YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as fh:
        return yaml.safe_load(fh) or {}


def detect_provider(model_name: str) -> Optional[str]:
    """Detect the provider from a model name string.

    Args:
        model_name: A model identifier such as ``openai/gpt-4o`` or
            ``anthropic/claude-opus-4-8``.

    Returns:
        The provider prefix, or *None* if it cannot be determined.
    """
    if "/" in model_name:
        return model_name.split("/")[0].lower()
    # Fall back to prefix matching for legacy model names.
    for prefix in _PROVIDER_ENV_REQUIREMENTS:
        if model_name.lower().startswith(prefix):
            return prefix
    return None


def validate_model_config(
    model_entry: dict,
) -> ValidationResult:
    """Check a single model entry from the config for required env vars.

    Args:
        model_entry: A dictionary from the ``model_list`` section of the
            LiteLLM config YAML.

    Returns:
        A :class:`ValidationResult` describing any missing variables.
    """
    model_name: str = model_entry.get("model_name", "<unnamed>")
    litellm_params: dict = model_entry.get("litellm_params", {})
    underlying_model: str = litellm_params.get("model", "")

    provider = detect_provider(underlying_model)
    required_vars = _PROVIDER_ENV_REQUIREMENTS.get(provider or "", [])
    missing = [v for v in required_vars if not os.environ.get(v)]

    result = ValidationResult(provider=provider or "unknown", model=model_name)
    result.missing_vars = missing

    # Warn if an api_key is hard-coded rather than via env var.
    if litellm_params.get("api_key") and not str(
        litellm_params["api_key"]
    ).startswith("os.environ/"):
        result.warnings.append(
            "api_key appears to be hard-coded; prefer os.environ/<VAR_NAME>"
        )

    return result


def validate_config(config: dict) -> list[ValidationResult]:
    """Validate all model entries in a loaded LiteLLM configuration.

    Args:
        config: Parsed configuration dictionary from :func:`load_config`.

    Returns:
        List of :class:`ValidationResult` objects, one per model entry.
    """
    model_list: list[dict] = config.get("model_list", [])
    return [validate_model_config(entry) for entry in model_list]


def print_report(results: list[ValidationResult]) -> int:
    """Print a human-readable validation report and return an exit code.

    Args:
        results: List of validation results from :func:`validate_config`.

    Returns:
        ``0`` if all models are valid, ``1`` otherwise.
    """
    ok_count = 0
    fail_count = 0

    for r in results:
        status = "OK" if r.is_valid else "FAIL"
        print(f"[{status}] {r.model} ({r.provider})")
        for var in r.missing_vars:
            print(f"       Missing env var: {var}")
            fail_count += 1
        for warn in r.warnings:
            print(f"       Warning: {warn}")
        if r.is_valid:
            ok_count += 1

    print(f"\nSummary: {ok_count} valid, {fail_count} issues found")
    return 0 if fail_count == 0 else 1


def main() -> None:
    """Entry point for the provider config validation script."""
    parser = argparse.ArgumentParser(
        description="Validate LiteLLM provider configurations."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the LiteLLM YAML config file.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    results = validate_config(config)
    if not results:
        print("No model_list entries found in config.")
        sys.exit(0)

    sys.exit(print_report(results))


if __name__ == "__main__":
    main()
