"""
Tests for litellm/router_utils/get_retry_from_policy.py

Covers get_num_retries_from_retry_policy, which resolves the number of
retries to use for a given exception based on a RetryPolicy (either a
global policy or a per-model-group override).
"""

from litellm.exceptions import AuthenticationError, RateLimitError
from litellm.router_utils.get_retry_from_policy import (
    get_num_retries_from_retry_policy,
)
from litellm.types.router import RetryPolicy


class TestGetNumRetriesFromRetryPolicy:
    def test_no_retry_policy_returns_none(self):
        exception = AuthenticationError(
            message="test", llm_provider="openai", model="gpt-5.4"
        )
        assert get_num_retries_from_retry_policy(exception=exception) is None

    def test_matching_exception_type_returns_configured_retries(self):
        exception = AuthenticationError(
            message="test", llm_provider="openai", model="gpt-5.4"
        )
        retry_policy = RetryPolicy(AuthenticationErrorRetries=3)
        assert (
            get_num_retries_from_retry_policy(
                exception=exception, retry_policy=retry_policy
            )
            == 3
        )

    def test_non_matching_exception_type_returns_none(self):
        exception = RateLimitError(
            message="test", llm_provider="openai", model="gpt-5.4"
        )
        retry_policy = RetryPolicy(AuthenticationErrorRetries=3)
        assert (
            get_num_retries_from_retry_policy(
                exception=exception, retry_policy=retry_policy
            )
            is None
        )

    def test_model_group_retry_policy_takes_precedence(self):
        exception = AuthenticationError(
            message="test", llm_provider="openai", model="gpt-5.4"
        )
        retry_policy = RetryPolicy(AuthenticationErrorRetries=1)
        model_group_retry_policy = {
            "my-group": RetryPolicy(AuthenticationErrorRetries=5)
        }
        assert (
            get_num_retries_from_retry_policy(
                exception=exception,
                retry_policy=retry_policy,
                model_group="my-group",
                model_group_retry_policy=model_group_retry_policy,
            )
            == 5
        )
