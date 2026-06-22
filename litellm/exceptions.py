# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

"""LiteLLM exception hierarchy — OpenAI-compatible error types.

All exceptions inherit from the corresponding ``openai`` exception so that
callers who already handle OpenAI SDK errors continue to work without
modification.  Each class adds three extra attributes to ease retry and
debugging logic:

- ``llm_provider``: The provider that raised the error (e.g. ``"openai"``,
  ``"anthropic"``, ``"bedrock"``).
- ``model``: The model string that was being called.
- ``litellm_debug_info``: Optional extra context set by LiteLLM's router
  or middleware layers.

Exception hierarchy (HTTP status code in brackets):

    AuthenticationError        [401]  → openai.AuthenticationError
    PermissionDeniedError      [403]  → openai.PermissionDeniedError
    NotFoundError              [404]  → openai.NotFoundError
    Timeout                    [408]  → openai.APITimeoutError
    UnprocessableEntityError   [422]  → openai.UnprocessableEntityError
    RateLimitError             [429]  → openai.RateLimitError
    InternalServerError        [500]  → openai.InternalServerError
    BadGatewayError            [502]  → openai.APIStatusError
    ServiceUnavailableError    [503]  → openai.APIStatusError
    BadRequestError            [400]  → openai.BadRequestError
        ContextWindowExceededError   (subclass — context too long)
        ContentPolicyViolationError  (subclass — safety filter triggered)
        RejectedRequestError         (subclass — proxy guardrail blocked)
        UnsupportedParamsError       (subclass — param not supported by provider)
        ImageFetchError              (subclass — image URL could not be fetched)
    APIError                         → openai.APIError  (catch-all for unexpected status codes)
    APIConnectionError               → openai.APIConnectionError
    APIResponseValidationError       → openai.APIResponseValidationError
        JSONSchemaValidationError    (structured output schema mismatch)
    BudgetExceededError              → Exception  (spend limit reached)
    OpenAIError                      → openai.OpenAIError  (pass-through)
    MidStreamFallbackError           → ServiceUnavailableError  (streaming failure)
    ModifyResponseException          → Exception  (guardrail synthetic response)
    GuardrailRaisedException         → Exception
    GuardrailInterventionNormalStringError → Exception
    BlockedPiiEntityError            → Exception
"""

## LiteLLM versions of the OpenAI Exception Types

from typing import Any, Dict, Optional

import httpx
import openai

from litellm.types.utils import LiteLLMCommonStrings

_MINIMAL_ERROR_RESPONSE: Optional[httpx.Response] = None


def _get_minimal_error_response() -> httpx.Response:
    """Get a cached minimal httpx.Response object for error cases."""
    global _MINIMAL_ERROR_RESPONSE
    if _MINIMAL_ERROR_RESPONSE is None:
        _MINIMAL_ERROR_RESPONSE = httpx.Response(
            status_code=400,
            request=httpx.Request(method="GET", url="https://litellm.ai"),
        )
    return _MINIMAL_ERROR_RESPONSE


class AuthenticationError(openai.AuthenticationError):  # type: ignore
    """Raised when API credentials are missing, invalid, or expired.

    Maps to HTTP 401.  Common causes:

    - Missing or malformed API key.
    - Key has been revoked or has insufficient permissions.
    - Wrong key used for the target provider.

    Args:
        message: Human-readable error description from the provider.
        llm_provider: Provider name (e.g. ``"openai"``, ``"anthropic"``).
        model: Model string that was being called.
        response: Raw ``httpx.Response`` from the provider, if available.
        litellm_debug_info: Optional extra context injected by LiteLLM
            middleware.
        max_retries: Maximum number of retries configured for this request.
        num_retries: Number of retries already attempted.
    """

    def __init__(
        self,
        message,
        llm_provider,
        model,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 401
        self.message = "litellm.AuthenticationError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        self.response = response or httpx.Response(
            status_code=self.status_code,
            request=httpx.Request(
                method="GET", url="https://litellm.ai"
            ),  # mock request object
        )
        super().__init__(
            self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


# raise when invalid models passed, example gpt-8
class NotFoundError(openai.NotFoundError):  # type: ignore
    """Raised when the requested model or resource does not exist.

    Maps to HTTP 404.  Typical causes:

    - Model ID is misspelled or not available in the target region.
    - Fine-tuned model ID refers to a deleted or inaccessible model.
    - API endpoint path is incorrect.

    Args:
        message: Human-readable error description.
        model: Model string that was being called.
        llm_provider: Provider name.
        response: Raw ``httpx.Response`` from the provider, if available.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
    """

    def __init__(
        self,
        message,
        model,
        llm_provider,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 404
        self.message = "litellm.NotFoundError: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        self.response = response or httpx.Response(
            status_code=self.status_code,
            request=httpx.Request(
                method="GET", url="https://litellm.ai"
            ),  # mock request object
        )
        super().__init__(
            self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class BadRequestError(openai.BadRequestError):  # type: ignore
    """Raised when the request payload is malformed or contains invalid parameters.

    Maps to HTTP 400.  Common causes:

    - Unsupported parameters for the target provider/model.
    - Invalid message format or missing required fields.
    - Input that is too long for the model (see also
      :class:`ContextWindowExceededError`).

    Args:
        message: Human-readable error description.
        model: Model string that was being called.
        llm_provider: Provider name.
        response: Raw ``httpx.Response`` from the provider, if available.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
        body: Optional parsed response body from the provider.
    """

    def __init__(
        self,
        message,
        model,
        llm_provider,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
        body: Optional[dict] = None,
    ):
        self.status_code = 400
        self.message = "litellm.BadRequestError: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        # Use response if it's a valid httpx.Response with a request, otherwise use minimal error response
        # Note: We check _request (not .request property) to avoid RuntimeError when _request is None
        if (
            response is not None
            and isinstance(response, httpx.Response)
            and hasattr(response, "_request")
            and getattr(response, "_request", None) is not None
        ):
            self.response = response
        else:
            self.response = _get_minimal_error_response()
        super().__init__(
            self.message, response=self.response, body=body
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class ImageFetchError(BadRequestError):
    """Raised when an image URL referenced in a multimodal request cannot be fetched.

    Subclass of :class:`BadRequestError` (HTTP 400).  LiteLLM raises this
    when it attempts to download an image before forwarding it to a provider
    that does not support URL-based image inputs and the download fails.
    """

    def __init__(
        self,
        message,
        model=None,
        llm_provider=None,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
        body: Optional[dict] = None,
    ):
        super().__init__(
            message=message,
            model=model,
            llm_provider=llm_provider,
            response=response,
            litellm_debug_info=litellm_debug_info,
            max_retries=max_retries,
            num_retries=num_retries,
            body=body,
        )


class UnprocessableEntityError(openai.UnprocessableEntityError):  # type: ignore
    """Raised when the request is well-formed but semantically invalid.

    Maps to HTTP 422.  Providers use this when the input passes schema
    validation but cannot be acted upon (e.g. a valid JSON payload
    containing a logically inconsistent configuration).

    Args:
        message: Human-readable error description.
        model: Model string that was being called.
        llm_provider: Provider name.
        response: Raw ``httpx.Response`` from the provider.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
    """

    def __init__(
        self,
        message,
        model,
        llm_provider,
        response: httpx.Response,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 422
        self.message = "litellm.UnprocessableEntityError: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        super().__init__(
            self.message, response=response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class Timeout(openai.APITimeoutError):  # type: ignore
    """Raised when a provider request exceeds the configured timeout.

    Maps to HTTP 408 by default (configurable via ``exception_status_code``).
    LiteLLM's router will retry timeout errors according to the configured
    retry policy before propagating this exception to the caller.

    Args:
        message: Human-readable error description.
        model: Model string that was being called.
        llm_provider: Provider name.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
        headers: HTTP response headers from the provider, if available
            (useful for ``Retry-After`` parsing).
        exception_status_code: Override the default 408 status code.
    """

    def __init__(
        self,
        message,
        model,
        llm_provider,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
        headers: Optional[dict] = None,
        exception_status_code: Optional[int] = None,
    ):
        request = httpx.Request(
            method="POST",
            url="https://api.openai.com/v1",
        )
        super().__init__(
            request=request
        )  # Call the base class constructor with the parameters it needs
        self.status_code = exception_status_code or 408
        self.message = "litellm.Timeout: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        self.headers = headers

    # custom function to convert to str
    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class PermissionDeniedError(openai.PermissionDeniedError):  # type: ignore
    """Raised when valid credentials are supplied but lack sufficient permissions.

    Maps to HTTP 403.  Distinct from :class:`AuthenticationError` (401) in
    that the key is recognized but the account does not have access to the
    requested model or feature (e.g. GPT-4 access not yet granted).

    Args:
        message: Human-readable error description.
        llm_provider: Provider name.
        model: Model string that was being called.
        response: Raw ``httpx.Response`` from the provider.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
    """

    def __init__(
        self,
        message,
        llm_provider,
        model,
        response: httpx.Response,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 403
        self.message = "litellm.PermissionDeniedError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        super().__init__(
            self.message, response=response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class RateLimitError(openai.RateLimitError):  # type: ignore
    """Raised when the provider's rate limit has been exceeded.

    Maps to HTTP 429.  LiteLLM's router will automatically retry requests
    that hit rate limits, backing off between attempts.  The ``code``
    attribute is set to ``"429"`` and ``type`` to ``"throttling_error"``
    for compatibility with provider-specific error formats.

    Args:
        message: Human-readable error description.
        llm_provider: Provider name.
        model: Model string that was being called.
        response: Raw ``httpx.Response`` from the provider, if available.
            Response headers are preserved to allow ``Retry-After`` parsing.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
    """

    def __init__(
        self,
        message,
        llm_provider,
        model,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 429
        self.message = "litellm.RateLimitError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        _response_headers = (
            getattr(response, "headers", None) if response is not None else None
        )
        self.response = httpx.Response(
            status_code=429,
            headers=_response_headers,
            request=httpx.Request(
                method="POST",
                url=" https://cloud.google.com/vertex-ai/",
            ),
        )
        super().__init__(
            self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs
        self.code = "429"
        self.type = "throttling_error"

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


# sub class of rate limit error - meant to give more granularity for error handling context window exceeded errors
class ContextWindowExceededError(BadRequestError):  # type: ignore
    """Raised when the total token count exceeds the model's context window.

    Subclass of :class:`BadRequestError` (HTTP 400).  Catch this error
    specifically when you want to handle context-length overflows differently
    from other bad-request errors (e.g. by truncating the prompt or switching
    to a model with a larger context window).

    Args:
        message: Human-readable error description.
        model: Model string that was being called.
        llm_provider: Provider name.
        response: Raw ``httpx.Response`` from the provider, if available.
        litellm_debug_info: Optional extra context.
    """

    def __init__(
        self,
        message,
        model,
        llm_provider,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
    ):
        self.status_code = 400
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        super().__init__(
            message=message,
            model=self.model,  # type: ignore
            llm_provider=self.llm_provider,  # type: ignore
            response=response,
            litellm_debug_info=self.litellm_debug_info,
        )  # Call the base class constructor with the parameters it needs

        # set after, to make it clear the raised error is a context window exceeded error
        self.message = "litellm.ContextWindowExceededError: {}".format(self.message)

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


# sub class of bad request error - meant to help us catch guardrails-related errors on proxy.
class RejectedRequestError(BadRequestError):  # type: ignore
    """Raised when a proxy guardrail blocks a request before it reaches the LLM.

    Subclass of :class:`BadRequestError` (HTTP 400).  Proxy guardrails raise
    this error when a request fails a safety or policy check.  The original
    request payload is preserved in ``request_data`` for auditing.

    Args:
        message: Human-readable description of why the request was rejected.
        model: Model string that was being called.
        llm_provider: Provider name.
        request_data: The full request dict that was rejected.
        litellm_debug_info: Optional extra context.
    """

    def __init__(
        self,
        message,
        model,
        llm_provider,
        request_data: dict,
        litellm_debug_info: Optional[str] = None,
    ):
        self.status_code = 400
        self.message = "litellm.RejectedRequestError: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        self.request_data = request_data
        request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        response = httpx.Response(status_code=400, request=request)
        super().__init__(
            message=self.message,
            model=self.model,  # type: ignore
            llm_provider=self.llm_provider,  # type: ignore
            response=response,
            litellm_debug_info=self.litellm_debug_info,
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class ContentPolicyViolationError(BadRequestError):  # type: ignore
    """Raised when a provider's safety system rejects the request content.

    Subclass of :class:`BadRequestError` (HTTP 400).  Example: OpenAI's
    content policy blocks image generation prompts that violate safety rules.
    Retrying the same request may succeed if the violation was a false positive.

    Args:
        message: Human-readable error description from the provider.
        model: Model string that was being called.
        llm_provider: Provider name.
        response: Raw ``httpx.Response`` from the provider, if available.
        litellm_debug_info: Optional extra context.
        provider_specific_fields: Additional structured fields from the
            provider's error response (e.g. flagged categories).
        body: Optional parsed response body.
    """

    #  Error code: 400 - {'error': {'code': 'content_policy_violation', 'message': 'Your request was rejected as a result of our safety system. Image descriptions generated from your prompt may contain text that is not allowed by our safety system. If you believe this was done in error, your request may succeed if retried, or by adjusting your prompt.', 'param': None, 'type': 'invalid_request_error'}}
    def __init__(
        self,
        message,
        model,
        llm_provider,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        provider_specific_fields: Optional[dict] = None,
        body: Optional[dict] = None,
    ):
        self.status_code = 400
        self.message = "litellm.ContentPolicyViolationError: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        self.provider_specific_fields = provider_specific_fields
        super().__init__(
            message=self.message,
            model=self.model,  # type: ignore
            llm_provider=self.llm_provider,  # type: ignore
            response=response,
            litellm_debug_info=self.litellm_debug_info,
            body=body,
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        return self._transform_error_to_string()

    def __repr__(self):
        return self._transform_error_to_string()

    def _transform_error_to_string(self) -> str:
        """
        Transform the error to a string
        """
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class ServiceUnavailableError(openai.APIStatusError):  # type: ignore
    """Raised when the provider's service is temporarily unavailable.

    Maps to HTTP 503.  Indicates a transient server-side problem.  LiteLLM's
    router will retry and fall back to other providers when this error occurs.

    Args:
        message: Human-readable error description.
        llm_provider: Provider name.
        model: Model string that was being called.
        response: Raw ``httpx.Response`` from the provider, if available.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
    """

    def __init__(
        self,
        message,
        llm_provider,
        model,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 503
        self.message = "litellm.ServiceUnavailableError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        _response_headers = (
            getattr(response, "headers", None) if response is not None else None
        )
        self.response = httpx.Response(
            status_code=self.status_code,
            headers=_response_headers,
            request=httpx.Request(
                method="POST",
                url=" https://cloud.google.com/vertex-ai/",
            ),
        )
        super().__init__(
            self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class BadGatewayError(openai.APIStatusError):  # type: ignore
    """Raised when an upstream gateway returns an error response.

    Maps to HTTP 502.  Typically indicates a networking or load-balancer
    issue between LiteLLM and the provider API.  Usually transient.

    Args:
        message: Human-readable error description.
        llm_provider: Provider name.
        model: Model string that was being called.
        response: Raw ``httpx.Response`` from the provider, if available.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
    """

    def __init__(
        self,
        message,
        llm_provider,
        model,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 502
        self.message = "litellm.BadGatewayError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        _response_headers = (
            getattr(response, "headers", None) if response is not None else None
        )
        self.response = httpx.Response(
            status_code=self.status_code,
            headers=_response_headers,
            request=httpx.Request(
                method="POST",
                url=" https://cloud.google.com/vertex-ai/",
            ),
        )
        super().__init__(
            self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class InternalServerError(openai.InternalServerError):  # type: ignore
    """Raised when the provider encounters an unexpected server-side error.

    Maps to HTTP 500.  The request was valid but the provider failed to
    process it.  LiteLLM's router will retry and fall back to other
    providers when this error occurs.

    Args:
        message: Human-readable error description.
        llm_provider: Provider name.
        model: Model string that was being called.
        response: Raw ``httpx.Response`` from the provider, if available.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
    """

    def __init__(
        self,
        message,
        llm_provider,
        model,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 500
        self.message = "litellm.InternalServerError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        _response_headers = (
            getattr(response, "headers", None) if response is not None else None
        )
        self.response = httpx.Response(
            status_code=self.status_code,
            headers=_response_headers,
            request=httpx.Request(
                method="POST",
                url=" https://cloud.google.com/vertex-ai/",
            ),
        )
        super().__init__(
            self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


# raise this when the API returns an invalid response object - https://github.com/openai/openai-python/blob/1be14ee34a0f8e42d3f9aa5451aa4cb161f1781f/openai/api_requestor.py#L401
class APIError(openai.APIError):  # type: ignore
    """Catch-all for unexpected HTTP error status codes not covered by other exceptions.

    Use this to handle provider errors that do not map cleanly to a more
    specific LiteLLM exception.  The ``status_code`` attribute holds the
    actual HTTP status returned by the provider.

    Args:
        status_code: The HTTP status code returned by the provider.
        message: Human-readable error description.
        llm_provider: Provider name.
        model: Model string that was being called.
        request: The outgoing ``httpx.Request`` that triggered the error.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
    """

    def __init__(
        self,
        status_code: int,
        message,
        llm_provider,
        model,
        request: Optional[httpx.Request] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = status_code
        self.message = "litellm.APIError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        if request is None:
            request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        super().__init__(self.message, request=request, body=None)  # type: ignore

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


# raised if an invalid request (not get, delete, put, post) is made
class APIConnectionError(openai.APIConnectionError):  # type: ignore
    """Raised when a network-level error prevents the request from reaching the provider.

    Does not indicate a provider-side failure; the request may not have been
    received at all.  Common causes: DNS resolution failure, TLS handshake
    error, connection reset, proxy misconfiguration.

    Args:
        message: Human-readable error description.
        llm_provider: Provider name.
        model: Model string that was being called.
        request: The outgoing ``httpx.Request`` that failed, if available.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
    """

    def __init__(
        self,
        message,
        llm_provider,
        model,
        request: Optional[httpx.Request] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.message = "litellm.APIConnectionError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.status_code = 500
        self.litellm_debug_info = litellm_debug_info
        self.request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        self.max_retries = max_retries
        self.num_retries = num_retries
        super().__init__(message=self.message, request=self.request)

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


# raised if an invalid request (not get, delete, put, post) is made
class APIResponseValidationError(openai.APIResponseValidationError):  # type: ignore
    """Raised when the provider's response cannot be parsed into the expected schema.

    LiteLLM parses provider responses into OpenAI-compatible objects.  When
    a provider returns a response in an unexpected format this error is raised
    instead of propagating a raw ``json.JSONDecodeError`` or ``KeyError``.

    Args:
        message: Human-readable error description.
        llm_provider: Provider name.
        model: Model string that was being called.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
    """

    def __init__(
        self,
        message,
        llm_provider,
        model,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.message = "litellm.APIResponseValidationError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        response = httpx.Response(status_code=500, request=request)
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        super().__init__(response=response, body=None, message=message)

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class JSONSchemaValidationError(APIResponseValidationError):
    """Raised when a structured-output response does not conform to the requested JSON schema.

    Subclass of :class:`APIResponseValidationError`.  Use ``e.raw_response``
    to inspect the raw string returned by the model and ``e.schema`` to see
    the schema it was validated against.

    Args:
        model: Model string that was being called.
        llm_provider: Provider name.
        raw_response: The raw response string returned by the model.
        schema: The JSON schema string the response was validated against.
    """

    def __init__(
        self, model: str, llm_provider: str, raw_response: str, schema: str
    ) -> None:
        self.raw_response = raw_response
        self.schema = schema
        self.model = model
        message = "litellm.JSONSchemaValidationError: model={}, returned an invalid response={}, for schema={}.\nAccess raw response with `e.raw_response`".format(
            model, raw_response, schema
        )
        self.message = message
        super().__init__(model=model, message=message, llm_provider=llm_provider)


class OpenAIError(openai.OpenAIError):  # type: ignore
    """Pass-through wrapper for raw OpenAI SDK errors.

    Used when an ``openai.OpenAIError`` is caught and needs to be re-raised
    with LiteLLM context attached.  The ``llm_provider`` is always
    ``"openai"``.

    Args:
        original_exception: The original ``openai.OpenAIError`` instance.
    """

    def __init__(self, original_exception=None):
        super().__init__()
        self.llm_provider = "openai"


class UnsupportedParamsError(BadRequestError):
    """Raised when a request includes parameters not supported by the target provider/model.

    Subclass of :class:`BadRequestError` (HTTP 400).  LiteLLM raises this
    when it detects — either through static provider configuration or a
    provider 400 response — that a parameter such as ``response_format``,
    ``tool_choice``, or ``logprobs`` is not supported for the given model.

    Args:
        message: Human-readable description of the unsupported parameter.
        llm_provider: Provider name.
        model: Model string that was being called.
        status_code: HTTP status code (default 400).
        response: Raw ``httpx.Response`` from the provider, if available.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
    """

    def __init__(
        self,
        message,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: int = 400,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 400
        self.message = "litellm.UnsupportedParamsError: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        response = response or httpx.Response(
            status_code=self.status_code,
            request=httpx.Request(
                method="GET", url="https://litellm.ai"
            ),  # mock request object
        )
        self.max_retries = max_retries
        self.num_retries = num_retries


LITELLM_EXCEPTION_TYPES = [
    AuthenticationError,
    NotFoundError,
    BadRequestError,
    UnprocessableEntityError,
    UnsupportedParamsError,
    Timeout,
    PermissionDeniedError,
    RateLimitError,
    ContextWindowExceededError,
    RejectedRequestError,
    ContentPolicyViolationError,
    InternalServerError,
    ServiceUnavailableError,
    BadGatewayError,
    APIError,
    APIConnectionError,
    APIResponseValidationError,
    OpenAIError,
    InternalServerError,
    JSONSchemaValidationError,
]


class BudgetExceededError(Exception):
    """Raised when a user's configured spending budget has been exceeded.

    Not an OpenAI-compatible error; this is specific to LiteLLM's budget
    management system.  The HTTP status code is 429 (matching rate-limit
    semantics) so that proxy clients can apply the same retry logic.

    Args:
        current_cost: The user's accumulated spend at the time of the error.
        max_budget: The configured maximum spend limit.
        message: Optional custom message; defaults to a formatted summary
            of current vs maximum cost.
    """

    def __init__(
        self, current_cost: float, max_budget: float, message: Optional[str] = None
    ):
        self.current_cost = current_cost
        self.max_budget = max_budget
        self.status_code = 429
        message = (
            message
            or f"Budget has been exceeded! Current cost: {current_cost}, Max budget: {max_budget}"
        )
        self.message = message
        super().__init__(message)


## DEPRECATED ##
class InvalidRequestError(openai.BadRequestError):  # type: ignore
    """Deprecated alias for :class:`BadRequestError`.

    Kept for backward compatibility.  New code should raise
    :class:`BadRequestError` directly.
    """

    def __init__(self, message, model, llm_provider):
        self.status_code = 400
        self.message = message
        self.model = model
        self.llm_provider = llm_provider
        self.response = httpx.Response(
            status_code=400,
            request=httpx.Request(
                method="GET", url="https://litellm.ai"
            ),  # mock request object
        )
        super().__init__(
            message=self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs


class MockException(openai.APIError):
    """Synthetic exception used in tests to simulate provider errors.

    Inherits from ``openai.APIError`` so it is caught by the same except
    clauses as real provider errors.  Should not appear in production code.

    Args:
        status_code: HTTP status code to simulate.
        message: Error message string.
        llm_provider: Provider name to attach to the error.
        model: Model string to attach to the error.
        request: Optional outgoing request object.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
    """

    # used for testing
    def __init__(
        self,
        status_code: int,
        message,
        llm_provider,
        model,
        request: Optional[httpx.Request] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = status_code
        self.message = "litellm.MockException: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        if request is None:
            request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        super().__init__(self.message, request=request, body=None)  # type: ignore


class LiteLLMUnknownProvider(BadRequestError):
    """Raised when a model string references a provider that LiteLLM does not recognise.

    The ``model`` argument should be the full model string as passed by the
    caller (e.g. ``"unknownprovider/gpt-4"``).  The error message is
    formatted using :attr:`~litellm.types.utils.LiteLLMCommonStrings.llm_provider_not_provided`.

    Args:
        model: The model string that could not be resolved to a provider.
        custom_llm_provider: Optional explicit provider name supplied by the
            caller (passed as ``custom_llm_provider`` kwarg).
    """

    def __init__(self, model: str, custom_llm_provider: Optional[str] = None):
        self.message = LiteLLMCommonStrings.llm_provider_not_provided.value.format(
            model=model, custom_llm_provider=custom_llm_provider
        )
        super().__init__(
            self.message, model=model, llm_provider=custom_llm_provider, response=None
        )

    def __str__(self):
        return self.message


class GuardrailRaisedException(Exception):
    """Raised when a guardrail encounters an error during execution.

    Distinct from :class:`RejectedRequestError` (which indicates a
    deliberate block): this exception is used when the guardrail itself
    fails unexpectedly.  The proxy catches this and returns an appropriate
    error response to the client.

    Args:
        guardrail_name: Identifier of the guardrail that raised the error.
        message: Description of what went wrong.
        should_wrap_with_default_message: When ``True`` (default), the
            message is wrapped with a standard prefix that includes the
            guardrail name.  Set to ``False`` to use *message* verbatim.
    """

    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        message: str = "",
        should_wrap_with_default_message: bool = True,
    ):
        default_message = f"Guardrail raised an exception, Guardrail: {guardrail_name}, Message: {message}"
        self.guardrail_name = guardrail_name
        self.message = default_message if should_wrap_with_default_message else message
        super().__init__(self.message)


class BlockedPiiEntityError(Exception):
    """Raised when a guardrail detects a blocked PII entity type in the request.

    Args:
        entity_type: The PII entity type that was detected and blocked
            (e.g. ``"PERSON"``, ``"EMAIL_ADDRESS"``).
        guardrail_name: Identifier of the guardrail that detected the entity.
    """

    def __init__(
        self,
        entity_type: str,
        guardrail_name: Optional[str] = None,
    ):
        """
        Raised when a blocked entity is detected by a guardrail.
        """
        self.entity_type = entity_type
        self.guardrail_name = guardrail_name
        self.message = f"Blocked entity detected: {entity_type} by Guardrail: {guardrail_name}. This entity is not allowed to be used in this request."
        super().__init__(self.message)


class MidStreamFallbackError(ServiceUnavailableError):  # type: ignore
    """Raised when a streaming response fails mid-stream and a fallback is required.

    Subclass of :class:`ServiceUnavailableError`.  Carries the content
    that was successfully generated before the failure (``generated_content``)
    so that fallback handlers can decide whether to retry from scratch or
    continue from the partial output.

    Args:
        message: Human-readable error description.
        model: Model string that was being called.
        llm_provider: Provider name.
        original_exception: The underlying exception that caused the stream
            to fail.
        response: Raw ``httpx.Response``, if available.
        litellm_debug_info: Optional extra context.
        max_retries: Maximum retries configured.
        num_retries: Retries already attempted.
        generated_content: The partial response content received before the
            failure.
        is_pre_first_chunk: ``True`` if the failure occurred before the first
            chunk was received (i.e. the model produced no content at all).
    """

    def __init__(
        self,
        message: str,
        model: str,
        llm_provider: str,
        original_exception: Optional[Exception] = None,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
        generated_content: str = "",
        is_pre_first_chunk: bool = False,
    ):
        original_status = getattr(original_exception, "status_code", None)
        self.status_code = int(original_status) if original_status is not None else 503
        self.message = f"litellm.MidStreamFallbackError: {message}"
        self.model = model
        self.llm_provider = llm_provider
        self.original_exception = original_exception
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        self.generated_content = generated_content
        self.is_pre_first_chunk = is_pre_first_chunk

        # Create a response if one wasn't provided
        if response is None:
            self.response = httpx.Response(
                status_code=self.status_code,
                request=httpx.Request(
                    method="POST",
                    url=f"https://{llm_provider}.com/v1/",
                ),
            )
        else:
            self.response = response

        # Save the original attributes before they are overridden by ServiceUnavailableError
        _saved_response = self.response
        _saved_request = getattr(self.response, "request", None) or httpx.Request(
            method="POST", url=f"https://{llm_provider}.com/v1/"
        )
        _saved_message = self.message

        # Call the parent constructor (which hardcodes status_code=503 and modifies the response object)
        super().__init__(
            message=self.message,
            llm_provider=llm_provider,
            model=model,
            response=self.response,
            litellm_debug_info=self.litellm_debug_info,
            max_retries=self.max_retries,
            num_retries=self.num_retries,
        )

        # Restore the propagated status and original response/request objects
        self.status_code = int(original_status) if original_status is not None else 503
        self.response = _saved_response
        self.request = _saved_request
        self.message = _saved_message
        self.args = (_saved_message,)

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        if self.original_exception:
            _message += f" Original exception: {type(self.original_exception).__name__}: {str(self.original_exception)}"
        return _message

    def __repr__(self):
        return self.__str__()


class ModifyResponseException(Exception):
    """Raised by a guardrail to replace the LLM response with a synthetic response.

    This exception carries the synthetic response that should be returned
    to the user instead of calling the LLM or instead of the LLM's response.
    It should be caught by the proxy and returned with a 200 status code.

    This is a base exception that all guardrails can use to replace responses,
    allowing violation messages to be returned as successful responses
    rather than errors.

    Args:
        message: The synthetic response content to return to the user.
        model: Model string that was being called.
        request_data: The full request dict that triggered the guardrail.
        guardrail_name: Identifier of the guardrail that raised this exception.
        detection_info: Optional structured dict with detection metadata
            (e.g. flagged categories, confidence scores).
    """

    def __init__(
        self,
        message: str,
        model: str,
        request_data: Dict[str, Any],
        guardrail_name: Optional[str] = None,
        detection_info: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.model = model
        self.request_data = request_data
        self.guardrail_name = guardrail_name
        self.detection_info = detection_info or {}
        super().__init__(message)


class GuardrailInterventionNormalStringError(
    Exception
):  # custom exception to raise when a guardrail intervenes, but we want to return a normal string to the user
    """Raised when a guardrail intercepts a request and wants to return a plain string response.

    Unlike :class:`ModifyResponseException`, which carries full structured
    response data, this exception is used when the guardrail simply wants
    to return a plain string message to the user (e.g. a polite refusal).
    The proxy catches this and wraps the string in a minimal response object.

    Args:
        message: The plain string to return to the user.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.__str__()
