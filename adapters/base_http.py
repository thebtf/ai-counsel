"""Base HTTP adapter with request/retry management."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# Configure progress logger for HTTP adapter debugging
progress_logger = logging.getLogger("ai_counsel.progress")
if not progress_logger.handlers:
    # Log to both console and dedicated progress file
    project_dir = Path(__file__).parent.parent
    progress_file = project_dir / "deliberation_progress.log"
    progress_handler = logging.FileHandler(progress_file, mode="a", encoding="utf-8")
    progress_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    progress_logger.addHandler(progress_handler)
    progress_logger.setLevel(logging.DEBUG)


def is_retryable_http_error(exception):
    """
    Determine if an HTTP error should be retried.

    Retries on:
    - 5xx server errors
    - 429 rate limit errors
    - Network errors (connection, timeout)

    Does NOT retry on:
    - 4xx client errors (bad request, auth, etc.)

    Args:
        exception: The exception to check

    Returns:
        bool: True if the error should be retried
    """
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on 5xx server errors and 429 rate limit
        return (
            exception.response.status_code >= 500
            or exception.response.status_code == 429
        )

    # Retry on network errors
    return isinstance(
        exception, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)
    )


class BaseHTTPAdapter(ABC):
    """
    Abstract base class for HTTP API adapters.

    Handles HTTP requests, timeout management, retry logic with exponential backoff,
    and error handling. Subclasses must implement build_request() and parse_response()
    for API-specific logic.

    Supports streaming mode for OpenAI-compatible APIs:
    - When use_streaming=True, sets stream=true in request
    - Reads SSE events line-by-line with activity-based timeout
    - Each event resets the timeout (heartbeat pattern)

    Example:
        class MyAdapter(BaseHTTPAdapter):
            def build_request(self, model, prompt):
                return ("/api/generate", {"Content-Type": "application/json"}, {"prompt": prompt})

            def parse_response(self, response_json):
                return response_json["text"]

        adapter = MyAdapter(base_url="http://localhost:8080", timeout=60, use_streaming=True)
        result = await adapter.invoke(prompt="Hello", model="my-model")
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 60,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        use_streaming: bool = False,
        activity_timeout: Optional[int] = None,
    ):
        """
        Initialize HTTP adapter.

        Args:
            base_url: Base URL for API (e.g., "http://localhost:11434")
            timeout: Request timeout in seconds (default: 60)
            max_retries: Maximum retry attempts for transient failures (default: 3)
            api_key: Optional API key for authentication
            headers: Optional default headers to include in all requests
            use_streaming: If True, use streaming mode with SSE events.
                Each SSE event acts as a heartbeat, resetting the activity timeout.
                Recommended for OpenAI-compatible APIs with long-running requests.
            activity_timeout: Seconds of inactivity before timeout when streaming.
                Only used when use_streaming=True. Defaults to timeout if not set.
        """
        self.base_url = base_url.rstrip("/")  # Remove trailing slash
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key
        self.default_headers = headers or {}
        self.use_streaming = use_streaming
        self.activity_timeout = activity_timeout if activity_timeout is not None else timeout

    @abstractmethod
    def build_request(
        self, model: str, prompt: str
    ) -> Tuple[str, dict[str, str], dict]:
        """
        Build API-specific request components.

        Args:
            model: Model identifier
            prompt: The prompt to send

        Returns:
            Tuple of (endpoint, headers, body):
            - endpoint: Full URL path (e.g., "/api/generate")
            - headers: Request headers dict
            - body: Request body dict (will be JSON-encoded)
        """
        pass

    @abstractmethod
    def parse_response(self, response_json: dict) -> str:
        """
        Parse API-specific response to extract model output.

        Args:
            response_json: Parsed JSON response from API

        Returns:
            Extracted model response text
        """
        pass

    async def invoke(
        self,
        prompt: str,
        model: str,
        context: Optional[str] = None,
        is_deliberation: bool = True,
        working_directory: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """
        Invoke the HTTP API with the given prompt and model.

        Args:
            prompt: The prompt to send to the model
            model: Model identifier
            context: Optional additional context to prepend to prompt
            is_deliberation: Whether this is part of a deliberation (unused for HTTP,
                           kept for API compatibility with BaseCLIAdapter)
            working_directory: Unused for HTTP adapters (kept for API compatibility)
            reasoning_effort: Unused for HTTP adapters (kept for API compatibility)

        Returns:
            Parsed response from the model

        Raises:
            TimeoutError: If request exceeds timeout
            httpx.HTTPStatusError: If API returns error status
            RuntimeError: If retries exhausted
        """
        # Build full prompt
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\n{prompt}"

        # Get request components from subclass
        endpoint, request_headers, body = self.build_request(model, full_prompt)

        # Enable streaming in request body if streaming mode is active
        if self.use_streaming:
            body["stream"] = True

        # Merge default headers with request-specific headers (request takes precedence)
        headers = {**self.default_headers, **request_headers}

        # Build full URL
        full_url = f"{self.base_url}{endpoint}"

        # Log request details for debugging
        logger = logging.getLogger(__name__)
        body_str = json.dumps(body, default=str)

        # Enhanced progress logging
        streaming_mode = "STREAMING" if self.use_streaming else "STANDARD"
        progress_logger.info(
            f"[START] HTTP REQUEST ({streaming_mode}) | Model: {model} | URL: {full_url}"
        )
        progress_logger.debug(f"   API Key present: {bool(self.api_key)}")
        progress_logger.debug(f"   Prompt length: {len(full_prompt)} chars")
        progress_logger.debug(f"   Body size: {len(body_str)} bytes")
        progress_logger.debug(f"   Headers: {list(headers.keys())}")
        progress_logger.debug(
            f"   Timeout: {self.timeout}s, Activity timeout: {self.activity_timeout}s"
        )

        start_time = datetime.now()

        # Execute request with appropriate method
        try:
            if self.use_streaming:
                # Streaming mode: read SSE events with heartbeat timeout
                result = await self._execute_streaming_request(
                    url=full_url, headers=headers, body=body, model=model
                )
                elapsed = (datetime.now() - start_time).total_seconds()
                progress_logger.info(
                    f"[SUCCESS] HTTP STREAMING | Model: {model} | Time: {elapsed:.2f}s"
                )
                return result
            else:
                # Standard mode: single request/response
                response_json = await self._execute_request_with_retry(
                    url=full_url, headers=headers, body=body
                )
                elapsed = (datetime.now() - start_time).total_seconds()
                progress_logger.info(
                    f"[SUCCESS] HTTP REQUEST | Model: {model} | Time: {elapsed:.2f}s"
                )
                progress_logger.debug(
                    f"   Response keys: {list(response_json.keys()) if isinstance(response_json, dict) else 'N/A'}"
                )
                return self.parse_response(response_json)

        except asyncio.TimeoutError:
            elapsed = (datetime.now() - start_time).total_seconds()
            progress_logger.error(
                f"[TIMEOUT] HTTP REQUEST | Model: {model} | Time: {elapsed:.2f}s"
            )
            raise TimeoutError(f"HTTP request timed out after {self.timeout}s")

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            progress_logger.error(
                f"[ERROR] HTTP REQUEST FAILED | Model: {model} | Time: {elapsed:.2f}s | Error: {type(e).__name__}: {str(e)[:200]}"
            )
            raise

    async def _execute_request_with_retry(
        self, url: str, headers: dict[str, str], body: dict
    ) -> dict:
        """
        Execute HTTP POST request with retry logic.

        Uses tenacity for exponential backoff retry on:
        - 5xx server errors
        - 429 rate limit errors
        - Network errors (connection, timeout)

        Does NOT retry on:
        - 4xx client errors (bad request, auth, etc.)

        Args:
            url: Full request URL
            headers: Request headers
            body: Request body (will be JSON-encoded)

        Returns:
            Parsed JSON response

        Raises:
            httpx.HTTPStatusError: On HTTP error (after retries exhausted for 5xx)
            httpx.NetworkError: On network error (after retries exhausted)
        """

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception(is_retryable_http_error),
            reraise=True,
        )
        async def _make_request():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                progress_logger.debug(f"   [POST] Making request to {url}")
                response = await client.post(url, headers=headers, json=body)
                progress_logger.debug(f"   [RESPONSE] Status: {response.status_code}")

                # Log error response body for 4xx errors (helps debugging)
                if 400 <= response.status_code < 500:
                    try:
                        error_body = response.json()
                        progress_logger.error(
                            f"   [HTTP_ERROR] {response.status_code}: {json.dumps(error_body, indent=2)}"
                        )
                    except Exception:
                        progress_logger.error(
                            f"   [HTTP_ERROR] {response.status_code} body: {response.text[:500]}"
                        )

                response.raise_for_status()  # Raise for 4xx/5xx
                return response.json()

        return await _make_request()

    async def _execute_streaming_request(
        self,
        url: str,
        headers: dict[str, str],
        body: dict,
        model: str,
    ) -> str:
        """
        Execute HTTP POST request with streaming response.

        Reads SSE (Server-Sent Events) line-by-line with activity-based timeout.
        Each SSE event resets the timeout (heartbeat pattern).

        OpenAI-compatible SSE format:
            data: {"choices": [{"delta": {"content": "token"}}]}
            data: [DONE]

        Args:
            url: Full request URL
            headers: Request headers
            body: Request body (should have stream=True)
            model: Model identifier (for logging)

        Returns:
            Assembled response text from all chunks

        Raises:
            TimeoutError: If no activity for activity_timeout seconds
            httpx.HTTPStatusError: On HTTP error
        """
        chunks: list[str] = []
        lines_received = 0

        progress_logger.debug(f"   [STREAM] Starting streaming request to {url}")

        # Use a longer connect timeout but activity-based read timeout
        timeout_config = httpx.Timeout(
            connect=30.0,  # Connection timeout
            read=self.activity_timeout,  # Activity timeout per chunk
            write=30.0,
            pool=30.0,
        )

        try:
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                async with client.stream("POST", url, headers=headers, json=body) as response:
                    # Check for HTTP errors
                    if response.status_code >= 400:
                        error_text = await response.aread()
                        progress_logger.error(
                            f"   [STREAM_ERROR] HTTP {response.status_code}: {error_text[:500]}"
                        )
                        response.raise_for_status()

                    progress_logger.debug(
                        f"   [STREAM] Connected, status: {response.status_code}"
                    )

                    # Read SSE events line-by-line
                    async for line in response.aiter_lines():
                        lines_received += 1

                        # Log heartbeat every 50 lines
                        if lines_received % 50 == 0:
                            progress_logger.debug(
                                f"   [STREAM] Heartbeat: {lines_received} lines, "
                                f"{len(chunks)} chunks"
                            )

                        # Parse SSE data lines
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix

                            # Check for end marker
                            if data_str.strip() == "[DONE]":
                                progress_logger.debug(
                                    f"   [STREAM] Received [DONE] marker"
                                )
                                break

                            # Parse JSON data
                            try:
                                data = json.loads(data_str)
                                text = self._extract_streaming_chunk(data)
                                if text:
                                    chunks.append(text)
                            except json.JSONDecodeError:
                                # Not valid JSON, skip
                                continue

        except httpx.ReadTimeout as err:
            progress_logger.warning(
                f"   [STREAM_TIMEOUT] No activity for {self.activity_timeout}s "
                f"(received {lines_received} lines, {len(chunks)} chunks)"
            )
            raise TimeoutError(
                f"Streaming timeout: no activity for {self.activity_timeout}s"
            ) from err

        result = "".join(chunks)
        progress_logger.debug(
            f"   [STREAM_COMPLETE] {lines_received} lines, {len(chunks)} chunks, "
            f"{len(result)} chars"
        )
        return result

    def _extract_streaming_chunk(self, data: dict) -> Optional[str]:
        """
        Extract text content from a streaming SSE event.

        Handles OpenAI-compatible formats:
        - Chat Completions: {"choices": [{"delta": {"content": "..."}}]}
        - Responses API: {"output": [{"content": "..."}]}

        Args:
            data: Parsed JSON from SSE data line

        Returns:
            Extracted text content, or None if no content found
        """
        # OpenAI Chat Completions format
        if "choices" in data:
            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                return delta.get("content")

        # OpenAI Responses API format
        if "output" in data:
            output = data.get("output", [])
            if output:
                return output[0].get("content")

        # Direct text field
        if "text" in data:
            return data.get("text")

        # Content field
        if "content" in data:
            return data.get("content")

        return None
