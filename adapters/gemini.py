"""Gemini CLI adapter."""

import json
import logging

from adapters.base import BaseCLIAdapter

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseCLIAdapter):
    """Adapter for gemini CLI tool (Google AI) with streaming support."""

    # Gemini API limits (conservative estimates based on production errors)
    # Gemini API rejects prompts around 30k+ tokens
    # Use 100k characters as safe threshold (~25k tokens at 4 chars/token)
    # This prevents "invalid argument" API errors seen in production
    MAX_PROMPT_CHARS = 100000

    # Streaming args for reliable heartbeat detection
    STREAMING_ARGS = ["--output-format", "stream-json"]

    def __init__(
        self,
        command: str = "gemini",
        args: list[str] | None = None,
        timeout: int = 60,
        activity_timeout: int | None = None,
        default_reasoning_effort: str | None = None,
        use_streaming: bool = True,
    ):
        """
        Initialize Gemini adapter.

        Args:
            command: Command to execute (default: "gemini")
            args: List of argument templates (from config.yaml)
            timeout: Timeout in seconds (default: 60)
            activity_timeout: Inactivity timeout in seconds (resets on output)
            default_reasoning_effort: Ignored (Gemini doesn't support reasoning effort)
            use_streaming: If True, use streaming JSON output for reliable heartbeat.

        Note:
            The gemini CLI uses `gemini -p "prompt"` or `gemini -m model -p "prompt"` syntax.
        """
        if args is None:
            raise ValueError("args must be provided from config.yaml")
        super().__init__(
            command=command,
            args=args,
            timeout=timeout,
            activity_timeout=activity_timeout,
            default_reasoning_effort=default_reasoning_effort,
            use_streaming=use_streaming,
        )

    def _adjust_args_for_context(self, is_deliberation: bool) -> list[str]:
        """
        Adjust arguments based on context.

        Adds streaming args if streaming is enabled. Uses base class helper
        to insert args before -p flag (Gemini CLI requires -p to be
        immediately followed by its argument).

        Args:
            is_deliberation: True if running as part of a deliberation

        Returns:
            Adjusted argument list
        """
        args = self.args.copy()

        if self.use_streaming:
            # Insert before -p flag (not between -p and {prompt})
            # Wrong: -p --output-format stream-json {prompt}
            # Right: --output-format stream-json -p {prompt}
            args = self._insert_streaming_args(
                args,
                self.STREAMING_ARGS,
                before_flag="-p",
                fallback_placeholder="{prompt}",
            )

        return args

    def validate_prompt_length(self, prompt: str) -> bool:
        """
        Validate that prompt length is within Gemini API limits.

        Args:
            prompt: The prompt text to validate

        Returns:
            True if prompt is valid length, False if too long
        """
        return len(prompt) <= self.MAX_PROMPT_CHARS

    def parse_output(self, raw_output: str) -> str:
        """
        Parse gemini output.

        Handles two output formats:
        1. Streaming JSON (--output-format stream-json): JSONL events
        2. Plain text: Clean output without header/footer

        Args:
            raw_output: Raw stdout from gemini

        Returns:
            Parsed model response
        """
        lines = raw_output.strip().split("\n")

        # Check if this is streaming JSON output (first non-empty line starts with {)
        first_content_line = next((line for line in lines if line.strip()), "")
        if first_content_line.startswith("{"):
            return self._parse_streaming_json(lines)

        # Plain text output - just strip whitespace
        return raw_output.strip()

    def _parse_streaming_json(self, lines: list[str]) -> str:
        """
        Parse streaming JSON output from Gemini CLI.

        Gemini --output-format stream-json outputs JSONL events.
        Look for text content in various event structures.

        Args:
            lines: List of JSONL lines

        Returns:
            Parsed model response
        """
        result_text = ""
        text_chunks = []

        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line)

                # Look for text content in various fields
                # Gemini may use different structures for different event types
                text = None

                # Direct text field
                if "text" in data:
                    text = data["text"]
                # Content field
                elif "content" in data:
                    text = data["content"]
                # Result field
                elif "result" in data:
                    text = data["result"]
                # Response field
                elif "response" in data:
                    text = data["response"]
                # Nested in candidates (Gemini API style)
                elif "candidates" in data:
                    candidates = data.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            text = parts[0].get("text", "")

                if text:
                    text_chunks.append(text)

            except json.JSONDecodeError:
                # Not valid JSON, might be raw text
                if not result_text:
                    result_text = line
                continue

        # Assemble text chunks
        if text_chunks:
            result_text = "".join(text_chunks)
            logger.debug(
                f"Assembled result from {len(text_chunks)} Gemini chunks: "
                f"{len(result_text)} chars"
            )

        if result_text:
            return result_text.strip()

        # Fallback: return raw output
        logger.warning("Could not extract result from Gemini JSON, returning raw output")
        return "\n".join(lines)
