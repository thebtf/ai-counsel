"""Codex CLI adapter."""

import json
import logging
from typing import Optional

from adapters.base import BaseCLIAdapter

logger = logging.getLogger(__name__)


class CodexAdapter(BaseCLIAdapter):
    """Adapter for codex CLI tool with streaming support.

    Reasoning effort is configured via the {reasoning_effort} placeholder in config.yaml.
    The base class handles substitution - this adapter just validates the value.
    """

    # Valid reasoning effort levels for Codex CLI
    # See: codex exec --help for current options
    VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high"}

    # Streaming arg for reliable heartbeat detection
    STREAMING_ARGS = ["--json"]

    def __init__(
        self,
        command: str = "codex",
        args: list[str] | None = None,
        timeout: int = 60,
        activity_timeout: Optional[int] = None,
        default_reasoning_effort: Optional[str] = None,
        use_streaming: bool = True,
    ):
        """
        Initialize Codex adapter.

        Args:
            command: Command to execute (default: "codex")
            args: List of argument templates (from config.yaml with {reasoning_effort} placeholder)
            timeout: Timeout in seconds (default: 60)
            activity_timeout: Inactivity timeout in seconds (resets on output)
            default_reasoning_effort: Default reasoning effort level (none/minimal/low/medium/high).
                Used when {reasoning_effort} placeholder is in args. Can be overridden per-participant.
            use_streaming: If True, use streaming JSON output for reliable heartbeat.
        """
        if args is None:
            raise ValueError("args must be provided from config.yaml")
        if (
            default_reasoning_effort is not None
            and default_reasoning_effort not in self.VALID_REASONING_EFFORTS
        ):
            raise ValueError(
                f"Invalid default_reasoning_effort '{default_reasoning_effort}' for Codex. "
                f"Valid values: {sorted(self.VALID_REASONING_EFFORTS)}"
            )
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

        Adds streaming args if streaming is enabled.

        Args:
            is_deliberation: True if running as part of a deliberation

        Returns:
            Adjusted argument list
        """
        args = self.args.copy()

        # Add streaming args if streaming is enabled
        if self.use_streaming:
            for arg in self.STREAMING_ARGS:
                if arg not in args:
                    # Insert streaming args before the prompt placeholder
                    if "{prompt}" in args:
                        prompt_idx = args.index("{prompt}")
                        args.insert(prompt_idx, arg)
                    else:
                        args.append(arg)

        return args

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
        Invoke Codex with optional reasoning_effort.

        Args:
            prompt: The prompt to send to the model
            model: Model identifier
            context: Optional additional context
            is_deliberation: Whether this is part of a deliberation
            working_directory: Optional working directory for subprocess execution
            reasoning_effort: Optional reasoning effort level (none, minimal, low, medium, high).
                Substituted into {reasoning_effort} placeholder by base class.

        Returns:
            Parsed response from the model

        Raises:
            ValueError: If reasoning_effort is invalid
            TimeoutError: If execution exceeds timeout
            RuntimeError: If CLI process fails
        """
        # Validate reasoning_effort if provided
        if (
            reasoning_effort is not None
            and reasoning_effort not in self.VALID_REASONING_EFFORTS
        ):
            raise ValueError(
                f"Invalid reasoning_effort '{reasoning_effort}' for Codex. "
                f"Valid values: {sorted(self.VALID_REASONING_EFFORTS)}"
            )

        # Base class handles {reasoning_effort} substitution via .format()
        return await super().invoke(
            prompt=prompt,
            model=model,
            context=context,
            is_deliberation=is_deliberation,
            working_directory=working_directory,
            reasoning_effort=reasoning_effort,
        )

    def parse_output(self, raw_output: str) -> str:
        """
        Parse codex output.

        Handles two output formats:
        1. Streaming JSON (--json): JSONL events with result in final message
        2. Plain text: Clean output without header/footer

        Args:
            raw_output: Raw stdout from codex

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
        Parse streaming JSON output from Codex CLI.

        Codex --json outputs JSONL events. Common patterns:
        - {"type": "message", "content": "text"}
        - Final response in last message event

        Args:
            lines: List of JSONL lines

        Returns:
            Parsed model response
        """
        result_text = ""
        message_contents = []

        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line)

                # Look for message content
                if data.get("type") == "message":
                    content = data.get("content", "")
                    if content:
                        message_contents.append(content)

                # Look for response/result fields
                for key in ["response", "result", "text", "output", "content"]:
                    if key in data and data[key]:
                        result_text = data[key]

            except json.JSONDecodeError:
                # Not valid JSON, might be raw text - use it as fallback
                if not result_text:
                    result_text = line
                continue

        # Prefer assembled messages if we got any
        if message_contents:
            result_text = "".join(message_contents)
            logger.debug(
                f"Assembled result from {len(message_contents)} Codex messages: "
                f"{len(result_text)} chars"
            )

        if result_text:
            return result_text.strip()

        # Fallback: return raw output
        logger.warning("Could not extract result from Codex JSON, returning raw output")
        return "\n".join(lines)
