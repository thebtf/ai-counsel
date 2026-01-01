"""Claude CLI adapter."""

import json
import logging

from adapters.base import BaseCLIAdapter

logger = logging.getLogger(__name__)


class ClaudeAdapter(BaseCLIAdapter):
    """Adapter for claude CLI tool with streaming support."""

    # Streaming args for reliable heartbeat detection
    STREAMING_ARGS = ["--verbose", "--output-format", "stream-json"]

    def __init__(
        self,
        command: str = "claude",
        args: list[str] | None = None,
        timeout: int = 60,
        activity_timeout: int | None = None,
        default_reasoning_effort: str | None = None,
        use_streaming: bool = True,
    ):
        """
        Initialize Claude adapter.

        Args:
            command: Command to execute (default: "claude")
            args: List of argument templates (from config.yaml)
            timeout: Timeout in seconds (default: 60)
            activity_timeout: Inactivity timeout in seconds (resets on output)
            default_reasoning_effort: Ignored (Claude doesn't support reasoning effort)
            use_streaming: If True, use streaming JSON output for reliable heartbeat.
                Recommended for reasoning models (Opus 4.5, Sonnet 4.5).
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
        Auto-detect context and adjust flags accordingly.

        For deliberations (multi-model debates), removes -p flag so Claude engages fully.
        For regular Claude Code work, adds -p flag for project context awareness.
        When streaming is enabled, adds streaming output args for reliable heartbeat.

        Args:
            is_deliberation: True if running as part of a deliberation

        Returns:
            Adjusted argument list with flags added/removed as needed
        """
        args = self.args.copy()

        if is_deliberation:
            # Remove -p flag for deliberations (we want full engagement)
            if "-p" in args:
                args.remove("-p")
        else:
            # Add -p flag for Claude Code work (project context awareness)
            if "-p" not in args:
                # Insert -p after --model argument if it exists
                if "--model" in args:
                    model_idx = args.index("--model")
                    # Insert after --model and its value
                    args.insert(model_idx + 2, "-p")
                else:
                    # Otherwise insert at the beginning
                    args.insert(0, "-p")

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

    def parse_output(self, raw_output: str) -> str:
        """
        Parse claude CLI output.

        Handles two output formats:
        1. Streaming JSON (--output-format stream-json): JSONL with result in last line
        2. Plain text: Header lines followed by model response

        Streaming JSON format:
            {"type":"system","subtype":"init",...}
            {"type":"stream_event","event":{"type":"content_block_delta",...}}
            {"type":"result","subtype":"success","result":"Full response text",...}

        Args:
            raw_output: Raw stdout from claude CLI

        Returns:
            Parsed model response
        """
        lines = raw_output.strip().split("\n")

        # Check if this is streaming JSON output (first non-empty line starts with {)
        first_content_line = next((l for l in lines if l.strip()), "")
        if first_content_line.startswith("{"):
            return self._parse_streaming_json(lines)

        # Plain text output - use original parsing logic
        return self._parse_plain_text(lines)

    def _parse_streaming_json(self, lines: list[str]) -> str:
        """
        Parse streaming JSON output from Claude CLI.

        Looks for the "result" line and extracts the response text.

        Args:
            lines: List of JSONL lines

        Returns:
            Parsed model response
        """
        result_text = ""

        for line in reversed(lines):  # Start from end - result is usually last
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "result":
                    # Found result line - extract the response
                    result_text = data.get("result", "")
                    if result_text:
                        logger.debug(
                            f"Extracted result from streaming JSON: {len(result_text)} chars"
                        )
                        return result_text
            except json.JSONDecodeError:
                # Not valid JSON, skip
                continue

        # Fallback: Try to assemble from stream_event deltas
        if not result_text:
            chunks = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") == "stream_event":
                        event = data.get("event", {})
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                chunks.append(text)
                except json.JSONDecodeError:
                    continue

            if chunks:
                result_text = "".join(chunks)
                logger.debug(
                    f"Assembled result from {len(chunks)} stream deltas: {len(result_text)} chars"
                )
                return result_text

        logger.warning("Could not extract result from streaming JSON, returning raw output")
        return "\n".join(lines)

    def _parse_plain_text(self, lines: list[str]) -> str:
        """
        Parse plain text output from Claude CLI.

        Skips header lines and extracts model response.

        Args:
            lines: List of output lines

        Returns:
            Parsed model response
        """
        # Skip header lines (typically start with "Claude Code", "Loading", etc.)
        # Find first line that looks like model output (substantial content)
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and not any(
                keyword in line.lower()
                for keyword in ["claude code", "loading", "version", "initializing"]
            ):
                start_idx = i
                break

        # Join remaining lines
        response = "\n".join(lines[start_idx:]).strip()
        return response
