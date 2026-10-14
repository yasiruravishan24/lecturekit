"""Exceptions with a stable `code` and an exit code, so the CLI and a backend can react to them."""


class LectureKitError(Exception):
    code = "error"
    exit_code = 1


class MissingInputError(LectureKitError):
    """A required file or setting is missing (recording, slides, transcript, API key)."""
    code = "missing_input"
    exit_code = 2


class ValidationError(LectureKitError):
    """Input exists but is not valid (bad rating value, unknown config, wrong file type)."""
    code = "validation"
    exit_code = 2


class ExternalToolError(LectureKitError):
    """ffmpeg, tesseract or another external program failed or is not installed."""
    code = "external_tool"


class LLMError(LectureKitError):
    """The Claude API call failed (key, credit, rate limit, network, bad reply)."""
    code = "llm"
