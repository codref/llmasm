"""File reader tool for inspecting local files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llmasm.schemas import RawText
from llmasm.tools.base import ToolSpec

_MAX_BYTES = 1_048_576  # 1 MB


class FileReaderTool:
    """Read the contents of a local file."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file.read",
            description="Read the contents of a file from the filesystem. Input is the file path.",
            input_schema="RawText",
            output_schema="RawText",
        )

    def invoke(self, input: BaseModel, provider: Any = None) -> BaseModel:
        path_str = getattr(input, "text", str(input)).strip()
        if not path_str:
            return RawText(text="No file path provided.")
        path = Path(path_str)
        if not path.exists():
            return RawText(text=f"File not found: {path_str}")
        if path.is_dir():
            return RawText(text=f"Path is a directory: {path_str}")
        try:
            if path.stat().st_size > _MAX_BYTES:
                return RawText(text=f"File exceeds {_MAX_BYTES:,} byte limit.")
            content = path.read_text(encoding="utf-8", errors="replace")
            return RawText(text=content)
        except PermissionError:
            return RawText(text=f"Permission denied: {path_str}")
        except OSError as exc:
            return RawText(text=f"File read error: {exc}")
