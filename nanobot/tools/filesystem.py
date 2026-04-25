"""File system tools: read, write, edit, list."""

from __future__ import annotations
import difflib
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type
import asyncio

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import BaseTool
from nanobot.tools import file_state
from nanobot.utils.helpers import build_image_content_blocks, detect_image_mime
from nanobot.core.config.paths import get_media_dir
from nanobot.tools.base import Tool, tool_parameters
from nanobot.tools.schema import BooleanSchema, IntegerSchema, StringSchema, tool_parameters_schema


def _resolve_path(
    path: str,
    workspace: Path | None = None,
    allowed_dir: Path | None = None,
    extra_allowed_dirs: list[Path] | None = None,
) -> Path:
    """Resolve path against workspace (if relative) and enforce directory restriction."""
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace:
        p = workspace / p
    resolved = p.resolve()
    if allowed_dir:
        media_path = get_media_dir().resolve()
        all_dirs = [allowed_dir] + [media_path] + (extra_allowed_dirs or []) 
        if not any(_is_under(resolved, d) for d in all_dirs):
            raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory.resolve())
        return True
    except ValueError:
        return False


class _FsTool(Tool):
    """Shared base for filesystem tools — common init and path resolution."""

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
        extra_allowed_dirs: list[Path] | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir
        self._extra_allowed_dirs = extra_allowed_dirs

    def _resolve(self, path: str) -> Path:
        return _resolve_path(path, self._workspace, self._allowed_dir, self._extra_allowed_dirs)


# --------------------------------------------------------------------------- 
# read_file
# ---------------------------------------------------------------------------


_BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/stdout", "/dev/stderr",
    "/dev/tty", "/dev/console",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})


def _is_blocked_device(path: str | Path) -> bool:
    """Check if path is a blocked device that could hang or produce infinite output."""
    import re
    raw = str(path)

    # Resolve symlinks to check the actual target
    try:
        resolved = str(Path(raw).resolve())
    except (OSError, ValueError):
        resolved = raw

    if raw in _BLOCKED_DEVICE_PATHS or resolved in _BLOCKED_DEVICE_PATHS:
        return True
    if re.match(r"/proc/\d+/fd/[012]$", raw) or re.match(r"/proc/self/fd/[012]$", raw):
        return True
    if re.match(r"/proc/\d+/fd/[012]$", resolved) or re.match(r"/proc/self/fd/[012]$", resolved):
        return True

    # Check if resolved path starts with /dev/ (covers symlinks to devices)
    if resolved.startswith("/dev/"):
        return True
    return False


def _parse_page_range(pages: str, total: int) -> tuple[int, int]:
    """Parse a page range like '2-5' into 0-based (start, end) inclusive."""
    parts = pages.strip().split("-")
    if len(parts) == 1:
        p = int(parts[0])
        return max(0, p - 1), min(p - 1, total - 1)
    start = int(parts[0])
    end = int(parts[1])
    return max(0, start - 1), min(end - 1, total - 1)


# 1. Define the Schema with Pydantic
class ReadFileSchema(BaseModel):
    path: str = Field(description="The file path to read")
    offset: int = Field(default=1, ge=1, description="Line number to start reading from")
    limit: int = Field(default=2000, ge=1, description="Maximum number of lines to read")
    pages: str | None = Field(default=None, description="PDF page range (e.g., '1-5')")


class ReadFileTool(BaseTool):
    """Read file contents with optional line-based pagination."""

    _MAX_CHARS = 128_000
    _DEFAULT_LIMIT = 2000
    _MAX_PDF_PAGES = 20

    name: str = "read_file"
    description: str = (
        "Read a file (text or image). Text output format: LINE_NUM|CONTENT. "
        "Images return visual content for analysis. "
        "Use offset and limit for large files. "
        "Cannot read non-image binary files. "
        "Reads exceeding ~128K chars are truncated."
    )
    args_schema: type[BaseModel] = ReadFileSchema

    workspace: Path
    allowed_dir: Path | None = None
    extra_allowed_dirs: list[Path] | None = None

    def _run(self, path: str, offset: int = 1, limit: int = 2000, pages: str | None = None) -> Any:
        return asyncio.run(self._arun(path, offset, limit, pages))

    async def _arun(self, path: str, offset: int = 1, limit: int = 2000, pages: str | None = None) -> Any:
        try:
            if not path:
                return "Error reading file: Unknown path"

            # Device path blacklist
            if _is_blocked_device(path):
                return f"Error: Reading {path} is blocked (device path that could hang or produce infinite output)."
            
            fp = _resolve_path(path, self.workspace, self.allowed_dir, self.extra_allowed_dirs)
            if _is_blocked_device(fp):
                return f"Error: Reading {fp} is blocked (device path that could hang or produce infinite output)."
            if not fp.exists():
                return f"Error: File not found: {path}"
            if not fp.is_file():
                return f"Error: Not a file: {path}"

            # PDF support
            if fp.suffix.lower() == ".pdf":
                return self._read_pdf(fp, pages)

            raw = fp.read_bytes()
            if not raw:
                return f"(Empty file: {path})"

            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if mime and mime.startswith("image/"):
                return build_image_content_blocks(raw, mime, str(fp), f"(Image file: {path})")

            # Read dedup: same path + offset + limit + unchanged mtime → stub
            # Always check for external modifications before dedup
            entry = file_state._state.get(str(fp.resolve()))
            try:
                current_mtime = os.path.getmtime(fp)
            except OSError:
                current_mtime = 0.0
            if entry and entry.can_dedup and entry.offset == offset and entry.limit == limit:
                if current_mtime != entry.mtime:
                    # File was modified externally - force full read and mark as not dedupable
                    entry.can_dedup = False
                    file_state.record_read(fp, offset=offset, limit=limit)  # Update state with new mtime
                    # Continue to read full content (don't return dedup message)
                else:
                    # File unchanged - return dedup message
                    # But only if content is actually unchanged (not just mtime)
                    current_hash = file_state._hash_file(str(fp))
                    if current_hash == entry.content_hash:
                        return f"[File unchanged since last read: {path}]"
                    else:
                        # Content changed despite same mtime - force full read
                        entry.can_dedup = False
                        file_state.record_read(fp, offset=offset, limit=limit)
            else:
                # No previous state or marked as not dedupable - read full content
                file_state.record_read(fp, offset=offset, limit=limit)
                # Force full read by setting can_dedup to False for this read
                if entry:
                    entry.can_dedup = False

            # Read the file content after dedup check
            raw = fp.read_bytes()
            try:
                text_content = raw.decode("utf-8")
            except UnicodeDecodeError:
                # Binary file - return error message
                mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
                if mime and mime.startswith("image/"):
                    return build_image_content_blocks(raw, mime, str(fp), f"(Image file: {path})")
                return f"Error: Cannot read binary file {path} (MIME: {mime or 'unknown'}). Only UTF-8 text and images are supported."

            # Normalize CRLF -> LF before line-splitting. Primarily a Windows
            # concern (git checkouts with autocrlf, editors saving CRLF) but
            # applied on all platforms so downstream StrReplace/Grep behavior
            # is consistent regardless of where the file was written.
            text_content = text_content.replace("\r\n", "\n")

            all_lines = text_content.splitlines()
            total = len(all_lines)

            if offset < 1:
                offset = 1
            if offset > total:
                return f"Error: offset {offset} is beyond end of file ({total} lines)"

            start = offset - 1
            end = min(start + (limit or self._DEFAULT_LIMIT), total)
            numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(all_lines[start:end])]
            result = "\n".join(numbered)

            if len(result) > self._MAX_CHARS:
                trimmed, chars = [], 0
                for line in numbered:
                    chars += len(line) + 1
                    if chars > self._MAX_CHARS:
                        break
                    trimmed.append(line)
                end = start + len(trimmed)
                result = "\n".join(trimmed)

            if end < total:
                result += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
            else:
                result += f"\n\n(End of file — {total} lines total)"
            file_state.record_read(fp, offset=offset, limit=limit)
            return result
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {e}"

    def _read_pdf(self, fp: Path, pages: str | None) -> str:
        try:
            import fitz  # pymupdf
        except ImportError:
            return "Error: PDF reading requires pymupdf. Install with: pip install pymupdf"

        try:
            doc = fitz.open(str(fp))
        except Exception as e:
            return f"Error reading PDF: {e}"

        total_pages = len(doc)
        if pages:
            try:
                start, end = _parse_page_range(pages, total_pages)
            except (ValueError, IndexError):
                doc.close()
                return f"Error: Invalid page range '{pages}'. Use format like '1-5'."
            if start > end or start >= total_pages:
                doc.close()
                return f"Error: Page range '{pages}' is out of bounds (document has {total_pages} pages)."
        else:
            start = 0
            end = min(total_pages - 1, self._MAX_PDF_PAGES - 1)

        if end - start + 1 > self._MAX_PDF_PAGES:
            end = start + self._MAX_PDF_PAGES - 1

        parts: list[str] = []
        for i in range(start, end + 1):
            page = doc[i]
            text = page.get_text().strip()
            if text:
                parts.append(f"--- Page {i + 1} ---\n{text}")
        doc.close()

        if not parts:
            return f"(PDF has no extractable text: {fp})"

        result = "\n\n".join(parts)
        if end < total_pages - 1:
            result += f"\n\n(Showing pages {start + 1}-{end + 1} of {total_pages}. Use pages='{end + 2}-{min(end + 1 + self._MAX_PDF_PAGES, total_pages)}' to continue.)"
        if len(result) > self._MAX_CHARS:
            result = result[:self._MAX_CHARS] + "\n\n(PDF text truncated at ~128K chars)"
        return result


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

# 1. Declarative Schema
class WriteFileSchema(BaseModel):
    path: str = Field(description="The file path to write to")
    content: str = Field(description="The full content to write to the file")

# 2. BaseTool Implementation
class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = (
        "Write content to a file. Overwrites if it exists; "
        "creates parent directories as needed."
    )
    args_schema: Type[BaseModel] = WriteFileSchema
    
    workspace: Path
    allowed_dir: Path | None = None
    
    def _run(self, path: str, content: str) -> str:
        return asyncio.run(self._arun(path, content))

    async def _arun(self, path: str, content: str) -> str:
        try:
            # Re-use the existing _resolve_path security guard
            fp = _resolve_path(path, self.workspace, self.allowed_dir)
            
            # Ensure parent exists
            fp.parent.mkdir(parents=True, exist_ok=True)
            
            # Write safely
            fp.write_text(content, encoding="utf-8")
            
            # Record state (your existing tracking logic)
            from nanobot.tools import file_state
            file_state.record_write(fp)
            
            return f"Successfully wrote {len(content)} characters to {path}"
        except PermissionError as e:
            return f"Error: Permission denied: {e}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------

_QUOTE_TABLE = str.maketrans({
    "\u2018": "'", "\u2019": "'",  # curly single → straight
    "\u201c": '"', "\u201d": '"',  # curly double → straight
    "'": "'", '"': '"',            # identity (kept for completeness)
})


def _normalize_quotes(s: str) -> str:
    return s.translate(_QUOTE_TABLE)


def _curly_double_quotes(text: str) -> str:
    parts: list[str] = []
    opening = True
    for ch in text:
        if ch == '"':
            parts.append("\u201c" if opening else "\u201d")
            opening = not opening
        else:
            parts.append(ch)
    return "".join(parts)


def _curly_single_quotes(text: str) -> str:
    parts: list[str] = []
    opening = True
    for i, ch in enumerate(text):
        if ch != "'":
            parts.append(ch)
            continue
        prev_ch = text[i - 1] if i > 0 else ""
        next_ch = text[i + 1] if i + 1 < len(text) else ""
        if prev_ch.isalnum() and next_ch.isalnum():
            parts.append("\u2019")
            continue
        parts.append("\u2018" if opening else "\u2019")
        opening = not opening
    return "".join(parts)


def _preserve_quote_style(old_text: str, actual_text: str, new_text: str) -> str:
    """Preserve curly quote style when a quote-normalized fallback matched."""
    if _normalize_quotes(old_text.strip()) != _normalize_quotes(actual_text.strip()) or old_text == actual_text:
        return new_text

    styled = new_text
    if any(ch in actual_text for ch in ("\u201c", "\u201d")) and '"' in styled:
        styled = _curly_double_quotes(styled)
    if any(ch in actual_text for ch in ("\u2018", "\u2019")) and "'" in styled:
        styled = _curly_single_quotes(styled)
    return styled


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _reindent_like_match(old_text: str, actual_text: str, new_text: str) -> str:
    """Preserve the outer indentation from the actual matched block."""
    old_lines = old_text.split("\n")
    actual_lines = actual_text.split("\n")
    if len(old_lines) != len(actual_lines):
        return new_text

    comparable = [
        (old_line, actual_line)
        for old_line, actual_line in zip(old_lines, actual_lines)
        if old_line.strip() and actual_line.strip()
    ]
    if not comparable or any(
        _normalize_quotes(old_line.strip()) != _normalize_quotes(actual_line.strip())
        for old_line, actual_line in comparable
    ):
        return new_text

    old_ws = _leading_ws(comparable[0][0])
    actual_ws = _leading_ws(comparable[0][1])
    if actual_ws == old_ws:
        return new_text

    if old_ws:
        if not actual_ws.startswith(old_ws):
            return new_text
        delta = actual_ws[len(old_ws):]
    else:
        delta = actual_ws

    if not delta:
        return new_text

    return "\n".join((delta + line) if line else line for line in new_text.split("\n"))


@dataclass(slots=True)
class _MatchSpan:
    start: int
    end: int
    text: str
    line: int


def _find_exact_matches(content: str, old_text: str) -> list[_MatchSpan]:
    matches: list[_MatchSpan] = []
    start = 0
    while True:
        idx = content.find(old_text, start)
        if idx == -1:
            break
        matches.append(
            _MatchSpan(
                start=idx,
                end=idx + len(old_text),
                text=content[idx : idx + len(old_text)],
                line=content.count("\n", 0, idx) + 1,
            )
        )
        start = idx + max(1, len(old_text))
    return matches


def _find_trim_matches(content: str, old_text: str, *, normalize_quotes: bool = False) -> list[_MatchSpan]:
    old_lines = old_text.splitlines()
    if not old_lines:
        return []

    content_lines = content.splitlines()
    content_lines_keepends = content.splitlines(keepends=True)
    if len(content_lines) < len(old_lines):
        return []

    offsets: list[int] = []
    pos = 0
    for line in content_lines_keepends:
        offsets.append(pos)
        pos += len(line)
    offsets.append(pos)

    if normalize_quotes:
        stripped_old = [_normalize_quotes(line.strip()) for line in old_lines]
    else:
        stripped_old = [line.strip() for line in old_lines]

    matches: list[_MatchSpan] = []
    window_size = len(stripped_old)
    for i in range(len(content_lines) - window_size + 1):
        window = content_lines[i : i + window_size]
        if normalize_quotes:
            comparable = [_normalize_quotes(line.strip()) for line in window]
        else:
            comparable = [line.strip() for line in window]
        if comparable != stripped_old:
            continue

        start = offsets[i]
        end = offsets[i + window_size]
        if content_lines_keepends[i + window_size - 1].endswith("\n"):
            end -= 1
        matches.append(
            _MatchSpan(
                start=start,
                end=end,
                text=content[start:end],
                line=i + 1,
            )
        )
    return matches


def _find_quote_matches(content: str, old_text: str) -> list[_MatchSpan]:
    norm_content = _normalize_quotes(content)
    norm_old = _normalize_quotes(old_text)
    matches: list[_MatchSpan] = []
    start = 0
    while True:
        idx = norm_content.find(norm_old, start)
        if idx == -1:
            break
        matches.append(
            _MatchSpan(
                start=idx,
                end=idx + len(old_text),
                text=content[idx : idx + len(old_text)],
                line=content.count("\n", 0, idx) + 1,
            )
        )
        start = idx + max(1, len(norm_old))
    return matches


def _find_matches(content: str, old_text: str) -> list[_MatchSpan]:
    """Locate all matches using progressively looser strategies."""
    for matcher in (
        lambda: _find_exact_matches(content, old_text),
        lambda: _find_trim_matches(content, old_text),
        lambda: _find_trim_matches(content, old_text, normalize_quotes=True),
        lambda: _find_quote_matches(content, old_text),
    ):
        matches = matcher()
        if matches:
            return matches
    return []


def _find_match_line_numbers(content: str, old_text: str) -> list[int]:
    """Return 1-based starting line numbers for the current matching strategies."""
    return [match.line for match in _find_matches(content, old_text)]


def _collapse_internal_whitespace(text: str) -> str:
    return "\n".join(" ".join(line.split()) for line in text.splitlines())


def _diagnose_near_match(old_text: str, actual_text: str) -> list[str]:
    """Return actionable hints describing why text was close but not exact."""
    hints: list[str] = []

    if old_text.lower() == actual_text.lower() and old_text != actual_text:
        hints.append("letter case differs")
    if _collapse_internal_whitespace(old_text) == _collapse_internal_whitespace(actual_text) and old_text != actual_text:
        hints.append("whitespace differs")
    if old_text.rstrip("\n") == actual_text.rstrip("\n") and old_text != actual_text:
        hints.append("trailing newline differs")
    if _normalize_quotes(old_text) == _normalize_quotes(actual_text) and old_text != actual_text:
        hints.append("quote style differs")

    return hints


def _best_window(old_text: str, content: str) -> tuple[float, int, list[str], list[str]]:
    """Find the closest line-window match and return ratio/start/snippet/hints."""
    lines = content.splitlines(keepends=True)
    old_lines = old_text.splitlines(keepends=True)
    window = max(1, len(old_lines))

    best_ratio, best_start = -1.0, 0
    best_window_lines: list[str] = []

    for i in range(max(1, len(lines) - window + 1)):
        current = lines[i : i + window]
        ratio = difflib.SequenceMatcher(None, old_lines, current).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, i
            best_window_lines = current

    actual_text = "".join(best_window_lines).replace("\r\n", "\n").rstrip("\n")
    hints = _diagnose_near_match(old_text.replace("\r\n", "\n").rstrip("\n"), actual_text)
    return best_ratio, best_start, best_window_lines, hints


def _find_match(content: str, old_text: str) -> tuple[str | None, int]:
    """Locate old_text in content with a multi-level fallback chain:

    1. Exact substring match
    2. Line-trimmed sliding window (handles indentation differences)
    3. Smart quote normalization (curly ↔ straight quotes)

    Both inputs should use LF line endings (caller normalises CRLF).
    Returns (matched_fragment, count) or (None, 0).
    """
    matches = _find_matches(content, old_text)
    if not matches:
        return None, 0
    return matches[0].text, len(matches)


class EditFileSchema(BaseModel):
    path: str = Field(description="The file path to edit")
    old_text: str = Field(description="The text to find and replace")
    new_text: str = Field(description="The text to replace with")
    replace_all: bool = Field(default=False, description="Replace all occurrences")

class EditFileTool(BaseTool):
    name: str = "edit_file"
    description: str = (
        "Edit a file by replacing old_text with new_text. "
        "Tolerates minor whitespace/indentation differences. "
        "If old_text matches multiple times, provide more context or set replace_all=true."
    )
    args_schema: type[BaseModel] = EditFileSchema

    workspace: Path
    allowed_dir: Path | None = None

    def _run(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        # We delegate to _arun to maintain the async flow of the rest of the bot
        return asyncio.run(self._arun(path, old_text, new_text, replace_all))

    async def _arun(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        try:
            # 1. Resolve path (Existing security guard)
            fp = _resolve_path(path, self.workspace, self.allowed_dir)
            if not fp.exists():
                return f"Error: File not found: {path}"

            # 2. Existing core logic (Read + Match + Replace)
            content = fp.read_text(encoding="utf-8").replace("\r\n", "\n")
            norm_old = old_text.replace("\r\n", "\n")
            
            # Using your existing _find_matches utility
            matches = _find_matches(content, norm_old)
            if not matches:
                return self._not_found_msg(old_text, content, path)
            
            # 3. Handle duplicates/replace_all
            if len(matches) > 1 and not replace_all:
                return "Warning: old_text appears multiple times. Provide more context or set replace_all=true."

            # 4. Perform Edit (Your existing string slicing logic)
            new_content = content
            selected = matches if replace_all else matches[:1]
            for match in reversed(selected):
                # Apply your existing quote/indentation preservation helpers
                replacement = _preserve_quote_style(norm_old, match.text, new_text)
                replacement = _reindent_like_match(norm_old, match.text, replacement)
                new_content = new_content[:match.start] + replacement + new_content[match.end():]
            
            fp.write_text(new_content, encoding="utf-8")
            from nanobot.tools import file_state
            file_state.record_write(fp)
            return f"Successfully edited {path}"

        except PermissionError as e:
            return f"Error: Permission denied: {e}"
        except Exception as e:
            return f"Error editing file: {str(e)}"

# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------

@tool_parameters(
    tool_parameters_schema(
        path=StringSchema("The directory path to list"),
        recursive=BooleanSchema(description="Recursively list all files (default false)"),
        max_entries=IntegerSchema(
            200,
            description="Maximum entries to return (default 200)",
            minimum=1,
        ),
        required=["path"],
    )
)
class ListDirTool(_FsTool):
    """List directory contents with optional recursion."""

    _DEFAULT_MAX = 200
    _IGNORE_DIRS = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", ".coverage", "htmlcov",
    }

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return (
            "List the contents of a directory. "
            "Set recursive=true to explore nested structure. "
            "Common noise directories (.git, node_modules, __pycache__, etc.) are auto-ignored."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self, path: str | None = None, recursive: bool = False,
        max_entries: int | None = None, **kwargs: Any,
    ) -> str:
        try:
            if path is None:
                raise ValueError("Unknown path")
            dp = self._resolve(path)
            if not dp.exists():
                return f"Error: Directory not found: {path}"
            if not dp.is_dir():
                return f"Error: Not a directory: {path}"

            cap = max_entries or self._DEFAULT_MAX
            items: list[str] = []
            total = 0

            if recursive:
                for item in sorted(dp.rglob("*")):
                    if any(p in self._IGNORE_DIRS for p in item.parts):
                        continue
                    total += 1
                    if len(items) < cap:
                        rel = item.relative_to(dp)
                        items.append(f"{rel}/" if item.is_dir() else str(rel))
            else:
                for item in sorted(dp.iterdir()):
                    if item.name in self._IGNORE_DIRS:
                        continue
                    total += 1
                    if len(items) < cap:
                        pfx = "📁 " if item.is_dir() else "📄 "
                        items.append(f"{pfx}{item.name}")

            if not items and total == 0:
                return f"Directory {path} is empty"

            result = "\n".join(items)
            if total > cap:
                result += f"\n\n(truncated, showing first {cap} of {total} entries)"
            return result
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {e}"
