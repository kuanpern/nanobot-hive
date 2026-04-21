"""Default web backend using httpx and ddgs (references core/tools/web.py)."""
from nanobot_hive.core.tools.web import WebFetchTool, WebSearchTool

__all__ = ["WebFetchTool", "WebSearchTool"]
