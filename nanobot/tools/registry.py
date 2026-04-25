"""Tool registry for dynamic tool management."""

from typing import Any

from langchain_core.tools import BaseTool

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get_definitions(self) -> list[dict[str, Any]]:
        # LangChain tools have a built-in method to export JSON schema
        return [tool.get_input_schema().model_json_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found."
        
        # BaseTool.invoke handles validation automatically
        try:
            return await tool.ainvoke(params)
        except Exception as e:
            return f"Error: {e}"