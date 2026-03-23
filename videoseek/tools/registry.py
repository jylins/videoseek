from typing import Callable


class ToolRegistry:
    def __init__(
        self,
        tools: dict[str, dict] | None = None,
        tool_functions: dict[str, Callable] | None = None,
    ) -> None:
        self._tools = dict(tools or {})
        self._tool_functions = dict(tool_functions or {})

    def register(self, name: str, tool: dict, function: Callable) -> None:
        self._tools[name] = tool
        self._tool_functions[name] = function

    def has_tool(self, name: str) -> bool:
        return name in self._tools and name in self._tool_functions

    def get_tool(self, name: str) -> dict:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def get_function(self, name: str) -> Callable:
        if name not in self._tool_functions:
            raise KeyError(f"Unknown tool function: {name}")
        return self._tool_functions[name]

    def resolve_tools(self, tool_names: list[str]) -> list[dict]:
        return [self.get_tool(name) for name in tool_names]

    @property
    def tools(self) -> dict[str, dict]:
        return dict(self._tools)

    @property
    def functions(self) -> dict[str, Callable]:
        return dict(self._tool_functions)
