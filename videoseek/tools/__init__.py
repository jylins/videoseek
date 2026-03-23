from .overview import overview_tool, execute_overview
from .skim import skim_tool, execute_skim
from .focus import focus_tool, execute_focus
from .answer import answer_tool, execute_answer
from .registry import ToolRegistry


TOOLS = {
    "overview": overview_tool,
    "skim": skim_tool,
    "focus": focus_tool,
    "answer": answer_tool,
}


TOOL_FUNCTIONS = {
    "overview": execute_overview,
    "skim": execute_skim,
    "focus": execute_focus,
    "answer": execute_answer,
}


DEFAULT_TOOL_REGISTRY = ToolRegistry(tools=TOOLS, tool_functions=TOOL_FUNCTIONS)