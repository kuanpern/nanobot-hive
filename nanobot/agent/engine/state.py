from typing import Annotated, TypedDict, Sequence, Any, Dict
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Appends new messages to the existing list
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Track iterations to respect AgentDefaults.max_tool_iterations
    iteration: int
    
    # Track ongoing subagent status for the spawn tool
    subagent_status: Dict[str, Any]
    
    # The 'my' tool scratchpad: persist across turns, not restarts
    scratchpad: Dict[str, Any]
    
    # Used for error handling and flow control
    is_error: bool
    last_error: str | None