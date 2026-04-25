import operator
from typing import Annotated, TypedDict, Sequence, Any, Dict
from langchain_core.messages import BaseMessage

from nanobot.identity import Identity
from nanobot.credential_manager import CredentialManager


class AgentState(TypedDict):
    # Appends new messages to the existing list
    messages: Annotated[Sequence[BaseMessage], operator.add]

    identity: Identity
    cred_manager: CredentialManager 

    # Track iterations to respect AgentDefaults.max_tool_iterations
    iteration: int
    
    # Track ongoing subagent status for the spawn tool
    subagent_status: Dict[str, Any]
    
    # The 'my' tool scratchpad: persist across turns, not restarts
    scratchpad: Dict[str, Any]
    
    # Used for error handling and flow control
    is_error: bool
    last_error: str | None