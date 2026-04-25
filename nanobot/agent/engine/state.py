# nanobot/agent/engine/state.py
from typing import Annotated, TypedDict, Sequence
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Appends new messages to the existing list
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Keep track of custom state for your consolidation/dream logic
    task_id: str | None
    next_step: str | None