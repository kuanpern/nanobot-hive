from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from nanobot.tools.base import ToolMessage
from .state import AgentState
from .runtime import _DEFAULT_ERROR_MESSAGE, build_length_recovery_message

import structlog
logger = structlog.get_logger()

# Configurable constants
MAX_ITERATIONS = 200

import json

def update_state_from_tools(state: AgentState):
    """Node that reconciles ToolMessages into the persistent AgentState."""
    new_scratchpad = state.get("scratchpad", {}).copy()
    
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "my":
            # Parse our custom signal from _arun
            if msg.content.startswith("PROTOCOL_SET_STATE:"):
                try:
                    _, key, val_json = msg.content.split(":", 2)
                    val = json.loads(val_json)
                    new_scratchpad[key] = val
                except Exception as e:
                    logger.error(f"Failed to update scratchpad from tool: {e}")
                    
    return {"scratchpad": new_scratchpad}

def _parse_my_tool_result(content: str) -> tuple[str, Any] | None:
    """
    Parses the signal: PROTOCOL_SET_STATE:{key}:{json_value}
    """
    if not content.startswith("PROTOCOL_SET_STATE:"):
        return None
    
    try:
        # Split into: PROTOCOL_SET_STATE, key, value_json
        _, key, val_json = content.split(":", 2)
        val = json.loads(val_json)
        return key, val
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse scratchpad update from tool: {e}")
        return None

def update_scratchpad_node(state: AgentState):
    """
    Looks for MyTool results in the messages and updates the state.scratchpad.
    """
    new_scratchpad = state.get("scratchpad", {}).copy()
    
    for msg in state["messages"]:
        # We look for ToolMessages named 'my'
        if isinstance(msg, ToolMessage) and msg.name == "my":
            parse_result = _parse_my_tool_result(msg.content)
            if parse_result:
                key, val = parse_result
                new_scratchpad[key] = val
    
    return {"scratchpad": new_scratchpad}

def create_agent_graph(provider, tools_registry, pending_queue):

    tools = list(tools_registry._tools.values())
    tool_node = ToolNode(tools)

    # 1. Routing Logic: The decision hub
    def router(state: AgentState):
        # A. Check iteration limit
        if state.get("iteration", 0) >= MAX_ITERATIONS:
            return END

        if state.get("is_error"):
            return "error_node" # Point to a recovery node

        # B. Check for Token Limit Hit
        last_msg = state["messages"][-1]
        if getattr(last_msg, "response_metadata", {}).get("finish_reason") == "length":
            return "recover_length"
            
        # C. Check for Tool Calls
        if last_msg.tool_calls:
            return "tools"
            
        return END

    # 2. Nodes
    def inject_node(state: AgentState):
        """Processes messages from the pending queue into the state."""
        new_messages = []
        while not pending_queue.empty():
            msg = pending_queue.get_nowait()
            new_messages.append(HumanMessage(content=msg.content))
        return {"messages": new_messages}

    async def call_llm(state: AgentState):
        """The core node for interacting with the LLM."""
        try:
            # 1. Retrieve messages and tools
            messages = state["messages"]
            tools = tools_registry.get_definitions()
            
            # 2. Extract provider settings (from provider.generation)
            settings = provider.generation
            
            # 3. Call the provider
            # Note: We use chat_with_retry to leverage your existing robust retry logic
            response = await provider.chat_with_retry(
                messages=messages,
                tools=tools,
                model=provider.get_default_model(),
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                reasoning_effort=settings.reasoning_effort
            )
            
            # 4. Handle tool use vs final text
            msg = AIMessage(
                content=response.content or "",
                additional_kwargs={"reasoning_content": response.reasoning_content}
            )
            if response.tool_calls:
                msg.tool_calls = [tc.to_openai_tool_call() for tc in response.tool_calls]
                
            # 5. Return updated state
            return {
                "messages": [msg],
                "iteration": state["iteration"] + 1,
                "is_error": False
            }

        except Exception as e:
            # Route to an error handler node or handle here
            return {
                "is_error": True,
                "last_error": str(e)
            }

    def recovery_node(state: AgentState):
        """Node for recovering from length limit errors."""
        return {
            "messages": [HumanMessage(content=build_length_recovery_message()["content"])],
            "iteration": state.get("iteration", 0) + 1 # Increment because we are triggering a new turn
        }

    # 3. Build Graph
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("injections", inject_node)
    builder.add_node("agent", call_llm)
    builder.add_node("tools", tool_node)
    builder.add_node("recovery", recovery_node)
    builder.add_node("update_state", update_state_from_tools) # New node

    # Edges
    builder.set_entry_point("injections")
    builder.add_edge("injections", "agent")
    
    builder.add_conditional_edges(
        "agent", 
        router, 
        {
            "recover_length": "recovery",
            "tools": "tools",
            END: END
        }
    )
    
    builder.add_edge("recovery", "agent")
    
    # After tools are executed, update the scratchpad state, 
    # then go back to injections to see if there are new user messages
    builder.add_edge("tools", "update_state")
    builder.add_edge("update_state", "injections") 
    
    return builder