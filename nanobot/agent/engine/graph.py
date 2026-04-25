# nanobot/agent/engine/graph.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from .state import AgentState
from .runtime import _DEFAULT_ERROR_MESSAGE

import structlog
logger = structlog.get_logger()

def check_for_length_limit(state: AgentState):
    """
    Checks if the last message indicated a token limit hit.
    If so, returns 'recover' to trigger your length_recovery_message logic.
    """
    last_msg = state["messages"][-1]
    # Check for OpenAI/LangChain finish reasons
    reason = getattr(last_msg, "response_metadata", {}).get("finish_reason")
    if reason == "length":
        return "recover"
    return "continue"

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def create_agent_graph(provider, tools_registry, pending_queue):
    
    # 1. Routing Logic (The "Brain")
    def router(state: AgentState):
        last_msg = state["messages"][-1]
        
        # Check for Token Limit Hit
        if getattr(last_msg, "response_metadata", {}).get("finish_reason") == "length":
            return "recover_length"
            
        # Check for Tool Calls
        if last_msg.tool_calls:
            return "tools"
            
        return END

    # 2. Nodes
    def inject_node(state: AgentState):
        new_messages = []
        while not pending_queue.empty():
            msg = pending_queue.get_nowait()
            new_messages.append(HumanMessage(content=msg.content))
        return {"messages": new_messages}

    def call_llm(state: AgentState):
        try:
            response = provider.chat_with_retry(
                messages=state["messages"], 
                tools=tools_registry.get_definitions()
            )
            return {"messages": [response]}
        except Exception as e:
            # Using your runtime constant
            return {"messages": [AIMessage(content=_DEFAULT_ERROR_MESSAGE)]}

    def recovery_node(state: AgentState):
        # Using your runtime constant
        return {"messages": [HumanMessage(content=build_length_recovery_message()["content"])]}

    # 3. Build Graph
    builder = StateGraph(AgentState)
    builder.add_node("injections", inject_node)
    builder.add_node("agent", call_llm)
    builder.add_node("tools", ToolNode(tools_registry))
    builder.add_node("recovery", recovery_node)

    # 4. Define Edges
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
    builder.add_edge("tools", "injections") # Back to start to pick up new messages
    
    return builder.compile()