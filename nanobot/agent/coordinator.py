# nanobot/agent/coordinator.py
from nanobot.agent.engine.runner import AgentRunner, AgentRunSpec

class AgentCoordinator:
    def __init__(self, provider, tools, sessions, workspace, model):
        self.runner = AgentRunner(provider)
        self.tools = tools
        self.sessions = sessions
        self.workspace = workspace
        self.model = model

    async def execute_turn(self, session_key: str, message_content: str, messages: list[dict]):
        session = self.sessions.get_or_create(session_key)
        
        # Assemble spec using the provided messages (history + new)
        spec = AgentRunSpec(
            initial_messages=messages,
            tools=self.tools,
            model=self.model,
            max_iterations=15, # Pull from config
            max_tool_result_chars=16000
        )
        
        result = await self.runner.run(spec)
        
        # Handle state persistence
        session.add_message("assistant", result.final_content or "")
        self.sessions.save(session)
        return result