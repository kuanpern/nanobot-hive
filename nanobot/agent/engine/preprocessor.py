# nanobot/agent/engine/preprocessor.py
from nanobot.utils.document import extract_documents
from nanobot.agent.context import ContextBuilder
from nanobot.agent.events import InboundMessage
from langchain_core.messages import HumanMessage

class MessagePreProcessor:
    def __init__(self, workspace, timezone):
        self.workspace = workspace
        self.timezone = timezone

    async def process(self, msg: InboundMessage) -> HumanMessage:
        content = msg.content
        media = msg.media or []

        # 1. Document Extraction
        if media:
            content, media = extract_documents(content, media)

        # 2. Build Runtime Metadata (the [Runtime Context] tag)
        runtime_ctx = ContextBuilder._build_runtime_context(
            msg.channel, msg.chat_id, self.timezone
        )
        
        # 3. Assemble final human message
        full_text = f"{runtime_ctx}\n\n{content}"
        
        # If there are remaining images (not handled by extract_documents), 
        # append them as Multimodal LangChain messages here
        return HumanMessage(content=[{"type": "text", "text": full_text}])