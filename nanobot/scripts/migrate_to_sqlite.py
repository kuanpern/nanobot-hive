import json
import sqlite3
import uuid
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def migrate_sessions(workspace: Path):
    sessions_dir = workspace / "sessions"
    db_path = sessions_dir / "checkpoints.sqlite"
    conn = sqlite3.connect(db_path)
    
    # LangGraph SQLite schema setup
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL,
            checkpoint BLOB NOT NULL,
            PRIMARY KEY (thread_id, checkpoint_ns)
        )
    """)

    for jsonl_file in sessions_dir.glob("*.jsonl"):
        session_key = jsonl_file.stem.replace("_", ":")
        messages = []
        
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data.get("_type") == "metadata": continue
                
                role = data.get("role")
                content = data.get("content")
                
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                elif role == "tool":
                    messages.append(ToolMessage(content=content, tool_call_id=data.get("tool_call_id")))

        # Serialize state as pickle (LangGraph uses pickle for blobs)
        import pickle
        state = {"messages": messages}
        blob = pickle.dumps(state)
        
        conn.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint) VALUES (?, ?, ?)",
            (session_key, "1", blob)
        )
        print(f"Migrated session {session_key}: {len(messages)} messages.")
    
    conn.commit()
    conn.close()
    print("Migration complete.")
