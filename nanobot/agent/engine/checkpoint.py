# nanobot/agent/engine/checkpoint.py
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path

def get_checkpointer(workspace: Path):
    # Save graph state to a sqlite file 
    db_path = workspace / "sessions" / "checkpoints.sqlite"
    return SqliteSaver.from_conn_string(str(db_path))