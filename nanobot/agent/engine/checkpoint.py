from typing import TypedDict
import sqlite3
from pathlib import Path
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing import Annotated

def get_checkpointer(workspace: Path):
    db_path = workspace / "sessions" / "checkpoints.sqlite"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)    
    checkpointer = AsyncSqliteSaver(conn)
    
    return checkpointer
