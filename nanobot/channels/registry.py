import importlib.metadata
import structlog
logger = structlog.get_logger()

def discover_all() -> dict[str, type]:
    """Discover all channels via entry_points + local websocket channel."""
    channels = {}
    
    # 1. Always load the mandatory local websocket channel
    try:
        from nanobot.channels.implementations.websocket import WebSocketChannel
        channels["websocket"] = WebSocketChannel
    except Exception as e:
        logger.error(f"Failed to load built-in websocket channel: {e}")

    # 2. Discover external plugins
    eps = importlib.metadata.entry_points(group="nanobot.channels")
    for ep in eps:
        try:
            channels[ep.name] = ep.load()
        except Exception as e:
            logger.error(f"Failed to load channel plugin {ep.name}: {e}")
            
    return channels