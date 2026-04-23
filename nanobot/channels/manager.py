from nanobot.channels.registry import discover_all

class ChannelManager:
    def __init__(self, config, bus):
        self.config = config
        self.bus = bus
        self._available_classes = discover_all()
        self.channels = {}

    async def start_all(self):
        enabled_names = self._get_enabled_names_from_config()
        for name in enabled_names:
            if name in self._available_classes:
                cls = self._available_classes[name]
                # Initialize channel
                instance = cls(getattr(self.config.channels, name), self.bus)
                self.channels[name] = instance
                await instance.start()