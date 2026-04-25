from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any, Dict, Optional
import structlog
from .identity import Identity
from .credential_backend.base import CredentialBackend

logger = structlog.get_logger()

class CredentialManager:
    def __init__(self, identity: Identity, backend: CredentialBackend):
        self.identity = identity
        self.backend = backend
        self._bundle: Optional[Any] = None

    async def _ensure_bundle(self):
        if self._bundle is None:
            self._bundle = await self.backend.fetch_bundle(self.identity)

    @asynccontextmanager
    async def authorize(self, scope: str, target: str) -> AsyncGenerator[Any, None]:
        # 1. Policy Check
        from .credential_policy import is_allowed
        if not is_allowed(self.identity, target, scope):
            raise PermissionError(f"Unauthorized: {scope} for {target}")

        # 2. Lazy load the bundle
        await self._ensure_bundle()
        secret = self._bundle.get(scope)
        
        try:
            yield secret
        finally:
            # Wipe secret from the local variable scope
            if isinstance(secret, (str, bytes)):
                secret = "*" * len(secret)
            logger.debug("Credential access cycle completed", target=target, scope=scope)