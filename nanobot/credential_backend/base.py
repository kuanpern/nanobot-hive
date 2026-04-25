from abc import ABC, abstractmethod
from ..identity import Identity
from ..credential_bundle import CredentialBundle

class CredentialBackend(ABC):
    @abstractmethod
    async def fetch_bundle(self, identity: Identity) -> CredentialBundle:
        """Fetch and return the bundle for this identity."""
        pass