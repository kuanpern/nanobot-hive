# nanobot/credential_backend/env.py
import os
from typing import Mapping
from .base import CredentialBackend
from ..identity import Identity
from ..credential_bundle import CredentialBundle

class EnvBackend(CredentialBackend):
    """
    Production-ready backend for environments where secrets are injected via 
    Kubernetes Secrets (as env vars) or cloud-provider sidecars.
    """
    
    async def fetch_bundle(self, identity: Identity) -> CredentialBundle:
        # We expect secrets prefixed like: NANOBOT_github_token, NANOBOT_web_fetch_cookie
        # This allows K8s to inject specific secrets per identity if needed.
        raw_data: dict[str, str] = {}
        prefix = "NANOBOT_"
        
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                # e.g., NANOBOT_github_token -> github_token
                scope = env_var[len(prefix):].lower()
                raw_data[scope] = value
                
        return CredentialBundle.from_dict(raw_data)