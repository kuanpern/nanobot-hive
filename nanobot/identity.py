# nanobot/identity.py
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True, slots=True)
class Identity:
    email: str
    name: Optional[str] = None

    def normalised(self) -> str:
        return self.email.strip().lower()

# nanobot/credential_policy.py
def is_allowed(identity: Identity, target: str, scope: str) -> bool:
    # A simple, extendable whitelist. In the future, this can query 
    # a database or a policy file.
    policy = {
        "web_fetch": {"browser_cookie", "tavily_api_key"},
        "exec": {"github_token"},
    }
    return target in policy.get(scope, set())