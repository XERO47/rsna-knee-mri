"""Token and session-secret hashing helpers.

Reader bootstrap tokens and browser session tokens are both high-entropy
random values (``secrets.token_urlsafe``), never low-entropy passwords, so a
plain salted SHA-256 hash is an appropriate strength/complexity trade-off; a
slow password hash (bcrypt/scrypt) is unnecessary and would only slow down
legitimate logins. Only the hash is ever persisted.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

TOKEN_BYTES = 32


def generate_token() -> str:
    """Return a fresh high-entropy URL-safe token (~256 bits)."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> str:
    """Return the hex SHA-256 digest of ``token``. Never store the raw token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)
