"""
Memory card: optional, user-held continuity across sessions.

Off by default because phones are often shared. At the end of a session Fanar
drafts a short summary with no names and no crisis details; the user edits it
and chooses whether to save. The card is encrypted with the user's passphrase
and stored only on the user's own device. The server never sees it.
"""

import base64
import os
from pathlib import Path


SUMMARY_PROMPT = """Write a SUMMARY of this conversation in at most 60 words so the user can pick up
next time. Do NOT include names, places, ages, or any mention of self-harm, suicide or crisis.
Write it in the user's language, in second person ("You talked about...")."""

DEFAULT_PATH = Path.home() / ".fadfid" / "memory_card.bin"


def available() -> bool:
    try:
        import cryptography.fernet  # noqa: F401
        return True
    except Exception:
        return False


def deriveKey(passphrase: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    kdf = Scrypt(salt=salt, length=32, n=2**15, r=8, p=1)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


def draftSummary(llm, history: list[dict]) -> str:
    return llm.generate(SUMMARY_PROMPT, history[-20:])


def saveCard(summary: str, passphrase: str, path: Path = DEFAULT_PATH):
    from cryptography.fernet import Fernet
    salt = os.urandom(16)
    token = Fernet(deriveKey(passphrase, salt)).encrypt(summary.encode("utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(salt + token)


def loadCard(passphrase: str, path: Path = DEFAULT_PATH) -> str | None:
    if not path.exists():
        return None
    from cryptography.fernet import Fernet, InvalidToken
    raw = path.read_bytes()
    try:
        return Fernet(deriveKey(passphrase, raw[:16])).decrypt(raw[16:]).decode("utf-8")
    except InvalidToken:
        return None


def deleteCard(path: Path = DEFAULT_PATH):
    if path.exists():
        path.unlink()
