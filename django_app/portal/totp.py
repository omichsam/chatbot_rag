"""Time-based one-time passwords (RFC 6238), as used by authenticator apps.

Standard library only: 6 digits, 30-second steps, HMAC-SHA1, one step of clock
drift allowed either side. A code is accepted once (the device remembers the last
step used), so a code seen over someone's shoulder can't be replayed.
"""
import base64
import hashlib
import hmac
import secrets
import struct
import time
from urllib.parse import quote

DIGITS = 6
STEP_SECONDS = 30
DRIFT_STEPS = 1
BACKUP_CODE_COUNT = 10


def new_secret():
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def _code(secret, step):
    key = base64.b32decode(secret + "=" * (-len(secret) % 8), casefold=True)
    digest = hmac.new(key, struct.pack(">Q", step), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    number = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(number % 10 ** DIGITS).zfill(DIGITS)


def current_step(now=None):
    return int((now if now is not None else time.time()) // STEP_SECONDS)


def code_at(secret, now=None):
    return _code(secret, current_step(now))


def matching_step(secret, code, after_step=0, now=None):
    """The time step the code belongs to (newer than after_step), or None."""
    code = "".join(ch for ch in str(code) if ch.isdigit())
    if len(code) != DIGITS:
        return None
    step = current_step(now)
    for candidate in range(step - DRIFT_STEPS, step + DRIFT_STEPS + 1):
        if candidate > after_step and hmac.compare_digest(_code(secret, candidate), code):
            return candidate
    return None


def provisioning_uri(secret, username, issuer="DocAssist"):
    """The otpauth:// link an authenticator app reads from a QR code."""
    label = quote(f"{issuer}:{username}")
    return f"otpauth://totp/{label}?secret={secret}&issuer={quote(issuer)}&digits={DIGITS}&period={STEP_SECONDS}"


def new_backup_codes():
    """(plain codes to show once, their hashes to store)."""
    codes = [f"{secrets.randbelow(10 ** 4):04d}-{secrets.randbelow(10 ** 4):04d}" for _ in range(BACKUP_CODE_COUNT)]
    return codes, [hash_backup_code(c) for c in codes]


def hash_backup_code(code):
    return hashlib.sha256("".join(ch for ch in code if ch.isdigit()).encode()).hexdigest()
