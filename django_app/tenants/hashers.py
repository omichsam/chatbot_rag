import hashlib

from django.contrib.auth.hashers import BasePasswordHasher, mask_hash
from django.utils.crypto import constant_time_compare


class LegacySHA256PasswordHasher(BasePasswordHasher):
    """Verifies unsalted SHA-256 hashes from the Streamlit app's users.json.

    Stored as "legacy_sha256$$<hex>". It is listed last in PASSWORD_HASHERS,
    so Django re-hashes the password with PBKDF2 on the next successful login.
    Never use it to create new passwords.
    """

    algorithm = "legacy_sha256"

    def salt(self):
        return ""

    def encode(self, password, salt):
        return f"{self.algorithm}$${hashlib.sha256(password.encode()).hexdigest()}"

    def decode(self, encoded):
        algorithm, _empty, hash_ = encoded.split("$", 2)
        return {"algorithm": algorithm, "hash": hash_, "salt": ""}

    def verify(self, password, encoded):
        return constant_time_compare(encoded, self.encode(password, ""))

    def safe_summary(self, encoded):
        return {"algorithm": self.algorithm, "hash": mask_hash(self.decode(encoded)["hash"])}

    def harden_runtime(self, password, encoded):
        pass

    def must_update(self, encoded):
        return True
