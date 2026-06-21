import base64
import json
import os
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class AppCrypto:
    """AES-256-GCM symmetric encryption helper for JSON payloads.

    Wraps Python's ``cryptography`` library with a simple interface for
    encrypting and decrypting arbitrary JSON-serializable dicts.  Each
    encrypt call generates a fresh random 12-byte nonce, so the same
    plaintext produces a different ciphertext on every call.

    Args:
        master_key: 32-byte raw key material for AES-256.  Must be kept
            secret and rotated out-of-band.  Use ``secrets.token_bytes(32)``
            or an equivalent CSPRNG to generate it.

    Raises:
        ValueError: If ``master_key`` is not exactly 32 bytes.

    Example::

        key = secrets.token_bytes(32)
        crypto = AppCrypto(key)
        enc = crypto.encrypt_json({"user_id": 42})
        dec = crypto.decrypt_json(enc)
        assert dec == {"user_id": 42}
    """

    def __init__(self, master_key: bytes) -> None:
        if len(master_key) != 32:
            raise ValueError("Master key must be 32 bytes for AES-256-GCM")
        self.key = master_key

    def encrypt_json(
        self, data: dict, aad: Optional[bytes] = None
    ) -> dict:
        """Encrypt a JSON-serializable dict with AES-256-GCM.

        Serializes ``data`` to UTF-8 JSON, then encrypts with a freshly
        generated random nonce.  The returned dict contains three
        base64-encoded fields that are required for decryption.

        Args:
            data: The dict to encrypt.  Must be JSON-serializable.
            aad: Optional Additional Authenticated Data (AAD).  This is
                mixed into the authentication tag but is **not** encrypted.
                The same ``aad`` bytes must be supplied to
                :meth:`decrypt_json`.  Pass ``None`` to omit AAD.

        Returns:
            A dict with three keys:

            - ``"nonce"`` (str): base64-encoded 12-byte random nonce.
            - ``"ciphertext"`` (str): base64-encoded encrypted payload.
            - ``"tag"`` (str): base64-encoded 16-byte GCM authentication tag.
        """
        aes = AESGCM(self.key)
        nonce = os.urandom(12)
        plaintext = json.dumps(data).encode("utf-8")
        ct = aes.encrypt(nonce, plaintext, aad)
        ciphertext, tag = ct[:-16], ct[-16:]
        return {
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "tag": base64.b64encode(tag).decode(),
        }

    def decrypt_json(
        self, enc: dict, aad: Optional[bytes] = None
    ) -> Any:
        """Decrypt a ciphertext dict produced by :meth:`encrypt_json`.

        Args:
            enc: The dict returned by :meth:`encrypt_json`, containing
                ``"nonce"``, ``"ciphertext"``, and ``"tag"`` keys.
            aad: The same Additional Authenticated Data that was passed to
                :meth:`encrypt_json`.  If the value does not match, decryption
                raises ``cryptography.exceptions.InvalidTag``.

        Returns:
            The original dict that was passed to :meth:`encrypt_json`.

        Raises:
            cryptography.exceptions.InvalidTag: If authentication fails,
                indicating the ciphertext or AAD has been tampered with.
            KeyError: If ``enc`` is missing a required key.
            json.JSONDecodeError: If the decrypted bytes are not valid JSON
                (should not happen with well-formed ciphertext).
        """
        aes = AESGCM(self.key)
        nonce = base64.b64decode(enc["nonce"])
        ct = base64.b64decode(enc["ciphertext"])
        tag = base64.b64decode(enc["tag"])
        data = aes.decrypt(nonce, ct + tag, aad)
        return json.loads(data.decode())
