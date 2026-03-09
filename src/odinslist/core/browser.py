"""Browser session with rotating TLS fingerprints for cover downloads."""
from __future__ import annotations

import logging
import os

from curl_cffi import requests as curl_requests

logger = logging.getLogger(__name__)


class BrowserSession:
    """Rotating curl_cffi session with browser TLS fingerprint impersonation."""
    IMPERSONATIONS = ["chrome120", "chrome124", "chrome131", "safari17_0", "edge101"]
    MAX_REQUESTS = max(1, int(os.environ.get("CURL_CFFI_ROTATE_EVERY", "10")))

    def __init__(self):
        self._session = None
        self._index = 0
        self._request_count = 0
        self._primed = False

    def get(self, url: str, **kwargs):
        """Make a GET request, auto-rotating and priming the session as needed."""
        self._maybe_rotate()
        self._maybe_prime()
        self._request_count += 1
        return self._session.get(url, **kwargs)

    def reset(self):
        """Reset the session (useful if Cloudflare starts blocking)."""
        if self._session:
            self._session.close()
        self._session = None
        self._primed = False
        self._request_count = 0
        logger.info("Image download session reset")

    def _maybe_rotate(self):
        if self._session is not None and self._request_count < self.MAX_REQUESTS:
            return
        if self._request_count >= self.MAX_REQUESTS:
            logger.info("Rotating session after %d requests", self._request_count)
        impersonate = self.IMPERSONATIONS[self._index % len(self.IMPERSONATIONS)]
        self._index += 1
        self._request_count = 0
        self._session = curl_requests.Session(impersonate=impersonate)
        logger.info("Created new curl_cffi session impersonating %s", impersonate)

    def _maybe_prime(self):
        if self._primed:
            return
        try:
            response = self._session.get("https://comicvine.gamespot.com/", timeout=15)
            if response.status_code == 200:
                logger.info("Session primed successfully")
            else:
                logger.warning("Session prime returned %d", response.status_code)
        except Exception as e:
            logger.warning("Session prime failed: %s", e)
        self._primed = True
