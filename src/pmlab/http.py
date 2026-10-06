"""Polite single-threaded HTTP: Cloudflare bans on concurrency, not volume."""
import json
import time

import requests

_session = requests.Session()
_last_call: dict[str, float] = {}
MIN_INTERVAL = 0.2  # seconds between calls to the same host (~5 rps)
RETRY_STATUS = (429, 500, 502, 503, 504)


def get_json(url: str, params: dict | None = None, retries: int = 5):
    """GET and parse JSON. HTTP 429 and 5xx, dropped connections and timeouts are retried, `retries` attempts in all,
    backing off 1, 2, 4 … s between them; the last attempt's error is raised (HTTPError, ConnectionError or Timeout),
    and any other HTTP error at once."""
    if retries < 1:
        raise ValueError("retries must be at least 1")
    host = url.split("/")[2]
    for attempt in range(retries):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()
        last = attempt == retries - 1
        try:
            r = _session.get(url, params=params, timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            if last:
                raise
        else:
            if r.status_code not in RETRY_STATUS or last:
                r.raise_for_status()
                # Gamma descriptions contain raw control characters, which strict JSON rejects.
                return json.loads(r.text, strict=False)
        time.sleep(2 ** attempt)
