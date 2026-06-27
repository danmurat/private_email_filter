import time
import urllib.request
import urllib.error
import pytest

LOCAL_BACKEND_URL = "http://0.0.0.0:80"
# just to test aws server. Not to be used for CI/CD tests!
AWS_BACKEND_URL = "http://35.178.170.58"

def test_rate_limiting_e2e():
    # Construct the endpoint URL
    url = f"{LOCAL_BACKEND_URL}/spamfilter/ts"
    status_codes = []
    
    # Send 5 rapid POST requests over the network (more than 1 should return 429 rate limit err)
    for _ in range(5):
        req = urllib.request.Request(url, method="POST")
        try:
            with urllib.request.urlopen(req) as response:
                status_codes.append(response.getcode())
        except urllib.error.HTTPError as e:
            status_codes.append(e.code)
        except urllib.error.URLError as e:
            pytest.fail(f"E2E test failed to connect to {url}. Is the server/proxy running? Error: {e.reason}")
            
    # Verify that the server (e.g., via Nginx reverse proxy) returned a 429 status code
    assert 429 in status_codes


def _post_status(url, data):
    """POST raw bytes and return the HTTP status code.

    Returns None if the connection is reset/refused, which nginx may legitimately
    do when it aborts an oversized upload mid-stream (see the >20MB case below).
    """
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            return response.getcode()
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, ConnectionResetError, BrokenPipeError):
        return None


def test_body_size_limit_e2e():
    # nginx is configured with `client_max_body_size 20M`. Verify a small upload
    # is let through and an oversized one is rejected.
    url = f"{LOCAL_BACKEND_URL}/spamfilter/ts"

    # The previous test exhausts the 1r/s rate limit. Wait for the bucket to
    # refill so we measure the body-size behaviour (413) and not the limit (429).
    time.sleep(2)

    # --- Small upload (< 20MB): nginx must NOT reject it for size ---
    # It reaches FastAPI, which returns a 4xx for the dummy (non-multipart)
    # payload (e.g. 422). The point is only that it is NOT a 413.
    small_body = b"x" * (1 * 1024 * 1024)  # 1MB
    small_status = _post_status(url, small_body)
    assert small_status is not None, "Small upload failed to connect — is the server/proxy running?"
    assert small_status != 413, f"Small upload was wrongly rejected as too large: {small_status}"

    time.sleep(2)  # refill the rate-limit bucket again

    # --- Large upload (> 20MB): nginx must reject it ---
    # Normally this is a 413. For a body well over the limit nginx may instead
    # reset the connection mid-upload (returned as None here); either outcome
    # means the upload was refused, which is what we want.
    large_body = b"x" * (21 * 1024 * 1024)  # 21MB
    large_status = _post_status(url, large_body)
    assert large_status in (413, None), (
        f"Expected oversized upload to be rejected (413 or connection reset), got {large_status}"
    )
