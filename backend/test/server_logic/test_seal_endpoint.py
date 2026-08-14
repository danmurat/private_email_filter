from fastapi.testclient import TestClient

from src.main import app


def test_raw_seal_endpoint_rejects_wrong_protocol_before_parsing_files():
    client = TestClient(app)
    response = client.post(
        "/spamfilter/seal",
        data={"protocol_version": "999", "vector_length": "150"},
        files={
            "parameters_file": ("parameters.bin", b"not-seal"),
            "galois_keys_file": ("galois.bin", b"not-seal"),
            "enc_email_file": ("cipher.bin", b"not-seal"),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported SEAL protocol version"


def test_raw_seal_endpoint_rejects_wrong_vector_length_before_parsing_files():
    client = TestClient(app)
    response = client.post(
        "/spamfilter/seal",
        data={"protocol_version": "1", "vector_length": "149"},
        files={
            "parameters_file": ("parameters.bin", b"not-seal"),
            "galois_keys_file": ("galois.bin", b"not-seal"),
            "enc_email_file": ("cipher.bin", b"not-seal"),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid encrypted vector length"
