"""Test fixtures. The full eval suite hits a live container at SERVICE_URL.

Default `SERVICE_URL=http://localhost:8080` matches `docker compose up`.
"""

import os

import httpx
import pytest


@pytest.fixture(scope="session")
def service_url() -> str:
    return os.environ.get("SERVICE_URL", "http://localhost:8080")


@pytest.fixture(scope="session")
def client(service_url: str) -> httpx.Client:
    with httpx.Client(base_url=service_url, timeout=90.0) as c:
        yield c
