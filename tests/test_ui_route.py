from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import pytest


@pytest.fixture
def client():
    with patch("main.build_agent", new_callable=AsyncMock), \
         patch("main.init_db"):
        from main import app
        with TestClient(app) as c:
            yield c


def test_root_returns_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
