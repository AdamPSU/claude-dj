"""Tests for FastAPI reaction endpoints."""
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

class TestReactionEndpoints:
    def test_get_score_default(self, client):
        resp = client.get("/reaction/score")
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert "sentiment" in data
        assert "confidence" in data

    def test_post_feedback_like(self, client):
        resp = client.post("/reaction/feedback", json={"feedback": "like"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentiment"] == "positive"

    def test_post_feedback_dislike(self, client):
        resp = client.post("/reaction/feedback", json={"feedback": "dislike"})
        assert resp.status_code == 200
        assert resp.json()["sentiment"] == "negative"

    def test_post_feedback_invalid(self, client):
        resp = client.post("/reaction/feedback", json={"feedback": "love"})
        assert resp.status_code == 400

    def test_get_summary(self, client):
        resp = client.get("/reaction/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "current_score" in data
        assert "trend_direction" in data

    def test_get_trend(self, client):
        resp = client.get("/reaction/trend")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["scores"], list)

    def test_post_track_context(self, client):
        resp = client.post("/reaction/track-context", json={"energy": 0.8, "cluster": "reggaeton"})
        assert resp.status_code == 200
        assert resp.json()["energy"] == 0.8

    def test_post_track_context_defaults(self, client):
        resp = client.post("/reaction/track-context", json={})
        assert resp.status_code == 200
        assert resp.json()["energy"] == 0.5
