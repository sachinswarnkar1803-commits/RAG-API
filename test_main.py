"""
Streamlined FastAPI Test suite for RAG API, Document Reader, and Multi-format Training.
"""

import io
import pytest
from fastapi.testclient import TestClient
from main import app, ml_model, NOTES_FILE

if NOTES_FILE.exists():
    ml_model.train(NOTES_FILE.read_text(encoding="utf-8"))

client = TestClient(app)


def test_home_and_login_pages():
    response = client.get("/")
    assert response.status_code == 200
    assert "FastAPI Educational Assistant" in response.text

    response = client.get("/login-page")
    assert response.status_code == 200
    assert "Sign In / Register" in response.text


def test_login_and_auth():
    resp = client.post("/login", data={"username": "admin", "password": "wrongpassword"})
    assert resp.status_code == 401

    resp = client.post("/login", data={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    token_data = resp.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    profile_resp = client.get("/profile", headers=headers)
    assert profile_resp.status_code == 200
    assert profile_resp.json()["current_user"] == "admin"


def test_user_registration_and_json_login():
    reg_data = {"username": "newteacher", "password": "teacherpass123"}
    reg_resp = client.post("/register", json=reg_data)
    assert reg_resp.status_code == 201
    assert "access_token" in reg_resp.json()

    dup_resp = client.post("/register", json=reg_data)
    assert dup_resp.status_code == 400

    login_resp = client.post("/login/json", json=reg_data)
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    profile_resp = client.get("/profile", headers=headers)
    assert profile_resp.status_code == 200
    assert profile_resp.json()["current_user"] == "newteacher"


def test_rag_exact_match_queries():
    resp = client.post("/login", data={"username": "student", "password": "student123"})
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    exact_queries = [
        "What is Natural Language Processing?",
        "What is tokenization?",
        "What is TF-IDF?",
        "What is VADER?"
    ]

    for q in exact_queries:
        res = client.post("/ask", json={"question": q}, headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["found"] is True
        assert data["score"] > 0


def test_rag_out_of_scope_query():
    resp = client.post("/login", data={"username": "student", "password": "student123"})
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    res = client.post("/ask", json={"question": "What is the current stock price of Apple?"}, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["found"] is False
    assert data["answer"] == "I could not find this topic in the provided notes."


def test_user_text_input_training():
    resp = client.post("/login", data={"username": "admin", "password": "admin123"})
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    user_text = """Deep learning utilizes neural networks with multiple layers to learn complex features from input data automatically."""
    
    train_resp = client.post("/train/text", json={"text": user_text, "title": "Neural Networks Notes"}, headers=headers)
    assert train_resp.status_code == 200
    assert train_resp.json()["temporary"] is True

    reader_resp = client.get("/model/reader", headers=headers)
    assert reader_resp.status_code == 200
    assert reader_resp.json()["title"] == "Neural Networks Notes"
    assert "neural networks with multiple layers" in reader_resp.json()["text"].lower()

    ask_resp = client.post("/ask", json={"question": "What does deep learning utilize?"}, headers=headers)
    assert ask_resp.status_code == 200
    assert ask_resp.json()["found"] is True

    if NOTES_FILE.exists():
        ml_model.train(NOTES_FILE.read_text(encoding="utf-8"))
