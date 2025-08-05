import pytest
from app import app
from unittest.mock import patch

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Medical Chatbot' in response.data

@patch('app.rag_chain')
def test_chat_route(mock_rag_chain, client):
    mock_rag_chain.invoke.return_value = {"answer": "Test response"}
    response = client.post('/get', data={'msg': 'What is diabetes?'})
    assert response.status_code == 200
    assert b'Test response' in response.data