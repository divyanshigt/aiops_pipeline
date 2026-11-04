from fastapi.testclient import TestClient
from flask import app

from main import create_app
import pytest

class FakeIrisModel:
    def predict(self, features):
        #Always return class [0]
        return [0]

def fake_model_loader():
        return FakeIrisModel()

def fake_version_loader():
        return "test-version"



@pytest.fixture
def client():
    app=create_app(model_loader=fake_model_loader, version_loader=fake_version_loader)
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify that the application is running aand can respond to a health check request"""

    response = client.get("/health")
    assert response.status_code == 200
    #to reduce the if else use assert statement to check the response body
    assert response.json() == {"status": "healthy"}

def test_valid_iris_prediction(client):
    """Verify that the application can return a valid prediiction"""

    #Define a valid request body
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    #Send a simulated POSt request with JSON input
    response = client.post("predict",json=payload)

    #Endpoint should accept the valid input
    assert response.status_code==200

    result=response.json()
    assert "prediction" in result
    assert "class_name" in result

    assert result["class_name"] in["setosa","versicolor","virginica"]
    assert result["prediction"] in [0,1,2]