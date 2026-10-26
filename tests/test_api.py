from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
def test_health_endpoint():

##Verify the application is running and can respond to a health check request.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_valid_iris_prediction():
    ##verify that we are getting valid prediction
    payload = {
    "sepal_length":3.4,
    "sepal_width":1.4,
    "petal_length":3.4,
    "petal_width":0.4
    }
     #send a simulated POST request with JSON input
    response = client.post("/predict", json=payload)

    #Endpoint should accept the valid input
    assert response.status_code == 200

    result = response.json()
    assert "prediction" in result
    assert "class_name" in result
    assert result["class_name"] in ["sentosa", "versicolor", "virginica"]
    assert result["prediction"] in [0, 1, 2]
