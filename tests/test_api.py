
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

VALID_VISITOR = {
    "Administrative": 1,
    "Administrative_Duration": 20.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 10,
    "ProductRelated_Duration": 300.0,
    "BounceRates": 0.02,
    "ExitRates": 0.04,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False,
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_valid_purchase_prediction():
    response = client.post("/predict", json=VALID_VISITOR)

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["purchase_probability"] <= 1
    assert 0 <= data["purchase_probability_percent"] <= 100
    assert data["threshold"] == 0.6
    assert isinstance(data["predicted_purchase"], bool)
    assert data["prediction"] in [
        "Purchase likely",
        "Purchase unlikely",
    ]

def test_invalid_bounce_rate_is_rejected():
    invalid_visitor = VALID_VISITOR.copy()
    invalid_visitor["BounceRates"] = 1.5

    response = client.post("/predict", json=invalid_visitor)

    assert response.status_code == 422


def test_missing_required_field_is_rejected():
    visitor = VALID_VISITOR.copy()
    del visitor["Month"]

    response = client.post("/predict", json=visitor)

    assert response.status_code == 422


def test_negative_bounce_rate_is_rejected():
    visitor = VALID_VISITOR.copy()
    visitor["BounceRates"] = -0.1

    response = client.post("/predict", json=visitor)

    assert response.status_code == 422


def test_same_input_returns_same_prediction():
    first_response = client.post("/predict", json=VALID_VISITOR)
    second_response = client.post("/predict", json=VALID_VISITOR)

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_result = first_response.json()
    second_result = second_response.json()

    assert (
        first_result["purchase_probability"]
        == second_result["purchase_probability"]
    )
    assert (
        first_result["predicted_purchase"]
        == second_result["predicted_purchase"]
    )
