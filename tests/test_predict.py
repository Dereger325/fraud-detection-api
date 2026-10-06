from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

FRAUD_PAYLOAD = {
        "Time": 57007.0, "V1": -1.27124419171437, "V2": 2.46267526851135, "V3": -2.85139500331783, "V4": 2.3244800653478, "V5": -1.37224488981369, "V6": -0.948195686538643, "V7": -3.06523436172054, "V8": 1.16692694787211, "V9": -2.26877058844813, "V10": -4.88114292689057, "V11": 2.25514748870463, "V12": -4.68638689759229, "V13": 0.652374668512965, "V14": -6.17428834800643, "V15": 0.594379608016446, "V16": -4.84969238709652, "V17": -6.53652073527011, "V18": -3.11909388163881, "V19": 1.71549441975915, "V20": 0.560478075726644, "V21": 0.652941051330455, "V22": 0.0819309763507574, "V23": -0.221347831198339, "V24": -0.523582159233306, "V25": 0.224228161862968, "V26": 0.756334522703558, "V27": 0.632800477330469, "V28": 0.250187092757197, "Amount": 0.01
    }

def test_predict_know_fraud_row():
    
    response = client.post("/predict",json=FRAUD_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"]=="FRAUD"
    assert data["fraud_probability"] >= 0.99

def test_missing_field_returns_422():
    payload = {"Time": 57007.0, "V1": -1.27124419171437} #missing 28 fields
    response = client.post("/predict",json=payload)
    assert response.status_code == 422

def test_wrong_type_returns_422():
    payload = FRAUD_PAYLOAD.copy()
    payload["Amount"] = "not_an_amount"
    response = client.post("/predict",json=payload)
    assert response.status_code == 422

def test_predict_columns_order_is_normalized():
    """ Documents that Pydantic reconstructs by field name, and not by json key order - so a client sending  scrambled field order (e.g. V2 before V1) still produces a correct prediction. This was a real bug when /predict took a raw dict. Pydantics schema closed it.
    """
    payload = {"Time": 57007.0,  "V2": 2.46267526851135, "V1": -1.27124419171437, "V3": -2.85139500331783, "V4": 2.3244800653478, "V5": -1.37224488981369, "V6": -0.948195686538643, "V7": -3.06523436172054, "V8": 1.16692694787211, "V9": -2.26877058844813, "V10": -4.88114292689057, "V11": 2.25514748870463, "V12": -4.68638689759229, "V13": 0.652374668512965, "V14": -6.17428834800643, "V15": 0.594379608016446, "V16": -4.84969238709652, "V17": -6.53652073527011, "V18": -3.11909388163881, "V19": 1.71549441975915, "V20": 0.560478075726644, "V21": 0.652941051330455, "V22": 0.0819309763507574, "V23": -0.221347831198339, "V24": -0.523582159233306, "V25": 0.224228161862968, "V26": 0.756334522703558, "V27": 0.632800477330469, "V28": 0.250187092757197, "Amount": 0.01}
    response = client.post("/predict",json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "FRAUD"
    assert data["fraud_probability"] > 0.99


from app import main as main_module 

class FailingModels:
    def generate_content(self,*args,**kwargs):
        raise RuntimeError("simulated gemini outage")

class FailingClient:
    models = FailingModels()

def test_explain_survives_gemini_failure(monkeypatch):
    monkeypatch.setattr(main_module, "client", FailingClient())
    response = client.post("/explain",json = FRAUD_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "FRAUD"
    assert data["explanation"] is None
    assert data["explanation_error"] is not None