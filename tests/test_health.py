from fastapi.testclient import TestClient

# Testeamos que existe el endpoint /health
# es decir la API ha arrancado y responde correctamente.
def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
