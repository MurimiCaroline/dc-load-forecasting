import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(built_pipeline):
    from dcforecast.api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_sites(client):
    sites = client.get("/sites").json()
    assert len(sites) == 6 and {"site_id", "mean_PUE"} <= sites[0].keys()


def test_kpis_filtered(client):
    r = client.get("/sites/SITE_001/kpis", params={"start": "2018-07-01", "end": "2018-07-07"})
    assert r.status_code == 200 and len(r.json()) == 7
    assert all(k["daily_PUE"] > 1 for k in r.json())


def test_forecast_historical_day(client):
    r = client.get("/forecast/SITE_004", params={"date": "2018-07-20"})
    body = r.json()
    assert r.status_code == 200
    assert len(body["hours"]) == 24 and body["in_sample"] is False
    assert body["evaluation"]["P_total_MW"]["MAE"] < body["evaluation"]["P_total_MW"]["baseline_lag24_MAE"] * 2
    h = body["hours"][12]["P_total_MW"]
    assert h["lower"] <= h["forecast"] <= h["upper"] and h["actual"] is not None


def test_what_if_heat_raises_pue(client):
    base = client.get("/forecast/SITE_002", params={"date": "2018-07-10"}).json()
    hot = client.get("/forecast/SITE_002", params={"date": "2018-07-10", "temp_offset_c": 8}).json()
    assert hot["summary"]["daily_PUE"] > base["summary"]["daily_PUE"]
    assert hot["evaluation"] is None


def test_unknown_site_404(client):
    assert client.get("/forecast/SITE_999", params={"date": "2018-07-20"}).status_code == 404


def test_out_of_range_date_422(client):
    assert client.get("/forecast/SITE_001", params={"date": "2018-05-02"}).status_code == 422


def test_future_day_needs_weather(client):
    r = client.post("/forecast", json={"site_id": "SITE_004", "date": "2018-08-01"})
    assert r.status_code == 422
    weather = [{"temp_C": 25, "dewpoint_C": 15, "rh_pct": 60} for _ in range(24)]
    r = client.post("/forecast", json={"site_id": "SITE_004", "date": "2018-08-01", "weather": weather})
    assert r.status_code == 200 and r.json()["evaluation"] is None


def test_predict_rows(client):
    example = client.app.openapi()["components"]["schemas"]["PredictRow"]["examples"][0]
    r = client.post("/predict", json={"rows": [example]})
    assert r.status_code == 200
    p = r.json()["predictions"][0]
    assert 20 < p["P_total_MW"] < 150 and 1.0 < p["PUE"] < 2.0
