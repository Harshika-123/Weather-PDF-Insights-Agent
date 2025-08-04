import pytest
from app.weather import fetch_weather

def test_fetch_weather_success(monkeypatch):
    def mock_get(*args, **kwargs):
        class _resp:
            status_code = 200
            def json(self):
                return {"main": {"temp": 30}, "weather": [{"description": "clear sky"}]}
        return _resp()
    monkeypatch.setattr("requests.get", mock_get)
    result = fetch_weather("Mumbai weather", {"weather_api_key": "123"})
    assert "main" in result
