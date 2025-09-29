import datetime as dt
from typing import Any, Dict, List

import pytest

from atmo_model.core import AtmosphericState, DeterministicAtmosphericModel
from atmo_model.llm_postprocessing import LLMPostProcessor
from atmo_model.pipeline import predict_three_day_soil_moisture


class DummyLLMClient:
    def __init__(self, response: Any):
        self.received_prompts: List[str] = []
        self.response = response

    def __call__(self, prompt: str) -> Any:
        self.received_prompts.append(prompt)
        return self.response


def _sample_state() -> AtmosphericState:
    return AtmosphericState(
        timestamp=dt.datetime(2024, 1, 1, 0, 0, 0),
        temperature_c=18.0,
        relative_humidity=60.0,
        precipitation_mm=0.5,
        soil_moisture=0.4,
    )


def _sample_weather(hours: int = 72) -> List[Dict[str, float]]:
    return [
        {
            "timestamp": (dt.datetime(2024, 1, 1) + dt.timedelta(hours=hour)).isoformat(),
            "temperature_c": 18.0 + hour * 0.05,
            "relative_humidity": 55.0 + hour * 0.1,
            "precipitation_mm": 0.2 if hour % 6 == 0 else 0.0,
        }
        for hour in range(hours)
    ]


def test_prompt_includes_features_and_states():
    state = _sample_state()
    weather = _sample_weather(3)
    processor = LLMPostProcessor(client=DummyLLMClient({"soil_moisture": [0.4] * 72}))
    model = DeterministicAtmosphericModel()

    projected_states = model.predict(state, weather, 3)
    features = {
        "mean_temperature": 19.0,
        "mean_humidity": 62.0,
        "total_precipitation": 0.4,
        "final_soil_moisture": 0.45,
    }

    prompt = processor.build_prompt(state, projected_states, features)

    assert "mean_temperature" in prompt
    assert "Projected states" in prompt
    assert "SM=" in prompt


def test_parse_response_handles_json():
    processor = LLMPostProcessor()
    response = {"soil_moisture": [0.5] * 100}
    parsed = processor.parse_response(response)
    assert parsed == [0.5] * processor.horizon_hours


def test_pipeline_with_mocked_llm_returns_forecast():
    state = _sample_state()
    weather = _sample_weather(72)
    fake_values = [0.4 + i * 0.001 for i in range(72)]
    processor = LLMPostProcessor(client=DummyLLMClient({"soil_moisture": fake_values}))

    forecast = predict_three_day_soil_moisture(state, weather, post_processor=processor)

    assert len(forecast.values) == 72
    assert forecast.values == fake_values
    assert "features" in forecast.metadata
    assert "prompt" in forecast.metadata
