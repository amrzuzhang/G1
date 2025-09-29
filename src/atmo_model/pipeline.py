"""High level orchestration for soil moisture predictions."""

from __future__ import annotations

import datetime as dt
from typing import Iterable, Optional

from .core import (
    AtmosphericState,
    DeterministicAtmosphericModel,
    SoilMoistureForecast,
    export_soil_moisture_features,
    load_numerical_weather_outputs,
)
from .llm_postprocessing import LLMPostProcessor


def predict_three_day_soil_moisture(
    initial_state: AtmosphericState,
    weather_data: Iterable[dict],
    *,
    model: Optional[DeterministicAtmosphericModel] = None,
    post_processor: Optional[LLMPostProcessor] = None,
) -> SoilMoistureForecast:
    """Predict soil moisture 72 hours in advance using the deterministic pipeline."""

    model = model or DeterministicAtmosphericModel()
    post_processor = post_processor or LLMPostProcessor()

    normalised_weather = load_numerical_weather_outputs(weather_data)

    hours_ahead = post_processor.horizon_hours
    projected_states = model.predict(initial_state, normalised_weather, hours_ahead)

    features = export_soil_moisture_features(projected_states)

    forecast = post_processor.postprocess(initial_state, projected_states, features)

    if len(forecast.values) < hours_ahead:
        extended = list(forecast.values)
        last_value = extended[-1] if extended else initial_state.soil_moisture
        while len(extended) < hours_ahead:
            extended.append(last_value)
        forecast.values = extended

    forecast.metadata.setdefault("issued_at", dt.datetime.utcnow().isoformat())
    forecast.metadata["features"] = features

    return forecast
