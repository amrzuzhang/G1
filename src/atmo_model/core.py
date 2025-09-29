"""Core data structures and deterministic atmospheric modelling utilities."""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence


def _ensure_timestamp(value: dt.datetime | str) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    return dt.datetime.fromisoformat(value)


@dataclass
class AtmosphericState:
    """Represents the atmospheric state at a specific timestamp."""

    timestamp: dt.datetime
    temperature_c: float
    relative_humidity: float
    precipitation_mm: float
    soil_moisture: float

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "AtmosphericState":
        return cls(
            timestamp=_ensure_timestamp(payload["timestamp"]),
            temperature_c=float(payload["temperature_c"]),
            relative_humidity=float(payload["relative_humidity"]),
            precipitation_mm=float(payload["precipitation_mm"]),
            soil_moisture=float(payload.get("soil_moisture", 0.0)),
        )


@dataclass
class SoilMoistureForecast:
    """Container for soil moisture forecasts at an hourly resolution."""

    issued_at: dt.datetime
    horizon_hours: int
    values: List[float] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_daily_averages(self, hours_per_day: int = 24) -> List[float]:
        """Return daily averages over the stored hourly predictions."""

        averages: List[float] = []
        for day in range(0, self.horizon_hours, hours_per_day):
            chunk = self.values[day : day + hours_per_day]
            if not chunk:
                break
            averages.append(sum(chunk) / len(chunk))
        return averages


class AtmosphericModel(ABC):
    """Abstract interface for atmospheric forecast models."""

    @abstractmethod
    def predict(
        self,
        initial_state: AtmosphericState,
        weather_inputs: Sequence[Dict[str, float]],
        hours_ahead: int,
    ) -> List[AtmosphericState]:
        """Predict the evolution of the atmospheric state."""


class DeterministicAtmosphericModel(AtmosphericModel):
    """A lightweight, fully deterministic atmospheric model.

    The model relies on simplified heuristics that provide deterministic
    outputs suitable for unit tests and offline experimentation.
    """

    def __init__(self, evaporation_rate: float = 0.015, percolation_rate: float = 0.01):
        self.evaporation_rate = evaporation_rate
        self.percolation_rate = percolation_rate

    def predict(
        self,
        initial_state: AtmosphericState,
        weather_inputs: Sequence[Dict[str, float]],
        hours_ahead: int,
    ) -> List[AtmosphericState]:
        states: List[AtmosphericState] = []
        current_state = initial_state
        time_step = dt.timedelta(hours=1)

        for hour in range(min(hours_ahead, len(weather_inputs))):
            weather = weather_inputs[hour]
            current_state = evolve_state(current_state, weather, time_step, self)
            states.append(current_state)
        return states


def load_numerical_weather_outputs(
    raw_weather_data: Iterable[Dict[str, object]]
) -> List[Dict[str, float]]:
    """Normalise numerical weather prediction outputs.

    Parameters
    ----------
    raw_weather_data:
        Iterable of mapping objects containing at least temperature, humidity and
        precipitation. Any missing field defaults to 0.0.
    """

    normalised: List[Dict[str, float]] = []
    for index, entry in enumerate(raw_weather_data):
        temperature = float(entry.get("temperature_c", 0.0))
        humidity = float(entry.get("relative_humidity", 0.0))
        precipitation = float(entry.get("precipitation_mm", 0.0))
        normalised.append(
            {
                "temperature_c": temperature,
                "relative_humidity": humidity,
                "precipitation_mm": precipitation,
                "hour_index": float(entry.get("hour_index", index)),
            }
        )
    return normalised


def evolve_state(
    state: AtmosphericState,
    weather: Dict[str, float],
    step: dt.timedelta,
    model: DeterministicAtmosphericModel,
) -> AtmosphericState:
    """Evolve the atmospheric state by one time step.

    The model applies a simplified mass balance between precipitation and
    evapotranspiration. Soil moisture is kept within the ``[0, 1]`` interval.
    """

    hours = step.total_seconds() / 3600.0
    temperature = weather["temperature_c"]
    humidity = weather["relative_humidity"]
    precipitation = weather["precipitation_mm"]

    evapotranspiration = (1 - humidity / 100.0) * model.evaporation_rate * hours
    infiltration = precipitation * model.percolation_rate * hours

    new_soil_moisture = state.soil_moisture + infiltration - evapotranspiration
    new_soil_moisture = max(0.0, min(1.0, new_soil_moisture))

    return AtmosphericState(
        timestamp=state.timestamp + step,
        temperature_c=temperature,
        relative_humidity=humidity,
        precipitation_mm=precipitation,
        soil_moisture=new_soil_moisture,
    )


def export_soil_moisture_features(
    states: Sequence[AtmosphericState],
) -> Dict[str, float]:
    """Export deterministic features that help the LLM estimate soil moisture."""

    if not states:
        return {
            "mean_temperature": 0.0,
            "mean_humidity": 0.0,
            "total_precipitation": 0.0,
            "final_soil_moisture": 0.0,
        }

    mean_temperature = sum(s.temperature_c for s in states) / len(states)
    mean_humidity = sum(s.relative_humidity for s in states) / len(states)
    total_precipitation = sum(s.precipitation_mm for s in states)
    final_soil_moisture = states[-1].soil_moisture

    return {
        "mean_temperature": mean_temperature,
        "mean_humidity": mean_humidity,
        "total_precipitation": total_precipitation,
        "final_soil_moisture": final_soil_moisture,
    }
