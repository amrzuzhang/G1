#!/usr/bin/env python3
"""CLI entry-point for the soil moisture prediction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from atmo_model.core import AtmosphericState
from atmo_model.pipeline import predict_three_day_soil_moisture


def _load_state(path: Path) -> AtmosphericState:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return AtmosphericState.from_dict(data)


def _load_weather(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("weather", [])
    if not isinstance(data, list):
        raise ValueError("Weather data must be a list of hourly entries")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial_state", type=Path, help="Path to a JSON file containing the initial atmospheric state")
    parser.add_argument("weather_data", type=Path, help="Path to a JSON file containing hourly weather forecasts")
    parser.add_argument("--output", type=Path, default=Path("soil_moisture_forecast.json"), help="Output path for the generated forecast")
    args = parser.parse_args()

    initial_state = _load_state(args.initial_state)
    weather_data = _load_weather(args.weather_data)

    forecast = predict_three_day_soil_moisture(initial_state, weather_data)

    serialisable = {
        "issued_at": forecast.issued_at.isoformat(),
        "horizon_hours": forecast.horizon_hours,
        "soil_moisture": forecast.values,
        "metadata": forecast.metadata,
    }

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(serialisable, handle, indent=2)

    print(f"Forecast saved to {args.output}")


if __name__ == "__main__":
    main()
