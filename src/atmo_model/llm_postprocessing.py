"""Utilities for formatting and parsing LLM powered soil moisture estimates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .core import AtmosphericState, SoilMoistureForecast

LLMClient = Callable[[str], Any]


@dataclass
class LLMPostProcessor:
    """Handle LLM based soil moisture post processing.

    Parameters
    ----------
    client:
        A callable that accepts a string prompt and returns a model response. The
        response may be raw text, a ``dict`` or a list containing a string.
    """

    client: Optional[LLMClient] = None
    horizon_hours: int = 72

    def build_prompt(
        self,
        initial_state: AtmosphericState,
        projected_states: Sequence[AtmosphericState],
        features: Dict[str, float],
    ) -> str:
        """Format a prompt describing the atmospheric evolution."""

        state_lines = []
        for state in projected_states:
            state_lines.append(
                f"{state.timestamp.isoformat()} | T={state.temperature_c:.1f}C | "
                f"RH={state.relative_humidity:.1f}% | P={state.precipitation_mm:.2f}mm | "
                f"SM={state.soil_moisture:.3f}"
            )

        feature_lines = "\n".join(f"- {name}: {value:.3f}" for name, value in sorted(features.items()))

        prompt = (
            "You are an expert hydrologist. Based on the projected atmospheric "
            "states, estimate the soil moisture for the next 72 hours in hourly "
            "increments. Return the results as a JSON object with the key "
            "'soil_moisture' containing a list of 72 decimal numbers between 0 and 1.\n"
            "Initial state: "
            f"T={initial_state.temperature_c:.1f}C, RH={initial_state.relative_humidity:.1f}%, "
            f"P={initial_state.precipitation_mm:.2f}mm, SM={initial_state.soil_moisture:.3f}.\n"
            "Projected states (hourly):\n"
            + "\n".join(state_lines)
            + "\nKey summary features:\n"
            + feature_lines
        )
        return prompt

    def _call_client(self, prompt: str) -> Any:
        if self.client is None:
            raise RuntimeError("No LLM client configured for post-processing")
        return self.client(prompt)

    def _coerce_response_text(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            return json.dumps(response)
        if isinstance(response, (list, tuple)) and response:
            return self._coerce_response_text(response[0])
        return str(response)

    def parse_response(self, response: Any) -> List[float]:
        """Parse an LLM response into 72 hourly soil moisture values."""

        text = self._coerce_response_text(response)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            numbers = [float(value) for value in re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)]
            return numbers[: self.horizon_hours]

        if isinstance(data, dict) and "soil_moisture" in data:
            values = data["soil_moisture"]
        else:
            values = data

        if not isinstance(values, Iterable):
            raise ValueError("LLM response does not contain an iterable of values")

        result: List[float] = []
        for value in values:
            try:
                result.append(float(value))
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError("Invalid soil moisture value in LLM response") from exc
        return result[: self.horizon_hours]

    def postprocess(
        self,
        initial_state: AtmosphericState,
        projected_states: Sequence[AtmosphericState],
        features: Dict[str, float],
    ) -> SoilMoistureForecast:
        """Run the LLM client and convert the response into a forecast."""

        prompt = self.build_prompt(initial_state, projected_states, features)
        response = self._call_client(prompt)
        values = self.parse_response(response)

        return SoilMoistureForecast(
            issued_at=projected_states[0].timestamp if projected_states else initial_state.timestamp,
            horizon_hours=self.horizon_hours,
            values=values,
            metadata={"prompt": prompt},
        )
