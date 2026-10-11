"""Read-only deterministic Demand analyst using the published message contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from mosaic.agents.messages import (
    AgentRole,
    AnalystRecommendation,
    AnalystRequest,
    ToolEvidence,
    ensure_tool_allowed,
)
from mosaic.simulator.actions import NoAction
from mosaic.tools.api import ToolRunner

_HISTORY_DAYS = 7
_FORECAST_HORIZON = 3
_FORECAST_WINDOW = 5
_SPIKE_RATIO = 1.5
_DROP_RATIO = 0.7

_DemandKey = tuple[str, str]
_DemandPoint = tuple[int, int]
_RiskSignal = tuple[float, str, str, int, float, str]


class DemandAgent:
    """Build a deterministic demand recommendation from ToolRunner evidence.

    The analyst has no action executor and can call only the demand-history and
    forecast tools. Its recommendation is a non-LLM coordination baseline, not
    an inference result.
    """

    def recommend(self, request: AnalystRequest, tools: ToolRunner) -> AnalystRecommendation:
        """Return a correlated read-only recommendation for the current episode day."""
        if request.to_role is not AgentRole.DEMAND:
            raise ValueError("DemandAgent requires a request addressed to the demand role")

        simulator = tools.context.simulator
        current_day = simulator.state.day
        if request.episode_seed != simulator.config.seed:
            raise ValueError("Request episode_seed does not match the active simulator")
        if request.day != current_day and not (current_day == 0 and request.day == 1):
            raise ValueError("Request day does not match the active simulator day")

        history_arguments: dict[str, object] = {"days": _HISTORY_DAYS}
        history_evidence, history_result = self._call_tool(
            tools, "get_demand_history", history_arguments
        )
        evidence = [history_evidence]
        if not history_evidence.success:
            return self._recommendation(
                request,
                evidence,
                "Demand history is unavailable; no forecast or spike/drop assessment can be made. "
                "No business action was executed.",
            )

        observations = self._read_history(history_result, current_day)
        if not observations:
            return self._recommendation(
                request,
                evidence,
                f"No demand history has been observed by simulator day {current_day}; "
                "the short-horizon forecast and spike/drop assessment are unavailable. "
                "No business action was executed.",
            )

        observations_by_pair: dict[_DemandKey, list[_DemandPoint]] = {}
        for day, region_id, sku_id, quantity in observations:
            observations_by_pair.setdefault((sku_id, region_id), []).append((day, quantity))

        forecasts: dict[_DemandKey, str] = {}
        forecasts_failed = 0
        for sku_id, region_id in sorted(observations_by_pair):
            arguments: dict[str, object] = {
                "sku_id": sku_id,
                "region_id": region_id,
                "horizon": _FORECAST_HORIZON,
                "window": _FORECAST_WINDOW,
            }
            forecast_evidence, forecast_result = self._call_tool(
                tools, "forecast_demand", arguments
            )
            evidence.append(forecast_evidence)
            if not forecast_evidence.success:
                forecasts_failed += 1
                continue
            forecast_points = self._read_forecast(forecast_result, current_day)
            if forecast_points:
                forecasts[(sku_id, region_id)] = ", ".join(
                    f"day {day}={quantity}" for day, quantity in forecast_points
                )
            else:
                forecasts_failed += 1

        signals: list[_RiskSignal] = []
        assessed_pairs = 0
        for (sku_id, region_id), points in observations_by_pair.items():
            points.sort(key=lambda point: point[0])
            if points[-1][0] != current_day:
                continue
            prior_quantities = [quantity for day, quantity in points if day < current_day]
            if not prior_quantities:
                continue
            assessed_pairs += 1
            latest = points[-1][1]
            prior_mean = sum(prior_quantities) / len(prior_quantities)
            if prior_mean == 0.0 and latest > 0:
                kind = "DEMAND_SPIKE"
            elif prior_mean > 0.0 and latest >= prior_mean * _SPIKE_RATIO:
                kind = "DEMAND_SPIKE"
            elif prior_mean > 0.0 and latest <= prior_mean * _DROP_RATIO:
                kind = "DEMAND_DROP"
            else:
                continue
            severity = abs(latest - prior_mean) / max(prior_mean, 1.0)
            signals.append((severity, sku_id, region_id, latest, prior_mean, kind))

        signals.sort(key=lambda signal: (-signal[0], signal[1], signal[2]))
        if signals:
            risk_text = "; ".join(
                f"{kind} for {sku_id}/{region_id} on day {current_day} "
                f"(observed {latest}; prior-observation mean {prior_mean:g})"
                for _, sku_id, region_id, latest, prior_mean, kind in signals[:3]
            )
            action_text = "Coordinator should review these demand signals with Inventory."
        elif assessed_pairs:
            risk_text = (
                f"No material demand spike/drop in {assessed_pairs} "
                "assessed SKU/region pairs."
            )
            action_text = "Continue monitoring the demand forecast."
        else:
            risk_text = (
                "Spike/drop assessment is unavailable because no SKU/region has both a "
                "current-day and an earlier observation."
            )
            action_text = "Continue monitoring as history becomes available."

        selected_pairs: list[_DemandKey] = []
        for _, sku_id, region_id, _, _, _ in signals:
            pair = (sku_id, region_id)
            if pair in forecasts and pair not in selected_pairs:
                selected_pairs.append(pair)
        for pair in sorted(forecasts):
            if pair not in selected_pairs:
                selected_pairs.append(pair)
        forecast_text = "; ".join(
            f"{sku_id}/{region_id}: {forecasts[(sku_id, region_id)]}"
            for sku_id, region_id in selected_pairs[:2]
        )
        if not forecast_text:
            forecast_text = "No short-horizon projection is available from the forecast tool."
        elif forecasts_failed:
            forecast_text += f"; {forecasts_failed} forecast tool call(s) were unavailable"

        summary = (
            "Deterministic Demand baseline (read-only tools): "
            f"{risk_text} Tool projections from forecast_demand: {forecast_text}. "
            f"{action_text} No business action was executed."
        )
        return self._recommendation(request, evidence, summary)

    @staticmethod
    def _call_tool(
        tools: ToolRunner, name: str, arguments: dict[str, object]
    ) -> tuple[ToolEvidence, dict[str, object] | None]:
        ensure_tool_allowed(AgentRole.DEMAND, name)
        response = tools.run(name, arguments)
        success = response.get("success") is True
        raw_result = response.get("result")
        result: dict[str, object] | None = None
        if isinstance(raw_result, dict):
            result = cast(dict[str, object], raw_result)
        elif not success:
            error = response.get("error")
            result = {"error": error} if error is not None else None

        evidence = ToolEvidence(
            tool=name,
            arguments=arguments,
            success=success,
            result=result,
        )
        return evidence, result

    @staticmethod
    def _read_history(
        result: dict[str, object] | None, current_day: int
    ) -> list[tuple[int, str, str, int]]:
        if result is None or not isinstance(result.get("history"), list):
            raise ValueError("get_demand_history returned an invalid result")
        observations: list[tuple[int, str, str, int]] = []
        for raw_entry in cast(Sequence[object], result["history"]):
            if not isinstance(raw_entry, dict):
                raise ValueError("get_demand_history returned an invalid history row")
            entry = cast(dict[str, object], raw_entry)
            day = entry.get("day")
            region_id = entry.get("region_id")
            sku_id = entry.get("sku_id")
            quantity = entry.get("quantity")
            if (
                not isinstance(day, int)
                or not isinstance(region_id, str)
                or not isinstance(sku_id, str)
                or not isinstance(quantity, int)
            ):
                raise ValueError("get_demand_history returned an invalid history row")
            if day > current_day:
                raise ValueError("get_demand_history returned unobserved future demand")
            observations.append((day, region_id, sku_id, quantity))
        return observations

    @staticmethod
    def _read_forecast(
        result: dict[str, object] | None, current_day: int
    ) -> list[tuple[int, int]]:
        if result is None or not isinstance(result.get("forecast"), list):
            raise ValueError("forecast_demand returned an invalid result")
        forecast: list[tuple[int, int]] = []
        for raw_entry in cast(Sequence[object], result["forecast"]):
            if not isinstance(raw_entry, dict):
                raise ValueError("forecast_demand returned an invalid forecast row")
            entry = cast(dict[str, object], raw_entry)
            day = entry.get("day")
            quantity = entry.get("quantity")
            if not isinstance(day, int) or not isinstance(quantity, int):
                raise ValueError("forecast_demand returned an invalid forecast row")
            if day <= current_day:
                raise ValueError("forecast_demand returned a non-future forecast day")
            forecast.append((day, quantity))
        return forecast

    @staticmethod
    def _recommendation(
        request: AnalystRequest, evidence: list[ToolEvidence], summary: str
    ) -> AnalystRecommendation:
        return AnalystRecommendation(
            message_id=f"{request.message_id}:demand",
            request_id=request.message_id,
            episode_seed=request.episode_seed,
            day=request.day,
            role=AgentRole.DEMAND,
            summary=summary,
            proposed_actions=[NoAction()],
            evidence=evidence,
        )
