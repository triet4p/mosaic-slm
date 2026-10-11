from typing import Any

import pytest
from pydantic import ValidationError

from mosaic.agents.messages import (
    KNOWN_TOOLS,
    UNIMPLEMENTED_MEMORY_TOOLS,
    AgentRole,
    AnalystRecommendation,
    AnalystRequest,
    CoordinatorDecision,
    MessageAdapter,
    ToolEvidence,
    allowed_tools,
    check_correlation,
    ensure_tool_allowed,
    executable_actions,
    role_tool_permissions,
)
from mosaic.simulator.actions import PlaceOrder, TransferStock
from mosaic.simulator.core import Simulator
from mosaic.simulator.episodes import EpisodeGenerator, EpisodeGeneratorConfig
from mosaic.simulator.supplier import SupplierBehavior
from mosaic.tools.api import ToolContext, ToolRunner


def make_request() -> AnalystRequest:
    return AnalystRequest(
        message_id="req-001",
        episode_seed=7,
        day=5,
        to_role=AgentRole.INVENTORY,
        instruction="Report inventory position and stockout risk for day 5.",
    )


def make_transfer() -> TransferStock:
    return TransferStock(
        source_warehouse="WH1",
        destination_warehouse="WH2",
        sku_id="SKU001",
        quantity=20,
    )


def make_recommendation() -> AnalystRecommendation:
    return AnalystRecommendation(
        message_id="rec-001",
        request_id="req-001",
        episode_seed=7,
        day=5,
        role=AgentRole.INVENTORY,
        summary="WH2 is low on SKU001; transfer 20 units from WH1.",
        proposed_actions=[make_transfer()],
        evidence=[
            ToolEvidence(
                tool="get_inventory_state",
                arguments={"sku_id": "SKU001"},
                success=True,
                result={"states": []},
            ),
            ToolEvidence(
                tool="calculate_transfer_cost",
                arguments={
                    "source_warehouse": "WH1",
                    "destination_warehouse": "WH2",
                    "sku_id": "SKU001",
                    "quantity": 20,
                },
                success=True,
                result={"transfer_cost": 8.0, "travel_time_days": 2},
            ),
        ],
        confidence=0.7,
    )


def make_decision() -> CoordinatorDecision:
    return CoordinatorDecision(
        message_id="dec-001",
        episode_seed=7,
        day=5,
        recommendation_ids=["rec-001"],
        rationale="Accept the inventory transfer; demand evidence is consistent.",
        final_actions=[make_transfer()],
        evidence=[
            ToolEvidence(
                tool="get_inventory_state",
                arguments={"sku_id": "SKU001"},
                success=True,
                result={"states": []},
            ),
        ],
    )


def test_exchange_round_trips_through_adapter_with_consistent_correlation() -> None:
    request = make_request()
    recommendation = make_recommendation()
    decision = make_decision()

    restored_request = MessageAdapter.validate_json(request.model_dump_json())
    restored_recommendation = MessageAdapter.validate_json(recommendation.model_dump_json())
    restored_decision = MessageAdapter.validate_json(decision.model_dump_json())

    assert isinstance(restored_request, AnalystRequest)
    assert isinstance(restored_recommendation, AnalystRecommendation)
    assert isinstance(restored_decision, CoordinatorDecision)
    assert restored_recommendation.request_id == request.message_id
    assert restored_decision.recommendation_ids == [recommendation.message_id]
    assert restored_decision.final_actions[0].type == "TRANSFER_STOCK"

    check_correlation(request, recommendation, decision)


def test_correlation_rejects_mismatched_day() -> None:
    decision = make_decision().model_copy(update={"day": 6})
    with pytest.raises(ValueError, match="day khong khop"):
        check_correlation(make_request(), make_recommendation(), decision)


def test_correlation_rejects_wrong_request_link() -> None:
    recommendation = make_recommendation().model_copy(update={"request_id": "req-999"})
    with pytest.raises(ValueError, match="request_id"):
        check_correlation(make_request(), recommendation, make_decision())


def test_correlation_rejects_unreferenced_recommendation() -> None:
    decision = make_decision().model_copy(update={"recommendation_ids": ["rec-999"]})
    with pytest.raises(ValueError, match="tham chieu"):
        check_correlation(make_request(), make_recommendation(), decision)


def test_request_rejects_non_coordinator_sender() -> None:
    with pytest.raises(ValidationError):
        AnalystRequest(
            message_id="req-002",
            episode_seed=7,
            day=5,
            from_role=AgentRole.DEMAND,
            to_role=AgentRole.INVENTORY,
            instruction="Out-of-authority request.",
        )


def test_request_rejects_coordinator_as_target() -> None:
    with pytest.raises(ValidationError):
        AnalystRequest(
            message_id="req-003",
            episode_seed=7,
            day=5,
            to_role=AgentRole.COORDINATOR,
            instruction="Coordinator cannot request itself.",
        )


def test_recommendation_rejects_coordinator_role() -> None:
    with pytest.raises(ValidationError):
        AnalystRecommendation(
            message_id="rec-002",
            request_id="req-001",
            episode_seed=7,
            day=5,
            role=AgentRole.COORDINATOR,
            summary="Coordinator must decide, not recommend.",
            proposed_actions=[make_transfer()],
            evidence=[ToolEvidence(tool="get_inventory_state")],
        )


def test_decision_rejects_analyst_role() -> None:
    with pytest.raises(ValidationError):
        CoordinatorDecision(
            message_id="dec-002",
            episode_seed=7,
            day=5,
            role=AgentRole.INVENTORY,
            recommendation_ids=["rec-001"],
            rationale="Analyst cannot issue a final decision.",
            final_actions=[make_transfer()],
        )


def test_recommendation_rejects_tool_outside_role_permission() -> None:
    with pytest.raises(ValidationError, match="khong duoc dung tool"):
        AnalystRecommendation(
            message_id="rec-003",
            request_id="req-001",
            episode_seed=7,
            day=5,
            role=AgentRole.DEMAND,
            summary="Demand must not read inventory state directly.",
            proposed_actions=[make_transfer()],
            evidence=[ToolEvidence(tool="get_inventory_state")],
        )


def test_evidence_rejects_unimplemented_memory_tool() -> None:
    with pytest.raises(ValidationError, match="Phase 3"):
        ToolEvidence(tool="write_private_memory")


def test_evidence_rejects_unknown_tool() -> None:
    with pytest.raises(ValidationError, match="khong ton tai"):
        ToolEvidence(tool="do_something")


def test_recommendation_reuses_business_action_validation() -> None:
    payload: dict[str, Any] = {
        "message_type": "recommendation",
        "message_id": "rec-004",
        "request_id": "req-001",
        "episode_seed": 7,
        "day": 5,
        "role": "demand",
        "summary": "Order more.",
        "proposed_actions": [
            {
                "type": "PLACE_ORDER",
                "supplier_id": "SUP_A",
                "warehouse_id": "WH1",
                "sku_id": "SKU001",
                "quantity": 0,
            }
        ],
        "evidence": [{"tool": "forecast_demand", "arguments": {}, "success": True, "result": {}}],
    }
    with pytest.raises(ValidationError):
        AnalystRecommendation.model_validate(payload)


def test_demand_recommendation_proposes_order_with_forecast_evidence() -> None:
    recommendation = AnalystRecommendation(
        message_id="rec-010",
        request_id="req-010",
        episode_seed=7,
        day=5,
        role=AgentRole.DEMAND,
        summary="Demand spike for SKU001 in R1; order 50 units to WH1.",
        proposed_actions=[
            PlaceOrder(supplier_id="SUP_A", warehouse_id="WH1", sku_id="SKU001", quantity=50)
        ],
        evidence=[
            ToolEvidence(
                tool="forecast_demand",
                arguments={"sku_id": "SKU001", "region_id": "R1"},
                success=True,
                result={"forecast": []},
            )
        ],
        confidence=0.8,
    )
    assert recommendation.proposed_actions[0].type == "PLACE_ORDER"
    with pytest.raises(TypeError):
        executable_actions(recommendation)


def test_executable_actions_only_accepts_coordinator_decision() -> None:
    actions = executable_actions(make_decision())
    assert len(actions) == 1
    assert actions[0].type == "TRANSFER_STOCK"
    with pytest.raises(TypeError):
        executable_actions(make_recommendation())
    with pytest.raises(TypeError):
        executable_actions(make_request())


def test_known_tools_match_deployed_tool_runner_registry() -> None:
    episode = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=1)).generate(seed=1)
    simulator = Simulator(
        episode.config,
        supplier_behavior=SupplierBehavior(
            seed=episode.config.seed, stochastic=episode.supplier_stochastic
        ),
    )
    runner = ToolRunner(ToolContext(simulator=simulator))
    assert sorted(spec.name for spec in runner.specs) == sorted(KNOWN_TOOLS)


def test_role_permissions_cover_expected_sets_without_memory_tools() -> None:
    permissions = role_tool_permissions()
    assert set(permissions) == {"demand", "inventory", "supplier_risk", "coordinator"}
    assert permissions["coordinator"] == sorted(KNOWN_TOOLS)
    assert set(permissions["demand"]) == {
        "forecast_demand",
        "get_demand_history",
    }
    assert set(permissions["inventory"]) == {
        "calculate_reorder_point",
        "calculate_transfer_cost",
        "forecast_demand",
        "get_demand_history",
        "get_inventory_state",
        "get_open_orders",
        "get_shipments",
    }
    assert set(permissions["supplier_risk"]) == {
        "calculate_order_cost",
        "get_inventory_state",
        "get_open_orders",
        "get_shipments",
        "get_supplier_status",
    }
    with pytest.raises(ValueError, match="khong duoc dung tool"):
        ensure_tool_allowed(AgentRole.SUPPLIER_RISK, "simulate_action")
    for tools in permissions.values():
        assert not set(tools) & set(UNIMPLEMENTED_MEMORY_TOOLS)
    assert allowed_tools(AgentRole.DEMAND) <= KNOWN_TOOLS


def test_envelope_rejects_unknown_message_type() -> None:
    with pytest.raises(ValidationError):
        MessageAdapter.validate_python({"message_type": "memo", "message_id": "m-1"})


def test_message_schemas_require_correlation_fields() -> None:
    request_schema = AnalystRequest.model_json_schema()
    assert {"message_id", "episode_seed", "day", "to_role", "instruction"} <= set(
        request_schema["required"]
    )
    assert request_schema["properties"]["message_type"]["const"] == "request"
    recommendation_schema = AnalystRecommendation.model_json_schema()
    assert {"request_id", "proposed_actions", "evidence"} <= set(recommendation_schema["required"])
    assert recommendation_schema["properties"]["message_type"]["const"] == "recommendation"
    decision_schema = CoordinatorDecision.model_json_schema()
    assert {"recommendation_ids", "rationale", "final_actions"} <= set(decision_schema["required"])
    assert decision_schema["properties"]["message_type"]["const"] == "decision"
