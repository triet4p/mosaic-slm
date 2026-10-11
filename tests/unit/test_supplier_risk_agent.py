from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import pytest
from pydantic import ValidationError

from mosaic.agents.messages import (
    AgentRole,
    AnalystRecommendation,
    AnalystRequest,
    ToolEvidence,
    allowed_tools,
    ensure_tool_allowed,
    executable_actions,
)
from mosaic.agents.supplier_risk import SupplierRiskAgent
from mosaic.simulator.actions import ActionExecutor, NoAction, PlaceOrder, TransferStock
from mosaic.simulator.core import Simulator
from mosaic.simulator.episodes import Episode, EpisodeGenerator, EpisodeGeneratorConfig
from mosaic.simulator.models import DisruptionEvent, DisruptionType, Supplier
from mosaic.simulator.supplier import SupplierBehavior
from mosaic.tools.api import ToolContext, ToolRunner

_SEED = 407
_SKU = "SKU001"


def make_supplier(
    supplier_id: str,
    *,
    base_lead_time: int,
    reliability: float,
    capacity: int,
    unit_cost: float,
    order_cost: float,
    supported_skus: list[str] | None = None,
) -> Supplier:
    return Supplier(
        supplier_id=supplier_id,
        supported_skus=supported_skus if supported_skus is not None else [_SKU],
        base_lead_time=base_lead_time,
        current_lead_time=base_lead_time,
        unit_cost=unit_cost,
        order_cost=order_cost,
        reliability_score=reliability,
        capacity=capacity,
    )


def make_environment(
    *,
    suppliers: list[Supplier] | None = None,
    disruptions: list[DisruptionEvent] | None = None,
) -> tuple[Episode, Simulator, ToolRunner]:
    generated = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=1)).generate(seed=_SEED)
    updates: dict[str, object] = {"disruptions": disruptions if disruptions is not None else []}
    if suppliers is not None:
        generated_suppliers = {
            supplier.supplier_id: supplier for supplier in generated.config.suppliers
        }
        updates["suppliers"] = [
            supplier.model_copy(
                update={
                    "supported_skus": sorted(
                        set(supplier.supported_skus)
                        | set(generated_suppliers[supplier.supplier_id].supported_skus)
                    )
                }
            )
            for supplier in suppliers
        ]
        assignments = dict(generated.config.initial_supplier_assignments)
        assignments[_SKU] = "SUP_A"
        updates["initial_supplier_assignments"] = assignments
    config = generated.config.model_copy(update=updates)
    episode = generated.model_copy(update={"config": config})
    simulator = Simulator(
        config,
        supplier_behavior=SupplierBehavior(seed=episode.seed, stochastic=False),
    )
    simulator.advance_day()
    return episode, simulator, ToolRunner(ToolContext(simulator=simulator))


def make_request(
    episode: Episode,
    simulator: Simulator,
    instruction: str,
    message_id: str = "req-supplier-risk",
) -> AnalystRequest:
    return AnalystRequest(
        message_id=message_id,
        episode_seed=episode.seed,
        day=simulator.state.day,
        to_role=AgentRole.SUPPLIER_RISK,
        instruction=instruction,
    )


def evidence_result(recommendation: AnalystRecommendation, tool: str) -> dict[str, object]:
    item = next(evidence for evidence in recommendation.evidence if evidence.tool == tool)
    assert item.success
    assert item.result is not None
    return cast(dict[str, object], item.result)


def assert_proposals_valid_on_copy(
    simulator: Simulator, recommendation: AnalystRecommendation
) -> None:
    validation_simulator = deepcopy(simulator)
    for action in recommendation.proposed_actions:
        assert isinstance(action, (NoAction, PlaceOrder))
        ActionExecutor(validation_simulator).apply(action)
    with pytest.raises(TypeError):
        executable_actions(recommendation)


def assert_order_quote_matches_action_record(
    simulator: Simulator, recommendation: AnalystRecommendation
) -> None:
    action = recommendation.proposed_actions[0]
    assert isinstance(action, PlaceOrder)
    validation_simulator = deepcopy(simulator)
    ActionExecutor(validation_simulator).apply(action)
    order = validation_simulator.state.purchase_orders[-1]
    quote_evidence = next(
        evidence
        for evidence in recommendation.evidence
        if evidence.tool == "calculate_order_cost"
        and evidence.arguments.get("supplier_id") == action.supplier_id
    )
    assert quote_evidence.result is not None
    quote_result = quote_evidence.result
    quoted_total = quote_result.get("total_cost")
    assert isinstance(quoted_total, (int, float))
    assert quoted_total == pytest.approx(order.order_cost + action.quantity * order.unit_cost)


def test_normal_supplier_proposal_is_correlated_tool_backed_and_read_only() -> None:
    episode, simulator, tools = make_environment(
        suppliers=[
            make_supplier(
                "SUP_A",
                base_lead_time=3,
                reliability=0.95,
                capacity=100,
                unit_cost=2.0,
                order_cost=10.0,
            ),
            make_supplier(
                "SUP_B",
                base_lead_time=4,
                reliability=0.9,
                capacity=100,
                unit_cost=1.5,
                order_cost=8.0,
            ),
        ]
    )
    simulator.state.sku_states[(_SKU, "WH1")].on_hand = 500
    ActionExecutor(simulator).apply(
        PlaceOrder(supplier_id="SUP_A", warehouse_id="WH1", sku_id=_SKU, quantity=5)
    )
    ActionExecutor(simulator).apply(
        TransferStock(
            source_warehouse="WH1",
            destination_warehouse="WH2",
            sku_id=_SKU,
            quantity=1,
        )
    )
    request = make_request(
        episode,
        simulator,
        "Assess this order: sku_id=SKU001; warehouse_id=WH1; quantity=25; supplier_id=SUP_A",
        "req-normal-supplier",
    )
    before_state = simulator.state.model_dump()
    before_config = simulator.config.model_dump()

    recommendation = SupplierRiskAgent().recommend(request, tools)

    assert (
        AnalystRecommendation.model_validate_json(recommendation.model_dump_json())
        == recommendation
    )
    assert recommendation.role is AgentRole.SUPPLIER_RISK
    assert recommendation.request_id == request.message_id
    assert recommendation.episode_seed == episode.seed
    assert recommendation.day == simulator.state.day == request.day
    assert recommendation.proposed_actions == [
        PlaceOrder(supplier_id="SUP_A", warehouse_id="WH1", sku_id=_SKU, quantity=25)
    ]
    assert {item.tool for item in recommendation.evidence} == {
        "get_supplier_status",
        "get_open_orders",
        "get_shipments",
        "get_inventory_state",
        "calculate_order_cost",
    }
    assert {item.tool for item in recommendation.evidence} <= allowed_tools(AgentRole.SUPPLIER_RISK)
    assert "Current matching open/partial orders: 1" in recommendation.summary
    assert "IN_TRANSIT=1" in recommendation.summary
    assert "60.00" in recommendation.summary
    assert_proposals_valid_on_copy(simulator, recommendation)
    assert_order_quote_matches_action_record(simulator, recommendation)
    assert simulator.state.model_dump() == before_state
    assert simulator.config.model_dump() == before_config
    assert SupplierRiskAgent().recommend(request, tools) == recommendation


def test_pending_orders_hide_unobserved_actuals_but_keep_expected_eta() -> None:
    episode, simulator, tools = make_environment(
        suppliers=[
            make_supplier(
                "SUP_A",
                base_lead_time=3,
                reliability=0.95,
                capacity=100,
                unit_cost=2.0,
                order_cost=10.0,
            ),
            make_supplier(
                "SUP_B",
                base_lead_time=4,
                reliability=0.9,
                capacity=100,
                unit_cost=1.5,
                order_cost=8.0,
            ),
        ]
    )
    simulator.state.sku_states[(_SKU, "WH1")].on_hand = 500
    ActionExecutor(simulator).apply(
        PlaceOrder(supplier_id="SUP_A", warehouse_id="WH1", sku_id=_SKU, quantity=5)
    )
    ActionExecutor(simulator).apply(
        TransferStock(
            source_warehouse="WH1",
            destination_warehouse="WH2",
            sku_id=_SKU,
            quantity=1,
        )
    )
    request = make_request(
        episode,
        simulator,
        "Assess this order: sku_id=SKU001; warehouse_id=WH1; quantity=25; supplier_id=SUP_A",
        "req-pending-observable",
    )

    recommendation = SupplierRiskAgent().recommend(request, tools)
    serialized = AnalystRecommendation.model_validate_json(recommendation.model_dump_json())

    order_rows = cast(list[Any], evidence_result(serialized, "get_open_orders")["orders"])
    assert len(order_rows) == 1
    published = cast(dict[str, Any], order_rows[0])
    assert published["quantity"] == 5
    assert published["expected_arrival"] == 4
    assert published["status"] == "OPEN"
    assert published["actual_arrival"] is None
    assert published["received_quantity"] == 0
    shipment_rows = cast(list[Any], evidence_result(serialized, "get_shipments")["shipments"])
    assert len(shipment_rows) == 1
    published_shipment = cast(dict[str, Any], shipment_rows[0])
    assert published_shipment["quantity"] == 1
    assert published_shipment["expected_arrival"] == 3
    assert published_shipment["actual_arrival"] is None
    assert "Current matching open/partial orders: 1" in serialized.summary
    assert_proposals_valid_on_copy(simulator, serialized)
    assert simulator.state.model_dump() == simulator.state.model_dump()


def test_identical_observable_history_with_different_hidden_arrivals_matches() -> None:
    generated = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=1)).generate(seed=_SEED)
    config = generated.config.model_copy(update={"disruptions": []})
    runners: list[tuple[Simulator, ToolRunner, int | None]] = []
    for behavior_seed in (407, 408):
        simulator = Simulator(
            config,
            supplier_behavior=SupplierBehavior(seed=behavior_seed, stochastic=True),
        )
        simulator.advance_day()
        simulator.state.suppliers["SUP_A"] = simulator.state.suppliers["SUP_A"].model_copy(
            update={"reliability_score": 0.5}
        )
        hidden = simulator.place_order("SUP_A", "WH1", _SKU, 5)
        assert hidden.actual_arrival is not None
        assert hidden.actual_arrival > simulator.state.day
        runners.append(
            (simulator, ToolRunner(ToolContext(simulator=simulator)), hidden.actual_arrival)
        )
    (simulator_a, tools_a, hidden_a), (simulator_b, tools_b, hidden_b) = runners
    assert hidden_a != hidden_b

    request_a = make_request(
        generated.model_copy(update={"config": config}),
        simulator_a,
        "sku_id=SKU001; warehouse_id=WH1; quantity=25; supplier_id=SUP_A",
        "req-hidden-outcome",
    )
    request_b = make_request(
        generated.model_copy(update={"config": config}),
        simulator_b,
        "sku_id=SKU001; warehouse_id=WH1; quantity=25; supplier_id=SUP_A",
        "req-hidden-outcome",
    )
    recommendation_a = SupplierRiskAgent().recommend(request_a, tools_a)
    recommendation_b = SupplierRiskAgent().recommend(request_b, tools_b)
    serialized_a = AnalystRecommendation.model_validate_json(recommendation_a.model_dump_json())
    serialized_b = AnalystRecommendation.model_validate_json(recommendation_b.model_dump_json())
    assert [item.model_dump(mode="json") for item in serialized_a.evidence] == [
        item.model_dump(mode="json") for item in serialized_b.evidence
    ]
    assert serialized_a.model_dump(mode="json") == serialized_b.model_dump(mode="json")
    for serialized in (serialized_a, serialized_b):
        order_entries = cast(list[Any], evidence_result(serialized, "get_open_orders")["orders"])
        published = cast(dict[str, Any], order_entries[0])
        assert published["expected_arrival"] == 4
        assert published["quantity"] == 5
        assert published["actual_arrival"] is None
        assert published["received_quantity"] == 0


def test_active_lead_time_and_reliability_risk_selects_safer_alternate() -> None:
    delay = DisruptionEvent(
        disruption_id="delay-a",
        type=DisruptionType.SUPPLIER_DELAY,
        start_day=1,
        end_day=3,
        subject_type="supplier",
        subject_id="SUP_A",
        params={"extra_days": 3},
    )
    episode, simulator, tools = make_environment(
        suppliers=[
            make_supplier(
                "SUP_A",
                base_lead_time=3,
                reliability=0.65,
                capacity=100,
                unit_cost=2.0,
                order_cost=10.0,
            ),
            make_supplier(
                "SUP_B",
                base_lead_time=2,
                reliability=0.9,
                capacity=100,
                unit_cost=1.5,
                order_cost=8.0,
            ),
        ],
        disruptions=[delay],
    )
    request = make_request(
        episode,
        simulator,
        "sku_id=SKU001; warehouse_id=WH1; quantity=25; supplier_id=SUP_A",
        "req-delay-alternate",
    )

    recommendation = SupplierRiskAgent().recommend(request, tools)

    assert recommendation.proposed_actions == [
        PlaceOrder(supplier_id="SUP_B", warehouse_id="WH1", sku_id=_SKU, quantity=25)
    ]
    status_result = evidence_result(recommendation, "get_supplier_status")
    status_rows = cast(list[dict[str, Any]], status_result["suppliers"])
    supplier_a = next(row for row in status_rows if row["supplier_id"] == "SUP_A")
    assert supplier_a["current_lead_time"] == 6
    assert [event["disruption_id"] for event in supplier_a["active_disruption_events"]] == [
        "delay-a"
    ]
    assert "SUPPLIER_DELAY" in recommendation.summary
    assert "6/3 days" in recommendation.summary
    assert "reliability 0.65" in recommendation.summary
    assert_proposals_valid_on_copy(simulator, recommendation)
    assert_order_quote_matches_action_record(simulator, recommendation)


def test_active_cancellation_risk_uses_viable_alternate_without_claiming_cancellation() -> None:
    cancellation = DisruptionEvent(
        disruption_id="cancel-a",
        type=DisruptionType.SUPPLIER_CANCELLATION,
        start_day=1,
        subject_type="supplier",
        subject_id="SUP_A",
    )
    episode, simulator, tools = make_environment(
        suppliers=[
            make_supplier(
                "SUP_A",
                base_lead_time=2,
                reliability=0.98,
                capacity=100,
                unit_cost=1.0,
                order_cost=5.0,
            ),
            make_supplier(
                "SUP_B",
                base_lead_time=3,
                reliability=0.9,
                capacity=100,
                unit_cost=1.5,
                order_cost=8.0,
            ),
        ],
        disruptions=[cancellation],
    )
    request = make_request(
        episode,
        simulator,
        "sku_id=SKU001; warehouse_id=WH1; quantity=25; supplier_id=SUP_A",
        "req-cancellation-alternate",
    )

    recommendation = SupplierRiskAgent().recommend(request, tools)

    assert recommendation.proposed_actions == [
        PlaceOrder(supplier_id="SUP_B", warehouse_id="WH1", sku_id=_SKU, quantity=25)
    ]
    assert "active SUPPLIER_CANCELLATION disruption" in recommendation.summary
    assert "Selected alternate supplier SUP_B" in recommendation.summary
    assert_proposals_valid_on_copy(simulator, recommendation)
    assert_order_quote_matches_action_record(simulator, recommendation)


def test_unavailable_requested_supplier_uses_an_actual_supported_alternate() -> None:
    episode, simulator, tools = make_environment(
        suppliers=[
            make_supplier(
                "SUP_A",
                base_lead_time=2,
                reliability=0.95,
                capacity=0,
                unit_cost=2.0,
                order_cost=10.0,
            ),
            make_supplier(
                "SUP_B",
                base_lead_time=3,
                reliability=0.9,
                capacity=100,
                unit_cost=1.5,
                order_cost=8.0,
            ),
        ]
    )
    request = make_request(
        episode,
        simulator,
        "sku_id=SKU001; warehouse_id=WH1; quantity=25; supplier_id=SUP_MISSING",
        "req-unavailable-supplier",
    )

    recommendation = SupplierRiskAgent().recommend(request, tools)

    assert recommendation.proposed_actions == [
        PlaceOrder(supplier_id="SUP_B", warehouse_id="WH1", sku_id=_SKU, quantity=25)
    ]
    assert "Requested supplier SUP_MISSING is not present" in recommendation.summary
    assert "Selected alternate supplier SUP_B" in recommendation.summary
    assert_proposals_valid_on_copy(simulator, recommendation)
    assert_order_quote_matches_action_record(simulator, recommendation)


def test_supplier_without_requested_sku_support_is_replaced_by_supported_supplier() -> None:
    episode, simulator, tools = make_environment()
    request = make_request(
        episode,
        simulator,
        "sku_id=SKU001; warehouse_id=WH1; quantity=25; supplier_id=SUP_B",
        "req-unsupported-preference",
    )

    recommendation = SupplierRiskAgent().recommend(request, tools)

    assert recommendation.proposed_actions == [
        PlaceOrder(supplier_id="SUP_A", warehouse_id="WH1", sku_id=_SKU, quantity=25)
    ]
    assert "Requested supplier SUP_B does not list SKU001 as supported" in recommendation.summary
    assert_proposals_valid_on_copy(simulator, recommendation)
    assert_order_quote_matches_action_record(simulator, recommendation)


def test_no_action_when_capacity_is_insufficient_or_target_is_unverified() -> None:
    episode, simulator, tools = make_environment(
        suppliers=[
            make_supplier(
                "SUP_A",
                base_lead_time=2,
                reliability=0.95,
                capacity=10,
                unit_cost=2.0,
                order_cost=10.0,
            ),
            make_supplier(
                "SUP_B",
                base_lead_time=3,
                reliability=0.9,
                capacity=10,
                unit_cost=1.5,
                order_cost=8.0,
            ),
        ]
    )
    request = make_request(
        episode,
        simulator,
        "sku_id=SKU001; warehouse_id=WH1; quantity=25; supplier_id=SUP_A",
        "req-insufficient-capacity",
    )

    recommendation = SupplierRiskAgent().recommend(request, tools)

    assert recommendation.proposed_actions == [NoAction()]
    assert "capacity 10 is below requested quantity 25" in recommendation.summary
    assert "NoAction: no supplier has confirmed SKU support, sufficient per-order capacity" in (
        recommendation.summary
    )
    assert any(item.tool == "get_supplier_status" for item in recommendation.evidence)
    assert not any(item.tool == "calculate_order_cost" for item in recommendation.evidence)
    assert_proposals_valid_on_copy(simulator, recommendation)

    unverified_request = make_request(
        episode,
        simulator,
        "sku_id=SKU001; warehouse_id=WH999; quantity=25",
        "req-unverified-warehouse",
    )
    unverified = SupplierRiskAgent().recommend(unverified_request, tools)
    assert unverified.proposed_actions == [NoAction()]
    assert "requested SKU/warehouse pair is not confirmed" in unverified.summary


def test_missing_quantity_and_forbidden_tool_return_explicit_read_only_result() -> None:
    episode, simulator, tools = make_environment()
    request = make_request(
        episode,
        simulator,
        "sku_id=SKU001; warehouse_id=WH1; supplier_id=SUP_A",
        "req-missing-quantity",
    )
    before_state = simulator.state.model_dump()
    before_config = simulator.config.model_dump()

    recommendation = SupplierRiskAgent().recommend(request, tools)

    assert recommendation.proposed_actions == [NoAction()]
    assert "request context is missing quantity" in recommendation.summary
    assert "quantity is never inferred" in recommendation.summary
    assert not any(item.tool == "calculate_order_cost" for item in recommendation.evidence)
    with pytest.raises(ValueError, match="khong duoc dung tool"):
        ensure_tool_allowed(AgentRole.SUPPLIER_RISK, "simulate_action")
    with pytest.raises(ValidationError, match="khong duoc dung tool"):
        AnalystRecommendation(
            message_id="rec-forbidden",
            request_id=request.message_id,
            episode_seed=episode.seed,
            day=simulator.state.day,
            role=AgentRole.SUPPLIER_RISK,
            summary="Forbidden evidence must not validate.",
            proposed_actions=[NoAction()],
            evidence=[ToolEvidence(tool="simulate_action")],
        )
    assert simulator.state.model_dump() == before_state
    assert simulator.config.model_dump() == before_config
