import pytest

from mosaic.simulator.models import DisruptionEvent, DisruptionType, PurchaseOrder, Supplier
from mosaic.simulator.supplier import OrderOutcomeReason, SupplierBehavior


def make_supplier(
    reliability: float = 1.0,
    current_lead_time: int = 3,
    supplier_id: str = "SUP_A",
) -> Supplier:
    return Supplier(
        supplier_id=supplier_id,
        base_lead_time=3,
        current_lead_time=current_lead_time,
        unit_cost=2.0,
        order_cost=10.0,
        reliability_score=reliability,
        capacity=500,
    )


def make_order(
    order_id: str = "PO1", placed_at: int = 5, supplier_id: str = "SUP_A"
) -> PurchaseOrder:
    return PurchaseOrder(
        order_id=order_id,
        supplier_id=supplier_id,
        warehouse_id="WH1",
        sku_id="SKU001",
        quantity=100,
        unit_cost=2.0,
        order_cost=10.0,
        placed_at=placed_at,
        expected_arrival=8,
    )


def make_cancellation(
    supplier_id: str = "SUP_A",
    start_day: int = 1,
    end_day: int | None = None,
    probability: float = 1.0,
) -> DisruptionEvent:
    return DisruptionEvent(
        disruption_id="DC1",
        type=DisruptionType.SUPPLIER_CANCELLATION,
        start_day=start_day,
        end_day=end_day,
        subject_type="supplier",
        subject_id=supplier_id,
        params={"cancel_probability": probability},
    )


def test_supplier_delivers_on_time_in_deterministic_mode():
    behavior = SupplierBehavior(seed=1, stochastic=False)
    outcome = behavior.resolve_order_outcome(make_supplier(), make_order(placed_at=5))
    assert outcome.cancelled is False
    assert outcome.actual_arrival == 8
    assert outcome.delay_days == 0
    assert outcome.reason is OrderOutcomeReason.ON_TIME


def test_supplier_uses_current_lead_time_for_expected_arrival():
    behavior = SupplierBehavior(seed=1, stochastic=False)
    supplier = make_supplier(current_lead_time=10)
    outcome = behavior.resolve_order_outcome(supplier, make_order(placed_at=5))
    assert outcome.expected_arrival == 15
    assert outcome.actual_arrival == 15


def test_supplier_outcome_is_deterministic_for_same_seed_and_order():
    first = SupplierBehavior(seed=7).resolve_order_outcome(
        make_supplier(reliability=0.5), make_order("PO_ABC", placed_at=3)
    )
    second = SupplierBehavior(seed=7).resolve_order_outcome(
        make_supplier(reliability=0.5), make_order("PO_ABC", placed_at=3)
    )
    assert first == second


def test_supplier_outcome_differs_between_seeds():
    supplier = make_supplier(reliability=0.5)
    first = [
        SupplierBehavior(seed=1).resolve_order_outcome(supplier, make_order(f"PO{i}", 3))
        for i in range(20)
    ]
    second = [
        SupplierBehavior(seed=2).resolve_order_outcome(supplier, make_order(f"PO{i}", 3))
        for i in range(20)
    ]
    assert first != second


def test_supplier_cancels_during_disruption_window():
    behavior = SupplierBehavior(seed=1, stochastic=False)
    disruptions = [make_cancellation(start_day=4, end_day=10)]
    outcome = behavior.resolve_order_outcome(
        make_supplier(), make_order(placed_at=5), disruptions=disruptions
    )
    assert outcome.cancelled is True
    assert outcome.actual_arrival is None
    assert outcome.reason is OrderOutcomeReason.DISRUPTION_CANCELLED


def test_supplier_cancellation_respects_window_bounds():
    behavior = SupplierBehavior(seed=1, stochastic=False)
    before = behavior.resolve_order_outcome(
        make_supplier(),
        make_order(placed_at=3),
        disruptions=[make_cancellation(start_day=4, end_day=10)],
    )
    after = behavior.resolve_order_outcome(
        make_supplier(),
        make_order(placed_at=11),
        disruptions=[make_cancellation(start_day=4, end_day=10)],
    )
    assert before.cancelled is False
    assert after.cancelled is False


def test_supplier_cancellation_probability_zero_never_cancels():
    behavior = SupplierBehavior(seed=1, stochastic=False)
    disruptions = [make_cancellation(start_day=1, end_day=30, probability=0.0)]
    outcome = behavior.resolve_order_outcome(
        make_supplier(), make_order(placed_at=5), disruptions=disruptions
    )
    assert outcome.cancelled is False
    assert outcome.reason is OrderOutcomeReason.ON_TIME


def test_supplier_cancellation_only_affects_target_supplier():
    behavior = SupplierBehavior(seed=1, stochastic=False)
    disruptions = [make_cancellation(supplier_id="SUP_B", start_day=1, end_day=30)]
    outcome = behavior.resolve_order_outcome(
        make_supplier(), make_order(placed_at=5), disruptions=disruptions
    )
    assert outcome.cancelled is False


def test_supplier_stochastic_delay_occurs_with_low_reliability():
    behavior = SupplierBehavior(seed=3, stochastic=True)
    supplier = make_supplier(reliability=0.1)
    outcomes = [
        behavior.resolve_order_outcome(supplier, make_order(f"PO{i}", 2)) for i in range(30)
    ]
    delayed = [o for o in outcomes if o.reason is OrderOutcomeReason.STOCHASTIC_DELAY]
    assert len(delayed) >= 1
    for outcome in delayed:
        assert 1 <= outcome.delay_days <= 5
        assert outcome.actual_arrival == outcome.expected_arrival + outcome.delay_days


def test_supplier_deterministic_mode_always_on_time_with_unreliable_supplier():
    behavior = SupplierBehavior(seed=3, stochastic=False)
    supplier = make_supplier(reliability=0.05)
    outcomes = [
        behavior.resolve_order_outcome(supplier, make_order(f"PO{i}", 2)) for i in range(50)
    ]
    assert all(o.reason is OrderOutcomeReason.ON_TIME for o in outcomes)


def test_supplier_rejects_order_of_another_supplier():

    behavior = SupplierBehavior(seed=1, stochastic=False)
    with pytest.raises(ValueError):
        behavior.resolve_order_outcome(make_supplier(), make_order(supplier_id="SUP_B"))
