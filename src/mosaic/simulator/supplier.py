"""Hành vi giao hàng của supplier (Sprint 1, task 3).

Quyết định kết quả giao hàng cho từng đơn dựa trên seed + order_id
(hash sha256) nên deterministic và không phụ thuộc thứ tự gọi.

Phân chia trách nhiệm:
- Delay do disruption được phản ánh qua `Supplier.current_lead_time`
  (disruption engine cập nhật) — module này dùng giá trị đó làm expected_arrival.
- Hủy đơn: sự kiện SUPPLIER_CANCELLATION đang hoạt động vào ngày đặt
  (tham số `cancel_probability`, mặc định 1.0).
- Trễ ngẫu nhiên: ở chế độ stochastic, xác suất đúng hạn = reliability_score.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mosaic.simulator.models import DisruptionEvent, DisruptionType, PurchaseOrder, Supplier

MAX_STOCHASTIC_DELAY_DAYS = 5


class OrderOutcomeReason(StrEnum):
    ON_TIME = "ON_TIME"
    STOCHASTIC_DELAY = "STOCHASTIC_DELAY"
    DISRUPTION_CANCELLED = "DISRUPTION_CANCELLED"


class OrderOutcome(BaseModel):
    """Kết quả giao hàng của một purchase order."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    expected_arrival: int = Field(ge=0)
    actual_arrival: int | None = Field(default=None, description="None nếu đơn bị hủy")
    cancelled: bool = False
    delay_days: int = Field(ge=0, default=0)
    reason: OrderOutcomeReason = OrderOutcomeReason.ON_TIME


class SupplierBehavior:
    """Quyết định kết quả giao hàng deterministic theo (seed, order_id)."""

    def __init__(
        self,
        seed: int,
        stochastic: bool = True,
        max_stochastic_delay_days: int = MAX_STOCHASTIC_DELAY_DAYS,
    ) -> None:
        self._seed = seed
        self._stochastic = stochastic
        self._max_stochastic_delay_days = max_stochastic_delay_days

    def resolve_order_outcome(
        self,
        supplier: Supplier,
        order: PurchaseOrder,
        disruptions: Sequence[DisruptionEvent] = (),
    ) -> OrderOutcome:
        """Kết quả giao hàng của `order` đặt cho `supplier` tại ngày `placed_at`."""
        if order.supplier_id != supplier.supplier_id:
            raise ValueError(
                f"order {order.order_id} thuộc supplier {order.supplier_id}, "
                f"không phải {supplier.supplier_id}"
            )
        expected_arrival = order.placed_at + supplier.current_lead_time
        cancellation = self._active_cancellation(order, disruptions)
        if cancellation is not None:
            rng = self._order_rng(order.order_id)
            probability = self._cancel_probability(cancellation)
            if rng.random() < probability:
                return OrderOutcome(
                    order_id=order.order_id,
                    expected_arrival=expected_arrival,
                    cancelled=True,
                    reason=OrderOutcomeReason.DISRUPTION_CANCELLED,
                )
        if self._stochastic:
            rng = self._order_rng(order.order_id)
            if rng.random() > supplier.reliability_score:
                delay_days = 1 + int(rng.random() * self._max_stochastic_delay_days)
                return OrderOutcome(
                    order_id=order.order_id,
                    expected_arrival=expected_arrival,
                    actual_arrival=expected_arrival + delay_days,
                    delay_days=delay_days,
                    reason=OrderOutcomeReason.STOCHASTIC_DELAY,
                )
        return OrderOutcome(
            order_id=order.order_id,
            expected_arrival=expected_arrival,
            actual_arrival=expected_arrival,
        )

    def _active_cancellation(
        self, order: PurchaseOrder, disruptions: Sequence[DisruptionEvent]
    ) -> DisruptionEvent | None:
        for event in disruptions:
            if event.type is not DisruptionType.SUPPLIER_CANCELLATION:
                continue
            if event.subject_type != "supplier" or event.subject_id != order.supplier_id:
                continue
            if event.start_day <= order.placed_at and (
                event.end_day is None or order.placed_at <= event.end_day
            ):
                return event
        return None

    @staticmethod
    def _cancel_probability(event: DisruptionEvent) -> float:
        probability = event.params.get("cancel_probability", 1.0)
        if not isinstance(probability, (int, float)):
            raise ValueError(
                f"Disruption {event.disruption_id} có tham số 'cancel_probability' không phải số"
            )
        return float(probability)

    def _order_rng(self, order_id: str) -> np.random.Generator:
        digest = hashlib.sha256(f"{self._seed}:{order_id}".encode()).digest()
        seed_value = int.from_bytes(digest[:8], byteorder="little")
        return np.random.default_rng(seed_value)
