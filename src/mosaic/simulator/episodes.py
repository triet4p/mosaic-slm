"""Episode generator (Sprint 1, task 8).

Sinh `SimulatorConfig` hoàn chỉnh từ spec + seed: 3 kho, 2 supplier,
20 sản phẩm, 3 region, schedule demand theo difficulty, disruptions
từ DisruptionEngine (các sự kiện demand được truyền thẳng vào
DemandGenerator khi sinh schedule).

Episode là bản ghi bất biến, serialize được toàn bộ (JSON) để replay
(task 9) mà không cần generator.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mosaic.simulator.core import SimulatorConfig
from mosaic.simulator.demand import DemandGenerator, DemandProfile
from mosaic.simulator.disruptions import DisruptionEngine, DisruptionEngineConfig
from mosaic.simulator.models import SKUState, Supplier, TransportRoute, Warehouse

DEMAND_NOISE_BY_DIFFICULTY: dict[int, float] = {1: 0.0, 2: 0.15, 3: 0.25}


class EpisodeGeneratorConfig(BaseModel):
    """Spec cấu hình sinh episode; có thể load từ YAML."""

    model_config = ConfigDict(extra="forbid")

    num_days: int = Field(ge=10, default=30)
    num_regions: int = Field(ge=1, default=3)
    num_warehouses: int = Field(ge=1, default=3)
    num_suppliers: int = Field(ge=1, default=2)
    num_products: int = Field(ge=1, default=20)
    min_disruptions: int = Field(ge=1, default=1)
    max_disruptions: int = Field(ge=1, default=5)
    difficulty: int = Field(ge=1, le=3, default=2)
    min_demand: int = Field(ge=1, default=5)
    max_demand: int = Field(ge=1, default=30)
    seasonality_amplitude: float = Field(ge=0.0, le=1.0, default=0.15)
    late_delivery_penalty_per_day: float = Field(ge=0.0, default=5.0)
    supplier_switch_cost: float = Field(ge=0.0, default=100.0)

    @model_validator(mode="after")
    def _validate_ranges(self) -> EpisodeGeneratorConfig:
        if self.max_disruptions < self.min_disruptions:
            raise ValueError("max_disruptions phải lớn hơn hoặc bằng min_disruptions")
        if self.max_demand < self.min_demand:
            raise ValueError("max_demand phải lớn hơn hoặc bằng min_demand")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> EpisodeGeneratorConfig:
        """Load spec từ file YAML."""
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)


class Episode(BaseModel):
    """Một episode hoàn chỉnh, deterministic theo seed."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    difficulty: int = Field(ge=1, le=3)
    config: SimulatorConfig
    supplier_stochastic: bool


class EpisodeGenerator:
    """Sinh episode hoàn chỉnh từ spec; cùng seed → cùng episode."""

    def __init__(self, spec: EpisodeGeneratorConfig) -> None:
        self._spec = spec

    def generate(self, seed: int) -> Episode:
        """Sinh một episode từ seed (deterministic)."""
        rng = np.random.default_rng(seed)
        warehouses = self._make_warehouses()
        products = self._product_ids()
        suppliers = self._make_suppliers(products)
        routes = self._make_routes(warehouses)
        profiles = self._make_profiles(rng, warehouses, products)
        num_disruptions = self._sample_num_disruptions(rng)
        disruptions = DisruptionEngine(
            DisruptionEngineConfig(
                seed=seed,
                difficulty=self._spec.difficulty,
                num_disruptions=num_disruptions,
                num_days=self._spec.num_days,
                regions=[w.region_id for w in warehouses],
                products=products,
                suppliers=[s.supplier_id for s in suppliers],
                warehouses=[w.warehouse_id for w in warehouses],
                routes=[r.route_id for r in routes],
            )
        ).generate()
        schedule = DemandGenerator(seed=seed, profiles=list(profiles.values())).generate_schedule(
            num_days=self._spec.num_days, disruptions=disruptions
        )
        sku_states = self._make_sku_states(warehouses, products, profiles, suppliers)
        config = SimulatorConfig(
            num_days=self._spec.num_days,
            seed=seed,
            warehouses=warehouses,
            suppliers=suppliers,
            routes=routes,
            initial_sku_states=sku_states,
            demand_schedule=schedule,
            disruptions=disruptions,
            late_delivery_penalty_per_day=self._spec.late_delivery_penalty_per_day,
            supplier_switch_cost=self._spec.supplier_switch_cost,
        )
        return Episode(
            seed=seed,
            difficulty=self._spec.difficulty,
            config=config,
            supplier_stochastic=self._spec.difficulty > 1,
        )

    def generate_dev_set(self, count: int = 100, base_seed: int = 0) -> list[Episode]:
        """Sinh dev set: count episode với seed liên tiếp base_seed..base_seed+count-1."""
        if count <= 0:
            raise ValueError("count phải lớn hơn 0")
        return [self.generate(base_seed + i) for i in range(count)]

    def _make_warehouses(self) -> list[Warehouse]:
        return [
            Warehouse(
                warehouse_id=f"WH{i + 1}",
                region_id=f"R{i + 1}",
                capacity=100000,
            )
            for i in range(self._spec.num_warehouses)
        ]

    def _product_ids(self) -> list[str]:
        return [f"SKU{idx:03d}" for idx in range(1, self._spec.num_products + 1)]

    def _make_suppliers(self, products: Sequence[str]) -> list[Supplier]:
        odd = [p for p in products if int(p[3:]) % 2 == 1]
        even = [p for p in products if int(p[3:]) % 2 == 0]
        return [
            Supplier(
                supplier_id="SUP_A",
                supported_skus=odd,
                base_lead_time=3,
                current_lead_time=3,
                unit_cost=2.0,
                order_cost=10.0,
                reliability_score=0.95,
                capacity=500,
            ),
            Supplier(
                supplier_id="SUP_B",
                supported_skus=even,
                base_lead_time=5,
                current_lead_time=5,
                unit_cost=1.5,
                order_cost=8.0,
                reliability_score=0.90,
                capacity=500,
            ),
        ]

    def _make_routes(self, warehouses: Sequence[Warehouse]) -> list[TransportRoute]:
        return [
            TransportRoute(
                route_id=f"RTE_{i + 1}",
                source_warehouse=warehouses[i].warehouse_id,
                destination_warehouse=warehouses[(i + 1) % len(warehouses)].warehouse_id,
                base_travel_time=2,
                current_travel_time=2,
                cost_per_unit=round(0.4 + 0.1 * (i % 3), 2),
            )
            for i in range(len(warehouses))
        ]

    def _make_profiles(
        self,
        rng: np.random.Generator,
        warehouses: Sequence[Warehouse],
        products: Sequence[str],
    ) -> dict[tuple[str, str], DemandProfile]:
        noise_std = DEMAND_NOISE_BY_DIFFICULTY[self._spec.difficulty]
        profiles: dict[tuple[str, str], DemandProfile] = {}
        for warehouse in warehouses:
            for product in products:
                base = int(rng.integers(self._spec.min_demand, self._spec.max_demand + 1))
                profiles[(warehouse.region_id, product)] = DemandProfile(
                    region_id=warehouse.region_id,
                    product_id=product,
                    base_demand=base,
                    seasonality_amplitude=self._spec.seasonality_amplitude,
                    seasonality_period=7,
                    noise_std=noise_std,
                )
        return profiles

    def _make_sku_states(
        self,
        warehouses: Sequence[Warehouse],
        products: Sequence[str],
        profiles: dict[tuple[str, str], DemandProfile],
        suppliers: Sequence[Supplier],
    ) -> list[SKUState]:
        supplier_by_product = {
            product: supplier.supplier_id
            for supplier in suppliers
            for product in supplier.supported_skus
        }
        supplier_map = {supplier.supplier_id: supplier for supplier in suppliers}
        states: list[SKUState] = []
        for warehouse in warehouses:
            for product in products:
                profile = profiles[(warehouse.region_id, product)]
                supplier = supplier_map[supplier_by_product[product]]
                base = profile.base_demand
                states.append(
                    SKUState(
                        sku_id=product,
                        warehouse_id=warehouse.warehouse_id,
                        on_hand=max(1, int(base * 4)),
                        reserved=0,
                        safety_stock=int(base * 1.5),
                        reorder_point=int(base * 2.5),
                        holding_cost=round(supplier.unit_cost * 0.02, 4),
                        stockout_cost=round(supplier.unit_cost * 1.0, 4),
                    )
                )
        return states

    def _sample_num_disruptions(self, rng: np.random.Generator) -> int:
        if self._spec.difficulty == 1:
            return 1
        return int(rng.integers(self._spec.min_disruptions, self._spec.max_disruptions + 1))
