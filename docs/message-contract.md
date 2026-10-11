# Message and Role Contract (Sprint 2 · Task 1)

Hop dong trao doi co cau truc giua bon logical role dung chung backbone
(Phase 0, `INITIALIZATION.md` §2.5). Tai lieu nay mo ta cach dung; dinh nghia
thuc thi nam trong `src/mosaic/agents/messages.py`, schema da cong bo nam
trong `schemas/messages/`.

## Roles

- `demand`: phan tich lich su nhu cau, du bao ngan han va demand risk; chi
  doc/forecast, khong thuc thi action.
- `inventory`: theo doi vi the ton kho, rui ro stockout, de xuat rebalancing.
- `supplier_risk`: theo doi lead time/reliability/disruption, de xuat dat
  hang hoac doi supplier.
- `coordinator`: giao viec, tong hop de xuat, chot business action cuoi cung.

Bon role khac nhau boi prompt, tool permission va role metadata; khong chay
bon model rieng.

## Messages

- `AnalystRequest` (`message_type = "request"`): Coordinator giao viec cho mot
  analyst trong mot ngay mo phong; mang `episode_seed`, `day`, `instruction`.
- `AnalystRecommendation` (`message_type = "recommendation"`): analyst tra de
  xuat kem `proposed_actions`, `evidence` tu tool, va `confidence`.
- `CoordinatorDecision` (`message_type = "decision"`): Coordinator chot
  `final_actions` kem `rationale` va tham chieu `recommendation_ids`.

Tuong quan mot vong trao doi duoc kiem tra bang `check_correlation`: khop
`episode_seed`/`day`, `recommendation.request_id` tro toi `request.message_id`,
va `decision.recommendation_ids` tham chieu `recommendation.message_id`.

## Execution authority

- `proposed_actions` cua analyst chi la de xuat. Khong dua truc tiep vao
  `ActionExecutor`.
- Chi `final_actions` cua `CoordinatorDecision` moi duoc thuc thi. Dung
  `executable_actions()` de lay actions thuc thi duoc; ham nay raise
  `TypeError` cho moi message cua analyst.
- Business action tai dung union `BusinessAction` hien co trong
  `src/mosaic/simulator/actions.py`; khong co union thu hai.

## Tool permissions

Bang quyen duoc cong bo trong `schemas/messages/role-tools.json` va kiem
chung bang `role_tool_permissions()`:

- `demand`: `forecast_demand`, `get_demand_history` (read-only).
- `inventory`: `calculate_reorder_point`, `calculate_transfer_cost`,
  `forecast_demand`, `get_demand_history`, `get_inventory_state`,
  `get_open_orders`, `get_shipments` (read-only; no `simulate_action`).
- `supplier_risk`: `calculate_order_cost`, `get_inventory_state`,
  `get_open_orders`, `get_shipments`, `get_supplier_status` (read-only; inventory
  state validates an explicitly requested SKU/warehouse only; no simulation or execution).
- `coordinator`: toan quyen 10 tool da trien khai.

`ToolEvidence` chi chap nhan 10 tool cua `ToolRunner` (Sprint 1, task 10).
Memory tools Phase 3 (`query_shared_graph`, `write_private_memory`,
`propose_shared_fact`) chua trien khai nen bi tu choi o bien validate; tool
la bi tu choi tuong tu.

`get_supplier_status` includes `active_disruption_events` on each supplier row.
The field contains only supplier delay/cancellation events active on the
simulator's current day; future scheduled disruptions are not exposed as
current evidence.

`get_open_orders` and `get_shipments` publish only currently observable
fields. Planned values (`quantity`, `expected_arrival`, `status`,
`placed_at`/`dispatch_time`) remain available for in-flight planning. Each
row's `actual_arrival` is exposed only after the simulator day has observed
it; pending future arrivals stay `None`, and an unobserved order's
`received_quantity` stays `0`. Inventory inbound screening uses the
observable `expected_arrival` estimate and does not consume hidden actual
outcomes.

## Published schemas

- `schemas/messages/request.schema.json`
- `schemas/messages/recommendation.schema.json`
- `schemas/messages/decision.schema.json`
- `schemas/messages/envelope.schema.json` (discriminated union qua
  `message_type`, dung cho consumer doc wire JSON)
- `schemas/messages/role-tools.json`

Tai sinh bang:

```powershell
uv run python -c "from pathlib import Path; from mosaic.agents.messages import export_schemas; export_schemas(Path('schemas/messages'))"
```

## Minimal consumer example

```python
from mosaic.agents.messages import (
    AnalystRecommendation,
    CoordinatorDecision,
    MessageAdapter,
    check_correlation,
    executable_actions,
)

request = ...  # AnalystRequest da khoi tao
recommendation = AnalystRecommendation(...)  # kem evidence tu ToolRunner
decision = CoordinatorDecision(...)  # tham chieu recommendation.message_id
check_correlation(request, recommendation, decision)

# Doc wire JSON bang envelope:
restored = MessageAdapter.validate_json(decision.model_dump_json())
assert isinstance(restored, CoordinatorDecision)

# Chi decision moi thuc thi duoc:
for action in executable_actions(decision):
    executor.apply(action)
executable_actions(recommendation)  # raise TypeError
```

## Demand Agent

`DemandAgent` (`mosaic.agents.demand`) is the deterministic, non-LLM baseline
for the demand analyst. It accepts an `AnalystRequest` and the active
`ToolRunner`, then returns the existing `AnalystRecommendation` contract:

```python
from mosaic.agents.demand import DemandAgent

recommendation = DemandAgent().recommend(request, tool_runner)
```

The request must target `demand` and match the active episode seed/day (the
first-day request may be made before day 1 has been advanced). The analyst
records `get_demand_history` for up to seven observed days and requests a
three-day projection from `forecast_demand` (its five-day calculation window).
Forecast values come only from that tool. A current-day value at least 1.5 times
the prior observed mean is flagged as a spike; at most 0.7 times is a drop.
Risk classification is unavailable without both a current-day and earlier
observation. The analyst never reads future scheduled demand, mutates simulator
state/configuration, or executes a business action. Demand-role permissions
reject `simulate_action` at both tool-call and recommendation-evidence
validation boundaries. Its `NO_ACTION` proposal leaves any business action to
the Coordinator after analyst review.

## Inventory Agent

`InventoryAgent` (`mosaic.agents.inventory`) is the deterministic, non-LLM
baseline for the inventory analyst. It accepts an `AnalystRequest` and active
`ToolRunner`, returning the existing `AnalystRecommendation` contract.

The request must target `inventory` and match the active episode seed/day (a
day-1 request may be made before the first advance). The analyst reads observed
demand history, requests a moving-average projection from `forecast_demand`,
reads inventory, open orders, and shipments, and obtains route-specific reorder
points and transfer quotes from their existing tools. It counts pending
purchase orders and in-flight shipments arriving by the candidate transfer ETA
when assessing inventory position. Route validity, reorder thresholds, and
transfer cost come from tool results; it does not read future demand or route
configuration directly.

The first forecast value for each observed SKU/region pair is summed into a
SKU-level daily demand estimate and passed to the reorder-point tool. Because
the exposed tools do not provide a region-to-warehouse mapping, that aggregate
is used as a conservative per-warehouse screening bound rather than reading
warehouse routing/configuration directly.

A transfer is proposed only when destination position is below the
tool-calculated reorder point and the source has available surplus above its
own route-specific reorder point. The proposed quantity is capped at the
destination's tool-reported safety-stock shortfall. The exact transfer-cost
quote must be lower than the estimated one-day stockout exposure (`quantity ×
destination stockout cost per unit`). This cost comparison is a transparent
screening estimate, not a simulated outcome.

The transfer-cost tool returns the simulator's current route charge, including
any active `TRANSPORT_DELAY` cost multiplier. It shares the simulator's
authoritative transport-cost calculation with shipment dispatch, so the quote
matches the current route's action cost. This remains a screening estimate,
not a simulated outcome.

Missing routes, insufficient source stock, inbound stock, unavailable forecasts,
or a quote at least as large as the estimate result in a truthful `NO_ACTION`
proposal. Inventory-role permissions reject `simulate_action` both when the
analyst calls tools and when validating recommendation evidence. The analyst
only proposes actions: it has no action executor and never mutates simulator
state/configuration.

```python
from mosaic.agents.inventory import InventoryAgent

recommendation = InventoryAgent().recommend(request, tool_runner)
```

## Supplier-Risk Agent

`SupplierRiskAgent` (`mosaic.agents.supplier_risk`) is the deterministic,
non-LLM supplier analyst. Like the Demand and Inventory agents, it accepts an
`AnalystRequest` and active `ToolRunner`, and returns the existing
`AnalystRecommendation` contract.

The request must target `supplier_risk` and match the active seed/day (a day-1
request may be made before the first advance). The agent reads supplier status,
open orders, and shipments through their existing tools. `get_supplier_status`
provides current lead time, base lead time, reliability, per-order capacity,
SKU support, and `active_disruption_events`; the latter contains only supplier
delay/cancellation events active on the simulator's current day. Future
scheduled disruptions are not used. The analyst reports current shipment
statuses without using hidden future arrival outcomes. Open-order reads exclude
completed/cancelled orders, so the agent reports active cancellation risk from
current disruption evidence and does not claim a historical cancellation it
cannot observe.

`AnalystRequest.instruction` carries actionable context as explicit key-value
tokens, for example `sku_id=SKU001; warehouse_id=WH1; quantity=50;
supplier_id=SUP_A`. The optional `supplier_id` is a preferred supplier. The
analyst never infers an order quantity. Before proposing an order, it confirms
the exact SKU/warehouse pair with `get_inventory_state`, verifies SKU support
and requested quantity against actual supplier status/capacity, and gets the
selected candidates' cost from `calculate_order_cost`. It does not reimplement
the simulator's order-cost formula or use inventory values to derive quantity.

An explicitly requested supplier remains preferred unless an eligible
alternative is no worse on observed lead time and reliability and strictly
better on at least one observed risk signal. A supplier with an active
cancellation disruption is not proposed. If no supplier was specified, viable
options are selected deterministically by active delay, current lead time,
reliability, tool-quoted order cost, then supplier ID. Missing/ambiguous context,
an unconfirmed SKU/warehouse, unsupported SKU, insufficient capacity, failed
required evidence, or missing cost quote returns an explicit `NO_ACTION`.

The agent records every tool result in recommendation evidence, remains
correlated to the request/episode/day, and has no `ActionExecutor` access.
Supplier-Risk permissions reject `simulate_action` at both tool-call and
recommendation-evidence validation boundaries; only the Coordinator can execute
an accepted action.

```python
from mosaic.agents.supplier_risk import SupplierRiskAgent

recommendation = SupplierRiskAgent().recommend(request, tool_runner)
```

