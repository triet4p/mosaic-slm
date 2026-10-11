# Decisions

## [2026-10-11] Keep supplier-risk order context explicit and disruptions current-day only

**Decision:** Keep the shared `AnalystRequest` schema unchanged, pass order targets as explicit `sku_id=...`, `warehouse_id=...`, `quantity=...` instruction tokens, and add current-day active supplier disruption details to the existing `get_supplier_status` result instead of adding a new tool.

**Alternatives considered:** Add task-specific typed fields to every analyst request; add a separate disruption-history tool; let Supplier-Risk evaluate `simulate_action` or future scheduled disruption data.

**Reason:** Task 4 needs an explicit supported SKU, destination, and quantity plus observable delay/cancellation evidence, while tasks 1–3 already established the shared message interface and the current status tool advertises disruption status but exposed no active event details. Filtering status evidence to the simulator's current day supplies the needed evidence without exposing future schedules or evaluating hidden outcomes.

**Consequences:** A Coordinator must include the explicit key-value tokens before Supplier-Risk can propose an order; missing or ambiguous values produce `NO_ACTION`. Supplier-Risk can cite only currently active delay/cancellation events and cannot claim historical cancellations that the open-order tool excludes. The added status field is available to other consumers, and `simulate_action` remains forbidden to this analyst.

## [2026-10-11] Mask unobserved order and shipment outcomes at the read-tool boundary

**Decision:** Publish `actual_arrival` and order `received_quantity` from `get_open_orders`/`get_shipments` only after the simulator day has observed them; pending rows keep planned `quantity`, `expected_arrival`, and `status` with masked actuals, and inventory inbound screening uses the observable expected arrival.
**Alternatives considered:** Filter future arrivals only in Supplier-Risk summaries while leaving raw tool payloads unchanged; delete open-order/shipment evidence wholesale; invent an inferred actual arrival from expected values.
**Reason:** Serialized `ToolEvidence.result` is Coordinator-facing, so a summary-only filter still leaks hidden realizations; deleting evidence removes legitimate in-flight planning signals, and invented arrivals fabricate unobserved outcomes. Masking at the shared tool surface preserves expected ETA/status for planning while keeping simulator ground truth untouched.
**Consequences:** Supplier-Risk and Inventory recommendations carry identical published evidence for identical observable histories even when hidden actuals differ; arrived outcomes retain truthful actual values once observed. Future tool consumers must treat `expected_arrival` as the planning estimate and `actual_arrival` as observed-only.

## [2026-10-11] Coordinate sequentially with observable need, conflict resolution, and bounded stop

**Decision:** Implement the Sprint 2 Coordinator as a boring sequential policy (Demand, then Inventory, then conditional Supplier-Risk from an observable safety-stock shortfall) with same-destination conflict resolution by cheapest tool quote, feasibility validation on a deep simulator copy, and explicit `NO_ACTION` bounded stops under `max_analyst_calls`/`max_revisions`.
**Alternatives considered:** Call all analysts unconditionally and concatenate proposals; add a parallel scheduler or shared-backbone/LLM abstraction in this task; re-ask analysts with free-form new instructions until agreement.
**Reason:** Task 5 needs evidence-backed synthesis without inventing supplier context, blind concatenation double-supplies destinations, and later tasks own parallel execution, persistence, backbone, and evaluation; bounded conditional calls with deterministic conflict/feasibility rules keep the baseline honest and replayable.
**Consequences:** Supplier-Risk only runs with valid observed SKU/warehouse/quantity context; conflicts resolve deterministically with traceable quotes; live simulator/config state is never mutated by the policy; exhausted budgets yield truthful `NO_ACTION` instead of success.
