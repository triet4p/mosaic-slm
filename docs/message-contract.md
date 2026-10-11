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
  `get_inventory_state`, `get_shipments`, `simulate_action`.
- `supplier_risk`: `calculate_order_cost`, `get_open_orders`,
  `get_supplier_status`, `simulate_action`.
- `coordinator`: toan quyen 10 tool da trien khai.

`ToolEvidence` chi chap nhan 10 tool cua `ToolRunner` (Sprint 1, task 10).
Memory tools Phase 3 (`query_shared_graph`, `write_private_memory`,
`propose_shared_fact`) chua trien khai nen bi tu choi o bien validate; tool
la bi tu choi tuong tu.

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
