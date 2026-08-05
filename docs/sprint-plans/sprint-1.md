# Sprint 1 Plan — Phase 1: Environment and Single-Agent Baseline

## Sprint Goal

Xây supply-chain simulator deterministic, episode generator, tool API và single-agent baseline để chứng minh bài toán đo lường được end-to-end.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] 1. **Domain models:** pydantic schemas cho Product, Warehouse, Supplier, CustomerRegion, SKUState, PurchaseOrder, CustomerOrder, Shipment, DisruptionEvent trong `src/mosaic/simulator/models.py`.
- [x] 2. **Demand generator:** sinh demand hằng ngày theo product/region (base + seasonality + noise), deterministic theo seed; hỗ trợ demand spike/drop từ disruption.
- [ ] 3. **Supplier behavior:** lead time (base/current), reliability score, capacity; logic giao trễ và hủy đơn theo disruption.
- [ ] 4. **Simulator core:** vòng lặp hằng ngày — fulfill demand → backorder/stockout → di chuyển shipment → nhận purchase order → cập nhật state; ghi cost đầy đủ.
- [ ] 5. **Action executor:** validate và áp dụng 6 business action (NO_ACTION, PLACE_ORDER, TRANSFER_STOCK, SWITCH_SUPPLIER, ADJUST_SAFETY_STOCK, ACCEPT_BACKORDER) với cost tương ứng.
- [ ] 6. **Reward function:** business reward theo công thức khóa (stockout + holding + ordering + transport + late + switch, cộng service bonus), deterministic với cùng state/action/seed.
- [ ] 7. **Disruption engine:** 6 loại sự kiện, lịch trình theo difficulty level 1–3, tích hợp vào episode.
- [ ] 8. **Episode generator:** sinh episode từ config (30 ngày, 20 SKU, 3 kho, 2 supplier, 1–5 disruption) theo seed; tạo dev set 100 episodes; cấu hình YAML trong `configs/environment/`.
- [ ] 9. **Replay:** serialize episode + tái chạy bit-identical không cần dịch vụ bên ngoài.
- [ ] 10. **Tool API:** read tools (get_inventory_state, get_demand_history, get_supplier_status, get_open_orders, get_shipments) và compute tools (forecast_demand, calculate_reorder_point, calculate_transfer_cost, calculate_order_cost, simulate_action).
- [ ] 11. **Heuristic baselines:** do-nothing, reorder-point, greedy transfer, oracle (full simulator state) — cần để normalize NTC.
- [ ] 12. **Single-agent runner:** vòng lặp agent (quan sát → tool call → action) với backbone pluggable; mặc định rule-based, interface OpenAI-compatible để swap vLLM ở Phase 4; ghi trajectory mỗi bước.
- [ ] 13. **Trajectory store:** lưu state/action/tool_call/reward mỗi bước ra JSONL theo episode trong `data/trajectories/`.
- [ ] 14. **Evaluation script:** tính NTC (normalize giữa oracle và heuristic baseline), cost breakdown, fill rate, stockout rate; xuất report trong `data/results/`.
- [ ] 15. **Tests:** unit (demand determinism, action validation, reward math, replay) + integration (một episode end-to-end hoàn chỉnh).
- [ ] 16. **Báo cáo Sprint 1:** metric cơ bản, xác nhận exit criteria Phase 1.

## Exit Criteria (Phase 1)

- Episode replay được (task 9).
- Reward deterministic (task 6).
- Ground truth rõ ràng — cost và fill rate truy vết được (task 4, 6).
- Agent hoàn thành end-to-end task (task 12, 14).

## Notes / Blockers

- **Backbone single-agent:** mặc định dùng rule-based (reorder-point + greedy transfer) vì vLLM serving thuộc Phase 4. Task 12 phải thiết kế interface pluggable để sau này swap bằng SLM client.
- **NTC phụ thuộc baselines:** task 14 cần kết quả oracle + heuristic từ task 11.
- **Memory chưa cần trong Sprint này:** Kuzu/SQLite thuộc Phase 3; Sprint 1 chỉ cần stateless single-agent.
- **Python:** 3.13 (requires-python >=3.12).
- Heuristic baseline là "non-LLM agent" bắt buộc trong INITIALIZATION.md §12.2 — nó đồng thời là backbone mặc định của single-agent runner.
