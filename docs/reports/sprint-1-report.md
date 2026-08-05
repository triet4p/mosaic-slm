# Sprint 1 Report — Phase 1: Environment and Single-Agent Baseline

**Ngày:** 2026-08-05
**Sprint:** [Sprint 1](../sprint-plans/sprint-1.md)
**Trạng thái:** Hoàn thành — 16/16 tasks

## 1. Mục tiêu

Xây supply-chain simulator deterministic, episode generator, tool API và
single-agent baseline; chứng minh bài toán đo lường được end-to-end.

## 2. Kết quả xây dựng

| Thành phần | Mô tả | Module |
| --- | --- | --- |
| Domain models | Product, Warehouse, Supplier, SKUState, PO, Shipment, Disruption... | `src/mosaic/simulator/models.py` |
| Demand generator | base + seasonality + noise, deterministic theo seed, hỗ trợ spike/drop | `src/mosaic/simulator/demand.py` |
| Supplier behavior | outcome (seed, order_id) hash-based, hủy đơn, trễ theo reliability | `src/mosaic/simulator/supplier.py` |
| Simulator core | vòng lặp ngày: inbound → shipments → fulfill demand → costs | `src/mosaic/simulator/core.py` |
| Action executor | 6 business actions, discriminated union, validation | `src/mosaic/simulator/actions.py` |
| Reward | `R = -(Σ 6 cost) + η·FillRate`, deterministic | `src/mosaic/simulator/reward.py` |
| Disruption engine | 6 loại sự kiện, difficulty 1–3, recurring (L3) | `src/mosaic/simulator/disruptions.py` |
| Episode generator | 30 ngày, 3 kho, 2 supplier, 20 SKU, 3 region; dev set 100 episodes | `src/mosaic/simulator/episodes.py` |
| Replay | save/load JSONL, run bit-identical, không mutate config | `src/mosaic/simulator/replay.py` |
| Tool API | 10 tool (5 read + 5 compute), JSON schema, simulate_action what-if | `src/mosaic/tools/api.py` |
| Heuristic baselines | do-nothing, reorder-point, greedy transfer, oracle (look-ahead) | `src/mosaic/simulator/baselines.py` |
| Single-agent runner | observe → decide → tools → actions → trajectory step | `src/mosaic/agents/runner.py` |
| Trajectory store | JSONL per episode, state snapshot mỗi bước | `src/mosaic/agents/trajectory.py` |
| Evaluation | NTC normalization, fill rate, stockout rate | `src/mosaic/evaluation/metrics.py` |

**Tests:** 152 tests (148 unit + 4 integration) — toàn bộ pass; ruff + pyright strict sạch.

## 3. Kết quả đánh giá — Dev set 100 episodes (difficulty 2)

Agent = single-agent runner với rule-based backbone (reorder-point + greedy transfer).

| Policy | Mean cost | Mean fill rate | Mean stockout rate |
| --- | ---: | ---: | ---: |
| Do-nothing | 48 441 | 0.132 | 0.868 |
| Reorder-point (baseline) | 17 644 | — | — |
| Oracle (best-reference) | 16 980 | — | — |
| **Agent (rule-based)** | **18 117** | **0.891** | **0.109** |

**NTC trung bình = 1.862** (1 = ngang baseline, 0 = oracle).

Nhận xét:

- Agent rule-based cải thiện lớn so với do-nothing (cost 18k vs 48k) nhưng
  **chưa vượt được reorder-point thuần** (NTC > 1) — greedy transfer thêm
  transport cost và dịch chuyển holding mà chưa đủ bù lợi ích. Đây là
  baseline hợp lý để so sánh với LLM agent ở Phase 2+.
- Oracle (best-reference: min cost của họ policy look-ahead buffer 0–2 +
  reorder-point) thấp hơn baseline ~4% — config khởi tạo đã khá tối ưu cho
  tỷ lệ holding/stockout hiện tại (1:50).
- `data/results/evaluation_report.json` lưu metric từng episode.
- `data/trajectories/trajectory_ep0.jsonl` (~3.8MB) là trajectory mẫu với
  state snapshot mỗi bước — dữ liệu nền cho SFT/RL (Phase 6+).

## 4. Exit Criteria Phase 1

| Tiêu chí | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Episode replay được | ✅ | `test_replay_*`, replay từ dev set bit-identical |
| Reward deterministic | ✅ | `test_reward_is_deterministic_*`, cùng state → cùng reward |
| Ground truth rõ ràng | ✅ | Cost ledger 6 thành phần + fill rate, truy vết được từng ngày |
| Agent hoàn thành end-to-end task | ✅ | Runner 30 ngày: 796 actions, fill rate 0.914 (episode 0) |

## 5. Bài học kỹ thuật (đã log vào `.agents/memory/lessons-learned.md`)

1. Sprint-plan status line phải gộp vào commit feature (không commit 1 dòng riêng).
2. Simulator phải deep-copy config objects — replay không được mutate episode.
3. argparse cần typed `Namespaces` cho pyright strict.
4. Quirk pydantic: `model_dump_json` serialize tuple key thành chuỗi và
   không validate ngược — trajectory dùng mã hóa `sku|wh` riêng.

## 6. Giới hạn và việc tiếp theo

**Giới hạn Phase 1:**

- Oracle là họ policy look-ahead, không phải optimal solver.
- Backbone LLM chưa chạy được (cần vLLM — Phase 4); `OpenAICompatibleBackbone`
  đã có interface nhưng chưa có server thật.
- Action không hợp lệ từ backbone sẽ raise (chưa có cơ chế retry/ghi lỗi).
- Chưa có multi-agent (Phase 2), memory (Phase 3), serving (Phase 4).

**Bước tiếp theo (Sprint 2 — Phase 2):**

- Demand, Inventory, Supplier-Risk, Coordinator agents (4 logical roles).
- Message schema + orchestrator, chạy tuần tự trước.
- So sánh single-agent vs multi-agent quality + token cost.
