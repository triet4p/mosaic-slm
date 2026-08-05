# 2. ARCHITECTURE PROPOSAL

## 2.1. Architecture Principles

Hệ thống được thiết kế theo các nguyên tắc:

1. Một base model, nhiều logical agent.
2. Agent specialization bằng prompt, tool và LoRA.
3. Private memory tách biệt với shared memory.
4. Shared memory có cấu trúc và provenance.
5. Coordinator không cần xử lý mọi dữ liệu chi tiết.
6. Inference scheduler nhận biết dependency giữa agent.
7. Mọi hành động đều có log để phục vụ training và evaluation.
8. Domain simulator cung cấp ground truth và reward.

---

## 2.2. High-Level Architecture

```text
                         ┌─────────────────────┐
                         │ Supply Chain        │
                         │ Simulator           │
                         └──────────┬──────────┘
                                    │
                             Events / State
                                    │
                         ┌──────────▼──────────┐
                         │ Agent Orchestrator  │
                         │ and Coordinator     │
                         └───────┬─────┬───────┘
                                 │     │
                 ┌───────────────┘     └────────────────┐
                 │                                      │
        ┌────────▼────────┐                    ┌────────▼────────┐
        │ Agent Runtime   │                    │ Tool Runtime    │
        │ vLLM            │                    │ DB / Forecast / │
        │ Shared SLM      │                    │ Optimization    │
        └────────┬────────┘                    └────────┬────────┘
                 │                                      │
        ┌────────▼──────────────────────────────────────▼───────┐
        │                   Memory Layer                       │
        │                                                     │
        │ Private Memory │ Shared Temporal KG │ Vector Index  │
        └────────┬────────────────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │ Trajectory and  │
        │ Telemetry Store │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │ Training and    │
        │ Evaluation      │
        └─────────────────┘
```

---

## 2.3. Main Components

### A. Supply Chain Simulator

Simulator tạo ra environment theo episode.

State có thể gồm:

```text
inventory levels
warehouse capacities
supplier lead times
supplier reliability
customer demand
open purchase orders
shipments
transport routes
operating costs
disruption events
```

Simulator phải:

* Có deterministic seed.
* Có ground truth.
* Cho phép replay episode.
* Tạo event theo thời gian.
* Tính business reward.
* Hỗ trợ nhiều mức độ khó.

### B. Agent Orchestrator

Orchestrator chịu trách nhiệm:

* Theo dõi state của episode.
* Duy trì dependency graph.
* Gọi agent.
* Chuyển message giữa agent.
* Áp dụng token budget.
* Quyết định agent nào được chạy song song.
* Thu thập output.
* Gửi quyết định cuối vào simulator.

Orchestrator ban đầu dùng rule-based policy; sau đó có thể thay bằng learned policy.

### C. Shared SLM Runtime

Một base SLM được dùng cho nhiều agent.

Phân biệt agent bằng:

* System prompt.
* Role identifier.
* Tool access.
* Private memory.
* LoRA adapter.
* Output schema.

Ví dụ:

```text
Base model: Qwen-family 3B–4B

Adapters:
- coordinator
- demand
- inventory
- supplier
- risk
```

Runtime chính:

* vLLM.
* OpenAI-compatible API.
* Prefix caching.
* Continuous batching.
* Multi-LoRA nếu khả thi.
* Metrics endpoint.
* Request metadata theo agent và episode.

### D. Tool Runtime

LLM không trực tiếp tính toán mọi thứ.

Các tool có thể gồm:

* Query inventory database.
* Query supplier history.
* Demand forecast.
* Safety stock calculator.
* Cost estimator.
* Route planner.
* Similar-event retrieval.
* Knowledge graph query.
* Mathematical optimizer.

Tool call phải dùng structured schema để dễ đánh giá.

### E. Private Memory

Private memory lưu:

* Working hypotheses.
* Recent observations.
* Agent-specific summaries.
* Role-specific preferences.
* Confidence.
* Tool results chưa cần chia sẻ.

Private memory có thể triển khai ban đầu bằng:

* JSON documents.
* SQLite.
* Key–value store.

### F. Shared Temporal Knowledge Graph

Shared graph lưu tri thức chung.

Node types:

```text
Product
Warehouse
Supplier
Shipment
Order
DemandEvent
Disruption
Forecast
Observation
Hypothesis
Decision
Outcome
Agent
```

Edge types:

```text
SUPPLIES
STORED_AT
SHIPPED_TO
AFFECTS
CAUSES
SUPPORTS
CONTRADICTS
SIMILAR_TO
PROPOSED_BY
RESULTED_IN
VALID_DURING
```

Mỗi fact có metadata:

```json
{
  "subject": "supplier_A",
  "relation": "HAS_LEAD_TIME",
  "object": "12_days",
  "valid_from": "episode_20_step_3",
  "valid_until": null,
  "source": "supplier_agent",
  "confidence": 0.87,
  "visibility": "shared",
  "evidence_ids": ["tool_result_381"]
}
```

### G. Memory Curator

Memory Curator có thể là rule-based hoặc một logical agent.

Nhiệm vụ:

* Validate schema.
* Deduplicate facts.
* Merge entities.
* Detect contradiction.
* Update confidence.
* Expire stale facts.
* Preserve provenance.
* Reject unsupported claims.

### H. Serving Scheduler

Scheduler nhận request với metadata:

```text
agent_role
episode_id
dependency_depth
deadline
estimated_prompt_tokens
estimated_output_tokens
prefix_id
adapter_id
priority
```

Các scheduler baseline:

* FIFO.
* Shortest-request-first.
* Priority-based.
* Dependency-aware.
* Prefix-aware.
* Hybrid dependency-prefix-aware.

### I. Telemetry Store

Ghi lại:

* Prompt tokens.
* Output tokens.
* Queue time.
* TTFT.
* Decode latency.
* GPU utilization.
* VRAM usage.
* KV-cache usage.
* Prefix-cache hit.
* Tool latency.
* Agent messages.
* Memory operations.
* Business reward.
* Final decisions.

Dữ liệu này được dùng cho:

* Benchmark.
* SFT dataset.
* Preference dataset.
* Offline RL.
* Error analysis.

---

## 2.4. End-to-End Data Flow

1. Simulator phát sinh event.
2. Coordinator nhận state tóm tắt.
3. Coordinator tạo task cho các agent.
4. Scheduler xếp các request.
5. Agent truy xuất private/shared memory.
6. Agent gọi tool khi cần.
7. Agent tạo observation hoặc hypothesis.
8. Memory Curator quyết định cập nhật shared graph.
9. Coordinator tổng hợp kết quả.
10. Coordinator chọn business action.
11. Simulator thực thi action.
12. Simulator trả outcome và reward.
13. Trajectory Store ghi lại toàn bộ episode.
14. Outcome được dùng để cập nhật memory và training data.
