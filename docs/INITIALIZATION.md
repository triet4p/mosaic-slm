# MOSAIC-SLM — PHASE 0 DECISIONS

## 0. Project Statement

**MOSAIC-SLM** là hệ thống Multi-Agent Small Language Model chạy trên một GPU consumer, trong đó nhiều logical agent sử dụng chung một base model, có private memory và shared temporal knowledge graph, nhằm giải quyết các bài toán ra quyết định nhiều bước dưới giới hạn latency, VRAM, token và compute budget.

Use case đầu tiên của hệ thống là:

> **Multi-Agent Inventory Rebalancing under Supply and Demand Disruptions**

---

# 1. Quyết định về Business Scenario

## 1.1. Scenario được chọn

Không xây dựng toàn bộ supply chain ngay từ đầu.

MVP chỉ tập trung vào:

> **Điều phối tồn kho giữa nhiều kho khi nhu cầu thay đổi và nhà cung cấp giao hàng không ổn định.**

Hệ thống vận hành một mạng lưới gồm:

* 3 kho hàng.
* 2 nhà cung cấp.
* 20 sản phẩm.
* 3 vùng khách hàng.
* 1–2 tuyến vận chuyển giữa mỗi cặp kho.
* Chu kỳ mô phỏng theo ngày.
* Mỗi episode kéo dài 30 ngày.

## 1.2. Các disruption chính

Phiên bản đầu hỗ trợ sáu loại sự kiện:

1. Nhu cầu tăng đột biến.
2. Nhu cầu giảm bất thường.
3. Supplier giao hàng chậm.
4. Supplier hủy đơn.
5. Một kho thiếu hàng.
6. Tuyến vận chuyển bị chậm hoặc tăng chi phí.

Chưa đưa vào MVP:

* Sản xuất.
* Manufacturing scheduling.
* Dynamic pricing.
* Hàng dễ hỏng.
* Đa tiền tệ.
* Hàng nghìn SKU.
* Tối ưu tuyến xe chi tiết.
* Đàm phán thực với supplier.

## 1.3. Business question

Ở mỗi ngày mô phỏng, hệ thống phải trả lời:

> Có cần đặt thêm hàng, chuyển hàng giữa các kho, đổi supplier, điều chỉnh safety stock hay giữ nguyên trạng thái không?

---

# 2. Quyết định về Agent Roles

MVP sử dụng **4 logical agent**, không dùng 6–8 agent ngay từ đầu.

## 2.1. Demand Agent

Trách nhiệm:

* Phân tích demand history.
* Phát hiện demand spike hoặc demand drop.
* Dự báo nhu cầu ngắn hạn.
* Ước lượng mức độ bất định.
* Publish demand event quan trọng vào shared memory.

## 2.2. Inventory Agent

Trách nhiệm:

* Theo dõi inventory position.
* Tính days of supply.
* Phát hiện stockout risk.
* Đề xuất chuyển hàng giữa các kho.
* Tính chi phí lưu kho và thiếu hàng.

## 2.3. Supplier and Risk Agent

Gộp Supplier Agent và Risk Agent trong MVP.

Trách nhiệm:

* Theo dõi lead time.
* Đánh giá supplier reliability.
* Phát hiện disruption.
* Truy xuất các disruption tương tự.
* Đề xuất đặt hàng hoặc đổi supplier.

## 2.4. Coordinator Agent

Trách nhiệm:

* Chọn agent cần gọi.
* Quản lý dependency giữa các agent.
* Kiểm soát token và tool-call budget.
* Tổng hợp recommendation.
* Chọn business action cuối cùng.
* Quyết định thời điểm dừng.

## 2.5. Agent architecture

Các agent:

* Không dùng model riêng biệt.
* Không chạy bốn model đồng thời.
* Dùng chung một base SLM.
* Phân biệt bằng system prompt, tool permission, memory namespace và role metadata.
* LoRA riêng cho từng agent chỉ được thêm ở giai đoạn SFT.

---

# 3. Quyết định về Environment State

State của simulator tại thời điểm (t):

[
S_t =
{
I_t,
D_t,
P_t,
H_t,
L_t,
R_t,
O_t,
E_t
}
]

Trong đó:

* (I_t): tồn kho hiện tại.
* (D_t): demand hiện tại và lịch sử demand.
* (P_t): purchase orders đang mở.
* (H_t): shipment đang vận chuyển.
* (L_t): lead time của supplier.
* (R_t): supplier reliability.
* (O_t): customer orders chưa hoàn thành.
* (E_t): disruption events.

## 3.1. Dữ liệu mỗi SKU tại mỗi kho

```text
sku_id
warehouse_id
on_hand
reserved
available
in_transit
backordered
safety_stock
reorder_point
holding_cost
stockout_cost
```

## 3.2. Dữ liệu supplier

```text
supplier_id
supported_skus
base_lead_time
current_lead_time
unit_cost
order_cost
reliability_score
capacity
active_disruptions
```

## 3.3. Dữ liệu shipment

```text
shipment_id
source
destination
sku_id
quantity
dispatch_time
expected_arrival
actual_arrival
transport_cost
status
```

---

# 4. Quyết định về Action Space

## 4.1. Business action space

Coordinator chỉ được chọn trong tập action hữu hạn:

```text
NO_ACTION

PLACE_ORDER(
  supplier_id,
  warehouse_id,
  sku_id,
  quantity
)

TRANSFER_STOCK(
  source_warehouse,
  destination_warehouse,
  sku_id,
  quantity
)

SWITCH_SUPPLIER(
  old_supplier,
  new_supplier,
  sku_id
)

ADJUST_SAFETY_STOCK(
  warehouse_id,
  sku_id,
  new_level
)

ACCEPT_BACKORDER(
  warehouse_id,
  sku_id,
  quantity
)

REQUEST_MORE_INFORMATION(
  agent_role,
  information_type
)
```

## 4.2. Agent coordination action space

Coordinator có các action:

```text
CALL_DEMAND_AGENT
CALL_INVENTORY_AGENT
CALL_SUPPLIER_RISK_AGENT
CALL_AGENTS_IN_PARALLEL
QUERY_SHARED_MEMORY
REQUEST_REVISION
MAKE_FINAL_DECISION
STOP
```

## 4.3. Memory action space

Mỗi agent có thể chọn:

```text
READ_PRIVATE
WRITE_PRIVATE
QUERY_SHARED
PROPOSE_SHARED_FACT
UPDATE_SHARED_FACT
INVALIDATE_SHARED_FACT
NO_MEMORY_ACTION
```

Các operation như entity merge, graph compaction và forgetting chưa để LLM quyết định trong MVP. Chúng được thực hiện bằng rule-based Memory Curator.

---

# 5. Quyết định về Reward

## 5.1. Business reward

Reward chính được tính hoàn toàn từ simulator, không dùng LLM-as-a-judge:

[
R_{\text{business}}
===================

-\left(
C_{\text{stockout}}
+
C_{\text{holding}}
+
C_{\text{ordering}}
+
C_{\text{transport}}
+
C_{\text{late}}
+
C_{\text{switch}}
\right)
+
B_{\text{service}}
]

Trong đó:

* (C_{\text{stockout}}): chi phí thiếu hàng.
* (C_{\text{holding}}): chi phí lưu kho.
* (C_{\text{ordering}}): chi phí đặt hàng.
* (C_{\text{transport}}): chi phí vận chuyển.
* (C_{\text{late}}): penalty giao trễ.
* (C_{\text{switch}}): chi phí đổi supplier.
* (B_{\text{service}}): bonus duy trì service level.

## 5.2. Service-level bonus

[
B_{\text{service}}
==================

\eta \cdot \text{FillRate}
]

với:

[
\text{FillRate}
===============

\frac{\text{Demand fulfilled immediately}}
{\text{Total demand}}
]

## 5.3. Compute-aware reward

Ở giai đoạn serving hoặc RL:

[
R_{\text{total}}
================

R_{\text{business}}
-\lambda_1 C_{\text{tokens}}
-\lambda_2 C_{\text{latency}}
-\lambda_3 C_{\text{GPU}}
-\lambda_4 C_{\text{communication}}
]

Trong Phase 0, các trọng số chưa cố định vĩnh viễn. Giá trị khởi đầu:

```text
lambda_tokens        = 0.001
lambda_latency       = 0.01
lambda_gpu           = 0.01
lambda_communication = 0.005
```

Các giá trị này sẽ được normalize lại sau khi thu thập baseline.

## 5.4. Nguyên tắc reward

* Business reward là mục tiêu chính.
* Compute reward là penalty phụ.
* Không cho phép policy đạt latency thấp bằng cách bỏ qua task.
* Quyết định sai nghiêm trọng phải bị phạt lớn hơn lượng compute tiết kiệm được.
* Reward phải deterministic với cùng state, action và random seed.

---

# 6. Quyết định về Ontology v1

Ontology v1 được cố định, chưa cho agent tự sinh relation mới.

## 6.1. Entity types

```text
Product
Warehouse
Supplier
CustomerRegion
PurchaseOrder
CustomerOrder
Shipment
DemandEvent
Disruption
Observation
Forecast
Hypothesis
Recommendation
Decision
Outcome
Agent
```

## 6.2. Core relations

```text
STORED_AT
SUPPLIED_BY
ORDERED_FROM
SHIPPED_FROM
SHIPPED_TO
SERVES_REGION
AFFECTS
CAUSES
DELAYED_BY
HAS_FORECAST
HAS_RISK
SUPPORTS
CONTRADICTS
SIMILAR_TO
PROPOSED_BY
SELECTED_AS
RESULTED_IN
VALID_DURING
SUPERSEDES
```

## 6.3. Shared fact schema

```json
{
  "fact_id": "fact_000001",
  "subject_id": "supplier_A",
  "relation": "HAS_RISK",
  "object": {
    "type": "Disruption",
    "value": "delivery_delay"
  },
  "valid_from": 12,
  "valid_until": null,
  "observed_at": 12,
  "source_agent": "supplier_risk_agent",
  "source_reference": "tool_result_381",
  "confidence": 0.87,
  "status": "active",
  "visibility": "shared"
}
```

## 6.4. Epistemic status

Mọi memory item phải thuộc một trong các loại:

```text
OBSERVATION
FORECAST
HYPOTHESIS
RECOMMENDATION
DECISION
OUTCOME
```

Không được lưu hypothesis như fact đã được xác nhận.

## 6.5. Temporal behavior

* Fact có `valid_from`.
* Fact có thể có `valid_until`.
* Fact mới có thể `SUPERSEDES` fact cũ.
* Fact hết hiệu lực không bị xóa vật lý.
* Retrieval mặc định ưu tiên fact còn hiệu lực.
* Historical query vẫn có thể lấy fact cũ.

---

# 7. Quyết định về Memory Architecture

## 7.1. Private memory

MVP dùng:

* SQLite.
* Một namespace cho mỗi agent và episode.
* Dữ liệu dạng JSON.
* Không dùng vector database cho private working memory.

Private memory lưu:

* Current hypothesis.
* Recent tool results.
* Agent-specific summary.
* Unpublished observations.
* Confidence state.

## 7.2. Shared memory

MVP dùng:

> **Kuzu embedded graph database**

Lý do:

* Chạy local.
* Không cần server riêng.
* Phù hợp với project một máy.
* Dễ đóng gói và replay experiment.
* Có thể giữ graph schema rõ ràng.

Có thể thay bằng Neo4j sau này, nhưng không dùng Neo4j làm dependency bắt buộc trong MVP.

## 7.3. Vector retrieval

Phase đầu chưa dùng vector retrieval làm cơ chế chính.

Thứ tự triển khai:

1. Structured graph query.
2. Temporal filtering.
3. Rule-based ranking.
4. Sau đó mới thêm embedding index để làm hybrid retrieval.

## 7.4. Memory Curator

Memory Curator ban đầu là deterministic service, không phải LLM agent.

Nó chịu trách nhiệm:

* Schema validation.
* Duplicate detection.
* Confidence normalization.
* Temporal validity.
* Entity ID resolution.
* Fact supersession.
* Provenance enforcement.

Điều này giúp tách lỗi reasoning khỏi lỗi graph corruption.

---

# 8. Quyết định về Model

## 8.1. Base model chính

> **Qwen3-4B**

Lý do lựa chọn:

* Quy mô đủ nhỏ để chạy trên GPU 16GB.
* Có khả năng instruction following và structured output tốt.
* Cho phép nghiên cứu cả chế độ reasoning và non-reasoning.
* Có context native 32K, nhưng dự án không sử dụng toàn bộ context này trong MVP. Model card chính thức công bố context native 32.768 token.

## 8.2. Chế độ chạy mặc định

* Non-thinking mode.
* BF16 nếu runtime và memory profiling ổn định.
* `max_model_len = 8192`.
* Output tối đa 512 token/request.
* Tensor parallel size = 1.
* GPU memory utilization ban đầu = 0.85.
* Không CPU-offload model weights trong baseline.

Reasoning mode chỉ được bật cho Coordinator trong một ablation riêng, không dùng mặc định cho mọi agent.

## 8.3. Model fallback

Nếu Qwen3-4B phát sinh incompatibility hoặc overhead quá lớn:

> **Qwen2.5-3B-Instruct**

Model này có context tối đa 32.768 token và generation tối đa 8.192 token theo model card, nhưng dự án vẫn giới hạn runtime ở 8K để giữ chỗ cho KV cache và concurrency.

## 8.4. Quantization

### Baseline chính

* BF16.
* Không quantize trong benchmark đầu tiên.

Mục đích là tạo baseline kernel và memory dễ giải thích.

### Benchmark thứ hai

* AWQ hoặc GPTQ 4-bit với Marlin-compatible path.
* So sánh chất lượng, VRAM, TTFT và throughput.

RTX 4060 thuộc kiến trúc Ada; bảng hỗ trợ phần cứng hiện tại của vLLM cho biết Marlin với GPTQ/AWQ được hỗ trợ trên Ada.

### llama.cpp baseline

* GGUF Q4_K_M.
* Dùng để đối chứng quantized local inference và CPU/GPU hybrid.

---

# 9. Quyết định về GPU Budget

Target hardware:

```text
GPU: RTX 4060
VRAM: 16GB
GPU count: 1
Tensor parallel: 1
```

## 9.1. VRAM allocation target

Ngân sách thiết kế:

```text
Model weights              8–9 GB tối đa
KV cache                    3–5 GB
CUDA/runtime workspace      1–2 GB
LoRA/profiling/headroom     1–2 GB
```

Không thiết kế baseline để dùng hết 15.9GB VRAM.

Mục tiêu vận hành:

> Peak VRAM không vượt quá khoảng 14.5GB trong benchmark ổn định.

## 9.2. Context decision

Dù model hỗ trợ context dài hơn, MVP dùng:

```text
Default context: 4K
Maximum context: 8K
```

Lý do:

* Multi-agent cần concurrency hơn là một request cực dài.
* Shared memory phải được truy xuất có chọn lọc.
* Context dài quá mức sẽ che giấu vấn đề memory architecture.
* KV cache dài làm giảm batch capacity.

## 9.3. Concurrency target

Benchmark các mức:

```text
1
2
4
8
16 concurrent requests
```

Kỳ vọng vận hành thực tế ban đầu:

```text
4–8 active agent requests
```

Không đặt mục tiêu 32 request đồng thời trong Phase 0.

---

# 10. Quyết định về Serving Runtime

## 10.1. Runtime chính

> **vLLM**

Dùng cho:

* Continuous batching.
* KV-cache management.
* Prefix caching.
* Request scheduling.
* OpenAI-compatible API.
* LoRA serving.
* Serving telemetry.

## 10.2. Runtime đối chứng

> **llama.cpp / llama-server**

Dùng để benchmark:

* GGUF quantization.
* CPU-only inference.
* GPU offloading.
* Hybrid CPU–GPU inference.
* Low-concurrency local serving.

## 10.3. Ollama

Không dùng trong experimental core.

Ollama chỉ được phép dùng cho:

* Smoke test.
* Demo thủ công.
* Kiểm tra nhanh model.

Kết quả Ollama không được dùng làm benchmark serving chính.

---

# 11. Quyết định về Shared Prefix

Prompt được chia thành bốn tầng:

```text
Global system prefix
└── Ontology and common tool schemas
    └── Agent role prefix
        └── Episode-specific context
            └── Request-specific observations
```

## 11.1. Global prefix

Dùng chung cho mọi agent:

* Output rules.
* Shared ontology.
* Memory operation schema.
* Common tool schema.
* Safety và consistency rules.

## 11.2. Role prefix

Riêng cho từng agent:

* Role objective.
* Available tools.
* Private memory rules.
* Expected output.

## 11.3. Mục tiêu nghiên cứu

So sánh:

```text
No prefix caching
Global-prefix caching
Role-prefix caching
Hierarchical prefix caching
```

---

# 12. Quyết định về Tool Set

LLM không được tự tính toán các đại lượng chính bằng text reasoning nếu đã có tool.

MVP có các tool:

```text
get_inventory_state
get_demand_history
forecast_demand
get_supplier_status
get_open_orders
get_shipments
calculate_reorder_point
calculate_transfer_cost
calculate_order_cost
query_shared_graph
write_private_memory
propose_shared_fact
simulate_action
```

## 12.1. Forecasting tool

Demand forecast ban đầu dùng phương pháp deterministic hoặc statistical đơn giản:

* Moving average.
* Exponential smoothing.
* Seasonal baseline nếu cần.

Không để LLM tự dự báo từ danh sách số liệu thô.

## 12.2. Optimization baseline

Có một non-LLM heuristic baseline:

* Reorder-point policy.
* Safety-stock policy.
* Greedy stock transfer.
* Supplier selection theo expected cost.

Đây là baseline bắt buộc để biết LLM agent có thực sự tạo thêm giá trị hay không.

---

# 13. Quyết định về Episode Generation

## 13.1. Episode format

```text
Duration: 30 simulated days
Decision interval: 1 day
Products: 20
Warehouses: 3
Suppliers: 2
Customer regions: 3
Disruptions per episode: 1–5
```

## 13.2. Difficulty levels

### Level 1 — Deterministic

* Demand ổn định.
* Lead time cố định.
* Một disruption.
* Không có conflicting observations.

### Level 2 — Stochastic

* Demand có noise.
* Lead time biến động.
* Nhiều disruption.
* Một số dữ liệu bị trễ.

### Level 3 — Partial information

* Agent nhận dữ liệu không đồng thời.
* Một số observation mâu thuẫn.
* Cần dùng historical memory.
* Có recurring disruptions.

## 13.3. Dataset split

```text
Train:      1,000 episodes
Validation:   200 episodes
Test:         300 episodes
```

Trong MVP trước training:

```text
Development set: 100 episodes
```

Dataset được sinh bằng seed và config, không lưu toàn bộ dữ liệu thủ công.

---

# 14. Quyết định về Metrics

## 14.1. Primary business metric

> **Normalized Total Operating Cost**

[
NTC =
\frac{C_{\text{agent}}-C_{\text{oracle}}}
{C_{\text{baseline}}-C_{\text{oracle}}}
]

Ý nghĩa:

* (NTC = 0): đạt mức oracle.
* (NTC = 1): ngang heuristic baseline.
* (NTC > 1): kém hơn baseline.

## 14.2. Primary technical metric

> **Business Reward per GPU-second**

[
BRPG =
\frac{\text{Normalized Business Reward}}
{\text{GPU-seconds}}
]

## 14.3. Serving metrics

Bắt buộc ghi:

* TTFT p50/p95.
* End-to-end latency p50/p95.
* Queue time.
* Prefill tokens/s.
* Decode tokens/s.
* Aggregate tokens/s.
* Peak VRAM.
* GPU utilization.
* KV-cache occupancy.
* Prefix-cache hit rate.
* GPU-seconds/episode.

## 14.4. Agent metrics

* Agent calls per episode.
* Tool calls per episode.
* Messages per episode.
* Duplicate message ratio.
* Invalid tool-call ratio.
* Structured-output success rate.
* Coordinator revision count.
* Tokens per successful episode.

## 14.5. Memory metrics

* Fact precision.
* Fact recall.
* Duplicate fact ratio.
* Stale fact retrieval rate.
* Unsupported fact rate.
* Provenance completeness.
* Retrieval recall@k.
* Retrieved-fact utilization.
* Memory size per episode.
* Decision improvement from memory.

## 14.6. Success criteria cho MVP

MVP được xem là thành công khi:

1. Multi-agent đạt business cost không kém heuristic baseline quá 10%.
2. Shared memory cải thiện kết quả ở recurring-disruption test ít nhất 5%.
3. Structured tool-call success trên 95%.
4. Peak VRAM dưới 14.5GB.
5. Hệ thống chạy ổn định ở concurrency 4.
6. Prefix caching hoặc batching giảm episode latency ít nhất 15% so với sequential serving.
7. Mọi shared fact đều có provenance.

---

# 15. Quyết định về Baselines

Các baseline bắt buộc:

## Business baselines

```text
Do-nothing policy
Reorder-point heuristic
Greedy transfer heuristic
Oracle policy with full simulator state
```

## Agent baselines

```text
Single-agent, no memory
Single-agent, full context
Multi-agent, no persistent memory
Multi-agent, summary memory
Multi-agent, shared temporal graph
```

## Serving baselines

```text
Sequential generation
Concurrent requests without custom policy
vLLM default scheduler
Dependency-aware heuristic scheduler
Prefix-aware heuristic scheduler
```

## Learning baselines

```text
Prompt-only
SFT
SFT + preference optimization
SFT + memory RL
SFT + coordination RL
```

---

# 16. Quyết định về RL Scope

Không làm end-to-end RL ngay từ đầu.

## RL thứ nhất: Memory policy

Policy học:

* Có truy xuất memory không.
* Truy xuất loại memory nào.
* Có lưu observation không.
* Lưu private hay đề xuất shared.
* Có invalidate fact không.

Đây là RL experiment đầu tiên.

## RL thứ hai: Coordinator policy

Policy học:

* Gọi agent nào.
* Có chạy song song không.
* Có cần thêm thông tin không.
* Khi nào đưa ra quyết định.
* Khi nào dừng.

## RL thứ ba: Scheduler policy

Chỉ triển khai sau khi đã có serving traces ổn định.

Policy học:

* Request nào chạy tiếp.
* Request nào đang nằm trên critical path.
* Background job nào bị trì hoãn.
* Có nên ưu tiên prefix-cache hit không.

## Business decision RL

Không phải ưu tiên đầu tiên.

Business policy trước hết dùng:

* Rule-based action proposal.
* SLM selection.
* Simulator reward.

Chỉ huấn luyện business decision RL khi memory và coordination policy đã ổn định.

---

# 17. Quyết định về Training Strategy

Thứ tự cố định:

```text
Prompt baseline
→ Synthetic trajectory generation
→ SFT structured behavior
→ Preference optimization
→ Offline RL
→ Online RL trong simulator
```

## 17.1. SFT targets

* Tool selection.
* Tool arguments.
* Entity extraction.
* Relation extraction.
* Memory action.
* Agent routing.
* Structured final decision.

## 17.2. LoRA strategy

* Một base model.
* LoRA riêng cho Coordinator.
* Một LoRA dùng chung cho ba analyst agent trong thử nghiệm đầu.
* Chỉ tách LoRA theo từng role nếu ablation cho thấy có lợi.

Không fine-tune full model trên RTX 4060.

---

# 18. Quyết định về Repository Structure

```text
mosaic-slm/
├── README.md
├── pyproject.toml
├── configs/
│   ├── environment/
│   ├── agents/
│   ├── models/
│   ├── serving/
│   └── experiments/
├── src/
│   └── mosaic/
│       ├── simulator/
│       ├── agents/
│       ├── orchestration/
│       ├── memory/
│       │   ├── private/
│       │   ├── graph/
│       │   ├── retrieval/
│       │   └── curator/
│       ├── tools/
│       ├── serving/
│       │   ├── clients/
│       │   ├── scheduler/
│       │   ├── workload/
│       │   └── telemetry/
│       ├── training/
│       │   ├── sft/
│       │   ├── preference/
│       │   └── rl/
│       └── evaluation/
├── schemas/
│   ├── ontology/
│   ├── messages/
│   ├── tools/
│   └── trajectories/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── replay/
├── scripts/
│   ├── run_simulator.py
│   ├── run_agents.py
│   ├── launch_vllm.py
│   ├── generate_episodes.py
│   ├── benchmark_serving.py
│   └── evaluate.py
├── experiments/
├── dashboards/
├── docs/
└── data/
    ├── generated/
    ├── trajectories/
    └── results/
```

---

# 19. Quyết định về Reproducibility

Mọi experiment phải lưu:

```text
git commit
model identifier
model revision
runtime version
CUDA version
driver version
GPU name
random seed
environment config
agent config
prompt version
ontology version
scheduler version
memory snapshot
```

Mỗi episode phải replay được mà không cần gọi dịch vụ bên ngoài.

---

# 20. Phase 0 Exit Criteria

Phase 0 hoàn thành khi repository có đủ:

1. Problem specification.
2. Scenario definition.
3. State schema.
4. Action schema.
5. Reward function.
6. Ontology v1.
7. Agent role definitions.
8. Model và runtime decision.
9. Hardware budget.
10. Baseline matrix.
11. Metric definitions.
12. Repository skeleton.
13. Development dataset configuration.
14. Success criteria.

## Final locked configuration

```text
Project:
  MOSAIC-SLM

Use case:
  Inventory rebalancing under supply-demand disruptions

Hardware:
  1 × RTX 4060 16GB

Primary runtime:
  vLLM

Comparison runtime:
  llama.cpp

Base model:
  Qwen3-4B

Fallback model:
  Qwen2.5-3B-Instruct

Default precision:
  BF16

Quantized experiment:
  AWQ/GPTQ 4-bit

Context:
  4K default
  8K maximum

Agents:
  Demand
  Inventory
  Supplier-Risk
  Coordinator

Memory:
  SQLite private memory
  Kuzu shared temporal knowledge graph

Episode:
  30 days
  20 products
  3 warehouses
  2 suppliers

Primary business metric:
  Normalized Total Operating Cost

Primary system metric:
  Business Reward per GPU-second

First RL target:
  Memory policy

Second RL target:
  Coordinator policy

Third RL target:
  Serving scheduler policy
```

Phase 0 vì vậy được xem là **đã đủ quyết định để bắt đầu Phase 1**, không cần tiếp tục mở rộng phạm vi nghiệp vụ trước khi xây simulator.