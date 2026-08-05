# 3. TECHNICAL PROPOSAL

## 3.1. Proposed Technology Stack

### Inference

* vLLM: runtime chính.
* llama.cpp: runtime benchmark đối chứng.
* Ollama: chỉ dùng cho smoke test hoặc demo đơn giản.

### Model

Giai đoạn đầu:

* SLM 1.5B–4B.
* BF16 hoặc FP16 nếu vừa VRAM.
* Context 4K–8K.

Giai đoạn sau:

* Model 7B–8B quantized.
* AWQ hoặc GPTQ cho vLLM.
* GGUF cho llama.cpp.

### Training

* PyTorch.
* Hugging Face Transformers.
* PEFT/LoRA.
* TRL hoặc framework tương đương.
* SFT trước.
* Preference optimization sau.
* Offline RL trước online RL.

### Memory

MVP:

* SQLite hoặc PostgreSQL.
* JSON facts.
* FAISS hoặc vector index nhẹ.

Giai đoạn sau:

* Neo4j, Kuzu hoặc graph database tương đương.
* Temporal graph layer.
* Hybrid graph–vector retrieval.

### Observability

* Prometheus.
* Grafana.
* NVIDIA Management Library.
* PyTorch Profiler.
* Nsight Systems ở giai đoạn tối ưu sâu.

---

## 3.2. Serving Proposal

### Baseline

Đầu tiên xây baseline với:

* Một model.
* Một agent.
* Một request mỗi lần.
* Không prefix cache.
* Không batching.
* Không LoRA.

Sau đó tăng dần:

1. Nhiều logical agent.
2. Concurrent requests.
3. Continuous batching.
4. Prefix caching.
5. Multi-LoRA.
6. Custom scheduler.
7. CPU–GPU hybrid workload.

### Workload Classes

#### Short memory query

```text
Input: 128–512 tokens
Output: 16–64 tokens
Priority: high
```

#### Analyst request

```text
Input: 1K–4K tokens
Output: 64–256 tokens
Priority: medium
```

#### Coordinator synthesis

```text
Input: 4K–8K tokens
Output: 128–512 tokens
Priority: critical path
```

#### Background summarization

```text
Input: 4K–16K tokens
Output: 128–512 tokens
Priority: low
```

### Serving Metrics

* Time to first token.
* Inter-token latency.
* End-to-end latency.
* Queue time.
* Prompt throughput.
* Decode throughput.
* Aggregate tokens per second.
* p50/p95/p99 latency.
* Peak VRAM.
* KV-cache occupancy.
* Prefix-cache hit rate.
* GPU utilization.
* GPU-seconds per episode.
* Completed episodes per hour.

### Custom Scheduling Objective

Scheduler tối ưu:

[
R_{\text{serving}} =
-\alpha L_{\text{episode}}
-\beta T_{\text{queue}}
-\gamma C_{\text{GPU}}
+\delta Q_{\text{task}}
]

Trong đó:

* (L_{\text{episode}}): thời gian hoàn thành episode.
* (T_{\text{queue}}): tổng queue time.
* (C_{\text{GPU}}): GPU-seconds.
* (Q_{\text{task}}): chất lượng quyết định.

---

## 3.3. Memory Proposal

### Memory Operations

```text
READ_PRIVATE
WRITE_PRIVATE
QUERY_SHARED
PROPOSE_SHARED_FACT
UPDATE_SHARED_FACT
INVALIDATE_FACT
MERGE_ENTITIES
SUMMARIZE
FORGET
NO_MEMORY_ACTION
```

### Retrieval Pipeline

1. Parse query.
2. Resolve entities.
3. Retrieve graph neighborhood.
4. Retrieve semantically similar facts.
5. Filter by time.
6. Rank by relevance, confidence và freshness.
7. Compress thành context.
8. Return provenance.

### Memory Evaluation

* Fact precision.
* Fact recall.
* Entity resolution accuracy.
* Relation accuracy.
* Temporal validity.
* Contradiction detection.
* Retrieval recall@k.
* Memory utilization rate.
* Memory growth.
* Duplicate ratio.
* Stale-fact ratio.
* Downstream decision improvement.
* Retrieval latency.

### Memory ROI

[
MemoryROI =
\frac{
Quality_{\text{with memory}}
----------------------------

Quality_{\text{without memory}}
}{
MemoryTokens + RetrievalLatency + StorageCost
}
]

---

## 3.4. Fine-Tuning Proposal

### Stage 1: Supervised Fine-Tuning

SFT cho các tác vụ rõ ràng:

* Entity extraction.
* Relation extraction.
* Tool selection.
* Tool argument generation.
* Fact validation.
* Agent routing.
* Structured decision output.

Trajectory format:

```text
state
role
available_tools
retrieved_memory
selected_action
tool_call
observation
memory_action
message
final_decision
```

### Stage 2: Preference Optimization

Tạo các cặp:

* Memory ngắn gọn so với memory chứa nhiều rác.
* Fact có provenance so với fact không có evidence.
* Message hữu ích so với message lặp lại.
* Tool call cần thiết so với tool call dư thừa.
* Quyết định cùng chất lượng nhưng ít token hơn.
* Retrieval đúng thời điểm so với retrieval quá sớm hoặc quá nhiều.

### Stage 3: Reinforcement Learning

Không huấn luyện mọi policy cùng lúc.

#### RL Phase A: Memory Policy

Policy học:

* Có lưu hay không.
* Lưu private hay shared.
* Truy xuất memory hay không.
* Cập nhật hoặc invalidate fact nào.
* Khi nào cần tổng hợp hoặc quên.

Reward:

[
R_{\text{memory}} =
aQ_{\text{downstream}}
+bP_{\text{fact}}
-cC_{\text{retrieval}}
-dC_{\text{storage}}
-eN_{\text{noise}}
]

#### RL Phase B: Communication Policy

Policy học:

* Có gửi message không.
* Gửi cho agent nào.
* Gửi chi tiết hay summary.
* Có cần escalate lên coordinator không.

Reward:

* Task quality tăng.
* Agent khác được unblock.
* Ít message hơn.
* Ít token hơn.
* Không lặp lại thông tin.

#### RL Phase C: Agent Scheduling Policy

Policy học:

* Agent nào chạy tiếp.
* Request nào được ưu tiên.
* Request nào chạy song song.
* Background task nào bị trì hoãn.
* Khi nào dừng một agent.

Reward:

* Episode latency thấp.
* GPU utilization cao.
* Ít starvation.
* Critical path hoàn thành sớm.
* Business quality không giảm.

#### RL Phase D: Business Decision Policy

Policy học:

* Đặt hàng.
* Chuyển kho.
* Đổi supplier.
* Điều chỉnh safety stock.
* Chấp nhận backorder.
* Không hành động.

Reward đến trực tiếp từ simulator.

---

## 3.5. Evaluation Proposal

### Baselines

#### Agent baseline

* Single-agent.
* Multi-agent không shared memory.
* Multi-agent với vector memory.
* Multi-agent với knowledge graph.
* Multi-agent với temporal knowledge graph.

#### Memory baseline

* No memory.
* Full context.
* Sliding window.
* Summary memory.
* Vector RAG.
* Graph retrieval.
* Hybrid graph–vector.

#### Scheduler baseline

* Sequential.
* FIFO.
* Continuous batching mặc định.
* Shortest job first.
* Priority scheduling.
* Dependency-aware.
* Prefix-aware.
* Learned scheduling policy.

#### Learning baseline

* Prompt-only.
* SFT.
* SFT + preference optimization.
* SFT + RL memory.
* SFT + RL communication.
* Full policy.

### Main Evaluation Dimensions

#### Task quality

* Decision correctness.
* Total business cost.
* Fill rate.
* Stockout.
* Late delivery.
* Recovery after disruption.

#### Agent quality

* Tool accuracy.
* Communication efficiency.
* Agent utilization.
* Duplicate reasoning.
* Coordinator correction rate.

#### Memory quality

* Fact correctness.
* Retrieval usefulness.
* Temporal consistency.
* Graph size.
* Provenance completeness.

#### System quality

* TTFT.
* End-to-end latency.
* Throughput.
* Peak VRAM.
* GPU-seconds.
* Tokens per successful episode.
* Quality per GPU-second.

### Primary Aggregate Metric

[
EfficiencyScore =
\frac{
NormalizedBusinessReward
}{
GPUSeconds + \lambda \cdot CPUSeconds
}
]

---

## 3.6. Expected Technical Contributions

1. Một kiến trúc private/shared temporal memory cho Multi-Agent SLM.
2. Một benchmark supply-chain nhiều episode có ground truth.
3. Một dependency-aware scheduler cho agent workload.
4. Một memory policy tối ưu theo task quality và compute cost.
5. Một bộ metric kết hợp business performance và serving efficiency.
6. Phân tích trade-off giữa graph memory, vector memory và full context.