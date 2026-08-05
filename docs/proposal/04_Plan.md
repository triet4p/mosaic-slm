# 4. PLAN PROPOSAL

## Phase 0 — Problem Framing

### Mục tiêu

Khóa phạm vi dự án.

### Công việc

* Chọn một supply-chain scenario.
* Xác định action space.
* Xác định state.
* Xác định reward.
* Xác định ontology v1.
* Chọn model và GPU target.
* Chọn metric chính.

### Deliverables

* Problem specification.
* Environment specification.
* Ontology v1.
* Evaluation matrix.
* Repository skeleton.

---

## Phase 1 — Environment and Single-Agent Baseline

### Mục tiêu

Xây simulator và chứng minh bài toán có thể đo lường.

### Công việc

* Xây supply-chain simulator.
* Tạo episode generator.
* Xây tool API.
* Tạo single-agent baseline.
* Ghi trajectory.
* Đánh giá business reward.

### Deliverables

* Simulator chạy được.
* Dataset episode đầu tiên.
* Single-agent baseline.
* Báo cáo metric cơ bản.

### Exit Criteria

* Episode replay được.
* Reward deterministic.
* Ground truth rõ ràng.
* Agent có thể hoàn thành end-to-end task.

---

## Phase 2 — Multi-Agent Baseline

### Mục tiêu

Chia bài toán thành nhiều vai trò.

### Công việc

* Xây Demand, Inventory, Supplier, Risk và Coordinator Agent.
* Định nghĩa message schema.
* Tạo orchestrator.
* Chạy tuần tự trước.
* So sánh single-agent với multi-agent.

### Deliverables

* Multi-agent workflow.
* Agent communication logs.
* Baseline coordination policy.
* Báo cáo chất lượng và token cost.

### Exit Criteria

* Multi-agent hoạt động ổn định.
* Vai trò không trùng lặp hoàn toàn.
* Có thể xác định agent nào đóng góp vào quyết định.

---

## Phase 3 — Private and Shared Memory

### Mục tiêu

Đưa memory dài hạn vào hệ thống.

### Công việc

* Thêm private memory.
* Xây shared ontology.
* Xây temporal knowledge graph.
* Thêm provenance.
* Thêm graph retrieval.
* Thêm Memory Curator.
* So sánh các memory baseline.

### Deliverables

* Private memory service.
* Shared temporal graph.
* Retrieval pipeline.
* Memory benchmark.
* Ablation report.

### Exit Criteria

* Memory cải thiện kết quả ở episode lặp lại.
* Fact có thể truy ngược về source.
* Hệ thống xử lý được stale và contradictory facts.

---

## Phase 4 — Serving Baseline and Profiling

### Mục tiêu

Hiểu rõ inference workload trước khi tối ưu.

### Công việc

* Deploy model bằng vLLM.
* Đo prefill và decode.
* Tạo agent workload generator.
* Đo concurrency 1, 2, 4, 8 và 16.
* Đo TTFT, throughput, VRAM và KV cache.
* Benchmark với llama.cpp.
* Profile bằng Nsight hoặc PyTorch Profiler.

### Deliverables

* Serving benchmark suite.
* Workload traces.
* Bottleneck analysis.
* vLLM–llama.cpp comparison.

### Exit Criteria

* Phân biệt được prefill bottleneck và decode bottleneck.
* Có baseline latency và throughput ổn định.
* Biết giới hạn concurrency trên GPU 16GB.

---

## Phase 5 — Serving Optimization

### Mục tiêu

Tối ưu nhiều agent trên một GPU.

### Công việc

* Bật continuous batching.
* Thiết kế shared prompt prefix.
* Bật prefix caching.
* Thử Multi-LoRA.
* Tối ưu context compression.
* Thiết kế dependency-aware scheduling.
* Thiết kế prefix-aware batching.
* Chuyển graph query và tool workload sang CPU.

### Deliverables

* Optimized serving runtime.
* Custom scheduler prototype.
* Prefix-sharing strategy.
* Performance report.

### Exit Criteria

* Throughput tăng rõ ràng so với sequential baseline.
* Episode latency giảm.
* Không làm giảm đáng kể business reward.
* VRAM nằm trong giới hạn phần cứng.

---

## Phase 6 — SFT and Preference Optimization

### Mục tiêu

Giảm phụ thuộc vào prompt và cải thiện structured behavior.

### Công việc

* Sinh training trajectories.
* Làm sạch dữ liệu.
* SFT tool use.
* SFT graph extraction.
* SFT agent routing.
* Xây preference pairs.
* Huấn luyện preference model hoặc DPO.

### Deliverables

* SFT dataset.
* LoRA adapters.
* Preference dataset.
* Model evaluation report.

### Exit Criteria

* Tool-call accuracy tăng.
* Structured output ổn định.
* Memory fact precision tăng.
* Token usage giảm hoặc không tăng đáng kể.

---

## Phase 7 — Reinforcement Learning for Memory

### Mục tiêu

Học policy lưu và truy xuất memory.

### Công việc

* Định nghĩa memory state.
* Định nghĩa memory action space.
* Thiết kế reward.
* Thu thập offline trajectories.
* Huấn luyện offline policy.
* So sánh với rule-based memory.
* Sau đó mới thử online RL trong simulator.

### Deliverables

* Memory policy.
* Reward analysis.
* Rule-based versus learned comparison.
* Memory ROI report.

### Exit Criteria

* Learned policy giảm memory noise.
* Retrieval cost giảm.
* Downstream reward giữ nguyên hoặc tăng.
* Policy không làm graph phát triển mất kiểm soát.

---

## Phase 8 — RL for Communication and Scheduling

### Mục tiêu

Học cách sử dụng agent và compute budget.

### Công việc

* Huấn luyện communication policy.
* Huấn luyện agent-selection policy.
* Huấn luyện scheduling policy.
* Gắn reward với GPU-seconds và episode latency.
* Kiểm tra fairness và starvation.
* So sánh với heuristic scheduler.

### Deliverables

* Learned coordination policy.
* Learned scheduling policy.
* Compute-aware evaluation.
* Ablation study.

### Exit Criteria

* Ít agent call hơn.
* Ít message lặp lại hơn.
* Episode latency giảm.
* Business quality không bị suy giảm.

---

## Phase 9 — Business Decision RL

### Mục tiêu

Tối ưu quyết định supply chain cuối cùng.

### Công việc

* Huấn luyện action policy trong simulator.
* Thử các disruption scenario.
* Kiểm tra generalization.
* Kiểm tra robustness với dữ liệu sai hoặc thiếu.
* So sánh với rule-based optimizer.

### Deliverables

* Business decision policy.
* Scenario benchmark.
* Robustness report.
* Cost and service-level analysis.

### Exit Criteria

* Tổng business cost thấp hơn baseline.
* Policy xử lý được scenario chưa thấy.
* Quyết định có thể giải thích bằng evidence trong memory.

---

## Phase 10 — Final Integration and Research Output

### Mục tiêu

Đóng gói thành hệ thống và kết quả nghiên cứu hoàn chỉnh.

### Công việc

* Tích hợp simulator, agent, memory, serving và RL.
* Chạy toàn bộ benchmark.
* Thực hiện ablation.
* Viết technical report.
* Xây demo dashboard.
* Đóng gói reproducible experiments.
* Chuẩn hóa configuration và seed.

### Deliverables

* End-to-end system.
* Reproducible benchmark.
* Technical paper hoặc thesis report.
* Demo.
* Public repository.
* Architecture documentation.
* Model card và limitations.