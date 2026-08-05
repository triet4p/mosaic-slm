# Global Project Plan

## Overview

**MOSAIC-SLM** là hệ thống Multi-Agent Small Language Model chạy trên một GPU consumer (RTX 4060 16GB), trong đó nhiều logical agent dùng chung một base model (Qwen3-4B), có private memory và shared temporal knowledge graph, nhằm giải quyết bài toán **Inventory Rebalancing under Supply and Demand Disruptions** dưới giới hạn latency, VRAM, token và compute budget.

Các quyết định Phase 0 đã được khóa tại [`INITIALIZATION.md`](./INITIALIZATION.md). Chi tiết đề xuất ban đầu: [`docs/proposal/`](./proposal/).

## Milestones

- [x] **Phase 0 — Problem Framing:** khóa scenario, action space, reward, ontology v1, model/runtime, metrics, repository skeleton.
- [ ] **Phase 1 — Environment and Single-Agent Baseline:** simulator chạy được, episode generator, tool API, single-agent baseline, trajectory, business reward đo được.
- [ ] **Phase 2 — Multi-Agent Baseline:** Demand, Inventory, Supplier-Risk, Coordinator agents + orchestrator.
- [ ] **Phase 3 — Private and Shared Memory:** SQLite private memory, Kuzu temporal KG, Memory Curator, retrieval.
- [ ] **Phase 4 — Serving Baseline and Profiling:** vLLM deployment, workload generator, TTFT/throughput/VRAM profiling, so sánh llama.cpp.
- [ ] **Phase 5 — Serving Optimization:** batching, prefix caching, Multi-LoRA, dependency-aware scheduling.
- [ ] **Phase 6 — SFT and Preference Optimization.**
- [ ] **Phase 7 — RL for Memory Policy.**
- [ ] **Phase 8 — RL for Communication and Scheduling.**
- [ ] **Phase 9 — Business Decision RL.**
- [ ] **Phase 10 — Final Integration and Research Output.**

## Active Sprints

- [Sprint 1](sprint-plans/sprint-1.md) — Phase 1: Environment and Single-Agent Baseline — *Status: Not Started*

## Completed Sprints

- None yet.

## Backlog / Future Work

- Phase 2: 4 logical agents (Demand, Inventory, Supplier-Risk, Coordinator) dùng chung base model.
- Phase 3: private/shared memory, temporal KG (Kuzu), Memory Curator, retrieval pipeline.
- Phase 4–5: vLLM serving baseline, profiling, prefix caching, custom scheduler.
- Phase 6–9: SFT → preference optimization → RL (memory → coordinator → scheduler → business decision).
- Phase 10: benchmark hoàn chỉnh, ablation, technical report, demo.
