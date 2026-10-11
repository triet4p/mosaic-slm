# Global Project Plan

## Overview

**MOSAIC-SLM** là hệ thống Multi-Agent Small Language Model chạy trên một GPU consumer (RTX 4060 16GB), trong đó nhiều logical agent dùng chung một base model (Qwen3-4B), có private memory và shared temporal knowledge graph, nhằm giải quyết bài toán **Inventory Rebalancing under Supply and Demand Disruptions** dưới giới hạn latency, VRAM, token và compute budget.

Các quyết định Phase 0 đã được khóa tại [`INITIALIZATION.md`](./INITIALIZATION.md). Chi tiết đề xuất ban đầu: [`docs/proposal/`](./proposal/).

## Milestones

- [x] **Phase 0 — Problem Framing:** khóa scenario, action space, reward, ontology v1, model/runtime, metrics, repository skeleton.
- [x] **Phase 1 — Environment and Single-Agent Baseline:** hoàn thành theo [Sprint 1 report](reports/sprint-1-report.md); simulator, replay, tool API, single-agent baseline, trajectory và business metrics đã có.
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

- [Sprint 2](sprint-plans/sprint-2.md) — Phase 2: Multi-Agent Baseline — *Status: In Progress; task 1 complete, task 2 active*. Tasks 1–9 full với evidence PASS và Bronze checkpoint riêng; task 10 light; final differential review được giao.

## Completed Sprints

- [Sprint 1](sprint-plans/sprint-1.md) — Phase 1: Environment and Single-Agent Baseline — 16/16 tasks hoàn thành; [report và giới hạn](reports/sprint-1-report.md).

## Backlog / Future Work

- Phase 2: triển khai qua Sprint 2; bốn logical roles, structured messages, sequential orchestration, comparative business/token evaluation. Endpoint SLM thực là prerequisite chưa xác minh; không coi rule-based coordination là bằng chứng LLM.
- Phase 3: private/shared memory, temporal KG (Kuzu), Memory Curator, retrieval pipeline.
- Phase 4–5: vLLM serving baseline, profiling, prefix caching, custom scheduler.
- Phase 6–9: SFT → preference optimization → RL (memory → coordinator → scheduler → business decision).
- Phase 10: benchmark hoàn chỉnh, ablation, technical report, demo.
