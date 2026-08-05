# MOSAIC-SLM

**Multi-Agent Small Language Model** chạy trên một GPU consumer duy nhất (RTX 4060 16GB), trong đó nhiều logical agent dùng chung một base model, có private memory và shared temporal knowledge graph, nhằm giải quyết các bài toán ra quyết định nhiều bước dưới giới hạn latency, VRAM, token và compute budget.

Use case đầu tiên:

> **Multi-Agent Inventory Rebalancing under Supply and Demand Disruptions**

## Bối cảnh

Hệ multi-agent truyền thống gọi nhiều LLM hoặc gọi một model lớn nhiều lần, dẫn đến chi phí inference cao, latency tăng theo số agent, cạnh tranh GPU memory / KV cache và context ngày càng dài. MOSAIC-SLM thay thế bằng một SLM phục vụ nhiều logical agent trên cùng một runtime (vLLM), kết hợp:

- **Private memory** cho từng agent (SQLite).
- **Shared temporal knowledge graph** (Kuzu) có timestamp, confidence và provenance.
- **Learned policy** (SFT → preference optimization → RL) thay vì prompt cố định.

## Thiết lập đã khóa (Phase 0)

| Thành phần | Lựa chọn |
| --- | --- |
| Hardware | 1 × RTX 4060 16GB |
| Runtime chính | vLLM |
| Runtime đối chứng | llama.cpp |
| Base model | Qwen3-4B (fallback: Qwen2.5-3B-Instruct) |
| Precision | BF16 (thử nghiệm: AWQ/GPTQ 4-bit) |
| Context | 4K mặc định, 8K tối đa |
| Agents | Demand, Inventory, Supplier-Risk, Coordinator |
| Memory | SQLite private + Kuzu shared temporal KG |
| Episode | 30 ngày, 20 sản phẩm, 3 kho, 2 supplier |
| Business metric | Normalized Total Operating Cost (NTC) |
| System metric | Business Reward per GPU-second (BRPG) |
| RL targets | Memory policy → Coordinator policy → Scheduler policy |

## Repository Structure

```text
mosaic-slm/
├── pyproject.toml
├── configs/               # environment, agents, models, serving, experiments
├── src/mosaic/
│   ├── simulator/         # supply-chain environment
│   ├── agents/            # logical agents
│   ├── orchestration/     # orchestrator và coordinator
│   ├── memory/            # private/, graph/, retrieval/, curator/
│   ├── tools/             # tool runtime
│   ├── serving/           # clients/, scheduler/, workload/, telemetry/
│   ├── training/          # sft/, preference/, rl/
│   └── evaluation/
├── schemas/               # ontology, messages, tools, trajectories
├── tests/                 # unit, integration, replay
├── scripts/               # entry points
├── experiments/
├── dashboards/
├── data/                  # generated, trajectories, results (không commit)
└── docs/                  # tài liệu dự án
```

## Trạng thái hiện tại

- **Phase 0 (Problem Framing)**: hoàn thành — quyết định đầy đủ trong [`docs/INITIALIZATION.md`](./docs/INITIALIZATION.md).
- **Phase 1 (Environment and Single-Agent Baseline)**: chưa bắt đầu — repository skeleton sẵn sàng.

## Tài liệu

- [`docs/README.md`](./docs/README.md) — index tài liệu.
- [`docs/INITIALIZATION.md`](./docs/INITIALIZATION.md) — quyết định Phase 0.
- [`docs/proposal/`](./docs/proposal/) — đề xuất ban đầu (problem, architecture, technical, plan).

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Mọi experiment phải lưu đầy đủ metadata (git commit, model, runtime, seed, config, ontology version) và episode phải replay được — xem mục 19 trong `docs/INITIALIZATION.md`.
