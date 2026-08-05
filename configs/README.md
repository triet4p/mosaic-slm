# configs

Cấu hình YAML cho từng nhóm, mỗi thư mục chứa cấu hình mặc định:

- `environment/` — simulator (episode length, demand, suppliers, disruptions).
- `agents/` — role definitions, system prompts, tool permissions.
- `models/` — model id, precision, context length, adapter.
- `serving/` — vLLM/llama.cpp runtime và scheduler.
- `experiments/` — seed, difficulty level, baseline matrix, metric targets.

Cấu hình phải được khai báo tường minh trong metadata experiment (xem mục 19 `docs/INITIALIZATION.md`).
