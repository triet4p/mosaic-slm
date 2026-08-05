# data

Dữ liệu được sinh bởi script, không commit vào git:

- `generated/` — dataset episode (seed + config, không lưu thủ công).
- `trajectories/` — trajectory và telemetry của từng episode.
- `results/` — kết quả benchmark và evaluation.

Mọi episode phải replay được mà không cần dịch vụ bên ngoài.
