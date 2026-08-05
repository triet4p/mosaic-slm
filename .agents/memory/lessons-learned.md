# Lessons Learned

## [2026-08-05] Separate one-line commit for sprint-plan status churn history

**Symptom:** Commit log đầy các commit 1 dòng kiểu `docs: mark sprint 1 task N done` ngay sau mỗi feature commit, làm lịch sử nhiễu và khó đọc.

**Root cause:** Quy trình implement-atomic-task yêu cầu cập nhật sprint plan (đánh dấu `[x]`) nhưng được thực hiện thành commit riêng thay vì gộp vào commit của chính feature đó.

**Fix / workaround:** Đánh dấu task done trong sprint plan và commit CÙNG với feature (cùng một commit: `git add src/... tests/... docs/sprint-plans/sprint-1.md && git commit -m "feat(...): ..."`). Với lịch sử đã bị nhiễu: `git rebase -i` + đổi `pick` thành `fixup` cho các commit chỉ chứa dòng status (xác định bằng `git log --format='%h %p %s'` để chắc parent là feat commit). Có thể tự động hóa bằng GIT_SEQUENCE_EDITOR.

**Watch out for:** Mọi commit update sprint plan (PLAN.md, sprint-N.md) đều phải nằm trong commit của công việc tương ứng. Ngoại lệ: commit chứa nội dung thực (tạo plan mới, tạo sprint plan) thì giữ riêng.

## [2026-08-05] Simulator mutated episode config via shared object references

**Symptom:** `run_episode()` chạy 2 lần trên cùng Episode cho kết quả khác nhau; `Simulator.advance_day` làm hỏng `episode.config` (on_hand, current_lead_time, current_travel_time bị thay đổi).

**Root cause:** `Simulator.__init__` đưa thẳng object từ `SimulatorConfig` vào state (SKUState, Supplier, TransportRoute) — config và state share cùng tham chiếu, mutation của simulator rò ngược vào config. Chiều ngược lại cũng có: `_find_route` đọc từ `self._config.routes` thay vì `self._state.routes`, nên disruption cập nhật route trong state không ảnh hưởng tới dispatch (chỉ lộ ra sau khi đã deep-copy fix chiều 1).

**Fix / workaround:** Deep-copy mọi object mutable khi khởi tạo state: `s.model_copy(deep=True)` cho sku_states/suppliers/routes. Mọi đọc/ghi phải đi qua state (`self._state.*`), không bao giờ đọc lại từ config sau khi đã khởi tạo.

**Watch out for:** Bất kỳ module nào nhận config/object rồi mutate (simulator, agent, tool) đều phải copy trước. Replay cùng episode phải luôn chạy được nhiều lần bit-identical — có test `test_run_episode_does_not_mutate_episode_config` bảo vệ.

## [2026-08-05] argparse Namespace attributes are Unknown under pyright strict

**Symptom:** `args.episode_file`, `args.index` bị pyright strict báo `reportUnknownVariableType`/`reportUnknownArgumentType`, làm toàn bộ biến downstream thành Unknown.

**Root cause:** `parser.parse_args()` trả về `argparse.Namespace` — class không khai báo attribute nên pyright strict coi mọi field là Unknown.

**Fix / workaround:** Khai báo subclass `class Args(argparse.Namespace): episode_file: str; index: int` và gọi `parser.parse_args(namespace=Args())`.

**Watch out for:** Mọi script dùng argparse trong repo này phải dùng typed Namespace pattern này.
