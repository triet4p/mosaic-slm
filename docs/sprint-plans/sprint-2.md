# Sprint 2 Plan — Phase 2: Multi-Agent Baseline

## Sprint Goal

Xây baseline bốn logical agent (Demand, Inventory, Supplier-Risk, Coordinator), orchestration tuần tự không persistent memory, và so sánh single-agent với multi-agent trên cùng episode bằng business metrics và token usage thực khi chạy SLM.

## Scope and Boundaries

- Nguồn quyết định: [INITIALIZATION.md](../INITIALIZATION.md) §2, §4.2, §12, §14, §15; kế thừa [Sprint 1 report](../reports/sprint-1-report.md).
- Bốn role dùng chung backbone; khác nhau bởi prompt, tool permission và role metadata. Không chạy bốn model riêng.
- Chạy sequential trước; không thêm parallel scheduler, memory/Kuzu/SQLite, LoRA, training hoặc GPU profiling trong sprint này.
- Dùng simulator, business actions, tool calculations, trajectory và evaluation conventions hiện có; giữ single-agent baseline hoạt động.
- Rule-based execution dùng để kiểm chứng coordination deterministic, không thay thế bằng chứng SLM quality hoặc token cost. SLM smoke dùng endpoint OpenAI-compatible thực nếu khả dụng; triển khai/profiling vLLM vẫn thuộc Phase 4. Không dùng mock response, token estimate hoặc số 0 để giả kết quả inference.

## Workflow and Authority

Status legend: [ ] pending / [~] in progress / [x] done.

Workflow: tasks 1–9 **full** (message/role/coordination/runtime/evaluation contracts); mỗi task cần worker behavior smoke → evidence-reviewer PASS với frozen manifest → Bronze-only exact-snapshot checkpoint trước task độc lập kế tiếp. Không shared atomic batch. Task 10 **light**, kiểm tra report với records và dữ liệu thực; không evidence gate riêng và không commit được giao. Giao **deep-reviewer** cuối sprint cho tương tác role permissions → recommendations → Coordinator → execution → metrics, sau mọi gate/checkpoint đã giao.

- Routing authority: người dùng đã cấp Main auto-routing cho Sprint 2 tasks 1–10, gồm correction/replacement trong scope. Không cấp model override hoặc advisor consent ngoài policy.
- Repository/worktree/Git `-C` root: `F:/ai-ml/mosaic-slm`; Git directory: `F:/ai-ml/mosaic-slm/.git`; target ref hiện đọc được: `refs/heads/main`. Worker/reviewer/executor phải kiểm tra lại identity/ref/base và ownership trước hành động liên quan.
- Commit/checkpoint authority: người dùng đã cấp Bronze-only exact-snapshot checkpoint cho full-route tasks 1–9, mỗi task một boundary riêng. Implementation workers không commit; task 10 light không được giao commit; không push.
- Staging ownership được cấp: chỉ task-owned changes và Main-authored `docs/PLAN.md`, `docs/sprint-plans/sprint-2.md` khi ownership được xác minh và nằm trong frozen manifest. Không foreign staged changes, mixed user hunks, blanket staging, history rewrite hoặc bypass hook/signing.
- Records: `artifacts/sprint-2/tasks/`, `reviews/`, `checkpoints/`; detailed handoff và Main temporary state tách riêng dưới `handoffs/` hoặc `state/`. Plan chỉ giữ outcome và links.

## Atomic Tasks

- [~] 1. **Message and role contract:** định nghĩa structured requests, analyst recommendations và Coordinator decisions; role identities, episode/day correlation và tool evidence references; công bố JSON schema trong `schemas/messages/` theo pattern hiện có. Acceptance: request → recommendation → final decision serialize/validate round-trip; business action payload tái dùng action schema hiện có; role/tool constraints không cho analyst thực thi business action; smoke chứng minh consumer đọc được schema và correlated messages. Gate: full, checkpoint riêng.
- [ ] 2. **Demand Agent:** short-horizon forecast và demand-risk recommendation bằng read/forecast tools; không tự thực thi action. Acceptance: demand spike/drop tạo recommendation có evidence từ tool và correlation đúng; dữ liệu calculation không do text reasoning tự tính; deterministic mode tái chạy được. Depends: 1. Gate: full, checkpoint riêng.
- [ ] 3. **Inventory Agent:** inventory position, stockout risk và đề xuất rebalancing qua inventory/reorder/transfer tools. Acceptance: recommendation chỉ dùng SKU/kho/tuyến hợp lệ; thiếu hàng và transfer tradeoff truy vết được về tool evidence; analyst không mutate simulator. Depends: 1. Gate: full, checkpoint riêng.
- [ ] 4. **Supplier-Risk Agent:** supplier status, reliability/lead-time/disruption và đề xuất ordering/supplier choice qua supplier/order tools. Acceptance: disruption làm thay đổi recommendation có evidence; không bịa supplier hoặc mutate simulator; role-specific tool permissions được thực thi. Depends: 1. Gate: full, checkpoint riêng.
- [ ] 5. **Coordinator policy:** gọi analyst, tổng hợp recommendation, request revision/information khi cần, make final decision và stop bằng coordination contract; chỉ Coordinator quyết định business actions. Acceptance: conflict có quyết định truy vết được tới recommendation; bounded calls/revisions không tạo vòng lặp vô hạn; budget và stop behavior được exercise; không dùng memory operations chưa triển khai. Depends: 2–4. Gate: full, checkpoint riêng.
- [ ] 6. **Shared backbone integration:** structured role prompts và OpenAI-compatible inference cho bốn role dùng cùng backbone/client, reuse existing single-agent conventions; deterministic non-LLM mode được ghi nhãn riêng. Acceptance: actual configured endpoint có thể nhận role-specific prompts/schema; provider token usage được giữ nguyên, thiếu usage là unavailable chứ không giả số; malformed output không thành decision thành công; single-agent contract vẫn hoạt động. Depends: 1–5. Gate: full, checkpoint riêng.
- [ ] 7. **Sequential episode runner and communication trajectory:** nối simulator → analyst calls → Coordinator → action execution → recorded day; lưu communication và evidence attribution cùng trajectory. Acceptance: episode 30 ngày hoàn tất ở deterministic mode, lặp cùng seed cho cùng action/cost và không mutate episode config; logs chứng minh analyst nào đóng góp và chỉ Coordinator output đi vào ActionExecutor; saved trajectory tái kiểm tra được không gọi server. Depends: 5–6. Gate: full, checkpoint riêng.
- [ ] 8. **Comparative evaluation and CLI:** chạy single-agent và multi-agent trên cùng episode/config/seed; report NTC, cost breakdown, fill/stockout rate, agent/tool/message calls và observed token usage. Acceptance: report phân biệt backbone mode và measured/unavailable usage; so sánh dev set 100 episodes ở deterministic mode; CLI thực chạy và xuất report đọc lại được; không đổi denominator/baseline normalization để làm đẹp kết quả. Depends: 7. Gate: full, checkpoint riêng.
- [ ] 9. **Real SLM comparison:** chạy single-agent và multi-agent với cùng actual model/endpoint trên bộ episode/seed được ghi rõ; lưu model/config/prompt metadata, business metrics, observed token usage và attribution. Acceptance: tối thiểu một episode 30 ngày cho mỗi mode hoàn tất trên endpoint thực, report không nhầm rule-based với SLM hoặc proxy token với measured token; mọi run thất bại và sample limits được ghi rõ. Không tuyên bố vượt baseline từ smoke nhỏ. Depends: 8; prerequisite: endpoint/model truy cập được. Gate: full, checkpoint riêng.
- [ ] 10. **Sprint report and documentation reconciliation:** report Phase 2 outcome/limits với links tới actual evaluation và passed records; đồng bộ root README, document index và status wording; sửa Sprint 1 do-nothing aggregate bằng dữ liệu JSON lưu thực. Acceptance: completed/unverified criteria phân biệt rõ; không claim multi-agent LLM/token comparison nếu task 9 chưa đạt; report số liệu khớp artifact; mọi link đọc được. Depends: 1–9. Gate: light, no per-task review/commit assigned; final differential review vẫn bắt buộc.

## Exit Criteria (Phase 2)

- Bốn logical role có nhiệm vụ khác biệt và structured communication; tool permissions được thực thi, không chỉ khai báo trong prompt.
- Sequential multi-agent episode chạy ổn định; quyết định/business action truy ngược được về analyst recommendations và tool evidence.
- Deterministic coordination tái chạy trên cùng episode mà không mutate config; single-agent baseline không bị thay thế hoặc hỏng.
- Có comparative report trên cùng dataset/config với business quality và observed call/message/token cost; real SLM comparison được chứng minh riêng, không substitute bằng deterministic mode.
- Tasks 1–9 đạt acceptance, evidence PASS và Bronze checkpoint riêng; task 10 verified; assigned final differential review không còn actionable findings.

## Decisions / Blockers

- Sprint 1 đã hoàn thành theo [sprint plan](sprint-1.md) và [report](../reports/sprint-1-report.md); root README còn stale, sẽ được reconciliation trong task 10.
- `INITIALIZATION.md` khóa bốn role (Supplier và Risk gộp); proposal năm role không phải contract hiện hành.
- Endpoint/model availability cho tasks 6/9 chưa được worker xác minh. Không coi optional dependencies hoặc client interface là bằng chứng có server; không đánh dấu Phase 2 complete nếu chưa có real SLM evidence đã giao.
- Task 1 đang khởi động; chưa có evidence PASS hoặc checkpoint Sprint 2. Endpoint/model availability vẫn là prerequisite chưa xác minh cho phần SLM.
