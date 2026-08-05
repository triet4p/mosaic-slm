# Lessons Learned

## [2026-08-05] Separate one-line commit for sprint-plan status churn history

**Symptom:** Commit log đầy các commit 1 dòng kiểu `docs: mark sprint 1 task N done` ngay sau mỗi feature commit, làm lịch sử nhiễu và khó đọc.

**Root cause:** Quy trình implement-atomic-task yêu cầu cập nhật sprint plan (đánh dấu `[x]`) nhưng được thực hiện thành commit riêng thay vì gộp vào commit của chính feature đó.

**Fix / workaround:** Đánh dấu task done trong sprint plan và commit CÙNG với feature (cùng một commit: `git add src/... tests/... docs/sprint-plans/sprint-1.md && git commit -m "feat(...): ..."`). Với lịch sử đã bị nhiễu: `git rebase -i` + đổi `pick` thành `fixup` cho các commit chỉ chứa dòng status (xác định bằng `git log --format='%h %p %s'` để chắc parent là feat commit). Có thể tự động hóa bằng GIT_SEQUENCE_EDITOR.

**Watch out for:** Mọi commit update sprint plan (PLAN.md, sprint-N.md) đều phải nằm trong commit của công việc tương ứng. Ngoại lệ: commit chứa nội dung thực (tạo plan mới, tạo sprint plan) thì giữ riêng.
