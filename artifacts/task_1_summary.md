# Task Summary: Sprint 1 — Task 1: Domain Models

**Sprint:** Sprint 1 — Phase 1: Environment and Single-Agent Baseline
**Task:** Task 1 — Domain models

## Summary of Work

Tạo các pydantic v2 domain models cho supply-chain simulator trong `src/mosaic/simulator/models.py`, khớp với schema mục 3.1–3.3 và entity list mục 6.1 của `docs/INITIALIZATION.md`:

- Enums: `DisruptionType` (6 loại disruption), `ShipmentStatus`, `OrderStatus`, `CustomerOrderStatus`.
- Entities: `Product`, `CustomerRegion`, `Warehouse`, `SKUState` (có `available = on_hand - reserved` qua `computed_field`), `Supplier` (mục 3.2), `TransportRoute`, `PurchaseOrder`, `Shipment` (mục 3.3), `CustomerOrder`, `DisruptionEvent` (params giới hạn kiểu JSON-scalar).
- Tất cả model `extra="forbid"`, ràng buộc `Field(ge=0)`/`le=1` cho số lượng và reliability; mutation trạng thái (vd `received_quantity`, `actual_arrival`) để dành cho simulator core (task 4).

## Files Modified

- [src/mosaic/simulator/models.py](src/mosaic/simulator/models.py) — domain models (task scope).
- [tests/unit/test_models.py](tests/unit/test_models.py) — 12 unit tests.
- [pyproject.toml](pyproject.toml) — chuyển dev deps sang uv `[dependency-groups]`, bỏ mypy.
- [pyrightconfig.json](pyrightconfig.json) — pyright strict mode (`typeCheckingMode: strict`).
- [.python-version](.python-version) — pin Python 3.13 cho uv.

## Testing

- **Test File:** [tests/unit/test_models.py](tests/unit/test_models.py)
- **Status:** Passed — 12/12
- **Execution Command:** `uv run pytest tests/unit/test_models.py -v`
- **Lint/Type:** `uv run ruff check src tests` (passed), `uv run ruff format --check src tests` (clean), `uv run pyright` (0 errors)

## Additional Notes

- Pyright 1.1.411 không nhận `strict: true` (boolean) trong config — phải dùng `typeCheckingMode: "strict"`.
- `in_transit` và `backordered` trong `SKUState` có default 0 để khởi tạo nhẹ nhàng.
- uv chọn interpreter theo `.python-version`; nếu không có file này uv tự chọn Python 3.12 trên máy.
- Chuẩn bị cho task 4: models có mutation field (`received_quantity`, `status`, `actual_arrival`) nhưng chưa có logic chuyển trạng thái — thuộc simulator core.
