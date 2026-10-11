# Changelog

## [Unreleased]

### Added

- Added a deterministic, read-only Inventory agent that assesses tool-backed stockout risk and proposes economically screened transfers through the existing recommendation contract.
- Added a deterministic, read-only Demand agent that reports tool-backed short-horizon forecasts and demand-shift risks through the existing recommendation contract.
- Added a deterministic, read-only Supplier-Risk agent that uses live supplier, disruption, order, shipment, inventory-target, and order-cost evidence to assess risk and propose only explicitly requested, capacity-valid orders.
- Added a deterministic, non-LLM sequential Coordinator policy that calls Demand, Inventory, and conditional Supplier-Risk analysts, resolves same-destination conflicts, validates final actions on a simulator copy, and stops explicitly with `NO_ACTION` when analyst/revision budgets are exhausted.

### Changed

- Exposed only currently active supplier delay/cancellation events from `get_supplier_status`, so assessments can cite observed disruptions without reading future schedules.
- Limited `get_open_orders` and `get_shipments` to currently observable arrival and receipt fields, so pending orders and shipments expose planned quantities and expected arrival dates without leaking unobserved actual outcomes.

### Fixed

- Corrected inventory transfer quotes to include active route transport-cost multipliers used by simulator dispatch.
- Corrected inventory inbound screening to use observable expected arrival estimates instead of hidden actual arrival outcomes.
