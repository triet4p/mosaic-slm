# Changelog

## [Unreleased]

### Added

- Added a deterministic, read-only Inventory agent that assesses tool-backed stockout risk and proposes economically screened transfers through the existing recommendation contract.
- Added a deterministic, read-only Demand agent that reports tool-backed short-horizon forecasts and demand-shift risks through the existing recommendation contract.

### Fixed

- Corrected inventory transfer quotes to include active route transport-cost multipliers used by simulator dispatch.
