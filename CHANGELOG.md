<!-- markdownlint-disable MD013 -->

# Changelog

Alle nennenswerten Änderungen dieses Projekts werden in dieser Datei dokumentiert.
Das Format orientiert sich an Keep a Changelog und SemVer (MAJOR.MINOR.PATCH).

## [0.1.0] – 2025-08-28

### Added

- Vergleichsskripte: `src/compare.py` (Achsenvergleich) und `src/compare_decisions.py` (Entscheidungs-Grid & Tabellen).
- Visualisierungen in `docs/`: `axis_comparison.png`, `decision_grid.png`.
- Entscheidungstabellen: `docs/decision_table.csv`, `docs/decision_table.md`.
- Run-Konfiguration: `configs/run_care_bias.yaml`.

### Changed

- README: Titel zu „Ethik Bias Tester“ geändert; Fallbeschreibung „Herr Herrmann“ präzisiert; Abschnitt zu Entscheidungstabellen ergänzt; Markdown-Lint-Fixes.
- `cases/herr_herrmann.txt` überarbeitet (präzisere Falldetails).

### Notes

- Ziel: Reproduzierbare Demo (Baseline/Deterministic/Care/Autonomy) mit CSV-Outputs und Diagrammen für den Pflegekongress.
- Judge deterministisch (temperature=0); Deutsche Sprache durchgängig.

[0.1.0]: https://github.com/goldikolb/pflege-ethik-bias-tester/releases/tag/v0.1.0
