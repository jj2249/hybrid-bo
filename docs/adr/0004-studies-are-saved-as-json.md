# Studies are saved as JSON

A study is saved as a folder of JSON files: `artifact.json` for the study's settings
and artifact (`study.json` with a `record` key before #85; `hb.load` reads both), and
one file per run (`runs/<method>/run-000.json`) holding its history, fitted
surrogates, scenarios, acquisition solutions and full record, with arrays as nested
lists. `hb.save` and `hb.load` are library API; the library never calls them itself.
JSON keeps saved results readable without the code that produced them (ADR 0003), is
diffable, needs no dependency, round-trips float64 exactly, and is already the format
of the regression baselines. At this scale (a few KB per run) compression buys nothing.

## Considered Options

- **Pickle** (the format before this decision): loading depends on the classes' module
  paths, so renaming a module breaks old results (as `hybrid_bo.results.ResultsBO`
  did), and unpickling runs code.
- **npz plus a JSON record**: compact, but binary and two files per run.
- **netCDF or Zarr through xarray** (ArviZ's choice): a heavy dependency for ragged data
  (runs of different lengths, one surrogate per iteration).
- **Parquet**: suits the tidy Polars view, but cannot hold the surrogates, so it would
  be a second format. It may come later as an export of the view.
