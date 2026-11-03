# Only unknown inequality constraints fail an evaluation

**Known inequality constraints are requirements on the result, not failure
conditions.** An evaluation that violates one succeeds, with states and an objective:
it is an infeasible evaluation, which can't be the incumbent but which the surrogate
model learns from. Only an unknown inequality constraint fails an evaluation (the
measurement raises `EvaluationFailed`), so the success labels are evidence about the
unknown inequality constraints alone, and the success classifier never relearns what
`c` already states. Runs catch `EvaluationFailed` and nothing broader: a root-finding
failure, or any other exception, stays an error and is never recorded as a failed
evaluation.

## Considered Options

- **Any violated constraint fails the evaluation** (the thesis this comes from,
  Schwickerath 2026): discards the states and objective the system did return, and
  mixes both kinds of constraint in the labels, so the classifier learns the known
  inequality constraints from 0/1 labels when they could be computed. The thesis
  acknowledges this conflation (§3.4).
- **Any exception in the measurement is a failed evaluation** (the thesis's oracle):
  reports bugs and root-finding failures as infeasibility, and with several roots
  makes feasibility depend on which start the rng picks first.

## Consequences

- Known inequality constraints say which evaluations count, not where we may evaluate.
  Constraints that must never be violated, such as a reactor's safe operating
  temperature and pressure, are a different problem (safe BO) and out of scope. A known
  inequality constraint that depends on `u` alone is imposed exactly in the acquisition
  problem, so trial points satisfy it; initial points are drawn without it.
- A real system that can't return states at some limit (an interlock that trips)
  raises `EvaluationFailed`. To the method, that limit is then an unknown inequality
  constraint, and the success classifier learns it as one.
- With no known inequality constraints, every evaluation that succeeds is feasible, and
  existing problems run as before.
