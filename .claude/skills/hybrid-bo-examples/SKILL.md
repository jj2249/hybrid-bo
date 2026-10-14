---
name: hybrid-bo-examples
description: Use when writing, modifying or running an example under examples/. Covers the files and exports an example needs, how to define the Problem with CasADi, the config.yaml fields, and common mistakes.
---

# Writing and running examples

`examples/illustrative_example/` is the reference. `examples/flash_unit/` is out
of scope for the current refactor, so leave it alone. This describes the **current**
contract. The target is DESIGN.md's "Target example": a single Python script with
named variables, no `config.yaml` and no callbacks. PR 9 switches the illustrative
example to it, and that PR rewrites this skill.

Use CONTEXT.md's terms when writing about examples. In the current contract, the
unknown states `y` are entries of `x` selected by `indices_x_output_gp`, `h_known`
is the known equations `g`, and `h_unknown` is the oracle.

## Layout

    examples/<name>/
      __init__.py          re-exports the functions below
      config.yaml          -> Config(**yaml)
      custom_problem.py    get_custom_problem
      user_definitions.py  get_u_initial, get_gp_*, get_optimizer_*
      plot_functions_*.py  create_plots for each routine (only if plotting)

## Required exports

| Function | Returns |
|---|---|
| `get_custom_problem(config)` | `Problem` |
| `get_u_initial(problem, rng)` | one run's initial points `(n_initial, n_u)`, drawn only from `rng` (use `hybrid_bo.lhs` over the `u` starting-point bounds) |
| `get_gp_hybrid_bo(n_inputs)` / `get_gp_standard_bo(n_inputs)` | `GP` |
| `get_optimizer_hybrid_bo()` / `get_optimizer_standard_bo()` | `Optimizer` (e.g. `MultiStartOptimizer(CasadiOptimizer("ipopt", ...))`) |
| `create_plots_{hybrid_bo,standard_bo,latin_hypercube_sampling,uniform_sampling}` | plot callbacks; the sampling ones take `(problem, config, f, u, x, n_points_initial, i_iteration)` (no GP) |

`main.py` spawns one generator per run (`config.rng.spawn(n_runs_bo)`), calls
`get_u_initial` once per run with that run's generator, and passes the generators
to the routine as `rngs`. Plot callbacks may use `config.rng`; it doesn't affect
the runs.

`examples/main.py` looks up `create_plots_<routine>` only when `create_plots` is
on, so an example without plot modules runs with `create_plots: False`.

## Defining the problem

See `illustrative_example/custom_problem.py`:

```python
u = SymbolicType.sym("u", config.n_u, 1)
x = SymbolicType.sym("x", config.n_x, 1)
# Objective, scalar (1, 1)
f = cas.Function("f", [u, x], [...])
# Known residuals
h_known = cas.Function("h_known", [u, x], [...])
# Oracle residuals, only used in the oracle simulation
h_unknown = cas.Function("h_unknown", [u, x], [...])
# Bounds: each of u_acq, x_acq, u_sp, x_sp is a (lower, upper) pair
Problem(f, h_known, h_unknown, *u_acq, *x_acq, *u_sp, *x_sp)
```

- Each function takes exactly `[u, x]`, as column vectors, and has one output.
  `n_u` and `n_x` are read from `f`.
- `h_known` and `h_unknown` together must have `n_x` equations, so the system is
  square for `fsolve`.
- All bounds are `(n, 1)` arrays.
  - Acquisition bounds (`*_acq`) may be `±inf` for `x`.
  - Starting-point bounds (`*_sp`) must be finite. They are the ranges for
    initial guesses and initial points.
- Use `hybrid_bo.type_aliases.SymbolicType` for symbols.

## config.yaml

- **Problem:**
  - `n_u` and `n_x` must match the `Problem`.
  - `indices_u_input_gp` and `indices_x_input_gp` select the GP inputs.
  - `indices_x_output_gp` selects the unknown states, which the GP predicts.
  - `indices_x_measured` selects the measured variables, which get the noise.
  - `use_output_gp_as_opt_var` makes the unknown states variables of the
    acquisition problem (default in the new design: `unknown_states_as_variables`).
  - `std_measurement_noise`
  - `n_starts_training_gp`
  - `n_starts_max_evaluate_problem`: the number of `fsolve` starts (which always
    use the CasADi Jacobian).
- **BO:**
  - `n_runs_bo` (independent runs) and `n_iterations_bo` (per run).
  - `create_plots`, `save_bo_results` and `save_extended_results`.
- **Acquisition:**
  - `formulation_acq` is `"ei"` or `"lcb"`.
  - `n_starts_acq_optimization`
  - `n_samples_gp`: the number of scenarios (hybrid only)
  - `factor_lcb_std`
  - `epsilon_ei_standard` is currently unused.
- **General:** `seed` (`null` means not reproducible).

## Common mistakes

- **GP inputs must total `n_u`.** Hybrid BO requires
  `len(indices_u_input_gp) + len(indices_x_input_gp) == n_u`, because `main.py`
  builds the GP with `problem.n_u` inputs (a Known issue; the new design takes the
  GP inputs from `h_inputs`).
- **Measured variables:** `len(indices_x_measured)` must equal the number of
  `h_unknown` equations (the number of unknown states), so the root-finding
  problem is square. Noise is drawn per measured variable.
- **Noise:** with `std_measurement_noise > 0`, remove `gp.noise_std.fix(0.0)` in
  `get_gp_*` so the GP can learn the noise.
- **Index lists** are lists, even with a single entry (`[0]`, not `0`), and an
  empty selection is `[]`.

## Running

Edit `example` and `routine` at the top of `examples/main.py`, then:

    pixi run -e dev python examples/main.py

Results go to `examples/<name>/<routine>/` and are git-ignored. Existing files
are never overwritten, so delete the folder to re-run.
