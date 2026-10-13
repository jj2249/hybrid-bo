---
name: hybrid-bo-examples
description: Use when writing, modifying or running an example under examples/. Covers the files and exports an example needs, how to define the Problem with CasADi, the config.yaml fields, and common mistakes.
---

# Writing and running examples

`examples/illustrative_example/` is the reference. `examples/flash_unit/` is out
of scope for the current refactor, so leave it alone. This describes the current
contract; ROADMAP stage 4 will simplify it, and this skill should be updated
alongside.

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
| `get_u_initial(problem, config)` | `list` of `n_runs_bo` arrays, each `(n_initial, n_u)` |
| `get_gp_hybrid_bo(n_inputs)` / `get_gp_standard_bo(n_inputs)` | `GP` |
| `get_optimizer_hybrid_bo()` / `get_optimizer_standard_bo()` | `Optimizer` (e.g. `MultiStartOptimizer(CasadiOptimizer("ipopt", ...))`) |
| `create_plots_{hybrid_bo,standard_bo,latin_hypercube_sampling,uniform_sampling}` | plot callbacks |

`examples/main.py` currently looks up all four `create_plots_*` even when
`create_plots: False`, so an example without them crashes there. Copy the
illustrative plot modules, or call the routines directly as
`tests/test_regression.py` does.

## Defining the problem

See `illustrative_example/custom_problem.py`:

```python
u = SymbolicType.sym("u", config.n_u, 1)
x = SymbolicType.sym("x", config.n_x, 1)
f = cas.Function("f", [u, x], [...])                  # scalar (1,1)
h_known = cas.Function("h_known", [u, x], [...])      # known residuals
h_unknown = cas.Function("h_unknown", [u, x], [...])  # true residuals, simulation only
Problem(f, h_known, h_unknown, *u_acq, *x_acq, *u_sp, *x_sp)  # each a (lower, upper) pair
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
  - `indices_x_output_gp` selects the states the GP predicts.
  - `indices_x_measured` selects the states that get measurement noise.
  - `use_output_gp_as_opt_var` makes the GP outputs explicit decision variables.
  - `std_measurement_noise`
  - `n_starts_training_gp`
  - `use_jacobian_evaluate_problem` and `n_starts_max_evaluate_problem` control
    `fsolve`.
- **BO:**
  - `n_runs_bo` (independent runs) and `n_iterations_bo` (per run).
  - `create_plots`, `save_bo_results` and `save_extended_results`.
- **Acquisition:**
  - `formulation_acq` is `"ei"` or `"lcb"`.
  - `n_starts_acq_optimization`
  - `n_samples_gp` (hybrid only)
  - `factor_lcb_std`
  - `epsilon_ei_standard` is currently unused.
- **General:** `seed` (`null` means not reproducible).

## Common mistakes

- **GP inputs must total `n_u`.** Hybrid BO requires
  `len(indices_u_input_gp) + len(indices_x_input_gp) == n_u`, because `main.py`
  builds the GP with `problem.n_u` inputs.
- **Measured states:** `len(indices_x_measured)` must equal the number of
  `h_unknown` equations. In practice measurement noise supports a single
  measured state.
- **Noise:** with `std_measurement_noise > 0`, remove `gp.noise_std.fix(0.0)` in
  `get_gp_*` so the GP can learn the noise.
- **Index lists** are lists, even with a single entry (`[0]`, not `0`), and an
  empty selection is `[]`.

## Running

Edit `example` and `routine` at the top of `examples/main.py`, then:

    pixi run -e dev python examples/main.py

Results go to `examples/<name>/<routine>/` and are git-ignored. Existing files
are never overwritten, so delete the folder to re-run.
