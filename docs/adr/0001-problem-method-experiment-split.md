# Problem, method and experiment are separate

A **problem** is a system plus an objective: decision variables, mechanistic and unknown
states with their bounds, the known equations `g`, and the *arguments* of the unknown
equations `h`, but not their form. The **method** (hybrid BO, standard BO, a baseline)
owns everything about the search, including the choice of surrogate model, so a
benchmark compares different methods on literally the same problem. The form of `h`
exists only as an **oracle** in test cases; it is not a field of the problem, because
real applications have none.

An evaluation is split at the **experiment**: a callable `experiment(u, rng) -> measured
variables`. A real application supplies only that. The library always solves the
root-finding problem for the unmeasured states and computes `f`, so there is one
root-finding path. In test cases the oracle simulation is the experiment, and it also
returns the oracle values used for benchmarking.

## Considered Options

- **`h_unknown` on the problem** (the code before the restructure): mixes the hidden
  truth into what the optimizer knows, and a real plant would still need a fake `h`.
- **The surrogate on the problem**: hybrid BO, standard BO and the baselines would each
  need a different problem for the same physical system.
