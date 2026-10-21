# GPs are data

A prior GP is a frozen dataclass, `hb.gp.GP(mean, kernel, noise_std)`, built from a
closed set of means and kernels that are frozen dataclasses too: `Zero`, and `Scale`,
`Matern12`, `Matern32`, `Matern52` and `SquaredExponential`. Each kernel declares its
hyperparameters as fields; a number fixes one, and `hb.gp.Param(init, bounds, log)`
fits it.
Behavior is plain functions that match on the type (`covariance`, `prior_mean`), as the
acquisition code does for approximations. The one operator, `outputscale * kernel`
with a number or a `Param`, is shorthand that builds `Scale(kernel, outputscale)`, so
an experiment shows the output scale where the math has it; it adds no behavior. A fitted GP is the prior GP with every
`Param` replaced by its value, held with its training data and scalers, so it describes
itself: `predict(fitted, X)` takes nothing else, a run saves its fitted GPs whole, and a
plot redraws an iteration from the saved run alone.

This amends [ADR 0003](0003-runs-are-pure-data.md): runs still hold only data, and
that data is now the whole GP. It replaces the convention that kernels are functions
and there are no kernel classes (DESIGN.md, Principles). Kernel hyperparameters are in
scaled units, as the fit sees them; the known observation noise `noise_std` is in
measured units, so an experiment states the noise of its measurement as it is.

## Considered Options

- **Kernels as lambdas over named hyperparameters** (`PriorGP`, `Hyper`, `h.<name>`,
  what this replaces): any kernel could be written in an experiment, but a GP couldn't
  be saved, so a run held only the hyperparameter values and every prediction, plot and
  replay needed the prior GP from the experiment. A plot then had to match the record's
  text of the GP against `default_gp`, and a known noise took a pinned `Hyper` and a
  hand conversion to scaled units.
- **Kernel composition** (`Sum`, a product of kernels, `+` and `kernel * kernel`): more
  kernels from the same set, but no experiment needs one yet; it can be added to the
  closed set later.
- **`Scale(kernel)` alone, no operator**: one way to write it, but it hides that the
  output scale is a hyperparameter of its own, and the factor reads as the math does.

## Consequences

There are no custom kernels: a new kernel or mean is a library change (a dataclass and
a case in `covariance` or `prior_mean`), in exchange for GPs that can be saved, loaded
and plotted without the code that built them. Saving tags each mean, kernel and `Param`
with its type's name. Studies saved before this change don't load: `hb.load` raises
`StudyFormatError`, and they are rerun.
