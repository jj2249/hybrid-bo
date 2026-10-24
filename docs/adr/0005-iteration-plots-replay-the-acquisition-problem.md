# Iteration plots replay the acquisition problem

The acquisition panels of `plots.iteration` (f under the scenarios and the acquisition
function, over a grid of u) are computed after the run by rebuilding that iteration's
acquisition problem from the stored surrogate, scenarios and incumbent, and solving it
with u fixed at each grid point. So the panels show exactly the function the method
minimized, with the states the solver would choose, and the stored acquisition value at
the trial point checks the replay. A run stores the acquisition problem's solution at
each iteration, but not the curves.

Amended in #154: the replay no longer solves the problem with IPOPT. With u fixed it
is a square system per scenario, so each scenario's states are a Newton root started
from the stored solution and continued outwards over the grid, and the value is the
problem's objective there. That is the root-finding rejected below, without its
objections: no random starts, and it stays on the solver's branch, which the trial
point checks (`ReplayWarning`). The panels still show the function the method
minimized.

## Considered Options

- **Root-finding the known equations per grid point and scenario** (the plots of #31):
  needs its own random starts, and where g has several roots it may pick a different
  one from the solver, so the plot can disagree with the method.
- **Storing the curves during the run**: works only for one decision variable, makes
  every run slower for something only plots use, and fixes the grid at run time.
