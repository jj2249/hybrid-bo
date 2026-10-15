# Iteration plots replay the acquisition problem

The acquisition panels of `plots.iteration` (f under the scenarios and the acquisition
function, over a grid of u) are computed after the run by rebuilding that iteration's
acquisition problem from the stored surrogate, scenarios and incumbent, and solving it
with u fixed at each grid point. So the panels show exactly the function the method
minimized, with the states the solver would choose, and the stored acquisition value at
the trial point checks the replay. A run stores the acquisition problem's solution at
each iteration, but not the curves.

## Considered Options

- **Root-finding the known equations per grid point and scenario** (the plots of #31):
  needs its own random starts, and where g has several roots it may pick a different
  one from the solver, so the plot can disagree with the method.
- **Storing the curves during the run**: works only for one decision variable, makes
  every run slower for something only plots use, and fixes the grid at run time.
