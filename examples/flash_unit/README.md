# Flash unit

* For higher resolution plots, set the parameter `n_eval_points` in `config.yaml` to larger values.
  Yet, this increases the execution time
* When creating data for method comparison,
  * Set the parameter `create_plots` in `config.yaml` to `False` in order to decrease the execution time drastically
  * Set the parameter `n_runs_bo` in `config.yaml` to a value `>1`
* You set the initial points in function `u_train_initial()` in file `config_gp.py`
* When you want to incorporate noise
  * Set the parameter `std_measurement_noise` in `config.yaml` to a value `>0.0`
  * Delete / comment out the line `gp.noise_std.fix(0.0)` in `config_gp.py`
* Some plots are annotated with `original problem`
  * For these plots, quantities are calculated with the `original` equations that do not check for physical feasibility
  * For the other plots, quantities are calculated with checking for physical feasibility and performing some heuristics for the case of infeasibility
