"""The top-level namespace holds only what a typical script uses."""

import types

import hybrid_bo as hb

USER_FACING = {
    # Define a problem
    "Var",
    "System",
    "Problem",
    "Measure",
    "oracle_measure",
    # Run a method
    "initial_points",
    "hybrid_bo",
    "HybridBOSettings",
    "standard_bo",
    "StandardBOSettings",
    "EI",
    "LCB",
    "SAA",
    "SoftSAA",
    "Linearization",
    "latin_hypercube_sampling",
    "uniform_sampling",
    # Save and load a run
    "oracle_optimum",
    "save_run",
    "load_run",
    # Look at the results
    "Run",
    "History",
    "incumbents",
    "oracle_incumbents",
    # Errors
    "HybridBOError",
    "NoSolutionError",
    "ReconstructionWarning",
    "ReplayWarning",
    "RootFindingError",
}


def test_top_level_holds_only_user_facing_names():
    """Apart from submodules (`hb.plots` among them), `hb.*` is exactly the
    user-facing set; internals such as `fit_gp` stay in their modules."""
    names = {
        name
        for name, value in vars(hb).items()
        if not name.startswith("_") and not isinstance(value, types.ModuleType)
    }

    assert names == USER_FACING


def test_the_study_machinery_has_left_the_library():
    """Studies run with Benchopt (ADR 0013), so these modules stay out of `hb`, which
    the check of the names above doesn't see, as it leaves out modules."""
    assert not {"cli", "studies", "views"} & set(vars(hb))


def test_internals_are_reachable_through_their_modules():
    """Submodules are attributes of `hb`, so `hb.gp.fit_gp` works after
    `import hybrid_bo as hb`."""
    assert callable(hb.gp.fit_gp)
    assert callable(hb.solvers.multistart)
    assert callable(hb.plots.iteration)
