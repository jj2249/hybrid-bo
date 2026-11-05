"""The trial point functions: one acquisition problem solved from fitted GPs."""

from typing import Any

import illustrative_example as ex
import numpy as np
import pytest
from scipy.stats import norm

import hybrid_bo as hb
from hybrid_bo.acquisition import hybrid_acquisition, standard_acquisition
from hybrid_bo.gp import fit_gp, predict
from hybrid_bo.hybrid_model import _pack

N_SCENARIOS = 4


def training_data() -> hb.Dataset:
    """Five points of the illustrative example, with y = sin(u) as if measured."""
    u = np.linspace(-1.8, 1.8, 5)[:, np.newaxis]
    rng = np.random.default_rng(1)
    return hb.reconstruct_states(ex.problem, u=u, measured=np.sin(u), rng=rng)


def stream_and_children(rng: np.random.Generator) -> tuple[Any, int]:
    """The generator's stream state and how many children it has spawned: `lhs` draws
    from a new child and leaves the stream alone."""
    seed_sequence = rng.bit_generator.seed_seq
    return rng.bit_generator.state, seed_sequence.n_children_spawned  # ty: ignore[unresolved-attribute]


# `parametrize` runs the test once per entry, with the entry as the argument
@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
def test_hybrid_trial_point_solves_the_acquisition_problem(unknown_states_as_variables):
    """The solution holds the trial point, the acquisition problem's objective there
    and each scenario's states, which satisfy the known equations."""
    # Setup: a surrogate model and the scenarios, drawn by the caller
    rng = np.random.default_rng(2)
    dataset = training_data()
    fitted = hb.fit_surrogate(ex.system, dataset, gp=ex.gp, rng=rng)
    scenarios = hb.saa_scenarios(N_SCENARIOS, ex.system.n_y, rng)
    incumbent = hb.incumbents(dataset)[-1]
    approximation = hb.SAA(
        n_scenarios=N_SCENARIOS, unknown_states_as_variables=unknown_states_as_variables
    )

    # Act
    solution = hb.hybrid_trial_point(
        ex.problem,
        fitted,
        scenarios,
        incumbent,
        rng,
        acquisition=hb.EI(),
        approximation=approximation,
    )

    # Check the shapes, and the bounds of u
    assert solution.x is not None and solution.y is not None
    assert solution.u.shape == (ex.system.n_u,)
    assert solution.x.shape == solution.y.shape == (N_SCENARIOS, 1)
    assert np.all(np.abs(solution.u) <= 2.0 + 1e-6)

    # Check the known equations of every scenario
    residuals = solution.x + np.exp(solution.x) - solution.y
    np.testing.assert_allclose(residuals, 0.0, atol=1e-6)

    # Check the value against the problem's objective at the solution
    built = hybrid_acquisition(
        ex.problem, fitted, scenarios, incumbent, hb.EI(), approximation
    )
    y = solution.y if unknown_states_as_variables else None
    expected = float(built.nlp.f(_pack(solution.u, solution.x, y)))
    assert solution.value == pytest.approx(expected, rel=1e-9)


def test_standard_trial_point_minimizes_analytic_ei():
    """The solution holds the trial point and -EI there, computed here from the GP's
    mean and std, and no scenario states."""
    # Setup: a GP of the objective
    rng = np.random.default_rng(2)
    dataset = training_data()
    fitted = fit_gp(ex.gp, dataset.u, dataset.f, rng)
    incumbent = float(hb.incumbents(dataset)[-1])

    # Act
    solution = hb.standard_trial_point(
        ex.problem, fitted, incumbent, rng, acquisition=hb.EI()
    )

    # Check against EI in closed form
    mean, std = predict(fitted, solution.u[np.newaxis])
    gamma = (incumbent - mean[0]) / std[0]
    expected_ei = (incumbent - mean[0]) * norm.cdf(gamma) + std[0] * norm.pdf(gamma)
    assert solution.value == pytest.approx(-expected_ei, rel=1e-6)
    assert solution.x is None and solution.y is None

    # Check that no other point of a grid has a better value
    grid = np.linspace(-2.0, 2.0, 201)[np.newaxis]
    built = standard_acquisition(ex.problem, fitted, incumbent, hb.EI())
    assert solution.value <= built.nlp.f.map(201)(grid).full().min() + 1e-9


def test_trial_point_functions_draw_only_their_starts():
    """Each draws one `lhs`, its starts, and nothing from the stream: the scenarios are
    the caller's, so an iteration's draws don't change with them."""
    # Setup: a surrogate model of each kind
    dataset = training_data()
    rng = np.random.default_rng(2)
    gp_of_f = fit_gp(ex.gp, dataset.u, dataset.f, rng)
    fitted = hb.fit_surrogate(ex.system, dataset, gp=ex.gp, rng=rng)
    scenarios = hb.nominal_scenario(ex.system.n_y)
    incumbent = float(hb.incumbents(dataset)[-1])

    # Act and check: the standard BO function
    state, children = stream_and_children(rng)
    hb.standard_trial_point(ex.problem, gp_of_f, incumbent, rng, acquisition=hb.EI())
    assert stream_and_children(rng) == (state, children + 1)

    # Act and check: the hybrid BO function
    hb.hybrid_trial_point(
        ex.problem,
        fitted,
        scenarios,
        incumbent,
        rng,
        acquisition=hb.EI(),
        approximation=hb.Linearization(),
    )
    assert stream_and_children(rng) == (state, children + 2)


def test_hybrid_trial_point_refuses_what_its_acquisition_problem_refuses():
    """The checks of `hybrid_acquisition` hold: linearization runs over the nominal
    scenario alone, and has no LCB."""
    rng = np.random.default_rng(2)
    dataset = training_data()
    fitted = hb.fit_surrogate(ex.system, dataset, gp=ex.gp, rng=rng)
    drawn = hb.saa_scenarios(N_SCENARIOS, ex.system.n_y, rng)

    # `raises` passes only if the block raises that exception
    with pytest.raises(ValueError, match="nominal scenario"):
        hb.hybrid_trial_point(
            ex.problem,
            fitted,
            drawn,
            0.0,
            rng,
            acquisition=hb.EI(),
            approximation=hb.Linearization(),
        )
    with pytest.raises(NotImplementedError, match="LCB"):
        hb.hybrid_trial_point(
            ex.problem,
            fitted,
            hb.nominal_scenario(ex.system.n_y),
            0.0,
            rng,
            acquisition=hb.LCB(std_factor=2.0),
            approximation=hb.Linearization(),
        )


def test_standard_trial_point_refuses_an_unknown_acquisition_function():
    """Anything but `EI` or `LCB` raises `TypeError`."""
    rng = np.random.default_rng(2)
    dataset = training_data()
    fitted = fit_gp(ex.gp, dataset.u, dataset.f, rng)

    with pytest.raises(TypeError, match="acquisition function"):
        hb.standard_trial_point(ex.problem, fitted, 0.0, rng, acquisition="EI")  # ty: ignore[invalid-argument-type]
