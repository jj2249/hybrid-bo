"""The S2MPJ benchmarks (`benchmarks/`): the rules that turn a problem into a hybrid
one, the Python they write, and the generated experiments in `benchmarks/s2mpj/`.

The rules and the printer are tested on hand-written problems. The tests that read
S2MPJ use the checkout that `pixi run -e benchmarks fetch-s2mpj` makes in
`benchmarks/.s2mpj/`, and are skipped without it, so the test suite passes without
S2MPJ.
"""

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import casadi as cas
import generate_s2mpj as gen
import numpy as np
import pytest
import s2mpj_source
from fetch_s2mpj import CHECKOUT
from s2mpj_source import Reason, Source, UnsupportedProblemError

from hybrid_bo import load
from hybrid_bo.sampling import lhs

BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"
GENERATED = sorted(gen.OUTPUT_DIR.glob("*.py"))

# `skipif` skips the tests it marks when its condition is true, giving the reason
needs_checkout = pytest.mark.skipif(
    not (CHECKOUT / "s2mpjlib.py").exists(),
    reason="no S2MPJ checkout: run pixi run -e benchmarks fetch-s2mpj",
)
needs_ruff = pytest.mark.skipif(shutil.which("ruff") is None, reason="no ruff")


def hs27(**changes):
    """HS27 written by hand as a `Source`, with `changes` to its fields:
    min 0.01 (v1 - 1)^2 + (v2 - v1^2)^2 s.t. v1 + 1 + v3^2 = 0, whose one element is
    v3^2, with f* = 0.04 at v* = (-1, 1, 0) from the start (2, 2, 2)."""
    v, s = cas.SX.sym("v", 3), cas.SX.sym("s", 1)
    f = cas.Function("f", [v], [0.01 * (v[0] - 1) ** 2 + (v[1] - v[0] ** 2) ** 2])
    c = cas.Function("c", [v, s], [v[0] + 1 + s])
    element_sums = cas.Function("element_sums", [v], [v[2] ** 2])
    source = Source(
        name="HS27",
        variable_names=("X1", "X2", "X3"),
        constraint_names=("CON1",),
        lower=np.full(3, -np.inf),
        upper=np.full(3, np.inf),
        start=np.array([2.0, 2.0, 2.0]),
        n_inequalities=0,
        nonlinear=(True,),
        element_variables=((2,),),
        stated_optima=("0.04",),
        f=f,
        c=c,
        element_sums=element_sums,
        evaluate=lambda point: (0.0, np.zeros(1)),  # replaced below
    )
    changed = dataclasses.replace(source, **changes)
    return dataclasses.replace(changed, evaluate=_evaluation_of(changed))


def _evaluation_of(source):
    """A `Source.evaluate` from the source's functions, as S2MPJ's float code would
    compute them."""

    def evaluate(point):
        """f and c at `point`."""
        sums = source.element_sums(point)
        return float(source.f(point)), source.c(point, sums).full().ravel()

    return evaluate


# The rules, on hand-written problems


def test_hs27_by_hand_splits_as_issue_180_works_it_through():
    """HS27 splits into u = (v2, v3) and x = v1, with y = v3^2 (CON1's element) and
    `h_inputs` v3. The bounds are v2 in [0, 3] and v3 in [-1, 3]: each is the hull of
    the start 2 and v*, widened by 1. f* = 0.04 is the stated value, and IPOPT
    agrees."""
    # The split, by index into (v1, v2, v3)
    chosen, origin = gen.split(hs27())

    assert chosen.decisions == (1, 2)
    assert chosen.states == (0,)
    assert chosen.hidden == 0
    assert chosen.h_inputs == (2,)
    np.testing.assert_array_equal(chosen.lower[1:], [0.0, -1.0])
    np.testing.assert_array_equal(chosen.upper[1:], [3.0, 3.0])
    np.testing.assert_array_equal(origin.solution, [-1.0, 1.0, 0.0])
    assert origin.optimum == 0.04
    assert origin.ipopt_optimum == pytest.approx(0.04, abs=1e-8)

    # y's start bounds: v3^2 over Latin hypercube points of v3 in [-1, 3], so nearly
    # its range [0, 9]
    low, high = chosen.y_start
    assert 0.0 <= low <= 0.01
    assert 8.5 <= high <= 9.0


# `parametrize` runs the test once per value in the list, each as its own test case
@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"n_inequalities": 1}, Reason.INEQUALITIES),
        ({"stated_optima": ()}, Reason.NO_SOLUTION),
        ({"stated_optima": ("???",)}, Reason.NO_SOLUTION),
        ({"stated_optima": ("0.04", "0.05")}, Reason.NO_SOLUTION),
        ({"nonlinear": (False,)}, Reason.ONLY_LINEAR_CONSTRAINTS),
        ({"stated_optima": ("0.05",)}, Reason.IPOPT_DISAGREES),
    ],
    ids=["inequalities", "none", "not_a_number", "two", "linear", "disagrees"],
)
def test_a_problem_outside_the_rules_is_refused_with_the_reason(changes, reason):
    """A change that puts HS27 outside the rules makes `split` raise
    `UnsupportedProblemError` with the reason, which its message names."""
    # `raises` passes only if the block raises the exception given
    with pytest.raises(UnsupportedProblemError) as error:
        gen.split(hs27(**changes))

    assert error.value.reason == reason
    assert str(reason) in str(error.value)


def test_a_problem_without_degrees_of_freedom_is_refused():
    """With as many constraints as variables there are no decision variables left."""
    v, s = cas.SX.sym("v", 3), cas.SX.sym("s", 3)
    c = cas.Function("c", [v, s], [v + s])
    sums = cas.Function("element_sums", [v], [v**2])
    source = hs27(
        constraint_names=("C1", "C2", "C3"),
        nonlinear=(True,) * 3,
        element_variables=((0,), (1,), (2,)),
        c=c,
        element_sums=sums,
    )

    with pytest.raises(UnsupportedProblemError) as error:
        gen.split(source)
    assert error.value.reason == Reason.NO_DEGREES_OF_FREEDOM


def test_an_infeasible_problem_is_refused_as_ipopt_failing():
    """When IPOPT finds no solution from the start, the reason is IPOPT_FAILED, not
    that it disagrees with SOLTN: v1 + 1 + v3^2 = 0 can't hold with v1 >= 0."""
    source = hs27(lower=np.array([0.0, -np.inf, -np.inf]))

    with pytest.raises(UnsupportedProblemError) as error:
        gen.split(source)
    assert error.value.reason == Reason.IPOPT_FAILED


def test_a_constraint_that_is_its_hidden_term_plus_a_constant_is_refused():
    """In HS7's form, c = (v1^2 + v2^2) - 4 with both terms elements, hiding the
    element sum leaves the known equation y - 4 = 0, which no state appears in."""
    v, s = cas.SX.sym("v", 3), cas.SX.sym("s", 1)
    source = hs27(
        c=cas.Function("c", [v, s], [s - 4]),
        element_sums=cas.Function("element_sums", [v], [v[0] ** 2 + v[2] ** 2]),
        element_variables=((0, 2),),
    )

    with pytest.raises(UnsupportedProblemError) as error:
        gen.split(source)
    assert error.value.reason == Reason.HIDDEN_TERM_IS_CONSTANT


def test_the_trace_check_refuses_a_trace_that_differs_from_s2mpj():
    """`check_trace` raises when the traced functions differ from S2MPJ's float
    evaluation, as an unpatched trace does, with NaN where a symbol became a float."""
    source = hs27()
    v = cas.SX.sym("v", 3)
    wrong = dataclasses.replace(
        source, f=cas.Function("f", [v], [source.f(v) + np.nan])
    )

    with pytest.raises(UnsupportedProblemError) as error:
        gen.check_trace(wrong, source.start[np.newaxis])
    assert error.value.reason == Reason.UNTRACEABLE


def test_states_are_picked_by_pivoted_qr_with_ties_outside_the_hidden_term():
    """The states are the columns a column-pivoted QR picks first, and of two equal
    columns the one outside the hidden term."""
    # v2's column is largest, so it is picked; v1 and v3 are equal
    assert gen.choose_states(np.array([[1.0, 3.0, 1.0]]), hidden=()) == (1,)
    jacobian = np.array([[1.0, 0.0, 1.0]])
    assert gen.choose_states(jacobian, hidden=(0,)) == (2,)
    assert gen.choose_states(jacobian, hidden=(2,)) == (0,)


def test_the_bounds_widen_the_hull_of_start_and_solution():
    """Each infinite bound is the hull's edge widened by max(1, half its width);
    finite bounds are kept."""
    lower, upper = gen.bounds(
        np.array([-np.inf, -np.inf, 0.5]),
        np.array([np.inf, np.inf, np.inf]),
        start=np.array([2.0, 10.0, 2.0]),
        solution=np.array([1.0, 0.0, 1.0]),
    )

    np.testing.assert_array_equal(lower, [0.0, -5.0, 0.5])
    np.testing.assert_array_equal(upper, [3.0, 15.0, 3.0])


@pytest.mark.parametrize(
    ("optimum", "expected"),
    [(0.0, 1e-3), (0.04, 4e-5), (-680.63, 0.68063), (1e-5, 1e-6)],
)
def test_the_temperature_is_a_thousandth_of_the_optimum_with_a_floor(optimum, expected):
    """SoftSAA's temperature is 1e-3 |f*|, at least 1e-6, and 1e-3 when the stated
    f* is exactly 0."""
    assert gen.temperature(optimum) == expected


def test_the_rules_draw_the_same_numbers_in_every_process():
    """The rules' random numbers are seeded from sha256 of the name, so new Python
    processes with different string hash seeds draw what this one draws."""
    code = "import generate_s2mpj; print(generate_s2mpj.rng_of('HS27').integers(10**9))"
    draws = {
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=BENCHMARKS_DIR,
            env=os.environ | {"PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for hash_seed in ("1", "2", "3")
    }

    assert draws == {str(gen.rng_of("HS27").integers(10**9))}


# The Python the generator writes


def test_printed_expressions_compute_the_same_values():
    """Each expression printed as Python, read back with the same symbols, gives
    exactly the same values: the parentheses keep its tree."""
    a, b, c = (cas.SX.sym(name) for name in "abc")
    expressions = [
        a - (b - c),
        (a + b) * c,
        a / (b * c),
        -(a**2),
        (-a) ** 2,
        a + -4.2 * cas.sin(b) ** 2,
        a ** (b**c),
        (a**b) ** c,
        2 + 0.01 * cas.exp(a - 1),
    ]
    v = types.SimpleNamespace(a=a, b=b, c=c)  # the symbols as g, h and f read them
    for expression in expressions:
        text = gen.to_python(expression)
        read_back = eval(text, {"cas": cas, "v": v})
        both = cas.Function("both", [a, b, c], [expression, read_back])
        values = both(0.7, 1.3, -0.4)
        assert float(values[0]) == float(values[1]), text


def test_printed_expressions_read_as_a_person_would_write_them():
    """Only the parentheses the tree needs, and a sum minus a product rather than
    plus a negative one."""
    a, b, c = (cas.SX.sym(name) for name in "abc")

    assert gen.to_python(a - (b - c)) == "v.a - (v.b - v.c)"
    assert gen.to_python((a + b) * c) == "(v.a + v.b) * v.c"
    assert gen.to_python(a + -4.2 * b**2) == "v.a - 4.2 * v.b ** 2"


@needs_ruff
def test_a_generated_experiment_is_the_problem_it_comes_from(tmp_path):
    """The experiment generated from HS27 by hand matches HS27's own evaluation at
    random points (`check_experiment`), and its oracle measurement at the solution's
    decision variables finds the solution's state and f*."""
    # Generate HS27 by hand into a temporary folder (`tmp_path`, pytest's fresh
    # folder for this test), and load it
    source = hs27()
    chosen, origin = gen.split(source)
    path = tmp_path / "hs27.py"
    path.write_text(gen.formatted(gen.experiment_text(source, chosen, origin)))
    experiment = gen.load_experiment(path)

    # It agrees with the source, and at u* the measurement finds v1 = -1
    gen.check_experiment(experiment, source)
    measured = experiment.measure(np.array([1.0, 0.0]), np.random.default_rng(0))
    np.testing.assert_allclose(measured.x, [-1.0], atol=1e-8)
    assert experiment.optimum == 0.04
    assert experiment.notes["hidden_constraint"] == "CON1"


# The generated experiments


def test_a_generated_benchmark_runs_prepares_and_resumes(tmp_path):
    """The committed HS27 benchmark runs as an ordinary experiment script: a prepared
    study, with one task run as a scheduler would, is finished by `resume`, and its
    artifact records where the experiment comes from."""
    # Prepare a study of Latin hypercube sampling, and run its first task
    script = gen.OUTPUT_DIR / "hs27.py"
    root = ["--results-root", str(tmp_path)]

    def run(*arguments):
        """Runs the script with `arguments` and the temporary results root."""
        subprocess.run(
            [sys.executable, str(script), *arguments, *root],
            check=True,
            capture_output=True,
        )

    run(
        "prepare",
        "--method",
        "latin_hypercube_sampling",
        "--n-runs",
        "2",
        "--name",
        "s",
    )
    study_dir = tmp_path / "hs27" / "s"
    run("run-task", "s", "--task-index", "0")

    # Resume runs the other task and finishes the study
    run("resume", "s")
    study = load(study_dir)
    assert [saved.method for saved in study.runs] == ["latin_hypercube_sampling"] * 2
    assert all(saved.history.u.shape == (20, 2) for saved in study.runs)
    assert "hidden_constraint: CON1" in study.artifact["notes"]
    assert json.loads((study_dir / "tasks.json").read_text())["notes"]["problem"] == (
        "HS27"
    )


@needs_checkout
# One test case per generated file, named by it
@pytest.mark.parametrize("path", GENERATED, ids=[path.stem for path in GENERATED])
def test_a_generated_benchmark_matches_s2mpj(path):
    """The generated f, g and h match S2MPJ's own float evaluation at random points
    within the experiment's bounds, and the stated f* still agrees with IPOPT."""
    experiment = gen.load_experiment(path)
    source = s2mpj_source.read(experiment.notes["problem"])

    gen.check_experiment(experiment, source)
    _, origin = gen.split(source)
    assert origin.ipopt_optimum == pytest.approx(
        experiment.optimum, rel=gen.OPTIMUM_RTOL, abs=gen.OPTIMUM_ATOL
    )


@needs_checkout
@needs_ruff
def test_regenerating_hs27_reproduces_the_committed_file(tmp_path):
    """The generator is deterministic and the committed file isn't edited by hand:
    generating HS27 again gives the committed file byte for byte."""
    path = gen.generate("HS27", output_dir=tmp_path)

    assert path.read_bytes() == (gen.OUTPUT_DIR / "hs27.py").read_bytes()


# S2MPJ's problems


# HS26 and HS46 write into float arrays, so they trace only with the patch
@needs_checkout
@pytest.mark.parametrize("name", ["HS27", "HS6", "HS26", "HS39", "HS46", "HS56", "BT9"])
def test_traced_functions_match_the_float_evaluation(name):
    """f and c traced with CasADi symbols equal S2MPJ's own float evaluation at random
    points around S2MPJ's start (`check_trace` raises otherwise)."""
    source = s2mpj_source.read(name)
    points = lhs(source.start - 2, source.start + 2, 10, np.random.default_rng(0))

    gen.check_trace(source, points)


@needs_checkout
def test_s2mpj_hs27_splits_as_issue_180_works_it_through():
    """HS27 read from S2MPJ splits as #180 works it through, as the hand-written
    one does: u = (X2, X3), x = X1, CON1 hidden with h_inputs X3."""
    chosen, origin = gen.split(s2mpj_source.read("HS27"))

    assert (chosen.decisions, chosen.states) == ((1, 2), (0,))
    assert (chosen.hidden, chosen.h_inputs) == (0, (2,))
    np.testing.assert_array_equal(chosen.lower[1:], [0.0, -1.0])
    np.testing.assert_array_equal(chosen.upper[1:], [3.0, 3.0])
    assert origin.optimum == 0.04


@needs_checkout
@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("HS21", Reason.INEQUALITIES),
        ("HS8", Reason.NO_DEGREES_OF_FREEDOM),
        ("HS28", Reason.ONLY_LINEAR_CONSTRAINTS),
        ("TRYmB", Reason.NO_SOLUTION),
        ("WACHBIEG", Reason.IPOPT_DISAGREES),
        ("HS7", Reason.HIDDEN_TERM_IS_CONSTANT),
        ("BT8", Reason.ILL_CONDITIONED),
        ("HS87", Reason.UNTRACEABLE),
    ],
)
def test_an_unsupported_s2mpj_problem_is_refused_with_the_reason(name, reason):
    """An S2MPJ problem outside the rules raises `UnsupportedProblemError` with the
    reason, from `read` or from `split`."""
    with pytest.raises(UnsupportedProblemError) as error:
        gen.split(s2mpj_source.read(name))

    assert error.value.reason == reason


@needs_checkout
def test_an_unknown_problem_is_a_value_error():
    """A name S2MPJ has no problem for, or that isn't a plain name, raises
    `ValueError`."""
    for name in ("NOSUCHPROBLEM", "../s2mpjlib"):
        with pytest.raises(ValueError, match="S2MPJ has no problem"):
            s2mpj_source.read(name)
