"""Fetches S2MPJ (Gratton & Toint, https://github.com/GrattonToint/S2MPJ, BSD-3) at the
pinned commit into the git-ignored `benchmarks/.s2mpj/`: a sparse checkout of
`s2mpjlib.py` and `python_problems/` only, as the repository is about 150 MB.

The `benchmark` pixi task runs it first; it does nothing when the checkout is already
at the commit. Run it alone as `pixi run -e benchmarks fetch-s2mpj`.
"""

import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/GrattonToint/S2MPJ"

# The S2MPJ commit every benchmark is built from (main on 2026-08-24). Changing it
# can change the problems, so the task list and the artifact record it
COMMIT = "fea6a70048eaad28b13a08703ddbfdbf65cd9c30"

CHECKOUT = Path(__file__).parent / ".s2mpj"
_SPARSE_PATHS = ("/s2mpjlib.py", "/python_problems/")


def checkout_commit() -> str | None:
    """The commit the checkout is at, or None if there is no checkout."""
    if not (CHECKOUT / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def fetch() -> None:
    """Makes the checkout and moves it to `COMMIT`, unless it is there already.

    Raises `subprocess.CalledProcessError` if a git command fails.
    """
    if checkout_commit() == COMMIT:
        return

    # A clone without file contents, then only the files the benchmarks read
    if not (CHECKOUT / ".git").exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--sparse"]
            + [REPOSITORY, str(CHECKOUT)],
            check=True,
        )
    git = ["git", "-C", str(CHECKOUT)]
    subprocess.run(
        [*git, "sparse-checkout", "set", "--no-cone", *_SPARSE_PATHS], check=True
    )
    subprocess.run([*git, "fetch", "--quiet", "origin", COMMIT], check=True)
    subprocess.run([*git, "checkout", "--quiet", "--detach", COMMIT], check=True)
    print(f"Fetched S2MPJ {COMMIT} into {CHECKOUT}", file=sys.stderr)


if __name__ == "__main__":
    fetch()
