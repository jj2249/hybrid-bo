"""The code version a run and its study's artifact record, which says whether the
study is reproducible: checked in temporary git repositories."""

import subprocess

import pytest

from hybrid_bo.run import code_version, is_clean


def git(repo, *args):
    """Runs `git *args` in `repo`."""
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


# A fixture: each test that names `repo` gets a fresh repository with one commit
@pytest.fixture
def repo(tmp_path):
    """A git repository in a temporary folder with `a.py` committed and `results/`
    ignored."""
    git(tmp_path, "init")
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / ".gitignore").write_text("/results/\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.name=x", "-c", "user.email=x@x", "commit", "-m", "a")
    return tmp_path


def test_a_committed_tree_is_clean(repo):
    """With every file committed, the version is the bare commit, and clean; ignored
    files don't count."""
    (repo / "results").mkdir()
    (repo / "results" / "study.json").write_text("{}")

    version = code_version(repo)

    assert "-" not in version
    assert is_clean(version)


def test_a_changed_file_is_dirty(repo):
    """An edit to a committed file, staged or not, marks the version `-dirty`."""
    (repo / "a.py").write_text("a = 2\n")

    version = code_version(repo)

    assert version.endswith("-dirty")
    assert not is_clean(version)


def test_an_untracked_file_is_untracked(repo):
    """A file never added to git, which `git describe --dirty` misses, marks the
    version `-untracked`."""
    (repo / "experiment.py").write_text("b = 1\n")

    version = code_version(repo)

    assert version.endswith("-untracked")
    assert not is_clean(version)


def test_a_changed_and_an_untracked_file_are_both_marked(repo):
    """Both marks appear when both apply."""
    (repo / "a.py").write_text("a = 2\n")
    (repo / "experiment.py").write_text("b = 1\n")

    assert code_version(repo).endswith("-dirty-untracked")


def test_outside_a_repository_the_version_is_unknown_and_not_clean(tmp_path):
    """Outside a git checkout the code isn't pinned, so the study isn't reproducible."""
    version = code_version(tmp_path)

    assert version == "unknown"
    assert not is_clean(version)
