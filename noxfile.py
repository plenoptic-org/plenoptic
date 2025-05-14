import nox
import yaml

with open(".pre-commit-config.yaml") as f:
    precommit_config = yaml.safe_load(f)
ruff_version = [r for r in precommit_config["repos"] if "ruff-pre-commit" in r["repo"]]
ruff_version = ruff_version[0]["rev"].replace("v", "")
numpydoc_version = [r for r in precommit_config["repos"] if "numpydoc" in r["repo"]]
numpydoc_version = numpydoc_version[0]["rev"].replace("v", "")


@nox.session
def format(session):
    # run linters
    session.install(f"ruff=={ruff_version}")
    session.run("ruff", "check", "--fix", "--config=pyproject.toml")
    session.run("ruff", "format", "--config=pyproject.toml")


@nox.session
def lint(session):
    # run linters
    session.install(f"ruff=={ruff_version}")
    session.install(f"numpydoc=={numpydoc_version}")
    session.run("ruff", "check", "--config=pyproject.toml")
    session.run("ruff", "format", "--check", "--config=pyproject.toml")
    session.run("numpydoc", "lint", "src")
    session.run("python", "tests/check_docstrings.py", "src")
    session.run("python", "tests/check_sphinx_directives.py", "src")


@nox.session(name="tests", python=["3.10", "3.11", "3.12"])
def tests(session):
    session.install(".[dev]")
    session.run("pytest")


@nox.session(name="doctests", python=["3.10", "3.11", "3.12"])
def doctests(session):
    session.install(".[dev]")
    session.run("pytest --doctest-modules --doctest-continue-on-failure src/")
