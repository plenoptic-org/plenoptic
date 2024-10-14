from pathlib import Path

import nox


@nox.session(name="lint")
def lint(session):
    # run linters
    session.install("ruff")
    session.run("ruff", "check")


@nox.session(name="tests", python=["3.10", "3.11", "3.12"])
def tests(session):
    # run tests
    session.install("pytest")
    # Install pytest-cov for coverage reporting
    session.install("pytest-cov")
    # Install dependencies listed in pyproject.toml
    session.install(
        "numpy>=1.1",
        "torch>=1.8,!=1.12.0",
        "pyrtools>=1.0.1",
        "scipy>=1.0",
        "matplotlib>=3.3",
        "tqdm>=4.29",
        "imageio>=2.5",
        "scikit-image>=0.15.0",
        "einops>=0.3.0",
        "importlib-resources>=6.0",
        "pooch>=1.5",
    )
    session.env["PYTHONPATH"] = str(Path().resolve() / "src")
    session.run("pytest")
    # queue up coverage session to run next
    session.notify("coverage")


@nox.session
def coverage(session):
    session.install("coverage")
    session.run("coverage")
