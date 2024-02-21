import nox


@nox.session(name="Run Tests", python=["3.10", "3.11"])
def tests(session):
    """Run the test suite."""
    session.run("pytest")


@nox.session(name="linter", python=["3.10", "3.11", "3.12"])
def linters(session):
    """Run linters"""
    session.run("ruff", "check", "--ignore", "D")

