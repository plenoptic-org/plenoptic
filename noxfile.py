import nox


@nox.session(name="tests", python=["3.10", "3.11", "3.12"])
def tests(session):
    """Run the test suite."""
    session.run("pytest")


@nox.session(name="linters")
def linters(session):
    """Run linters"""
    session.run("ruff", "check", "--ignore", "D")

