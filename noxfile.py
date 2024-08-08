import nox


@nox.session(name="lint")
def lint(session):
    # run linters
    session.install("ruff")
    session.run("ruff", "check", "--ignore", "D")


@nox.session(name="tests", python=["3.10", "3.11", "3.12"])
def tests(session):
    # run tests
    session.install("pytest")
    session.run("pytest")
