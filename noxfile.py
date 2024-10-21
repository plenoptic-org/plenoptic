import nox


@nox.session(name="lint")
def lint(session):
    # run linters
    session.install("ruff")
    session.run("ruff", "check")


@nox.session(name="tests", python=["3.10", "3.11", "3.12"])
def tests(session):
    session.install(".[dev]")
    session.run("pytest")
    # queue up coverage session to run next
    session.notify("coverage")


@nox.session
def coverage(session):
    session.install("coverage")
    session.run("coverage")
