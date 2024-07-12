import nox

@nox.session(name="lint")
def lint(session):
    # run linters
    session.install("ruff")
    session.run("ruff", "check", "--ignore", "D")

@nox.session(name="tests")
def tests(session):
    # run tests
    session.install("pytest")
    session.run("pytest")