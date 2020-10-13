"""Full test suite designed for testing against multiple Python environments."""
import nox


def setup_pipenv(session):
    """Sets up virtual environment for use in nox."""
    session.run("pipenv", "install")
    session.run("pipenv", "install", "--dev")


@nox.session(python=["3.8"])
def tests(session):
    """Runs tests."""
    setup_pipenv(session)
    session.run("pytest")


@nox.session(python=["3.8"])
def lint(session):
    """Lints and typechecks files."""
    setup_pipenv(session)
    session.run("pylint", "text_data")
    session.run("pylint", "tests")
    session.run("mypy", "text_data")
