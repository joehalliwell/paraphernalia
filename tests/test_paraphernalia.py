from paraphernalia import (
    running_in_colab,
    running_in_github_action,
    running_in_jupyter,
    seed,
    setup,
)


def test_setup_function():
    # NB: test_setup is called for every test
    setup()


def test_jupyter():
    assert not running_in_jupyter()


def test_colab():
    assert not running_in_colab()


def test_github():
    # Just a smoketest
    running_in_github_action()


def test_seed():
    # Just a smoketest
    seed(42)
