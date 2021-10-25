import paraphernalia as pa


def test_setup_function():
    # NB: test_setup is called for every test
    pa.setup()


def test_jupyter():
    assert not pa.running_in_jupyter()


def test_colab():
    assert not pa.running_in_colab()


def test_github():
    # Just a smoketest
    pa.running_in_github_action()


def test_seed():
    # Just a smoketest
    pa.seed(42)
