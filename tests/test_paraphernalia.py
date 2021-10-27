import time

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


def test_cache_home():
    cache = pa.cache_home()
    assert str(pa.cache_home("FOO")) == "FOO"
    pa.cache_home(cache)


def test_data_home():
    data = pa.data_home()
    assert str(pa.data_home("FOO")) == "FOO"
    pa.data_home(data)


def test_seed():
    # Just a smoketest
    pa.seed(42)


def test_default_seed():
    s1 = pa.seed()
    s2 = pa.seed()
    assert s1 is not None
    assert s1 == s2
    s3 = pa.seed(s1 + 1)
    assert s1 != s3
