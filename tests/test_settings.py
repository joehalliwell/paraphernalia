from paraphernalia import settings


def test_global_settings():
    assert settings().cache_home.exists()
    assert settings().project_home.exists()
    assert isinstance(settings().seed, int)


def test_cache_home(tmpdir):
    assert settings().cache_home.exists()
    assert settings().cache_home.is_dir()

    settings().cache_home = tmpdir  # Unique to test invocation
    assert settings().cache_home.exists()
    assert settings().cache_home.is_dir()


def test_default_seed():
    s1 = settings().seed
    s2 = settings().seed
    assert s1 is not None
    assert s1 == s2
    settings().seed = s1 + 1
    s3 = settings().seed
    assert s1 != s3
