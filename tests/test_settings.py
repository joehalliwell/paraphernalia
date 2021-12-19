from paraphernalia import settings


def test_global_settings():
    assert settings().cache_home.exists()
    assert settings().project_home.exists()


def test_cache_home(tmpdir):
    assert settings().cache_home.exists()
    assert settings().cache_home.is_dir()

    settings().cache_home = tmpdir  # Unique to test invocation
    assert settings().cache_home.exists()
    assert settings().cache_home.is_dir()
