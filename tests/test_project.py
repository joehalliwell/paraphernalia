from paraphernalia import Project, get_seed, project, set_seed, settings


def test_basic_usage(monkeypatch, tmpdir):
    """
    Test output directory creation, and activate flag
    """
    monkeypatch.setattr(settings(), "project_home", tmpdir)

    p1 = Project(title="Project #1")
    assert p1.path.exists()
    assert p1.path.parent == tmpdir
    assert p1.creator == settings().creator
    assert project() == p1

    p2 = Project(title="Project #2", creator="Pseudonymous creator")
    assert p2.path.exists()
    assert p2.path.parent == tmpdir
    assert p2.path != p1.path
    assert p2.creator == "Pseudonymous creator"
    assert project() == p2

    p3 = Project(title="Project #3", activate=False)
    assert project() == p2
    p3.activate()
    assert project() == p3


def test_project_seed(monkeypatch, tmpdir):
    """Test that setting the project seed sets the global seed."""
    monkeypatch.setattr(settings(), "project_home", tmpdir)

    seed = 123456
    set_seed(seed + 1)
    assert get_seed() != seed

    p1 = Project(title="Project #1", seed=seed)

    assert p1.seed == seed
    assert get_seed() == seed


def test_default_seed():
    """Check that seed defaults to the global"""
    set_seed(654321)
    p = Project(title="Project #1")

    assert p.seed == get_seed()
