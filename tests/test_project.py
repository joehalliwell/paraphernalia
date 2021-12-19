import paraphernalia as pa
from paraphernalia._project import Project, project
from paraphernalia.utils import get_seed


def test_basic_usage(monkeypatch, tmpdir):
    """
    Test output directory creation, and activate flag
    """
    monkeypatch.setattr(pa.settings(), "project_home", tmpdir)

    p1 = Project(title="Project #1")
    assert p1.path.exists()
    assert p1.path.parent == tmpdir
    assert p1.creator == pa.settings().creator
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
    monkeypatch.setattr(pa.settings(), "project_home", tmpdir)
    p1 = Project(title="Project #1", seed=123456)

    assert p1.seed == 123456
    assert get_seed() == p1.seed
