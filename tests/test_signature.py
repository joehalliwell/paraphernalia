import pytest
from click.testing import CliRunner
from PIL import Image

from paraphernalia.signature import XMP, sign


def inject_fixture(name, params):
    @pytest.fixture(scope="module", params=params)
    def my_fixture(request):
        return request.param

    globals()[name] = my_fixture


inject_fixture("filename", ["test.png", "test.jpg", "test.tif"])
inject_fixture("creator", ["David Hockney", "Joan Miró"])
inject_fixture("title", ["An uninvited guest", "𝖀𝖙𝖔𝖕𝖎𝖆"])
inject_fixture("tags", [["foo"], ["bar", "baz"]])
inject_fixture("description", ["A gesamukunstwerk"])
inject_fixture("rights", ["All rights reversed"])


def test_roundtrip(tmpdir, filename, creator, title, tags, description, rights):
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    filename = str(tmpdir.join(filename))
    img.save(filename)

    tags = ["procgen", "paraphernalia"]
    description = "A gesamkunstwerk"
    rights = "All rights reversed"

    with XMP(filename) as sig:
        sig.creator = creator
        sig.title = title
        sig.tags = tags
        sig.description = description
        sig.rights = rights

    with XMP(filename) as sig:
        assert sig.creator == creator
        assert sig.creators == [creator]
        assert sig.title == title
        assert sig.tags == tags
        assert sig.description == description
        assert sig.rights == rights

    with XMP(filename) as sig:
        sig.clear()

    with XMP(filename) as sig:
        assert sig.title == None


def test_sign(filename, creator, title, tags, description, rights):
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    runner = CliRunner()
    with runner.isolated_filesystem():
        img.save(filename)

        tag_args = []
        if tags:
            tag_args = ["--tag"] + " --tag ".join(tags).split()

        result = runner.invoke(
            sign,
            [filename, "--creator", creator, "--title", title, *tag_args],
        )
        assert result.exit_code == 0

        with XMP(filename) as sig:
            assert sig.creator == creator
            assert sig.title == title
            assert sig.tags == tags
            assert sig.description == None
            assert sig.rights == None

        # Add description and change creator
        result = runner.invoke(
            sign,
            [
                filename,
                "--creator",
                "NOT " + creator,
                "--description",
                description,
                "--rights",
                rights,
            ],
        )
        assert result.exit_code == 0

        # Check that creator has changed, but not other metadata
        with XMP(filename) as sig:
            assert sig.creator != creator
            assert sig.description == description
            assert sig.rights == rights
            assert sig.title == title
            assert sig.tags == tags

        # Test clearing
        results = runner.invoke(sign, [filename, "--clear"])
        with XMP(filename) as sig:
            assert sig.creator is None
            assert sig.creators == []
            assert sig.title is None
            assert sig.tags == []
            assert sig.description is None
            assert sig.rights is None
