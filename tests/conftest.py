import pkg_resources
from PIL import Image
from pytest import fixture


@fixture
def studio():
    return Image.open(pkg_resources.resource_filename(__name__, "studio.jpg"))
