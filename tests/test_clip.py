import PIL
import pkg_resources
import torch
import torchvision.transforms as T

from paraphernalia.torch.clip import CLIP


def test_basic():
    with torch.no_grad():
        clip = CLIP("three acrobats")
        assert clip.prompts.shape == (1, 512)
        assert clip.detail_prompts.shape == (1, 512)


def test_studio():
    with torch.no_grad():
        img = PIL.Image.open(pkg_resources.resource_filename(__name__, "studio.jpg"))
        img = T.functional.resize(img, 256)
        img = T.functional.to_tensor(img)
        img = img.unsqueeze(0)
        clip = CLIP("an artists studio")

        similarity1 = clip.forward(img).detach()
        assert similarity1.shape == (1,)
        assert similarity1[0] > 0.0
        assert similarity1[0] < 1.0

        clip = CLIP("a cute kitten playing on the grass")
        similarity2 = clip.forward(img).detach()
        assert similarity2.shape == (1,)
        assert similarity2[0] < 1.0
        assert similarity2[0] > 0.0

        assert similarity1[0] > similarity2[0]


def test_grads():
    img = PIL.Image.open(pkg_resources.resource_filename(__name__, "studio.jpg"))
    img = T.functional.resize(img, 256)
    img = T.functional.to_tensor(img)
    img = img.unsqueeze(0)
    clip = CLIP("an artists studio")
    similarity = clip.forward(img)
    similarity.backward()