import torch
import torchvision.transforms as T

from paraphernalia.torch.clip import CLIP
from tests import require_cuda


def test_basic():
    with torch.no_grad():
        clip = CLIP("three acrobats")
        assert clip.prompts.shape == (1, 512)
        assert clip.detail_prompts.shape == (1, 512)


def test_studio(studio):
    with torch.no_grad():
        studio = T.functional.resize(studio, 256)
        studio = T.functional.to_tensor(studio)
        studio = studio.unsqueeze(0)
        clip = CLIP("an artists studio")

        similarity1 = clip.forward(studio).detach()
        assert similarity1.shape == (1,)
        assert similarity1[0] > 0.0
        assert similarity1[0] < 1.0

        clip = CLIP("a cute kitten playing on the grass")
        similarity2 = clip.forward(studio).detach()
        assert similarity2.shape == (1,)
        assert similarity2[0] < 1.0
        assert similarity2[0] > 0.0

        assert similarity1[0] > similarity2[0]


@require_cuda
def test_grads(studio):
    studio = T.functional.resize(
        studio, 777
    )  # The model is supposed to handle any size
    studio = T.functional.to_tensor(studio)
    studio = studio.unsqueeze(0)
    clip = CLIP("an artists studio")
    clip.encoder.requires_grad_(True)
    similarity = clip.forward(studio)
    similarity.backward()
