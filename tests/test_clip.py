import pytest
import torch
import torchvision.transforms as T

from paraphernalia.torch.clip import CLIP, EncodedImage, encode_text
from paraphernalia.torch.modules import (
    Not,
    Perceiver,
    SimilarTo,
    SimilarToAny,
    WeightedSum,
)
from tests import skipif_no_cuda


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
        studio = studio.to(clip.device)

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


@skipif_no_cuda
def test_grads(studio):
    studio = T.functional.resize(
        studio, 777
    )  # The model is supposed to handle any size
    studio = T.functional.to_tensor(studio)
    studio = studio.unsqueeze(0)
    clip = CLIP("an artists studio")
    studio = studio.to(clip.device)

    clip.encoder.requires_grad_(True)
    similarity = clip.forward(studio)
    similarity.backward()


def test_similarity(studio):
    """
    Check that factored similarity modules are working as
    expected with CLIP.
    """
    clip = CLIP("NOT USED")

    studio = T.functional.resize(studio, 256)
    studio = T.functional.to_tensor(studio)
    studio = studio.unsqueeze(0)
    studio = clip.encode_image(studio)

    assert studio.shape[0] == 1

    good_caption = "An artists studio"
    bad_caption = "A cute kitten"

    similarity1 = SimilarTo(clip.encode_text(good_caption))(studio)
    assert similarity1.shape == (1,)
    assert similarity1[0] > 0.0
    assert similarity1[0] < 1.0

    similarity2 = SimilarTo(clip.encode_text(bad_caption))(studio)
    assert similarity2.shape == (1,)
    assert similarity2[0] > 0.0
    assert similarity2[0] < 1.0

    # Studio is a better description than kitten!
    assert similarity2[0] < similarity1[0]

    # Any of good or bad, is as good as good
    similarity3 = SimilarToAny(clip.encode_text([good_caption, bad_caption]))(studio)
    assert similarity3[0] > 0.0
    assert similarity3[0] < 1.0
    assert torch.isclose(similarity3, similarity1)


@pytest.mark.parametrize("lenses", [[T.RandomCrop(224)], [T.RandomCrop(224)] * 5])
def test_perceiver(studio, lenses):
    """Smoketest Perceiver"""

    studio = T.functional.resize(studio, 256)
    studio = T.functional.to_tensor(studio)
    studio = studio.unsqueeze(0)

    p1 = Perceiver(lenses, EncodedImage(SimilarTo(encode_text("An artists studio"))))
    similarity1 = p1(studio)
    assert similarity1.shape == (1,)
    assert similarity1[0] > 0.0
    assert similarity1[0] < 1.0

    p2 = Perceiver(lenses, EncodedImage(SimilarTo(encode_text("A cute kitten"))))
    similarity2 = p2(studio)
    assert similarity2.shape == (1,)
    assert similarity2[0] > 0.0
    assert similarity2[0] < 1.0

    # Studio is a better description than kitten!
    assert similarity2[0] < similarity1[0]


def test_compound_prompt(studio):

    studio = T.functional.resize(studio, 256)
    studio = T.functional.to_tensor(studio)
    studio = studio.unsqueeze(0)

    lenses = [T.RandomCrop(224)]

    p1 = Perceiver(
        lenses,
        EncodedImage(
            WeightedSum(
                prompt=SimilarTo(encode_text("An artists studio")),
                antiprompt=Not(SimilarTo(encode_text("A spaceship"))),
            )
        ),
    )
    similarity1 = p1(studio)
    assert similarity1.shape == (1,)
    assert similarity1[0] > 0.0
    assert similarity1[0] < 1.0

    p2 = Perceiver(
        lenses,
        EncodedImage(
            WeightedSum(
                prompt=SimilarTo(encode_text("An artists studio")),
                antiprompt=Not(SimilarTo(encode_text("A cluttered office"))),
            )
        ),
    )
    similarity2 = p2(studio)
    assert similarity2.shape == (1,)
    assert similarity2[0] > 0.0
    assert similarity2[0] < 1.0

    # The second compound is worse than the first
    assert similarity2[0] < similarity1[0]
