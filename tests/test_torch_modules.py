import torch
from torch import Tensor

from paraphernalia.torch.modules import (
    Constant,
    SimilarTo,
    SimilarToAny,
    Parallel,
    AdaptiveMultiLoss,
    WeightedSum,
)


def test_constant():
    u = Tensor([0, 1, 2])
    v = Tensor([1, 2, 3])
    m = Constant(v)
    assert torch.equal(v, m(u))


def test_weighted():
    m = WeightedSum(
        one=Constant(Tensor([1, 0, 0, 0])),
        two=Constant(Tensor([0, 1, 0, 0])),
        three=Constant(Tensor([0, 0, 1, 0])),
        four=Constant(Tensor([0, 0, 0, 1])),
    )
    x = Tensor([0])
    assert torch.equal(torch.full((4,), 0.25), m(x))

    m.set_weight("three", 0)
    m.set_weight("four", 0)

    assert torch.equal(Tensor([0.5, 0.5, 0, 0]), m(x))


def test_similar():
    a = Tensor([1, 0, 0]).unsqueeze(0)
    b = Tensor([0, 1, 0]).unsqueeze(0)
    c = Tensor([0, 0, 1]).unsqueeze(0)
    both = torch.cat([a, b])
    assert both.shape == (2, 3)
    trio = torch.cat([a, b, c])

    m = SimilarTo(a)
    assert torch.equal(m(a), Tensor([1.0]))
    assert torch.equal(m(b), Tensor([0.0]))
    assert torch.equal(m(both), Tensor([1.0, 0.0]))
    assert torch.equal(m(trio), Tensor([1.0, 0.0, 0.0]))

    m = SimilarTo(both)
    assert torch.equal(m(a), Tensor([0.5]))
    assert torch.equal(m(b), Tensor([0.5]))
    assert torch.equal(m(both), Tensor([0.5, 0.5]))
    assert torch.equal(m(trio), Tensor([0.5, 0.5, 0.0]))


def test_similar_any():
    a = Tensor([1, 0, 0]).unsqueeze(0)
    b = Tensor([0, 1, 0]).unsqueeze(0)
    c = Tensor([0, 0, 1]).unsqueeze(0)
    both = torch.cat([a, b])
    assert both.shape == (2, 3)
    trio = torch.cat([a, b, c])

    m = SimilarToAny(a)
    assert torch.equal(m(a), Tensor([1.0]))
    assert torch.equal(m(b), Tensor([0.0]))
    assert torch.equal(m(both), Tensor([1.0, 0.0]))
    assert torch.equal(m(trio), Tensor([1.0, 0.0, 0.0]))

    m = SimilarToAny(both)
    assert torch.equal(m(a), Tensor([1.0]))
    assert torch.equal(m(b), Tensor([1.0]))
    assert torch.equal(m(both), Tensor([1.0, 1.0]))
    assert torch.equal(m(trio), Tensor([1.0, 1.0, 0.0]))


def test_multi_loss():
    ml = AdaptiveMultiLoss(2)
    loss = ml(Tensor([1.0, 3.0]))
    assert loss == 2.0


def test_parallel():
    c1 = Tensor([0])
    c2 = Tensor([1])
    m = Parallel([Constant(c1), Constant(c2)])
    result = m()
    assert result == [c1, c2]
