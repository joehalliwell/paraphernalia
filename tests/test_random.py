from paraphernalia._random import get_seed, set_seed


def test_seed():
    s1 = get_seed()
    assert s1 is not None
    assert s1 == get_seed()

    set_seed(s1 + 1)
    assert s1 != get_seed()


def test_negative_seed():
    set_seed(-3612378101266877635)
