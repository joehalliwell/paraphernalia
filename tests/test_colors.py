from paraphernalia import colors


def test_basic():
    b, w = colors.BW.as_rgb()
    assert b[0] < 10 and b[0] >= 0
    assert w[0] > 250 and w[0] <= 255


def test_bw_unit():
    b, w = colors.BW.as_unit_rgb()
    assert b[0] < 0.1 and b[0] >= 0
    assert w[0] > 0.95 and w[0] <= 1.0
