from paraphernalia import jupyter

"""
These functions don't do much outside Jupyter, but they should be safe to invoke.
"""


def test_ding():
    jupyter.ding()


def test_say():
    jupyter.say("Hello world!")
