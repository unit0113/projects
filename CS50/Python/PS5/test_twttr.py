from twttr import shorten


def test_lower():
    assert shorten("Hello") == "Hll"

def test_upper():
    assert shorten("HELLO") == "HLL"

def test_alphanumeric():
    assert shorten("123ABCdef") == "123BCdf"

def test_punctuation():
    assert shorten("Hello, World!") == "Hll, Wrld!"