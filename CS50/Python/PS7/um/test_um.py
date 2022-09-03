from um import count


def test_solo():
    assert count("um") == 1

def test_solo_caps():
    assert count("Um") == 1

def test_solo_punc():
    assert count("um,") == 1

def test_fake():
    assert count("album") == 0

def test_multiple():
    assert count("Um, thanks, um...") == 2

def test_fake_plus_solo():
    assert count("Um, thanks for the album.") == 1

def test_fake_plus_solo():
    assert count("Um? Mum? Is this that album where, um, umm, the clumsy alums play drums?") == 2
